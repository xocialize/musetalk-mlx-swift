import CoreGraphics
import Foundation
import MLX
import MLXToolKit
import MuseTalk
import WhisperMLX

/// MLXEngine package: MuseTalk 1.5 (audio-driven lip-sync) exposing the canonical `talkingHead`
/// surface. One loaded unit = the MuseTalk VAE+UNet pipeline + the shared WhisperMLX audio
/// encoder + the bisenet face-parser; Vision supplies the per-frame face crop.
///
/// Engine-owned lifecycle (C13): constructed from a `TalkingHeadConfiguration`, paged in by
/// `load()`, driven by `run(_:)`, reclaimed by `unload()`. Lifecycle is `InferenceActor`-isolated;
/// the non-`Sendable` MLX models are actor-isolated state. Cancellation is honored per frame-batch.
@InferenceActor
public final class TalkingHeadPackage: ModelPackage {
    public typealias Configuration = TalkingHeadConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // MuseTalk weights are MIT; this port code is MIT (mirrors upstream).
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "mlx-community/MuseTalk-1.5-fp16",
                revision: "main",
                tier: 3
            ),
            requirements: RequirementsManifest(
                // Measured peak (Python port): ~7 GB fp16 at bs=8 realtime; declare just above.
                // Quant variants shrink the UNet only (VAE+audio fp16) — slightly lower peak.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 8_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 7_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 6_500_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .pro
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.5),
            ],
            surfaces: [
                TalkingHeadContract.descriptor(
                    name: "musetalk-talkinghead",
                    summary: "Audio-driven lip-sync (MuseTalk 1.5, MLX). Re-renders the mouth of a "
                        + "source face video to match driving speech via single-step latent "
                        + "inpainting — realtime on Apple Silicon. A still portrait is driven by a "
                        + "looped source video.",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// Resident components, paged in by `load()`.
    private var pipeline: MuseTalkPipeline?
    private var whisper: Whisper?
    private var whisperPosEmb: MLXArray?
    private var bisenet: BiSeNet?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let dir = try await resolveModelDirectory()

        // MuseTalk VAE + UNet (UNet quantized for q8/q4; VAE always fp16/fp32).
        let vae = AutoencoderKL()
        try MuseTalkWeights.load(vae, from: dir.appendingPathComponent("vae.safetensors"))
        let unet = UNet2DConditionModel()
        let q: (groupSize: Int, bits: Int)? = {
            switch configuration.quant {
            case .int8: return (64, 8)
            case .int4: return (64, 4)
            default: return nil
            }
        }()
        try MuseTalkWeights.load(unet, from: dir.appendingPathComponent("unet.safetensors"), quant: q)
        pipeline = MuseTalkPipeline(vae: vae, unet: unet)

        // Shared whisper-tiny encoder (mlx-examples-naming dir) + its stored embed_positions.
        if let whisperDir = configuration.whisperEncoderDirectory {
            let model = try WhisperLoader.fromDirectory(whisperDir)
            whisper = model
            // HF embed_positions (if shipped alongside) matches the oracle; else computed sinusoids.
            let posURL = whisperDir.appendingPathComponent("embed_positions.safetensors")
            whisperPosEmb = (try? loadArrays(url: posURL))?["embed_positions"]
        }

        // bisenet face-parser (blend mask).
        if let bisenetURL = configuration.bisenetWeights {
            let net = BiSeNet()
            net.train(false)
            try MuseTalkWeights.load(net, from: bisenetURL)
            bisenet = net
        }
    }

    public func unload() async {
        pipeline = nil
        whisper = nil
        whisperPosEmb = nil
        bisenet = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .talkingHead:
            guard let req = request as? TalkingHeadRequest else {
                throw PackageError.configurationMismatch(
                    expected: "TalkingHeadRequest", got: String(describing: type(of: request)))
            }
            return try await runTalkingHead(req, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surface

    private func runTalkingHead(_ request: TalkingHeadRequest, pipeline: MuseTalkPipeline) async throws
        -> TalkingHeadResponse
    {
        try Task.checkCancellation()
        guard let whisper else { throw PackageError.notLoaded }

        // 1. decode source frames + driving audio -> per-frame audio cross-attn chunks.
        let frames = try TalkingHeadIO.decodeFrames(request.source.data)
        let fps = request.fps ?? request.source.frameRate ?? 25
        let wav = try TalkingHeadIO.decodeAudioPCM(request.audio.data)
        let mel = try TalkingHeadIO.logMel80(wav)                              // (1,80,3000*)
        let stacked = whisper.encoderHiddenStates(mel, positionEmbedding: whisperPosEmb)
        let chunks = AudioFeatures.getWhisperChunk(stacked, librosaLength: wav.dim(0), fps: Int(fps))

        // 2. per-frame: Vision crop -> latent -> UNet(t=0, audio) -> decode -> bisenet blend -> paste.
        let n = min(frames.count, chunks.dim(0))
        var out: [CGImage] = []
        out.reserveCapacity(n)
        for i in 0 ..< n {
            try Task.checkCancellation()
            let frame = frames[i]
            guard let box = FaceCrop.crop(cgImage: frame) else { out.append(frame); continue }
            let cropBGR = try TalkingHeadIO.cropResizeBGR(frame, box: box, size: MuseTalkConstants.resizedImg)
            let latent = pipeline.getLatentsForUnet(cropBGR)
            let chunk = chunks[i ... i]                                        // (1,50,384)
            let recon = pipeline.generateFaces(latent, chunk)                  // (1,256,256,3) BGR uint8
            eval(recon)
            let blended = try TalkingHeadIO.pasteBack(frame: frame, recon: recon, box: box, bisenet: bisenet)
            out.append(blended)
        }

        try Task.checkCancellation()
        let mp4 = try TalkingHeadIO.encodeMP4(frames: out, fps: fps)
        return TalkingHeadResponse(
            video: Video(format: .mp4, data: mp4, durationSeconds: Double(n) / fps, frameRate: fps))
    }

    /// Resolution: explicit `modelDirectory` → HF download of `repo`.
    private func resolveModelDirectory() async throws -> URL {
        if let dir = configuration.modelDirectory { return dir }
        throw PackageError.configurationMismatch(
            expected: "modelDirectory (HF auto-download pending)", got: "nil")
    }
}

extension TalkingHeadPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(TalkingHeadPackage.self)
    }
}
