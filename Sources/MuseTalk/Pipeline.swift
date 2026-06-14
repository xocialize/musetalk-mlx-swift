// MuseTalk MLX pipeline — single-step latent inpainting.
//
// 1:1 translation of musetalk_mlx/pipeline_mlx.py (face-level neural path) +
// whisper/audio2feature.py (audio framing). crop 256² -> VAE.encode(masked)⊕encode(ref)=8ch
// -> UNet(t=0, audio cross-attn) -> VAE.decode -> recon BGR face. Face detect / crop / blend
// are upstream preprocessing wired in by the caller; this owns the neural generation.
import Foundation
import MLX
import MLXFFT
import MLXNN

private let NORM_MEAN: Float = 0.5
private let NORM_STD: Float = 0.5
private func rgbFromBGR() -> MLXArray { MLXArray([Int32(2), 1, 0]) }   // channel reverse (BGR<->RGB)

public struct MuseTalkPipeline {
    public let vae: AutoencoderKL
    public let unet: UNet2DConditionModel
    public let scalingFactor: Float

    public init(vae: AutoencoderKL, unet: UNet2DConditionModel, scalingFactor: Float? = nil) {
        self.vae = vae
        self.unet = unet
        self.scalingFactor = scalingFactor ?? vae.scalingFactor
    }

    // ---- image preprocessing (mirrors preprocess_img / get_mask_tensor) ----

    /// Upper half kept (1), lower half (mouth) masked (0). (size, size).
    static func maskTensor(_ size: Int) -> MLXArray {
        let half = size / 2
        return MLX.concatenated([MLXArray.ones([half, size]), MLXArray.zeros([size - half, size])], axis: 0)
    }

    /// BGR uint8 HxWx3 (256²) -> normalized NCHW float32 (1,3,256,256).
    static func preprocessImg(_ imgBGR: MLXArray, halfMask: Bool, size: Int = MuseTalkConstants.resizedImg) -> MLXArray {
        let rgb = MLX.take(imgBGR.asType(.float32), rgbFromBGR(), axis: 2) / 255.0   // HWC RGB
        var x = rgb.transposed(2, 0, 1)                                              // CHW
        if halfMask {
            x = x * (maskTensor(size) .> 0.5)                                        // broadcast over channel
        }
        x = (x - NORM_MEAN) / NORM_STD
        return x.expandedDimensions(axis: 0)                                         // (1,3,256,256)
    }

    // ---- VAE wrapper (mirrors get_latents_for_unet / decode_latents) ----

    /// 256² BGR face -> 8-ch latent (masked⊕ref). Deterministic: posterior mean.
    public func getLatentsForUnet(_ cropBGR: MLXArray) -> MLXArray {
        let masked = Self.preprocessImg(cropBGR, halfMask: true)
        let ref = Self.preprocessImg(cropBGR, halfMask: false)
        let ml = scalingFactor * vae.encode(masked).mean
        let rl = scalingFactor * vae.encode(ref).mean
        return MLX.concatenated([ml, rl], axis: 1)                                   // (1,8,32,32) NCHW
    }

    /// 4-ch latent -> BGR uint8 (B,256,256,3).
    public func decodeLatents(_ latents: MLXArray) -> MLXArray {
        var img = vae.decode(latents / scalingFactor)                               // NCHW
        img = MLX.clip(img / 2 + 0.5, min: 0, max: 1)
        img = img.transposed(0, 2, 3, 1)                                            // NHWC RGB
        let u8 = MLX.round(img * 255)
        return MLX.take(u8, rgbFromBGR(), axis: 3).asType(.uint8)                   // RGB -> BGR
    }

    /// latentBatch (B,8,32,32) + audioChunks (B,50,384) -> recon BGR uint8 (B,256,256,3).
    public func generateFaces(_ latentBatch: MLXArray, _ audioChunks: MLXArray) -> MLXArray {
        let audio = AudioFeatures.applyPE(audioChunks)
        let pred = unet(latentBatch, MLXArray([Int32(MuseTalkConstants.unetTimestep)]), audio)
        return decodeLatents(pred)
    }
}

// --------------------------------------------------------------------------- //
// audio framing (mirrors whisper/audio2feature.py) — MuseTalk-specific, not WhisperMLX
// --------------------------------------------------------------------------- //
public enum AudioFeatures {
    /// Sinusoidal PE matching musetalk/models/unet.py PositionalEncoding (interleave sin/cos).
    public static func positionalEncoding(_ seqLen: Int, _ dModel: Int = MuseTalkConstants.audioFeatureDim) -> MLXArray {
        let pos = MLXArray(0 ..< seqLen).asType(.float32).reshaped(seqLen, 1)
        let div = MLX.exp(MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32) * (-Foundation.log(10000.0) / Float(dModel)))
        let angles = pos * div.reshaped(1, dModel / 2)
        let inter = MLX.stacked([MLX.sin(angles), MLX.cos(angles)], axis: -1).reshaped(seqLen, dModel)
        return inter.expandedDimensions(axis: 0)                                     // (1, seq, d)
    }

    /// Add sinusoidal PE to (B, seq, d) audio features.
    public static func applyPE(_ x: MLXArray) -> MLXArray {
        x + positionalEncoding(x.dim(1), x.dim(2)).asType(x.dtype)
    }

    // ---- 80-mel log-mel frontend (mirrors whisper/log_mel.py; HF whisper-tiny exact) ----
    private static let nFFT = 400, hop = 160, nSamples = 480_000

    private static func melFilters() -> MLXArray {
        guard let url = Bundle.module.url(forResource: "mel_filters_80", withExtension: "safetensors"),
              let w = try? loadArrays(url: url), let mel = w["mel_filters"] else {
            fatalError("missing mel_filters_80.safetensors resource")
        }
        return mel   // (201, 80)
    }

    private static func hann(_ n: Int) -> MLXArray {
        let k = MLXArray(0 ..< n).asType(.float32)
        return 0.5 - 0.5 * MLX.cos(2.0 * Float.pi * k / Float(n))
    }

    private static func reversedGather(_ x: MLXArray, _ lo: Int, _ count: Int) -> MLXArray {
        // x[lo ..< lo+count] reversed (no flip primitive in mlx-swift)
        let idx = MLXArray((0 ..< count).reversed().map { Int32(lo + $0) })
        return MLX.take(x, idx, axis: 0)
    }

    /// 1-D 16 kHz waveform -> (1, 80, 3000) log-mel. STFT n_fft=400/hop=160, periodic Hann,
    /// center+reflect pad, power, drop the trailing frame, slaney 80-mel filterbank, log10 clamp.
    public static func logMel80(_ audio: MLXArray) -> MLXArray {
        var a = audio.asType(.float32)
        let n = a.dim(0)
        if n > nSamples { a = a[0 ..< nSamples] }
        else if n < nSamples { a = MLX.concatenated([a, MLXArray.zeros([nSamples - n])], axis: 0) }

        // center=True reflect pad by n_fft/2
        let pad = nFFT / 2
        let len = a.dim(0)
        let x = MLX.concatenated([reversedGather(a, 1, pad), a, reversedGather(a, len - pad - 1, pad)], axis: 0)

        let nFrames = 1 + (x.dim(0) - nFFT) / hop
        let cols = MLXArray(0 ..< nFFT).reshaped(1, nFFT)
        let rows = (MLXArray(0 ..< nFrames) * Int32(hop)).reshaped(nFrames, 1)
        let frames = MLX.take(x, cols + rows, axis: 0) * hann(nFFT).reshaped(1, nFFT)   // (nFrames, nFFT)

        let spec = MLXFFT.rfft(frames, axis: -1)                  // (nFrames, 201) complex
        let power = MLX.abs(spec).square()[0 ..< (nFrames - 1)]   // drop last frame -> (3000, 201)
        let mel = power.matmul(melFilters())                     // (3000, 80)
        var log = MLX.log10(MLX.maximum(mel, 1e-10))
        log = MLX.maximum(log, log.max() - 8.0)
        log = (log + 4.0) / 4.0
        return log.transposed(1, 0).expandedDimensions(axis: 0)  // (1, 80, 3000)
    }

    /// stacked (1, seq, nHidden, 384) -> (numFrames, 10*nHidden, 384). Faithful port of
    /// AudioProcessor.get_whisper_chunk.
    public static func getWhisperChunk(_ stacked: MLXArray, librosaLength: Int, fps: Int = 25,
                                       audioFps: Int = 50, sr: Int = 16000,
                                       padLeft: Int = 2, padRight: Int = 2) -> MLXArray {
        let featLenPerFrame = 2 * (padLeft + padRight + 1)                           // 10
        let idxMult = Float(audioFps) / Float(fps)
        let numFrames = Int((Float(librosaLength) / Float(sr) * Float(fps)).rounded(.down))
        let actualLength = Int((Float(librosaLength) / Float(sr) * Float(audioFps)).rounded(.down))
        var wf = stacked[0..., 0 ..< actualLength, 0..., 0...]
        let pad = Int(idxMult.rounded(.up))
        let zerosL = MLXArray.zeros(like: wf[0..., 0 ..< (pad * padLeft), 0..., 0...])
        let zerosR = MLXArray.zeros(like: wf[0..., 0 ..< (pad * 3 * padRight), 0..., 0...])
        wf = MLX.concatenated([zerosL, wf, zerosR], axis: 1)

        var clips: [MLXArray] = []
        for fi in 0 ..< numFrames {
            let ai = Int((Float(fi) * idxMult).rounded(.down))
            clips.append(wf[0..., ai ..< (ai + featLenPerFrame), 0..., 0...])        // (1,10,nHidden,384)
        }
        let prompts = MLX.concatenated(clips, axis: 0)                               // (T,10,nHidden,384)
        let (t, c, h, w) = (prompts.dim(0), prompts.dim(1), prompts.dim(2), prompts.dim(3))
        return prompts.reshaped(t, c * h, w)                                         // (T, 50, 384)
    }
}
