import ArgumentParser
import Foundation
import MLX
import MuseTalk
#if canImport(Vision)
import CoreGraphics
import ImageIO
#endif

struct CropEntry: Decodable { let frame: String; let coords: [Double] }

@main
struct MuseTalkCLI: AsyncParsableCommand {
    @Option(help: "S0 VAE key-contract gate: published vae.safetensors")
    var vaeKeys: String?

    @Option(help: "S0 UNet key-contract gate: published unet.safetensors")
    var unetKeys: String?

    @Option(help: "S1 VAE forward parity golden (.safetensors with input/moments/recon)")
    var vaeGolden: String?

    @Option(help: "Published vae.safetensors weights (for forward gates)")
    var vaeWeights: String?

    @Option(help: "S1 UNet forward parity golden (.safetensors with latent/audio/pred)")
    var unetGolden: String?

    @Option(help: "Published unet.safetensors weights (for forward gates)")
    var unetWeights: String?

    @Option(help: "Quantize the UNet before load (bits: 8 or 4; group_size 64). Implies GPU.")
    var unetQuant: Int?

    @Option(help: "S2 pipeline parity golden (.safetensors: img/latents/pred/recon); needs --vae-weights")
    var pipelineGolden: String?

    @Option(help: "S2 audio-framing golden (.safetensors: stacked/chunks/librosa_length)")
    var audioChunkGolden: String?

    @Option(help: "80-mel log-mel golden (.safetensors: wav/mel)")
    var melGolden: String?

    @Option(help: "bisenet parity golden (.safetensors: input/feat_out/argmax); needs --bisenet-weights")
    var bisenetGolden: String?

    @Option(help: "Converted bisenet_mlx.safetensors weights")
    var bisenetWeights: String?

    @Option(help: "Vision face-crop validation golden (JSON [{frame, coords:[x1,y1,x2,y2]}])")
    var visionCropGolden: String?

    @Flag(help: "Run on GPU (default CPU = true fp32 parity)")
    var gpu = false

    func relMax(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self) / (MLX.abs(b).max().item(Float.self) + 1e-9)
    }

    func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.asType(.float32).flattened(), bf = b.asType(.float32).flattened()
        let num = (af * bf).sum().item(Float.self)
        let den = Foundation.sqrt((af * af).sum().item(Float.self) * (bf * bf).sum().item(Float.self))
        return num / (den + 1e-9)
    }

    /// Structural gate: weight-free module key set == sanitized on-disk key set (C-S0).
    func keyContractGate(label: String, moduleKeys: Set<String>, weightsPath: String) throws {
        let ckpt = try MuseTalkWeights.checkpointKeySet(url: URL(fileURLWithPath: weightsPath))
        let missing = ckpt.subtracting(moduleKeys)   // on disk, not in module
        let extra = moduleKeys.subtracting(ckpt)      // in module, not on disk
        print("[S0] \(label): module \(moduleKeys.count) keys, checkpoint \(ckpt.count) keys")
        if missing.isEmpty, extra.isEmpty {
            print("[S0] \(label): ✅ exact key-contract match")
        } else {
            print("[S0] \(label): ❌ missing-in-module \(missing.count): \(missing.sorted().prefix(10))")
            print("[S0] \(label):    extra-in-module \(extra.count): \(extra.sorted().prefix(10))")
        }
    }

    func run() async throws {
        try Device.withDefaultDevice(Device(gpu ? .gpu : .cpu)) { try runImpl() }
    }

    func runImpl() throws {
        if let vaeKeys {
            let vae = AutoencoderKL()
            let keys = Set(vae.parameters().flattened().map(\.0))
            try keyContractGate(label: "VAE", moduleKeys: keys, weightsPath: vaeKeys)
        }

        if let unetKeys {
            let unet = UNet2DConditionModel()
            let keys = Set(unet.parameters().flattened().map(\.0))
            try keyContractGate(label: "UNet", moduleKeys: keys, weightsPath: unetKeys)
        }

        if let unetGolden, let unetWeights {
            let unet = UNet2DConditionModel()
            let quant: (Int, Int)? = unetQuant.map { (64, $0) }
            try MuseTalkWeights.load(unet, from: URL(fileURLWithPath: unetWeights), quant: quant)
            let g = try loadArrays(url: URL(fileURLWithPath: unetGolden))
            // latent NCHW (B,8,32,32); audio (B,50,384); timestep 0; pred NCHW (B,4,32,32)
            let pred = unet(g["latent"]!, MLXArray([Int32(0)]), g["audio"]!)
            eval(pred)
            if let bits = unetQuant {
                // S6: quantized matmul runs on GPU (float noise ~1e-3) — gate on cosine vs fp16.
                print(String(format: "[S6] UNet q%d forward    cosine=%.5f  rel=%.3e  shape=%@",
                             bits, cosine(pred, g["pred"]!), relMax(pred, g["pred"]!), "\(pred.shape)"))
            } else {
                print(String(format: "[S1] UNet forward       rel=%.3e  shape=%@", relMax(pred, g["pred"]!), "\(pred.shape)"))
            }
        }

        if let vaeGolden, let vaeWeights {
            let vae = AutoencoderKL()
            try MuseTalkWeights.load(vae, from: URL(fileURLWithPath: vaeWeights))
            let g = try loadArrays(url: URL(fileURLWithPath: vaeGolden))
            // all golden tensors NCHW: input (1,3,256,256), enc_* (1,4,32,32), recon (1,3,256,256)
            let gauss = vae.encode(g["input"]!)
            eval(gauss.mean, gauss.logvar)
            print(String(format: "[S1] VAE encode mean    rel=%.3e", relMax(gauss.mean, g["enc_mean"]!)))
            print(String(format: "[S1] VAE encode logvar  rel=%.3e", relMax(gauss.logvar, g["enc_logvar"]!)))
            let recon = vae.decode(g["latent"]!)
            eval(recon)
            print(String(format: "[S1] VAE decode recon   rel=%.3e", relMax(recon, g["recon"]!)))
        }

        if let pipelineGolden, let vaeWeights {
            let vae = AutoencoderKL()
            try MuseTalkWeights.load(vae, from: URL(fileURLWithPath: vaeWeights))
            let pipe = MuseTalkPipeline(vae: vae, unet: UNet2DConditionModel())
            let g = try loadArrays(url: URL(fileURLWithPath: pipelineGolden))
            // stage 1: img (256,256,3 uint8 BGR) -> 8-ch latent
            let latents = pipe.getLatentsForUnet(g["img"]!)
            eval(latents)
            print(String(format: "[S2] pipeline latents   rel=%.3e", relMax(latents, g["latents"]!)))
            // stage 3: pred (1,4,32,32) -> recon uint8 BGR; gate on integer pixel diff (≤2/255 doctrine)
            let recon = pipe.decodeLatents(g["pred"]!)
            eval(recon)
            let diff = MLX.abs(recon.asType(.int32) - g["recon"]!.asType(.int32))
            print(String(format: "[S2] pipeline recon     max|d|=%d/255  mean|d|=%.4f  shape=%@",
                         diff.max().item(Int32.self), diff.asType(.float32).mean().item(Float.self), "\(recon.shape)"))
        }

        if let bisenetGolden, let bisenetWeights {
            let net = BiSeNet()
            net.train(false)   // BatchNorm: use running stats (eval)
            try MuseTalkWeights.load(net, from: URL(fileURLWithPath: bisenetWeights))
            let g = try loadArrays(url: URL(fileURLWithPath: bisenetGolden))
            let pred = net(g["input"]!)
            eval(pred)
            let maxAbs = MLX.abs(pred - g["feat_out"]!).max().item(Float.self)
            let predArg = argMax(pred, axis: 1).asType(.int32)
            let agree = (predArg .== g["argmax"]!.asType(.int32)).asType(.float32).mean().item(Float.self)
            print(String(format: "[BISENET] feat_out max_abs=%.3e  argmax agree=%.4f%%  shape=%@",
                         maxAbs, agree * 100, "\(pred.shape)"))
        }

        if let visionCropGolden {
            try validateVisionCrop(visionCropGolden)
        }

        if let melGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: melGolden))
            let mel = AudioFeatures.logMel80(g["wav"]!)
            eval(mel)
            print(String(format: "[S2] log-mel 80         rel=%.3e  max|Δ|=%.3e  shape=%@",
                         relMax(mel, g["mel"]!), MLX.abs(mel - g["mel"]!).max().item(Float.self), "\(mel.shape)"))
        }

        if let audioChunkGolden {
            let g = try loadArrays(url: URL(fileURLWithPath: audioChunkGolden))
            let len = Int(g["librosa_length"]!.item(Int32.self))
            let chunks = AudioFeatures.getWhisperChunk(g["stacked"]!, librosaLength: len)
            eval(chunks)
            print(String(format: "[S2] audio chunks       rel=%.3e  shape=%@", relMax(chunks, g["chunks"]!), "\(chunks.shape)"))
        }
    }

    /// Validate the Vision-derived crop box against the dvisual DWPose golden crops (IoU + edge offset).
    func validateVisionCrop(_ path: String) throws {
        #if canImport(Vision)
        let entries = try JSONDecoder().decode([CropEntry].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        var ious: [Double] = []
        var sdx1 = 0.0, sdy1 = 0.0, sdx2 = 0.0, sdy2 = 0.0
        var detected = 0
        for e in entries {
            guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: e.frame) as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let b = FaceCrop.crop(cgImage: img) else { continue }
            detected += 1
            let g = e.coords.map { Int($0) }   // [x1,y1,x2,y2]
            let ix1 = max(b.x1, g[0]), iy1 = max(b.y1, g[1]), ix2 = min(b.x2, g[2]), iy2 = min(b.y2, g[3])
            let inter = Double(max(0, ix2 - ix1) * max(0, iy2 - iy1))
            let union = Double((b.x2 - b.x1) * (b.y2 - b.y1) + (g[2] - g[0]) * (g[3] - g[1])) - inter
            ious.append(union > 0 ? inter / union : 0)
            sdx1 += Double(b.x1 - g[0]); sdy1 += Double(b.y1 - g[1])
            sdx2 += Double(b.x2 - g[2]); sdy2 += Double(b.y2 - g[3])
        }
        let n = Double(max(detected, 1))
        print(String(format: "[VISION-CROP] frames=%d detected=%d (%.1f%%)  mean IoU=%.4f",
                     entries.count, detected, Double(detected) / Double(entries.count) * 100, ious.reduce(0, +) / n))
        print(String(format: "[VISION-CROP] mean signed Δ (vision - golden): x1=%+.1f y1=%+.1f x2=%+.1f y2=%+.1f px",
                     sdx1 / n, sdy1 / n, sdx2 / n, sdy2 / n))
        #else
        print("[VISION-CROP] Vision unavailable on this platform")
        #endif
    }
}
