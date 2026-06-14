// Frozen upstream configs — the oracle. Mirrors musetalk_mlx/config.py.
// Captured from MuseTalk pinned commit 0a89dec + the component HF repos. Do NOT "improve".
import Foundation

/// stabilityai/sd-vae-ft-mse — AutoencoderKL (diffusers 0.4.2).
public struct VAEConfig: Sendable {
    public var blockOutChannels: [Int] = [128, 256, 512, 512]
    public var inChannels = 3
    public var outChannels = 3
    public var latentChannels = 4
    public var layersPerBlock = 2
    public var normNumGroups = 32
    public init() {}
}

/// musetalkV15/musetalk.json — UNet2DConditionModel (SD1.x topology, diffusers 0.6.0.dev0).
public struct UNetConfig: Sendable {
    public var blockOutChannels: [Int] = [320, 640, 1280, 1280]
    public var inChannels = 8                  // 4 masked-target latent ⊕ 4 reference latent
    public var outChannels = 4
    public var crossAttentionDim = 384         // whisper-tiny feature dim (NOT 768 text)
    public var layersPerBlock = 2
    public var normNumGroups = 32
    public var flipSinToCos = true
    public var freqShift = 0.0
    // down/up block cross-attn flags (CrossAttnDownBlock2D ×3 + DownBlock2D; UpBlock2D + CrossAttn ×3)
    public var downHasCross: [Bool] = [true, true, true, false]
    public var upHasCross: [Bool] = [false, true, true, true]
    public init() {}
}

public enum MuseTalkConstants {
    public static let vaeScalingFactor: Float = 0.18215
    public static let resizedImg = 256         // face crop size
    public static let latentSize = 32          // 256 / 8
    public static let unetTimestep = 0         // single-step inpainting: fixed t=0
    public static let audioFeatureDim = 384    // whisper-tiny encoder hidden

    /// diffusers eps quirks (lessons M3/M5): VAE GroupNorm 1e-6; UNet resnets 1e-5 but the
    /// Transformer2DModel pre-proj GroupNorm is 1e-6 — two different eps within one UNet.
    public static let vaeGroupNormEps: Float = 1e-6
    public static let unetResnetEps: Float = 1e-5
    public static let unetTransformerGroupNormEps: Float = 1e-6
    public static let nHeads = 8               // constant across the UNet (head_dim = ch/8)
}
