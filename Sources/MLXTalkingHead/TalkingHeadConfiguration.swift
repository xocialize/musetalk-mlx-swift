import Foundation
import MLXToolKit

/// Init-time configuration for `TalkingHeadPackage` (C9): which published MuseTalk variant +
/// where the component checkpoints live. Per-request source/audio/fps ride the canonical
/// `TalkingHeadRequest`, not here.
///
/// Checkpoint resolution at `load()`:
///   - `modelDirectory` → a resolved MuseTalk dist dir (`{vae,unet,whisper_encoder}.safetensors`
///     + `config.json`); else HF download of `repo`.
///   - `whisperEncoderDirectory` / `bisenetWeights` resolve the shared audio encoder + face parser.
public struct TalkingHeadConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Published MuseTalk variant repo id (also the provenance source).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the chosen variant (fp16 / int8 / int4) — UNet only; VAE+audio stay fp16.
    public var quant: Quant
    /// Realtime batch size for the per-frame UNet pass (datagen-style; 8 ≈ 34 fps).
    public var batchSize: Int

    /// Resolved local MuseTalk checkpoint dir. Environment-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    /// whisper-tiny encoder dir (mlx-examples-naming `weights.safetensors` + `config.json`).
    public var whisperEncoderDirectory: URL?
    /// Converted bisenet face-parser weights (`bisenet_mlx.safetensors`).
    public var bisenetWeights: URL?
    /// Engine-chosen models root (future auto-materialization target). Excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/MuseTalk-1.5-fp16",
        revision: String? = nil,
        quant: Quant = .fp16,
        batchSize: Int = 8,
        modelDirectory: URL? = nil,
        whisperEncoderDirectory: URL? = nil,
        bisenetWeights: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.batchSize = batchSize
        self.modelDirectory = modelDirectory
        self.whisperEncoderDirectory = whisperEncoderDirectory
        self.bisenetWeights = bisenetWeights
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The published int8 variant (UNet quantized; VAE+audio fp16).
    public static var q8: TalkingHeadConfiguration {
        TalkingHeadConfiguration(repo: "mlx-community/MuseTalk-1.5-q8", quant: .int8)
    }

    /// The published int4 variant.
    public static var q4: TalkingHeadConfiguration {
        TalkingHeadConfiguration(repo: "mlx-community/MuseTalk-1.5-q4", quant: .int4)
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, batchSize
    }
}
