import CoreGraphics
import Foundation
import MLX
import MuseTalk

/// Media I/O + blend for the talkingHead surface — the non-MLX frontier of the pipeline.
///
/// These wrap container/codec + compositing work (AVFoundation video decode/encode, an 80-mel
/// log-mel for the whisper-tiny encoder, the bisenet-masked paste-back). They are stubbed pending
/// the in-app validation phase, where the real encode/decode path is wired and exercised end to
/// end; the neural orchestration in `TalkingHeadPackage.runTalkingHead` is complete and calls them.
public enum TalkingHeadIO {
    /// Decode a source video container to per-frame `CGImage`s.
    public static func decodeFrames(_ data: Data) throws -> [CGImage] {
        throw TalkingHeadError.notWired("video decode (AVFoundation)")
    }

    /// Decode audio to a mono 16 kHz PCM waveform (MLXArray, shape (samples,)).
    public static func decodeAudioPCM(_ data: Data) throws -> MLXArray {
        throw TalkingHeadError.notWired("audio decode (AVFoundation)")
    }

    /// 80-mel log-mel spectrogram for the whisper-tiny encoder (parity-gated, rel 1.4e-4 vs the
    /// Python reference). Delegates to the core `AudioFeatures.logMel80`.
    public static func logMel80(_ wav: MLXArray) throws -> MLXArray {
        AudioFeatures.logMel80(wav)
    }

    /// Crop `box` from `frame`, resize to `size`², return normalized BGR uint8 (HxWx3) for the VAE.
    public static func cropResizeBGR(_ frame: CGImage, box: FaceCrop.Box, size: Int) throws -> MLXArray {
        throw TalkingHeadError.notWired("crop+resize (CoreGraphics)")
    }

    /// Composite the regenerated face (`recon`, BGR uint8) back into `frame` at `box`, feathered by
    /// the bisenet face-parse mask (or a soft rectangle when bisenet is absent).
    public static func pasteBack(frame: CGImage, recon: MLXArray, box: FaceCrop.Box, bisenet: BiSeNet?) throws -> CGImage {
        throw TalkingHeadError.notWired("bisenet-masked paste-back")
    }

    /// Encode `frames` to an H.264 mp4 at `fps`.
    public static func encodeMP4(frames: [CGImage], fps: Double) throws -> Data {
        throw TalkingHeadError.notWired("video encode (AVFoundation)")
    }
}

public enum TalkingHeadError: Error, CustomStringConvertible {
    case notWired(String)
    public var description: String {
        switch self { case let .notWired(what): return "talkingHead I/O not yet wired: \(what)" }
    }
}
