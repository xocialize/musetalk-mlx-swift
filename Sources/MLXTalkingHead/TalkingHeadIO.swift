import AVFoundation
import CoreGraphics
import CoreImage
import Foundation
import MLX
import MuseTalk

/// Media I/O + blend for the talkingHead surface — the non-MLX frontier of the pipeline.
///
/// These wrap container/codec + compositing work (AVFoundation video decode/encode, mono-16 kHz PCM
/// for the whisper-tiny encoder, the bisenet-masked paste-back). The neural orchestration in
/// `TalkingHeadPackage.runTalkingHead` is complete and calls them.
///
/// Wired 2026-06-14 (Xcode-agent handoff). Tunable knobs flagged inline: the paste-back blend
/// (bisenet class-set + feather radius) and audio muxing (see `encodeMP4`).
public enum TalkingHeadIO {
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Decode a source video container to per-frame `CGImage`s (AVAssetReader, 32BGRA).
    public static func decodeFrames(_ data: Data) throws -> [CGImage] {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "th-src-\(UUID().uuidString).mp4")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let asset = AVURLAsset(url: url)
        // Synchronous track access — `decodeFrames` is called synchronously from the @InferenceActor
        // `runTalkingHead`, and the asset is a local temp file so tracks are available immediately.
        // (The async `loadTracks` API can't be bridged here without a non-Sendable-closure race.)
        guard let videoTrack = asset.tracks(withMediaType: .video).first else {
            throw TalkingHeadError.notWired("no video track in source")
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw TalkingHeadError.notWired("cannot add video reader output") }
        reader.add(output)
        guard reader.startReading() else {
            throw TalkingHeadError.notWired("video reader start failed: \(String(describing: reader.error))")
        }

        var frames: [CGImage] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let ci = CIImage(cvPixelBuffer: pixelBuffer)
            if let cg = ciContext.createCGImage(ci, from: ci.extent) { frames.append(cg) }
        }
        if reader.status == .failed {
            throw TalkingHeadError.notWired("video read failed: \(String(describing: reader.error))")
        }
        guard !frames.isEmpty else { throw TalkingHeadError.notWired("decoded 0 frames") }
        return frames
    }

    /// Decode audio to a mono 16 kHz Float32 PCM waveform (MLXArray, shape (samples,)).
    public static func decodeAudioPCM(_ data: Data) throws -> MLXArray {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "th-aud-\(UUID().uuidString).wav")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let file = try AVAudioFile(forReading: url)
        guard let srcBuffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else { throw TalkingHeadError.notWired("audio buffer alloc") }
        try file.read(into: srcBuffer)

        // Convert to mono Float32 @ 16 kHz (whisper input rate).
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: file.processingFormat, to: outFormat)
        else { throw TalkingHeadError.notWired("audio converter setup") }

        let ratio = 16_000.0 / file.processingFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(file.length) * ratio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity)
        else { throw TalkingHeadError.notWired("audio out buffer alloc") }

        var fed = false
        var convError: NSError?
        converter.convert(to: outBuffer, error: &convError) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return srcBuffer
        }
        if let convError { throw TalkingHeadError.notWired("audio convert: \(convError.localizedDescription)") }
        guard let ch = outBuffer.floatChannelData else { throw TalkingHeadError.notWired("no PCM channel data") }
        let n = Int(outBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: ch[0], count: n))
        return MLXArray(samples)
    }

    /// 80-mel log-mel spectrogram for the whisper-tiny encoder (parity-gated, rel 1.4e-4 vs the
    /// Python reference). Delegates to the core `AudioFeatures.logMel80`.
    public static func logMel80(_ wav: MLXArray) throws -> MLXArray {
        AudioFeatures.logMel80(wav)
    }

    /// Crop `box` from `frame`, resize to `size`², return BGR uint8 (size×size×3) for the VAE.
    public static func cropResizeBGR(_ frame: CGImage, box: FaceCrop.Box, size: Int) throws -> MLXArray {
        let bx = max(0, box.x1), by = max(0, box.y1)
        let bw = max(1, min(box.x2, frame.width) - bx)
        let bh = max(1, min(box.y2, frame.height) - by)
        guard let cropped = frame.cropping(to: CGRect(x: bx, y: by, width: bw, height: bh)) else {
            throw TalkingHeadError.notWired("crop out of bounds (box \(box), frame \(frame.width)x\(frame.height))")
        }
        let rgba = try drawRGBA(cropped, width: size, height: size)   // size*size*4, RGBA
        var bgr = [UInt8](repeating: 0, count: size * size * 3)
        for p in 0..<(size * size) {
            bgr[p * 3 + 0] = rgba[p * 4 + 2]   // B
            bgr[p * 3 + 1] = rgba[p * 4 + 1]   // G
            bgr[p * 3 + 2] = rgba[p * 4 + 0]   // R
        }
        return MLXArray(bgr).reshaped([size, size, 3])
    }

    /// Composite the regenerated face (`recon`, (1,size,size,3) BGR uint8) back into `frame` at `box`,
    /// feathered by the bisenet face-parse mask (or a soft rectangle when bisenet is absent).
    public static func pasteBack(
        frame: CGImage, recon: MLXArray, box: FaceCrop.Box, bisenet: BiSeNet?
    ) throws -> CGImage {
        let size = recon.dim(1)                                  // 256
        let bx = max(0, box.x1), by = max(0, box.y1)
        let bw = max(1, min(box.x2, frame.width) - bx)
        let bh = max(1, min(box.y2, frame.height) - by)

        // recon BGR uint8 → RGBA bytes at size².
        let reconU8 = recon.reshaped([size, size, 3]).asType(.uint8)
        eval(reconU8)
        let bgr = reconU8.asArray(UInt8.self)

        // Feathered face-blend alpha (1 = use recon, 0 = keep source).
        let mask = faceBlendMask(reconBGR: reconU8, size: size, bisenet: bisenet)

        var maskedRGBA = [UInt8](repeating: 0, count: size * size * 4)
        for p in 0..<(size * size) {
            maskedRGBA[p * 4 + 0] = bgr[p * 3 + 2]   // R
            maskedRGBA[p * 4 + 1] = bgr[p * 3 + 1]   // G
            maskedRGBA[p * 4 + 2] = bgr[p * 3 + 0]   // B
            maskedRGBA[p * 4 + 3] = UInt8((mask[p] * 255).rounded())
        }
        let maskedCG = try cgImage(rgba: maskedRGBA, width: size, height: size)

        // Composite: draw the full source frame, then the masked recon resized into the box rect.
        let outW = frame.width, outH = frame.height
        guard let ctx = CGContext(
            data: nil, width: outW, height: outH, bitsPerComponent: 8, bytesPerRow: outW * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw TalkingHeadError.notWired("paste-back context") }
        ctx.draw(frame, in: CGRect(x: 0, y: 0, width: outW, height: outH))
        // CoreGraphics origin is bottom-left; box is top-left pixel coords → flip y.
        ctx.interpolationQuality = .high
        ctx.draw(maskedCG, in: CGRect(x: bx, y: outH - by - bh, width: bw, height: bh))

        guard let out = ctx.makeImage() else { throw TalkingHeadError.notWired("paste-back makeImage") }
        return out
    }

    /// Encode `frames` to an H.264 mp4 at `fps`.
    /// NOTE (tune, flagged for the external): video-only — `encodeMP4(frames:fps:)` takes no audio, so
    /// the driving speech is NOT muxed. For A/V-synced delivery, either add an `audio:` param (and
    /// update the `runTalkingHead` call) or mux in a wrapper post-step. First runs validate lip motion.
    public static func encodeMP4(frames: [CGImage], fps: Double) throws -> Data {
        guard let first = frames.first else { throw TalkingHeadError.notWired("encode 0 frames") }
        let w = first.width, h = first.height
        let url = FileManager.default.temporaryDirectory
            .appending(path: "th-out-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h])
        guard writer.canAdd(input) else { throw TalkingHeadError.notWired("cannot add video input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw TalkingHeadError.notWired("encode startWriting: \(String(describing: writer.error))")
        }
        writer.startSession(atSourceTime: .zero)

        let timescale = CMTimeScale(600)
        let frameDuration = CMTime(value: CMTimeValue((600.0 / fps).rounded()), timescale: timescale)
        for (i, frame) in frames.enumerated() {
            guard let pool = adaptor.pixelBufferPool else { throw TalkingHeadError.notWired("no pixel buffer pool") }
            let buffer = try pixelBuffer(from: frame, pool: pool, width: w, height: h)
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            guard adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(i))) else {
                throw TalkingHeadError.notWired("encode append frame \(i): \(String(describing: writer.error))")
            }
        }
        input.markAsFinished()
        let sema = DispatchSemaphore(value: 0)
        writer.finishWriting { sema.signal() }
        sema.wait()
        guard writer.status == .completed, FileManager.default.fileExists(atPath: url.path) else {
            throw TalkingHeadError.notWired("encode incomplete: status=\(writer.status.rawValue) err=\(String(describing: writer.error))")
        }
        return try Data(contentsOf: url)
    }

    // MARK: - helpers

    /// Feathered face-blend mask at `size`² in [0,1]. Uses the bisenet face-parse (skin + nose + lips +
    /// mouth of the 19-class CelebAMask map) when available, else a soft rectangle.
    /// TUNE (external inspect→tune loop): the class-set + feather radius are the blend knobs; the
    /// ImageNet normalization mirrors the zllrunning face-parsing pre-proc (CLAUDE.md bisenet note).
    private static func faceBlendMask(reconBGR: MLXArray, size: Int, bisenet: BiSeNet?) -> [Float] {
        if let net = bisenet {
            // recon BGR uint8 [H,W,3] → RGB float, ImageNet-normalized NCHW (1,3,H,W).
            let rgb = take(reconBGR.asType(.float32), MLXArray([Int32(2), 1, 0]), axis: 2) / 255.0
            let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 3])
            let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 3])
            let norm = ((rgb - mean) / std).transposed(2, 0, 1)[.newAxis]      // (1,3,H,W)
            let parse = net(norm).argMax(axis: 1).squeezed()                  // (H,W) Int32
            let faceClasses: [Int32] = [1, 10, 11, 12, 13]                    // skin, nose, mouth, u_lip, l_lip
            var m = MLXArray.zeros([size, size], dtype: .float32)
            for c in faceClasses { m = m + (parse .== c).asType(.float32) }
            eval(m)
            return feather(clip(m, min: 0, max: 1).asArray(Float.self), size: size, radius: max(4, size / 24))
        }
        // soft rectangle: 1 in the interior, feathered to 0 over an inset border.
        var rect = [Float](repeating: 1, count: size * size)
        let inset = max(4, size / 8)
        for y in 0..<size {
            for x in 0..<size {
                let e = Float(min(min(x, size - 1 - x), min(y, size - 1 - y)))
                rect[y * size + x] = min(1, e / Float(inset))
            }
        }
        return rect
    }

    /// Separable box-blur a [size*size] mask to feather edges (`radius` px each axis).
    private static func feather(_ src: [Float], size: Int, radius: Int) -> [Float] {
        if radius <= 0 { return src }
        var tmp = [Float](repeating: 0, count: size * size)
        var out = [Float](repeating: 0, count: size * size)
        let win = Float(2 * radius + 1)
        for y in 0..<size {
            for x in 0..<size {
                var s: Float = 0
                for k in -radius...radius { s += src[y * size + min(size - 1, max(0, x + k))] }
                tmp[y * size + x] = s / win
            }
        }
        for x in 0..<size {
            for y in 0..<size {
                var s: Float = 0
                for k in -radius...radius { s += tmp[min(size - 1, max(0, y + k)) * size + x] }
                out[y * size + x] = s / win
            }
        }
        return out
    }

    /// Draw a CGImage into a `width`×`height` RGBA8 (premultiplied-last) buffer and return the bytes.
    private static func drawRGBA(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var buf = [UInt8](repeating: 0, count: width * height * 4)
        let ok = buf.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { throw TalkingHeadError.notWired("rgba context") }
        return buf
    }

    private static func cgImage(rgba: [UInt8], width: Int, height: Int) throws -> CGImage {
        let cf = CFDataCreate(nil, rgba, rgba.count)!
        guard let provider = CGDataProvider(data: cf),
              let img = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw TalkingHeadError.notWired("cgImage from rgba") }
        return img
    }

    private static func pixelBuffer(from image: CGImage, pool: CVPixelBufferPool, width: Int, height: Int) throws -> CVPixelBuffer {
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let buffer = out else { throw TalkingHeadError.notWired("pixel buffer alloc") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw TalkingHeadError.notWired("pixel buffer context") }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

public enum TalkingHeadError: Error, CustomStringConvertible {
    case notWired(String)
    public var description: String {
        switch self { case let .notWired(what): return "talkingHead I/O: \(what)" }
    }
}
