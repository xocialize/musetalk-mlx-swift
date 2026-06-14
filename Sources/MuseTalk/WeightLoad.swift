// Weight loading for the published MLX MuseTalk variants (mlx-community/MuseTalk-1.5-*).
//
// The published safetensors come from the Python port's save_native — keys are already
// MLX-native module names and conv weights are already MLX layout (O,H,W,I), so NO
// transpose is needed. The one diffusers quirk to bridge is the FeedForward sparse index:
// the checkpoint stores `...ff.net.0.proj` (GEGLU) + `...ff.net.2` (Linear); our clean
// module names them `ff.geglu.proj` + `ff.linear` (a heterogeneous [GEGLU, Dropout, Linear]
// array is awkward in MLX-Swift — the workspace convention is a load-time key sanitizer).
import Foundation
import MLX
import MLXNN

public enum MuseTalkWeights {
    /// Bridge the diffusers FeedForward sparse index to our clean module key names.
    public static func sanitizeKey(_ key: String) -> String {
        key.replacingOccurrences(of: ".ff.net.0.", with: ".ff.geglu.")
            .replacingOccurrences(of: ".ff.net.2.", with: ".ff.linear.")
    }

    /// Sanitized on-disk key set (what the module must expose), for the S0 structural gate.
    public static func checkpointKeySet(url: URL) throws -> Set<String> {
        let raw = try loadArrays(url: url)
        return Set(raw.keys.map(sanitizeKey))
    }

    /// Load published weights into `module`, enforcing an exact key match (refuse partial loads).
    ///
    /// `quant`: when set, the UNet Linears are quantized (group_size/bits) BEFORE loading so the
    /// module's key set gains the packed `.scales`/`.biases` entries the q8/q4 checkpoints carry.
    /// Quantized weights are packed uint32 (+ fp16 scales/biases) — do NOT dtype-cast them.
    public static func load(_ module: Module, from url: URL, dtype: DType = .float32,
                            quant: (groupSize: Int, bits: Int)? = nil) throws {
        if let quant {
            quantize(model: module, groupSize: quant.groupSize, bits: quant.bits)
        }
        let raw = try loadArrays(url: url)
        let sanitized = raw.map { (sanitizeKey($0.key), quant == nil ? $0.value.asType(dtype) : $0.value) }

        let onDisk = Set(sanitized.map(\.0))
        let expected = Set(module.parameters().flattened().map(\.0))
        let missing = expected.subtracting(onDisk)
        let unused = onDisk.subtracting(expected)
        guard missing.isEmpty, unused.isEmpty else {
            throw MuseTalkError.keyMismatch(missing: missing.sorted(), unused: unused.sorted())
        }

        try module.update(parameters: ModuleParameters.unflattened(sanitized), verify: .none)
        eval(module)
    }
}

public enum MuseTalkError: Error, CustomStringConvertible {
    case keyMismatch(missing: [String], unused: [String])
    public var description: String {
        switch self {
        case let .keyMismatch(missing, unused):
            return "key mismatch — missing \(missing.count): \(missing.prefix(8)) … unused \(unused.count): \(unused.prefix(8))"
        }
    }
}
