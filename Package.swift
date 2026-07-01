// swift-tools-version: 6.2
//
// MLX-Swift port of MuseTalk 1.5 (TMElyralab) — realtime lip-sync via single-step latent
// inpainting. The `MuseTalk` core (VAE + UNet + pipeline + bisenet + Vision crop) mirrors the
// Python musetalk-mlx package; `MLXTalkingHead` wraps it as an MLXEngine ModelPackage (the
// `talkingHead` capability), consuming the shared WhisperMLX audio encoder. See CLAUDE.md.
//
import PackageDescription

let package = Package(
    name: "MuseTalk",
    platforms: [
        .macOS(.v26),   // matches the MLXEngine contract (MLXToolKit) the wrapper target links
        .iOS(.v17),
    ],
    products: [
        .library(name: "MuseTalk", targets: ["MuseTalk"]),
        .library(name: "MLXTalkingHead", targets: ["MLXTalkingHead"]),
        .executable(name: "musetalk-cli", targets: ["musetalk-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // MLXEngine contract (Foundation-only) for the wrapper target.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.17.0"),
        // Shared whisper-tiny audio encoder core (v0.2.0+ = swift-transformers 1.x, so it
        // co-resolves with the engine / wan-core 1.x ecosystem in the app graph — E12).
        .package(url: "https://github.com/xocialize/whisper-mlx-swift.git", from: "0.2.0"),
    ],
    targets: [
        .target(
            name: "MuseTalk",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            resources: [.copy("Resources/mel_filters_80.safetensors")]
        ),
        // MLXEngine ModelPackage wrapper (the `talkingHead` surface). Engine-aware; the core
        // `MuseTalk` target stays engine-agnostic.
        .target(
            name: "MLXTalkingHead",
            dependencies: [
                "MuseTalk",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "WhisperMLX", package: "whisper-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "musetalk-cli",
            dependencies: [
                "MuseTalk",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
    ]
)
