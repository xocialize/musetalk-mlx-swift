// swift-tools-version: 6.0
//
// MLX-Swift port of MuseTalk 1.5 (TMElyralab) — realtime lip-sync via single-step
// latent inpainting. Mirrors the Python musetalk-mlx package (VAE + UNet + pipeline);
// the whisper-tiny audio encoder is the shared WhisperMLX core (added with the pipeline).
// Module/file names track the diffusers reference for 1:1 parity diffing. See CLAUDE.md.
//
import PackageDescription

let package = Package(
    name: "MuseTalk",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "MuseTalk", targets: ["MuseTalk"]),
        .executable(name: "musetalk-cli", targets: ["musetalk-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MuseTalk",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
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
