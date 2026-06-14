// MLX-Swift port of the MuseTalk face-parsing BiSeNet (ResNet18 backbone + BiSeNet head).
//
// 1:1 translation of refs/MuseTalk/musetalk/utils/face_parsing/{model.py,resnet.py}. Produces a
// 19-class face-parse map; MuseTalk argmaxes it to build the blend mask for pasting the
// regenerated mouth back. PyTorch port (mlx-porting): BatchNorm eval stats + bilinear
// align_corners are the parity-sensitive ops. NHWC internally, NCHW public (matches the golden).
import Foundation
import MLX
import MLXNN

// --------------------------------------------------------------------------- //
// building blocks
// --------------------------------------------------------------------------- //
final class ConvBNReLU: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm
    init(_ inCh: Int, _ outCh: Int, ks: Int = 3, stride: Int = 1, padding: Int = 1) {
        self._conv.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: IntOrPair(ks), stride: IntOrPair(stride), padding: IntOrPair(padding), bias: false)
        self._bn.wrappedValue = BatchNorm(featureCount: outCh)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(conv(x))) }
}

/// ResNet downsample shortcut: Sequential(Conv1x1/s, BN) -> clean keys conv/bn (converted offline).
final class DownsampleBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm
    init(_ inCh: Int, _ outCh: Int, stride: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 1, stride: IntOrPair(stride), bias: false)
        self._bn.wrappedValue = BatchNorm(featureCount: outCh)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { bn(conv(x)) }
}

final class BasicBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "downsample") var downsample: DownsampleBlock?

    init(_ inCh: Int, _ outCh: Int, stride: Int = 1) {
        self._conv1.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, stride: IntOrPair(stride), padding: 1, bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: outCh)
        self._conv2.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, stride: 1, padding: 1, bias: false)
        self._bn2.wrappedValue = BatchNorm(featureCount: outCh)
        self._downsample.wrappedValue = (inCh != outCh || stride != 1) ? DownsampleBlock(inCh, outCh, stride: stride) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = relu(bn1(conv1(x)))
        residual = bn2(conv2(residual))
        let shortcut = downsample != nil ? downsample!(x) : x
        return relu(shortcut + residual)
    }
}

final class Resnet18: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [BasicBlock]
    @ModuleInfo(key: "layer2") var layer2: [BasicBlock]
    @ModuleInfo(key: "layer3") var layer3: [BasicBlock]
    @ModuleInfo(key: "layer4") var layer4: [BasicBlock]
    // NOTE: mlx-swift Pool applies its `padding` to the wrong axes for NHWC (pads W+C, not H+W),
    // so we pad H,W ourselves with -inf and use padding: 0.
    let maxpool = MaxPool2d(kernelSize: 3, stride: 2, padding: 0)
    private func pad1(_ x: MLXArray) -> MLXArray {
        MLX.padded(x, widths: [IntOrPair(0), IntOrPair(1), IntOrPair(1), IntOrPair(0)],
                   mode: .constant, value: MLXArray(-Float.infinity))
    }

    override init() {
        self._conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 64, kernelSize: 7, stride: 2, padding: 3, bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: 64)
        func layer(_ inCh: Int, _ outCh: Int, stride: Int) -> [BasicBlock] {
            [BasicBlock(inCh, outCh, stride: stride), BasicBlock(outCh, outCh, stride: 1)]
        }
        self._layer1.wrappedValue = layer(64, 64, stride: 1)
        self._layer2.wrappedValue = layer(64, 128, stride: 2)
        self._layer3.wrappedValue = layer(128, 256, stride: 2)
        self._layer4.wrappedValue = layer(256, 512, stride: 2)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var h = maxpool(pad1(relu(bn1(conv1(x)))))
        for b in layer1 { h = b(h) }
        var feat8 = h; for b in layer2 { feat8 = b(feat8) }
        var feat16 = feat8; for b in layer3 { feat16 = b(feat16) }
        var feat32 = feat16; for b in layer4 { feat32 = b(feat32) }
        return (feat8, feat16, feat32)
    }
}

final class AttentionRefinementModule: Module {
    @ModuleInfo(key: "conv") var conv: ConvBNReLU
    @ModuleInfo(key: "conv_atten") var convAtten: Conv2d
    @ModuleInfo(key: "bn_atten") var bnAtten: BatchNorm
    init(_ inCh: Int, _ outCh: Int) {
        self._conv.wrappedValue = ConvBNReLU(inCh, outCh, ks: 3, stride: 1, padding: 1)
        self._convAtten.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 1, bias: false)
        self._bnAtten.wrappedValue = BatchNorm(featureCount: outCh)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feat = conv(x)
        var atten = feat.mean(axes: [1, 2], keepDims: true)   // NHWC global avg pool
        atten = sigmoid(bnAtten(convAtten(atten)))
        return feat * atten
    }
}

final class FeatureFusionModule: Module {
    @ModuleInfo(key: "convblk") var convblk: ConvBNReLU
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    init(_ inCh: Int, _ outCh: Int) {
        self._convblk.wrappedValue = ConvBNReLU(inCh, outCh, ks: 1, stride: 1, padding: 0)
        self._conv1.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh / 4, kernelSize: 1, bias: false)
        self._conv2.wrappedValue = Conv2d(inputChannels: outCh / 4, outputChannels: outCh, kernelSize: 1, bias: false)
        super.init()
    }
    func callAsFunction(_ fsp: MLXArray, _ fcp: MLXArray) -> MLXArray {
        let feat = convblk(MLX.concatenated([fsp, fcp], axis: 3))   // NHWC: concat channels
        var atten = feat.mean(axes: [1, 2], keepDims: true)
        atten = sigmoid(conv2(relu(conv1(atten))))
        return feat * atten + feat
    }
}

final class BiSeNetOutput: Module {
    @ModuleInfo(key: "conv") var conv: ConvBNReLU
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(_ inCh: Int, _ midCh: Int, _ nClasses: Int) {
        self._conv.wrappedValue = ConvBNReLU(inCh, midCh, ks: 3, stride: 1, padding: 1)
        self._convOut.wrappedValue = Conv2d(inputChannels: midCh, outputChannels: nClasses, kernelSize: 1, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { convOut(conv(x)) }
}

final class ContextPath: Module {
    @ModuleInfo(key: "resnet") var resnet: Resnet18
    @ModuleInfo(key: "arm16") var arm16: AttentionRefinementModule
    @ModuleInfo(key: "arm32") var arm32: AttentionRefinementModule
    @ModuleInfo(key: "conv_head32") var convHead32: ConvBNReLU
    @ModuleInfo(key: "conv_head16") var convHead16: ConvBNReLU
    @ModuleInfo(key: "conv_avg") var convAvg: ConvBNReLU

    override init() {
        self._resnet.wrappedValue = Resnet18()
        self._arm16.wrappedValue = AttentionRefinementModule(256, 128)
        self._arm32.wrappedValue = AttentionRefinementModule(512, 128)
        self._convHead32.wrappedValue = ConvBNReLU(128, 128, ks: 3, stride: 1, padding: 1)
        self._convHead16.wrappedValue = ConvBNReLU(128, 128, ks: 3, stride: 1, padding: 1)
        self._convAvg.wrappedValue = ConvBNReLU(512, 128, ks: 1, stride: 1, padding: 0)
        super.init()
    }

    /// nearest 2× (NHWC) — exact for a 512-input crop where every CP stage is a clean ×2.
    private func up2x(_ x: MLXArray) -> MLXArray { repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2) }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (feat8, feat16, feat32) = resnet(x)
        let avg = convAvg(feat32.mean(axes: [1, 2], keepDims: true))               // (B,1,1,128)
        let avgUp = MLX.broadcast(avg, to: [avg.dim(0), feat32.dim(1), feat32.dim(2), avg.dim(3)])
        var feat32Up = up2x(arm32(feat32) + avgUp)
        feat32Up = convHead32(feat32Up)
        var feat16Up = up2x(arm16(feat16) + feat32Up)
        feat16Up = convHead16(feat16Up)
        return (feat8, feat16Up, feat32Up)
    }
}

// --------------------------------------------------------------------------- //
// top-level BiSeNet
// --------------------------------------------------------------------------- //
public final class BiSeNet: Module {
    @ModuleInfo(key: "cp") var cp: ContextPath
    @ModuleInfo(key: "ffm") var ffm: FeatureFusionModule
    @ModuleInfo(key: "conv_out") var convOut: BiSeNetOutput
    @ModuleInfo(key: "conv_out16") var convOut16: BiSeNetOutput
    @ModuleInfo(key: "conv_out32") var convOut32: BiSeNetOutput

    public init(nClasses: Int = 19) {
        self._cp.wrappedValue = ContextPath()
        self._ffm.wrappedValue = FeatureFusionModule(256, 256)
        self._convOut.wrappedValue = BiSeNetOutput(256, 256, nClasses)
        self._convOut16.wrappedValue = BiSeNetOutput(128, 64, nClasses)   // aux (loaded, unused at inference)
        self._convOut32.wrappedValue = BiSeNetOutput(128, 64, nClasses)
        super.init()
    }

    /// xNCHW: (B,3,H,W) normalized -> parse logits NCHW (B,nClasses,H,W).
    public func callAsFunction(_ xNCHW: MLXArray) -> MLXArray {
        let (h, w) = (xNCHW.dim(2), xNCHW.dim(3))
        let x = xNCHW.transposed(0, 2, 3, 1)                              // NHWC
        let (featRes8, featCp8, _) = cp(x)
        let featFuse = ffm(featRes8, featCp8)
        let out = convOut(featFuse)                                       // (B, h/8, w/8, nClasses)
        let scale = Float(h) / Float(out.dim(1))
        let up = Upsample(scaleFactor: .float(scale), mode: .linear(alignCorners: true))(out)
        return up.transposed(0, 3, 1, 2)                                 // NCHW
    }
}
