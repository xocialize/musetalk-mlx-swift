// MLX-Swift port of the MuseTalk v1.5 UNet (diffusers UNet2DConditionModel, SD1.x topology).
//
// 1:1 translation of musetalk_mlx/models/unet.py. in=8 (masked⊕ref latent), out=4,
// cross_attention_dim=384 (whisper audio), 8 heads everywhere (head_dim = ch/8). Single-step
// inpainting: called at fixed timestep 0, but the time-embedding path still runs. NHWC
// internally, NCHW public. Two GroupNorm eps in one net (M5): resnets/conv_norm_out 1e-5,
// Transformer2DModel.norm 1e-6.
import Foundation
import MLX
import MLXFast
import MLXNN

private let RESNET_EPS = MuseTalkConstants.unetResnetEps
private let TF_GN_EPS = MuseTalkConstants.unetTransformerGroupNormEps
private let N_HEADS = MuseTalkConstants.nHeads

// --------------------------------------------------------------------------- //
// timestep embedding
// --------------------------------------------------------------------------- //
func getTimestepEmbedding(_ timesteps: MLXArray, dim: Int, flipSinToCos: Bool = true,
                          downscaleFreqShift: Float = 1.0, maxPeriod: Float = 10000) -> MLXArray {
    let half = dim / 2
    var exponent = -Foundation.log(maxPeriod) * MLXArray(0 ..< half).asType(.float32)
    exponent = exponent / (Float(half) - downscaleFreqShift)
    let freqs = MLX.exp(exponent)
    let emb = timesteps.asType(.float32).reshaped(timesteps.dim(0), 1) * freqs.reshaped(1, half)
    var out = MLX.concatenated([MLX.sin(emb), MLX.cos(emb)], axis: -1)
    if flipSinToCos {
        out = MLX.concatenated([out[0..., half...], out[0..., ..<half]], axis: -1)
    }
    return out
}

final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ inDim: Int, _ timeDim: Int) {
        self._linear1.wrappedValue = Linear(inDim, timeDim)
        self._linear2.wrappedValue = Linear(timeDim, timeDim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

// --------------------------------------------------------------------------- //
// resnet / sampling
// --------------------------------------------------------------------------- //
final class UNetResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "time_emb_proj") var timeEmbProj: Linear
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(_ inCh: Int, _ outCh: Int, timeDim: Int, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inCh, eps: RESNET_EPS, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, padding: 1)
        self._timeEmbProj.wrappedValue = Linear(timeDim, outCh)
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outCh, eps: RESNET_EPS, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outCh, outputChannels: outCh, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = inCh != outCh
            ? Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 1) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = h + timeEmbProj(silu(temb)).reshaped(temb.dim(0), 1, 1, h.dim(3))   // NHWC broadcast
        h = conv2(silu(norm2(h)))
        let res = convShortcut != nil ? convShortcut!(x) : x
        return res + h
    }
}

final class UNetDownsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, stride: 2, padding: 1)  // symmetric (UNet)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class UNetUpsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ ch: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2))
    }
}

// --------------------------------------------------------------------------- //
// attention / transformer
// --------------------------------------------------------------------------- //
final class CrossAttention: Module {
    let heads: Int
    let dimHead: Int
    let scale: Float
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]   // to_out.0 (to_out.1 = dropout)

    init(queryDim: Int, crossDim: Int? = nil, heads: Int = N_HEADS) {
        let cross = crossDim ?? queryDim
        self.heads = heads
        self.dimHead = queryDim / heads
        self.scale = Foundation.pow(Float(dimHead), -0.5)
        self._toQ.wrappedValue = Linear(queryDim, queryDim, bias: false)
        self._toK.wrappedValue = Linear(cross, queryDim, bias: false)
        self._toV.wrappedValue = Linear(cross, queryDim, bias: false)
        self._toOut.wrappedValue = [Linear(queryDim, queryDim)]
        super.init()
    }

    private func split(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        return x.reshaped(b, n, heads, dimHead).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let ctx = context ?? x
        let q = split(toQ(x)), k = split(toK(ctx)), v = split(toV(ctx))
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let (b, h, n, d) = (out.dim(0), out.dim(1), out.dim(2), out.dim(3))
        return toOut[0](out.transposed(0, 2, 1, 3).reshaped(b, n, h * d))
    }
}

final class GEGLU: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ dimIn: Int, _ dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut * 2)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(proj(x), parts: 2, axis: -1)
        return parts[0] * gelu(parts[1])
    }
}

/// diffusers FeedForward [GEGLU, Dropout, Linear] = net.0 / net.1 / net.2. Dropout (net.1) is
/// parameter-free; our clean names geglu/linear are bridged from net.0/net.2 at load (sanitizer).
final class FeedForward: Module {
    @ModuleInfo(key: "geglu") var geglu: GEGLU
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ dim: Int, mult: Int = 4) {
        let inner = dim * mult
        self._geglu.wrappedValue = GEGLU(dim, inner)
        self._linear.wrappedValue = Linear(inner, dim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(geglu(x)) }
}

final class BasicTransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn1") var attn1: CrossAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn2") var attn2: CrossAttention
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "ff") var ff: FeedForward

    init(_ dim: Int, crossDim: Int, heads: Int = N_HEADS) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._attn1.wrappedValue = CrossAttention(queryDim: dim, crossDim: nil, heads: heads)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim)
        self._attn2.wrappedValue = CrossAttention(queryDim: dim, crossDim: crossDim, heads: heads)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim)
        self._ff.wrappedValue = FeedForward(dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ context: MLXArray) -> MLXArray {
        var h = attn1(norm1(x)) + x
        h = attn2(norm2(h), context: context) + h
        h = ff(norm3(h)) + h
        return h
    }
}

final class Transformer2DModel: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projIn: Conv2d
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [BasicTransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Conv2d

    init(_ ch: Int, crossDim: Int, nBlocks: Int = 1, groups: Int = 32) {
        self._norm.wrappedValue = GroupNorm(groupCount: groups, dimensions: ch, eps: TF_GN_EPS, pytorchCompatible: true)
        self._projIn.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
        self._transformerBlocks.wrappedValue = (0 ..< nBlocks).map { _ in BasicTransformerBlock(ch, crossDim: crossDim) }
        self._projOut.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ context: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let res = x
        var y = projIn(norm(x)).reshaped(b, h * w, c)
        for blk in transformerBlocks { y = blk(y, context) }
        y = projOut(y.reshaped(b, h, w, c))
        return y + res
    }
}

// --------------------------------------------------------------------------- //
// down / up / mid blocks
// --------------------------------------------------------------------------- //
final class DownBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [UNetResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Transformer2DModel]?
    @ModuleInfo(key: "downsamplers") var downsamplers: [UNetDownsample2D]?

    init(_ inCh: Int, _ outCh: Int, timeDim: Int, nLayers: Int, addDownsample: Bool, crossDim: Int?) {
        self._resnets.wrappedValue = (0 ..< nLayers).map { UNetResnetBlock2D($0 == 0 ? inCh : outCh, outCh, timeDim: timeDim) }
        self._attentions.wrappedValue = crossDim != nil
            ? (0 ..< nLayers).map { _ in Transformer2DModel(outCh, crossDim: crossDim!) } : nil
        self._downsamplers.wrappedValue = addDownsample ? [UNetDownsample2D(outCh)] : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray, _ context: MLXArray) -> (MLXArray, [MLXArray]) {
        var h = x
        var res: [MLXArray] = []
        for (i, resnet) in resnets.enumerated() {
            h = resnet(h, temb)
            if let attentions { h = attentions[i](h, context) }
            res.append(h)
        }
        if let downsamplers {
            h = downsamplers[0](h)
            res.append(h)
        }
        return (h, res)
    }
}

final class UpBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [UNetResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Transformer2DModel]?
    @ModuleInfo(key: "upsamplers") var upsamplers: [UNetUpsample2D]?

    init(_ inCh: Int, prevCh: Int, _ outCh: Int, timeDim: Int, nLayers: Int, addUpsample: Bool, crossDim: Int?) {
        var blocks: [UNetResnetBlock2D] = []
        for i in 0 ..< nLayers {
            let resSkip = i == nLayers - 1 ? inCh : outCh
            let resIn = i == 0 ? prevCh : outCh
            blocks.append(UNetResnetBlock2D(resIn + resSkip, outCh, timeDim: timeDim))
        }
        self._resnets.wrappedValue = blocks
        self._attentions.wrappedValue = crossDim != nil
            ? (0 ..< nLayers).map { _ in Transformer2DModel(outCh, crossDim: crossDim!) } : nil
        self._upsamplers.wrappedValue = addUpsample ? [UNetUpsample2D(outCh)] : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ resList: [MLXArray], _ temb: MLXArray, _ context: MLXArray) -> MLXArray {
        var h = x
        for (i, resnet) in resnets.enumerated() {
            h = MLX.concatenated([h, resList[i]], axis: -1)   // NHWC: concat channels
            h = resnet(h, temb)
            if let attentions { h = attentions[i](h, context) }
        }
        if let upsamplers { h = upsamplers[0](h) }
        return h
    }
}

final class UNetMidBlock2DCrossAttn: Module {
    @ModuleInfo(key: "resnets") var resnets: [UNetResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Transformer2DModel]

    init(_ ch: Int, timeDim: Int, crossDim: Int) {
        self._resnets.wrappedValue = [UNetResnetBlock2D(ch, ch, timeDim: timeDim), UNetResnetBlock2D(ch, ch, timeDim: timeDim)]
        self._attentions.wrappedValue = [Transformer2DModel(ch, crossDim: crossDim)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray, _ context: MLXArray) -> MLXArray {
        var h = resnets[0](x, temb)
        h = attentions[0](h, context)
        h = resnets[1](h, temb)
        return h
    }
}

// --------------------------------------------------------------------------- //
// top-level UNet
// --------------------------------------------------------------------------- //
public final class UNet2DConditionModel: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "time_embedding") var timeEmbedding: TimestepEmbedding
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownBlock2D]
    @ModuleInfo(key: "mid_block") var midBlock: UNetMidBlock2DCrossAttn
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    let inDim: Int
    let flipSinToCos: Bool

    public init(_ cfg: UNetConfig = UNetConfig()) {
        let boc = cfg.blockOutChannels
        let nLayers = cfg.layersPerBlock
        let crossDim = cfg.crossAttentionDim
        let timeDim = boc[0] * 4
        self.inDim = boc[0]
        self.flipSinToCos = cfg.flipSinToCos

        self._convIn.wrappedValue = Conv2d(inputChannels: cfg.inChannels, outputChannels: boc[0], kernelSize: 3, padding: 1)
        self._timeEmbedding.wrappedValue = TimestepEmbedding(boc[0], timeDim)

        var downs: [DownBlock2D] = []
        var outCh = boc[0]
        for i in 0 ..< boc.count {
            let inCh = outCh
            outCh = boc[i]
            downs.append(DownBlock2D(inCh, outCh, timeDim: timeDim, nLayers: nLayers,
                                     addDownsample: i != boc.count - 1,
                                     crossDim: cfg.downHasCross[i] ? crossDim : nil))
        }
        self._downBlocks.wrappedValue = downs

        self._midBlock.wrappedValue = UNetMidBlock2DCrossAttn(boc[boc.count - 1], timeDim: timeDim, crossDim: crossDim)

        var ups: [UpBlock2D] = []
        let rev = Array(boc.reversed())            // [1280,1280,640,320]
        outCh = rev[0]
        for i in 0 ..< rev.count {
            let prevCh = outCh
            outCh = rev[i]
            let inCh = rev[Swift.min(i + 1, boc.count - 1)]
            ups.append(UpBlock2D(inCh, prevCh: prevCh, outCh, timeDim: timeDim, nLayers: nLayers + 1,
                                 addUpsample: i != boc.count - 1,
                                 crossDim: cfg.upHasCross[i] ? crossDim : nil))
        }
        self._upBlocks.wrappedValue = ups

        self._convNormOut.wrappedValue = GroupNorm(groupCount: cfg.normNumGroups, dimensions: boc[0], eps: RESNET_EPS, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: boc[0], outputChannels: cfg.outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    /// sampleNCHW: (B,8,32,32); timesteps: (B,) or (1,); encoderHiddenStates: (B,50,384). -> (B,4,32,32)
    public func callAsFunction(_ sampleNCHW: MLXArray, _ timesteps: MLXArray, _ encoderHiddenStates: MLXArray) -> MLXArray {
        var tEmb = getTimestepEmbedding(timesteps, dim: inDim, flipSinToCos: flipSinToCos, downscaleFreqShift: 1.0)
        tEmb = tEmb.asType(sampleNCHW.dtype)
        let temb = timeEmbedding(tEmb)

        var x = convIn(sampleNCHW.transposed(0, 2, 3, 1))    // NCHW -> NHWC
        var resSamples: [MLXArray] = [x]
        for down in downBlocks {
            let (nx, res) = down(x, temb, encoderHiddenStates)
            x = nx
            resSamples.append(contentsOf: res)
        }

        x = midBlock(x, temb, encoderHiddenStates)

        for up in upBlocks {
            let n = up.resnets.count
            let res = Array(resSamples.suffix(n).reversed())
            resSamples.removeLast(n)
            x = up(x, res, temb, encoderHiddenStates)
        }

        x = convOut(silu(convNormOut(x)))
        return x.transposed(0, 3, 1, 2)                       // NHWC -> NCHW
    }
}
