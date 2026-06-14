# CLAUDE.md — musetalk-mlx-swift

MLX-Swift port of **MuseTalk 1.5** (TMElyralab) — realtime lip-sync via single-step latent
inpainting (NOT diffusion: one UNet forward per frame at fixed t=0). Module `MuseTalk`. Mirrors
the Python `musetalk-mlx` package (`/Volumes/DEV_ARCHIVE/musetalk-mlx`, published
`mlx-community/MuseTalk-1.5-{fp16,q8,q4}`). Part of the MuseTalk → MLXEngine `talkingHead` port.

## Ground rules
- **Port = transpose, not redesign.** 1:1 with `musetalk_mlx/models/{vae,unet}.py` (diffusers-
  isomorphic). A reader should diff Swift vs Python and see only syntax.
- **Run correctness parity on CPU** (`--gpu` opts in); GPU fp32 matmul is tf32-like. Quantized
  forwards are GPU-only (gate on cosine, not rel).
- Build with `xcrun swift build` (full Xcode toolchain); gates are `musetalk-cli` modes — plain
  `swift run` does GPU inference fine (metallib resolves from the products dir).

## Components & status (2026-06-14)
| Net | Source | Gate |
|---|---|---|
| VAE (`AutoencoderKL`, SD1.x 4-ch) | `sd-vae-ft-mse` | S0 key-contract 248/248 ✅ · S1 forward **bit-exact** ✅ |
| UNet (`UNet2DConditionModel`, 8→4ch, cross=384, t=0) | `musetalkV15/unet.pth` | S0 686/686 ✅ · S1 **bit-exact** ✅ · q8 cosine **1.00000** / q4 **0.99984** ✅ |
| Pipeline (face path + audio framing) | `pipeline_mlx.py` + `audio2feature.py` | S2 img→latents rel 0.000 · pred→recon **max\|Δ\|=0/255** · getWhisperChunk rel 0.000 ✅ |
| Whisper-tiny audio encoder | shared `WhisperMLX` core (v0.1.0) | gated there (1.5e-5); composed by transitivity, wired at the engine wrapper (where the mlx-swift graph resolves) |

S1 parity is **bit-exact** (rel 0.000) because both sides are `mlx::core` — gated against the
Python-MLX reference on the *same published fp16 weights, fp32 compute* (cross-validate same-
fixture; isolates implementation parity from fp16 weight-rounding). Quant cosine matches the
published `unet_cosine_vs_fp16` exactly.

## Port specifics / gotchas
- **Published weights are MLX-native** (`save_native`): conv layout already (O,H,W,I), keys already
  MLX module names — **no transpose** in the loader.
- **diffusers `ff.net` sparse index**: checkpoint has `ff.net.0.proj` (GEGLU) + `ff.net.2` (Linear);
  a heterogeneous `[GEGLU,Dropout,Linear]` array is awkward in MLX-Swift, so clean names
  `ff.geglu`/`ff.linear` are bridged by a **load-time key sanitizer** (`MuseTalkWeights.sanitizeKey`).
  `to_out.0` IS dense → native `[Linear]`. (Convention lifted from `qwen-image-edit-swift`.)
- **Two GroupNorm eps in the UNet** (M5): resnets/conv_norm_out 1e-5, `Transformer2DModel.norm`
  1e-6. VAE GroupNorm is 1e-6 throughout (M3). `pytorchCompatible: true` everywhere.
- `attention_head_dim: 8` is **num-heads** (M4): 8 heads, head_dim = ch/8.
- VAE downsample = asymmetric pad (bottom/right) then stride-2 pad-0; UNet downsample = symmetric
  stride-2 pad-1. attn `to_q/k/v` have NO bias; `to_out.0` does (M6).
- **Quant**: `quantize(model:groupSize:64,bits:)` before load; packed uint32 weights must NOT be
  dtype-cast. The loader's exact key-match doubles as a structural check on the quant scope
  (184 Linears / 1054 keys).

## Gates (CPU fp32 unless noted)
```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
DIST=/Volumes/DEV_ARCHIVE/musetalk-mlx/dist
# fixtures (musetalk-mlx venv):
/Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_vae_golden.py
/Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_unet_golden.py
# S0 key contract
xcrun swift run musetalk-cli --vae-keys $DIST/MuseTalk-1.5-MLX-fp16/vae.safetensors
xcrun swift run musetalk-cli --unet-keys $DIST/MuseTalk-1.5-MLX-fp16/unet.safetensors
# S1 forward parity
xcrun swift run musetalk-cli --vae-golden goldens/vae_golden.safetensors --vae-weights $DIST/MuseTalk-1.5-MLX-fp16/vae.safetensors
xcrun swift run musetalk-cli --unet-golden goldens/unet_golden.safetensors --unet-weights $DIST/MuseTalk-1.5-MLX-fp16/unet.safetensors
# S6 quant (GPU)
xcrun swift run musetalk-cli --gpu --unet-golden goldens/unet_golden.safetensors --unet-weights $DIST/MuseTalk-1.5-MLX-q8/unet.safetensors --unet-quant 8
```

## Deps / version note
mlx-swift `from: 0.30.0` (resolves 0.31.4), matching `qwen-image-edit-swift`. The shared
`WhisperMLX` core is pinned at 0.21.0 — **reconcile to one mlx-swift graph at wrapper link time**
(the talkingHead package will depend on both).

## Next
Neural pipeline complete + bit-exact. Remaining: face preprocessing ports (S3FD/DWPose/bisenet,
faithful — each needs a PyTorch→MLX pass, no Python-MLX donor) → `talkingHead` engine wrapper
(adds the WhisperMLX dep + reconciles the mlx-swift graph; wires encoder→getWhisperChunk).
