# musetalk-mlx-swift

MLX-Swift port of **[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk)** (TMElyralab) —
realtime lip-sync via **single-step latent-space inpainting** (one UNet forward per frame, not
diffusion). MIT-licensed, commercial-OK. Swift port of the Python
[musetalk-mlx](https://huggingface.co/mlx-community/MuseTalk-1.5-fp16) (`mlx-community/MuseTalk-1.5-{fp16,q8,q4}`).

> **Status:** neural core ported + parity-locked — VAE and UNet are **bit-exact** vs the Python-MLX
> reference (rel 0.000 on shared weights); quantized UNet matches the published cosine (q8 1.00000,
> q4 0.99984). Pipeline + audio wiring (the shared `WhisperMLX` encoder) and the face-preprocessing
> stages are in progress. See [CLAUDE.md](CLAUDE.md).

## What it does

Given a face crop and driving audio, MuseTalk regenerates the lower-face/mouth region per frame to
match the speech, in a single UNet forward conditioned on whisper-tiny audio features:

```
audio ─► WhisperMLX encoder (5 stacked hidden states, 384-d) ─► chunk[2,2] ─► sinusoidal PE ─┐ (cross-attn)
face  ─► crop 256² ─► VAE.encode(masked) ⊕ VAE.encode(ref) ─► 8-ch latent ─► UNet(t=0) ─► VAE.decode ─► paste
```

## Build

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
xcrun swift build
```

Parity gates are `musetalk-cli` modes; fixtures regenerate from `scripts/capture_*.py` (run in the
Python musetalk-mlx venv). See [CLAUDE.md](CLAUDE.md) for the full gate commands and port notes.

## License

MIT (mirrors upstream). Dependency models keep their own permissive licenses.
