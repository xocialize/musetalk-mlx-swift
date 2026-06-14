"""Export the VAE S1 parity golden for the Swift port — SAME published fp16 weights,
fp32 compute on CPU, so the gate isolates Swift-vs-Python *implementation* parity from
fp16 weight-rounding (cross-validate same-fixture; the skill's S1 doctrine).

Run in the musetalk-mlx venv:
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_vae_golden.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Volumes/DEV_ARCHIVE/musetalk-mlx")
mx.set_default_device(mx.cpu)  # true fp32 reference

from musetalk_mlx.models.vae import AutoencoderKL
from musetalk_mlx.utils.weights import load_native

DIST = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16")
SRC = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/goldens/vae_golden.npz")
OUT = Path("goldens/vae_golden.safetensors")
OUT.parent.mkdir(parents=True, exist_ok=True)


def cast_fp32(tree):
    if isinstance(tree, dict):
        return {k: cast_fp32(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [cast_fp32(v) for v in tree]
    return tree.astype(mx.float32)


vae = AutoencoderKL()
load_native(vae, DIST / "vae.safetensors")        # fp16 published weights
vae.update(cast_fp32(vae.parameters()))           # -> fp32 compute (match Swift loader)
mx.eval(vae.parameters())
vae.eval()

d = np.load(SRC)
img = mx.array(d["img"].astype(np.float32))        # NCHW (1,3,256,256)
latent = mx.array(d["dec_latent"].astype(np.float32))

gauss = vae.encode(img)
mean, logvar = gauss.mean, gauss.logvar
recon = vae.decode(latent)
mx.eval(mean, logvar, recon)

mx.save_safetensors(str(OUT), {
    "input": img, "enc_mean": mean, "enc_logvar": logvar,
    "latent": latent, "recon": recon,
})
print(f"wrote {OUT}  (same fp16 weights, fp32 compute)")
