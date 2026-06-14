"""Export the UNet S1 parity golden for the Swift port — same published fp16 weights,
fp32 compute on CPU (cross-validate same-fixture). Run in the musetalk-mlx venv:
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_unet_golden.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Volumes/DEV_ARCHIVE/musetalk-mlx")
mx.set_default_device(mx.cpu)

from musetalk_mlx.models.unet import UNet2DConditionModel
from musetalk_mlx.utils.weights import load_native

DIST = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16")
SRC = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/goldens/unet_golden.npz")
OUT = Path("goldens/unet_golden.safetensors")
OUT.parent.mkdir(parents=True, exist_ok=True)


def cast_fp32(tree):
    if isinstance(tree, dict):
        return {k: cast_fp32(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [cast_fp32(v) for v in tree]
    return tree.astype(mx.float32)


unet = UNet2DConditionModel()
load_native(unet, DIST / "unet.safetensors")       # fp16 (unquantized variant)
unet.update(cast_fp32(unet.parameters()))
mx.eval(unet.parameters())
unet.eval()

d = np.load(SRC)
latent = mx.array(d["latent"].astype(np.float32))   # (1,8,32,32)
audio = mx.array(d["audio"].astype(np.float32))      # (1,50,384)
pred = unet(latent, mx.array([0]), audio)
mx.eval(pred)

mx.save_safetensors(str(OUT), {"latent": latent, "audio": audio, "pred": pred})
print(f"wrote {OUT}  pred {pred.shape}  (same fp16 weights, fp32 compute)")
