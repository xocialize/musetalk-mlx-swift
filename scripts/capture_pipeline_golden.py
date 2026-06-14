"""Export the pipeline S2 parity golden for the Swift port — same published fp16 VAE weights,
fp32 compute on CPU. Stage 1 = img -> get_latents_for_unet; stage 3 = pred -> decode_latents
(stage 2 UNet is separately bit-exact gated). Run in the musetalk-mlx venv:
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_pipeline_golden.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Volumes/DEV_ARCHIVE/musetalk-mlx")
mx.set_default_device(mx.cpu)

from musetalk_mlx.models.vae import AutoencoderKL
from musetalk_mlx.pipeline_mlx import MuseTalkPipeline
from musetalk_mlx.utils.weights import load_native

DIST = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/dist/MuseTalk-1.5-MLX-fp16")
SRC = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/goldens/pipeline_golden.npz")
OUT = Path("goldens/pipeline_golden.safetensors")
OUT.parent.mkdir(parents=True, exist_ok=True)


def cast_fp32(tree):
    if isinstance(tree, dict):
        return {k: cast_fp32(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [cast_fp32(v) for v in tree]
    return tree.astype(mx.float32)


vae = AutoencoderKL()
load_native(vae, DIST / "vae.safetensors")
vae.update(cast_fp32(vae.parameters()))
mx.eval(vae.parameters())
vae.eval()
pipe = MuseTalkPipeline(vae, None, None, scaling_factor=0.18215)

src = np.load(SRC)
img = src["img"]                                     # (256,256,3) uint8 BGR
pred = mx.array(src["pred"].astype(np.float32))      # (1,4,32,32)
latents = pipe.get_latents_for_unet(img)             # stage 1
recon = pipe.decode_latents(pred)                    # stage 3 -> np uint8 BGR (1,256,256,3)
mx.eval(latents)

mx.save_safetensors(str(OUT), {
    "img": mx.array(img),                            # uint8
    "latents": latents,                              # (1,8,32,32)
    "pred": pred,                                    # (1,4,32,32)
    "recon": mx.array(recon),                        # uint8 (1,256,256,3)
})
print(f"wrote {OUT}  latents {latents.shape}  recon {recon.shape}  (same fp16 weights, fp32)")
