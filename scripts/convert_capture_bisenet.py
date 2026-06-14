"""Convert the MuseTalk bisenet face-parser (79999_iter.pth) to MLX-native safetensors AND
capture a PyTorch parity golden for the Swift port.

- Weights -> clean MLX-native keys (downsample.0/.1 -> downsample.conv/.bn, drop
  num_batches_tracked, conv (O,I,H,W)->(O,H,W,I)) so the Swift loader is an exact match.
- Golden: fixed numpy input injected into torch BiSeNet -> feat_out logits + argmax parse map.

Run in the musetalk-mlx venv:
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/convert_capture_bisenet.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

MUSETALK = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx")
sys.path.insert(0, str(MUSETALK / "refs" / "MuseTalk" / "musetalk" / "utils"))

# legacy-tar checkpoints (resnet18 backbone + 79999_iter) need weights_only=False (torch 2.6+ flip, M12)
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from face_parsing.model import BiSeNet  # noqa: E402

WEIGHTS = MUSETALK / "weights" / "face-parse-bisent"
OUT_W = WEIGHTS / "bisenet_mlx.safetensors"
OUT_G = Path("goldens/bisenet_golden.safetensors")
OUT_G.parent.mkdir(parents=True, exist_ok=True)

# --- load torch model ---
net = BiSeNet(str(WEIGHTS / "resnet18-5c106cde.pth"))
net.load_state_dict(torch.load(WEIGHTS / "79999_iter.pth", map_location="cpu", weights_only=False))
net.eval()

# --- convert weights -> clean MLX-native keys ---
out = {}
for k, v in net.state_dict().items():
    if k.endswith("num_batches_tracked"):
        continue
    nk = k.replace("downsample.0.", "downsample.conv.").replace("downsample.1.", "downsample.bn.")
    w = v.detach().cpu().float().numpy()
    if w.ndim == 4:                                  # conv (O,I,H,W) -> (O,H,W,I)
        w = np.transpose(w, (0, 2, 3, 1))
    out[nk] = mx.array(w)
mx.eval(list(out.values()))
mx.save_safetensors(str(OUT_W), out)
print(f"wrote {len(out)} tensors -> {OUT_W}")

# --- capture golden (fixed input injected into torch) ---
rng = np.random.default_rng(0)
inp = (rng.standard_normal((1, 3, 512, 512)).astype(np.float32) * 1.5)   # ~normalized-image range
with torch.no_grad():
    feat_out = net(torch.from_numpy(inp))[0]          # (1,19,512,512) post-bilinear logits
feat = feat_out.numpy().astype(np.float32)
argmax = feat.argmax(axis=1).astype(np.int32)         # (1,512,512) parse map

mx.save_safetensors(str(OUT_G), {
    "input": mx.array(inp),                            # NCHW (1,3,512,512)
    "feat_out": mx.array(feat),                        # NCHW (1,19,512,512)
    "argmax": mx.array(argmax),                        # (1,512,512)
})
print(f"wrote golden input {inp.shape} feat_out {feat.shape} -> {OUT_G}")
