"""Export the audio-framing (get_whisper_chunk) golden for the Swift port. Pure array op —
no weights. Derives librosa_length from the golden's num_frames and verifies it reproduces the
stored chunks. Run in the musetalk-mlx venv:
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_audio_chunk_golden.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Volumes/DEV_ARCHIVE/musetalk-mlx")
from musetalk_mlx.whisper.audio2feature import get_whisper_chunk

SRC = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx/goldens/audio_golden.npz")
OUT = Path("goldens/audio_chunk_golden.safetensors")
OUT.parent.mkdir(parents=True, exist_ok=True)

d = np.load(SRC)
stacked = mx.array(d["stacked"].astype(np.float32))   # (1,1500,5,384)
chunks_ref = d["chunks"].astype(np.float32)            # (200,50,384)
num_frames = int(d["num_frames"])
sr, fps = 16000, 25
librosa_length = num_frames * sr // fps                # exact L reproducing num_frames + actual_length

ch = np.array(get_whisper_chunk(stacked, librosa_length, fps=fps))
assert ch.shape == chunks_ref.shape and np.allclose(ch, chunks_ref), \
    f"librosa_length {librosa_length} does not reproduce golden chunks ({ch.shape} vs {chunks_ref.shape})"

mx.save_safetensors(str(OUT), {
    "stacked": stacked,
    "chunks": mx.array(chunks_ref),
    "librosa_length": mx.array(np.int32(librosa_length)),
})
print(f"wrote {OUT}  librosa_length={librosa_length}  chunks {chunks_ref.shape}  (verified == golden)")
