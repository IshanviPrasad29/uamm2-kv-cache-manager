import json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
def load(n): return json.loads((RES / f"{n}.json").read_text())
def mean(rs, k): return float(np.mean([r[k] for r in rs]))
def std(rs, k): return float(np.std([r[k] for r in rs]))
def ms(rs, k, f="{:.3g}"): return (f + " ± " + f).format(mean(rs, k), std(rs, k))
MODELS = {"TinyLlama-like": (22, 4, 64), "Llama-3-8B-like": (32, 8, 128), "7B MHA-style": (32, 32, 128), "Llama-3-70B-like": (80, 8, 128)}
def bpt(L, H, D, b=2): return 2 * L * H * D * b
