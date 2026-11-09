"""Bit-exact Python golden models of the RTL in hw/rtl, plus test-vector generation.

Everything here mirrors the integer arithmetic in the Verilog line for line. The RTL testbenches
compare against vectors written by `write_vectors`.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np

W_IMP, W_REC, W_FRQ = 154, 64, 38          # sums to 256 (~ 60% / 25% / 15%)
TH_HOT, TH_WARM, EVICT_P = 174, 89, 243
NBLK, LANES = 64, 16
PRESSURES = (0, 242, 243, 255)


# ------------------------------------------------------------------ scorer
def pack_meta(imp: int, rec: int, frq: int, prot: int) -> int:
    return (prot << 24) | (frq << 16) | (rec << 8) | imp


def scorer_model(imp: int, rec: int, frq: int, prot: int, pressure: int) -> Tuple[int, int, int]:
    """Returns (score, tier, evict). tier: 0 hot, 1 warm, 2 cold."""
    score = (imp * W_IMP + rec * W_REC + frq * W_FRQ) >> 8
    if prot or score >= TH_HOT:
        tier = 0
    elif score >= TH_WARM:
        tier = 1
    else:
        tier = 2
    evict = int((not prot) and pressure >= EVICT_P and score < TH_WARM)
    return score, tier, evict


def _score_to_meta_for(target_score: int, rng) -> Tuple[int, int, int]:
    """Find (imp, rec, frq) whose score == target_score (for boundary vectors)."""
    lo, hi = max(0, target_score - 20), min(255, target_score + 20)
    for _ in range(200000):
        imp = int(rng.integers(lo, hi + 1))
        frq = int(rng.integers(lo, hi + 1))
        base = W_IMP * imp + W_FRQ * frq
        recs = [r for r in range(256) if ((base + W_REC * r) >> 8) == target_score]
        if recs:
            return imp, int(rng.choice(recs)), frq
    raise RuntimeError(f"no vector found for score {target_score}")


def scorer_datasets(seed: int = 3) -> List[List[Tuple[int, int, int, int]]]:
    rng = np.random.default_rng(seed)
    ds = []
    # dataset 0: directed boundaries around every threshold, with and without `protected`
    d0 = []
    for tgt in (0, 1, 88, 89, 90, 173, 174, 175, 254, 255):
        imp, rec, frq = _score_to_meta_for(tgt, rng)
        d0.append((imp, rec, frq, 0))
        d0.append((imp, rec, frq, 1))
    corners = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (1, 1, 1)]
    for c in corners:
        d0.append((*c, 0))
    while len(d0) < NBLK:
        d0.append((*(int(x) for x in rng.integers(0, 256, 3)), 0))
    ds.append(d0[:NBLK])
    # datasets 1..3: random
    for _ in range(3):
        ds.append([(*(int(x) for x in rng.integers(0, 256, 3)), int(rng.random() < 0.2))
                   for _ in range(NBLK)])
    return ds


# ------------------------------------------------------------------ codec
def codec_params(mn: int, mx: int) -> Tuple[int, int, int]:
    rng = mx - mn
    if rng == 0:
        return mn, 0, 0
    inv = min(65535, int(round(15 * 4096 / rng)))
    step = min(65535, int(round(rng * 256 / 15)))
    return mn, inv, step


def quant4_model(x: List[int], zp: int, inv: int) -> List[int]:
    out = []
    for xi in x:
        dx = xi - zp if xi >= zp else 0
        r = (dx * inv + 2048) >> 12
        out.append(min(r, 15))
    return out


def dequant4_model(q: List[int], zp: int, step: int) -> List[int]:
    return [min(255, zp + ((qi * step + 128) >> 8)) for qi in q]


def _pack_nibbles(q: List[int]) -> int:
    return sum(v << (4 * i) for i, v in enumerate(q))


def _pack_bytes(x: List[int]) -> int:
    return sum(v << (8 * i) for i, v in enumerate(x))


def codec_vectors(n: int = 400, seed: int = 5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mode = i % 4
        if mode == 0:       # full range
            x = [int(v) for v in rng.integers(0, 256, LANES)]
        elif mode == 1:     # narrow range around a random center
            c = int(rng.integers(0, 230))
            x = [int(c + v) for v in rng.integers(0, 16, LANES)]
        elif mode == 2:     # constant group (range == 0)
            x = [int(rng.integers(0, 256))] * LANES
        else:               # extremes
            x = [int(v) for v in rng.choice([0, 255, 1, 254], LANES)]
        mn, mx = min(x), max(x)
        zp, inv, step = codec_params(mn, mx)
        # also exercise zp larger than some x (clamp path) on a subset
        if i % 7 == 0 and mx > mn:
            zp = min(255, mn + int(rng.integers(0, max(1, (mx - mn) // 2) + 1)))
        # zp BELOW the true min (stale / conservative statistics): exercises the q>15 clamp
        if i % 5 == 0 and mn > 0:
            zp = max(0, mn - int(rng.integers(1, min(mn, 40) + 1)))
        q = quant4_model(x, zp, inv)
        deq = dequant4_model(q, zp, step)
        rows.append((zp, inv, step, x, q, deq))
    return rows


def write_vectors(out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = scorer_datasets()
    with open(out / "scorer_meta.hex", "w") as f:
        for d in ds:
            for imp, rec, frq, prot in d:
                f.write(f"{pack_meta(imp, rec, frq, prot):07x}\n")
    with open(out / "scorer_exp.hex", "w") as f:
        for p in PRESSURES:
            for d in ds:
                for imp, rec, frq, prot in d:
                    s, t, e = scorer_model(imp, rec, frq, prot, p)
                    f.write(f"{(e << 10) | (t << 8) | s:03x}\n")
    rows = codec_vectors()
    with open(out / "codec_in.hex", "w") as fi, open(out / "codec_exp.hex", "w") as fe:
        for zp, inv, step, x, q, deq in rows:
            fi.write(f"{(zp << 160) | (inv << 144) | (step << 128) | _pack_bytes(x):042x}\n")
            fe.write(f"{(_pack_bytes(deq) << 64) | _pack_nibbles(q):048x}\n")
    with open(out / "meta.txt", "w") as f:
        f.write(f"NBLK={NBLK}\nDS={len(ds)}\nNPRESS={len(PRESSURES)}\nNCODEC={len(rows)}\n")
