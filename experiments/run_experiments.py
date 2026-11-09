"""Runs every study used in the report and stores raw results as JSON (experiments/results/)."""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from uamm2 import Config, Simulator                      # noqa: E402
from uamm2.ablation import variants                      # noqa: E402
from uamm2.attention import softmax                      # noqa: E402
from uamm2.codecs import (calibrate, decode, encode, parse_codec,   # noqa: E402
                          perchannel_key_roundtrip, pertoken_key_roundtrip)
from uamm2.config import TierSpec                        # noqa: E402
from uamm2.workload import SyntheticWorkload             # noqa: E402

OUT = ROOT / "experiments" / "results"
SEEDS = (11, 12, 13, 14, 15)


def slim(r):
    keep = {k: v for k, v in r.items() if k != "note"}
    keep["overflow_total"] = sum(r["capacity_overflow_bytes_max"].values())
    return keep


def _job(args):
    cfg, n, seed, want_trace = args
    sim = Simulator(cfg, seed)
    r = slim(sim.run(n))
    if want_trace:
        r["trace"] = sim.trace
    return r


def pmap(jobs):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(_job, jobs))


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj))
    print("saved", name, flush=True)


def main():
    t0 = time.time()
    base = Config()
    V = variants(base)

    # E1: main ablation, 5 seeds
    jobs, keys = [], []
    for name, cfg in V.items():
        for s in SEEDS:
            jobs.append((cfg, 1536, s, False)); keys.append((name, s))
    res = pmap(jobs)
    e1 = {}
    for (name, s), r in zip(keys, res):
        e1.setdefault(name, []).append(r)
    save("e1_ablation", e1)

    # E2: context scaling
    ctxs = (512, 768, 1024, 1536, 2048, 3072)
    sel = {"v1_threshold_fp16": V["v1_threshold_fp16"], "v2_planner_int8_kivi4": V["v2_planner_int8_kivi4"],
           "v2_planner_all_codecs": V["v2_planner_all_codecs"]}
    jobs, keys = [], []
    for name, cfg in sel.items():
        for n in ctxs:
            for s in SEEDS[:3]:
                jobs.append((cfg, n, s, False)); keys.append((name, n))
    res = pmap(jobs)
    e2 = {}
    for (name, n), r in zip(keys, res):
        e2.setdefault(name, {}).setdefault(str(n), []).append(r)
    save("e2_context_scaling", e2)

    # E3: attention block budget
    budgets = (4, 8, 12, 16, 24, 32, 48)
    jobs, keys = [], []
    for b in budgets:
        for s in SEEDS[:3]:
            jobs.append((replace(base, attn_block_budget=b), 1536, s, False)); keys.append(b)
    res = pmap(jobs)
    e3 = {}
    for b, r in zip(keys, res):
        e3.setdefault(str(b), []).append(r)
    save("e3_budget", e3)

    # E4: lambda sweep, hysteresis margin, plan interval, block size, decay
    def sweep(field, values, transform=None):
        jobs, keys = [], []
        for v in values:
            cfg = transform(v) if transform else replace(base, **{field: v})
            for s in SEEDS[:3]:
                jobs.append((cfg, 1536, s, False)); keys.append(v)
        out = {}
        for v, r in zip(keys, pmap(jobs)):
            out.setdefault(str(v), []).append(r)
        return out
    save("e4_lambda", sweep("lambda_quality", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)))
    save("e4_hysteresis", sweep("hysteresis_margin", (0.0, 0.1, 0.25, 0.5, 1.0)))
    save("e4_residency", sweep("min_residency_steps", (0, 8, 16, 32, 64)))
    save("e4_interval", sweep("plan_interval", (2, 4, 8, 16, 32, 64)))
    save("e4_decay", sweep("h2o_decay", (0.5, 0.8, 0.9, 0.95, 0.98, 0.995)))
    save("e4_aggregation", sweep("h2o_aggregation", ("mean", "max", "sum")))

    def bs_t(bs):
        return replace(base, block_size=bs, attn_block_budget=max(1, 24 * 16 // bs))
    save("e4_blocksize", sweep("block_size", (8, 16, 32, 64), bs_t))

    # E5: capacity budget scaling
    def cap_t(f):
        tiers = tuple(TierSpec(t.name, int(t.capacity_bytes * f), t.base_latency_ms, t.bandwidth_gbps)
                      for t in base.tiers)
        return replace(base, tiers=tiers)
    def cap_v1(f):
        c = cap_t(f)
        return replace(c, policy="threshold", codecs=("fp16",), use_entropy_coding=False)
    factors = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    e5 = {"v2": sweep("cap", factors, cap_t), "v1": sweep("cap", factors, cap_v1)}
    save("e5_capacity", e5)

    # E5b: codec set study (which codecs enabled)
    sets = {"fp16": ("fp16",), "fp16+int8": ("fp16", "int8"), "fp16+int8+kivi4": ("fp16", "int8", "kivi4"),
            "all(+kivi2)": ("fp16", "int8", "kivi4", "kivi2"), "kivi4_only": ("fp16", "kivi4"),
            "kivi2_only": ("fp16", "kivi2")}
    jobs, keys = [], []
    for name, cs in sets.items():
        for ent in (False, True):
            for s in SEEDS[:3]:
                jobs.append((replace(base, codecs=cs, use_entropy_coding=ent), 1536, s, False)); keys.append((name, ent))
    e5b = {}
    for (name, ent), r in zip(keys, pmap(jobs)):
        e5b.setdefault(f"{name}|entropy={ent}", []).append(r)
    save("e5b_codecsets", e5b)

    # E6: time-series traces (v1 vs v2) with one seed
    tr = pmap([(V["v1_threshold_fp16"], 1536, 11, True), (V["v2_planner_all_codecs"], 1536, 11, True),
               (V["v2_planner_int8_kivi4"], 1536, 11, True)])
    save("e6_traces", {"v1": tr[0], "v2_all": tr[1], "v2_int8_kivi4": tr[2]})

    # E7: codec micro-study
    rng = np.random.default_rng(0)
    codec_rows = []
    for outl in (False, True):
        for name in ("fp16", "int8", "kivi4", "kivi2", "int8z", "kivi4z", "kivi2z"):
            errs, sizes = [], []
            for _ in range(200):
                d = 64
                scale = np.ones(d)
                if outl:
                    scale[rng.choice(d, 4, replace=False)] = 6.0
                k = (rng.normal(size=(16, d)) * scale).astype(np.float32)
                v = rng.normal(size=(16, d)).astype(np.float32)
                e, s = calibrate(k, v, [name], 32)
                errs.append(e[name]); sizes.append(s[name])
            codec_rows.append({"codec": name, "outliers": outl, "err_mean": float(np.mean(errs)),
                               "err_p95": float(np.percentile(errs, 95)), "bytes": float(np.mean(sizes)),
                               "ratio_vs_fp16": float(np.mean(sizes)) / (16 * 64 * 2 * 2)})
    pc = []
    for bits in (2, 4, 8):
        for outl in (False, True):
            ec, et = [], []
            for _ in range(200):
                d = 64
                scale = np.ones(d)
                if outl:
                    scale[rng.choice(d, 4, replace=False)] = 6.0
                k = (rng.normal(size=(16, d)) * scale).astype(np.float32)
                ec.append(float(((perchannel_key_roundtrip(k, bits) - k) ** 2).sum() / (k ** 2).sum()))
                et.append(float(((pertoken_key_roundtrip(k, bits) - k) ** 2).sum() / (k ** 2).sum()))
            pc.append({"bits": bits, "outliers": outl, "per_channel": float(np.mean(ec)), "per_token": float(np.mean(et))})
    save("e7_codecs", {"codecs": codec_rows, "key_axis": pc})

    # E8: attention structure of the workload (dense oracle)
    d = 64
    w = SyntheticWorkload(d, 11)
    n = 1536
    K = np.zeros((n, d)); cum_curves = []; blk_mass = []; ent = []
    for t in range(n):
        q, k, v = w.step(t); K[t] = k
        if t >= 255 and t % 64 == 0:
            p = softmax(K[:t + 1] @ q / np.sqrt(d))
            s = np.sort(p)[::-1]
            cum_curves.append(np.cumsum(s)[:256].tolist() + [1.0] * max(0, 256 - len(s)))
            nb = (t + 1) // 16
            bm = np.array([p[i * 16:(i + 1) * 16].sum() for i in range(nb)])
            blk_mass.append(np.sort(bm)[::-1][:48].tolist() + [0.0] * max(0, 48 - nb))
            ent.append(float(-(p * np.log(p + 1e-12)).sum()))
    save("e8_attention", {"cum_token": np.mean(cum_curves, axis=0).tolist(),
                          "block_mass_sorted": np.mean(blk_mass, axis=0).tolist(), "entropy": ent})
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
