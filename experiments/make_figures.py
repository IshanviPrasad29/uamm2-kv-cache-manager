"""Data figures for the report: everything here is drawn from experiments/results/*.json, the VCD
dumps in docs/wave/, the Yosys stat files in docs/synth/ and captured tool output in docs/captures/."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "experiments" / "results"
FIG = ROOT / "docs" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

PAL = {"v1": "#c53030", "v2fp16": "#718096", "v2i8": "#222222", "v2all": "#2f855a", "drop": "#b7791f",
       "t0": "#2f855a", "t1": "#222222", "t2": "#6b46c1", "t3": "#c05621"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def ms(rs, k):
    v = np.array([r[k] for r in rs], float)
    return v.mean(), v.std()


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig", name)


# ===================================================================== analytic (problem statement)
MODELS = {"TinyLlama-like (22L, 4 KV heads, d=64)": (22, 4, 64),
          "Llama-3-8B-like (32L, 8 KV heads, d=128)": (32, 8, 128),
          "7B MHA-style (32L, 32 KV heads, d=128)": (32, 32, 128),
          "Llama-3-70B-like (80L, 8 KV heads, d=128)": (80, 8, 128)}


def kv_bytes_per_token(L, H, D, bytes_per=2):
    return 2 * L * H * D * bytes_per


def f_kv_growth():
    ctx = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 1024
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (name, (L, H, D)), col in zip(MODELS.items(), ["#2f855a", "#222222", "#6b46c1", "#c53030"]):
        ax.plot(ctx, kv_bytes_per_token(L, H, D) * ctx / 2 ** 30, marker="o", ms=3.5, label=name, color=col)
    for gb, lab in ((2, "2 GB"), (4, "4 GB"), (8, "8 GB")):
        ax.axhline(gb, color="gray", ls="--", lw=0.8)
        ax.text(1100, gb * 1.08, f"{lab} free DRAM", fontsize=7.5, color="gray")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("KV cache size, FP16 (GiB, one sequence)")
    ax.set_title("KV cache growth with context length (batch = 1)")
    ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "f01_kv_growth.png")


def f_kv_bandwidth():
    ctx = np.array([1, 2, 4, 8, 16, 32, 64, 128]) * 1024
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bw = 12.8e9
    for (name, (L, H, D)), col in zip(MODELS.items(), ["#2f855a", "#222222", "#6b46c1", "#c53030"]):
        ax.plot(ctx, bw / (kv_bytes_per_token(L, H, D) * ctx), marker="o", ms=3.5, label=name, color=col)
    ax.axhline(10, color="gray", ls="--", lw=0.8); ax.text(1100, 11, "10 tokens/s", fontsize=7.5, color="gray")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("upper bound on decode tokens/s from KV reads alone")
    ax.set_title("Dense attention reads the whole KV cache every token\n(12.8 GB/s DRAM, weights and compute ignored)")
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "f02_kv_bandwidth_bound.png")


def f_kv_compression_effect():
    ctx = np.array([4, 8, 16, 32, 64, 128]) * 1024
    L, H, D = MODELS["Llama-3-8B-like (32L, 8 KV heads, d=128)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    base = kv_bytes_per_token(L, H, D) * ctx / 2 ** 30
    for lab, ratio, col in (("FP16", 1.0, "#c53030"), ("INT8 (0.5x)", 0.5, "#222222"),
                            ("4-bit KIVI-style (~0.34x measured here)", 0.344, "#2f855a"),
                            ("2-bit KIVI-style (~0.22x measured here)", 0.219, "#6b46c1")):
        ax.plot(ctx, base * ratio, marker="o", ms=3.5, label=lab, color=col)
    ax.axhline(4, color="gray", ls="--", lw=0.8); ax.text(4200, 4.2, "4 GB budget", fontsize=7.5, color="gray")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)"); ax.set_ylabel("KV size (GiB)")
    ax.set_title("Effect of KV quantization on the memory wall (Llama-3-8B-like geometry)")
    ax.legend(fontsize=7.5)
    save(fig, "f03_quant_effect_memory_wall.png")


def f_coverage_matrix():
    methods = ["PagedAttention/vLLM", "H2O", "StreamingLLM", "SnapKV", "Quest", "InfiniGen", "KIVI", "KVQuant",
               "Palu (low-rank)", "KV-Compress", "EvicPress", "TensorRT-LLM KV compression", "taitashaw engine",
               "Ara / RVV", "MLC LLM", "UAMM-2 (this work)"]
    caps = ["paged blocks", "importance\nscoring", "token\neviction", "query-aware\nselection", "KV\nquantization",
            "multi-tier\nplacement", "joint codec +\nplacement", "edge HW\ncodec", "vector\ncompute", "portable\nruntime"]
    # 2 = primary contribution, 1 = partial / supported, 0 = not addressed (from the cited abstracts / docs)
    M = np.array([
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # PagedAttention
        [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # H2O
        [0, 1, 2, 0, 0, 0, 0, 0, 0, 0],   # StreamingLLM
        [0, 2, 2, 0, 0, 0, 0, 0, 0, 0],   # SnapKV
        [0, 1, 0, 2, 0, 0, 0, 0, 0, 0],   # Quest
        [0, 1, 0, 2, 0, 2, 0, 0, 0, 0],   # InfiniGen
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],   # KIVI
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],   # KVQuant
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],   # Palu
        [1, 1, 2, 0, 0, 0, 0, 0, 0, 0],   # KV-Compress (block eviction in paged cache)
        [0, 0, 2, 0, 2, 2, 2, 0, 0, 0],   # EvicPress
        [1, 0, 2, 0, 2, 2, 1, 0, 0, 0],   # TRT-LLM
        [0, 0, 2, 0, 2, 0, 0, 2, 0, 0],   # taitashaw
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],   # Ara
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 2],   # MLC (weight quant; paged KV in engine - not verified)
        [2, 2, 1, 2, 2, 2, 2, 1, 2, 1],   # UAMM-2
    ])
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=2.4, aspect="auto")
    ax.set_xticks(range(len(caps))); ax.set_xticklabels(caps, fontsize=7.5)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=8)
    ax.xaxis.tick_top()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, ["", "o", "#"][M[i, j]], ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] == 2 else "black")
    ax.set_xticks(np.arange(-.5, len(caps)), minor=True); ax.set_yticks(np.arange(-.5, len(methods)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_title("Capability coverage of related work   (# = main contribution, o = partial/supported)", pad=40, fontsize=10)
    save(fig, "f04_coverage_matrix.png")


# ===================================================================== results
def f_ablation():
    e1 = load("e1_ablation")
    names = ["v1_threshold_fp16", "v2_planner_fp16", "v2_planner_int8_kivi4", "v2_planner_all_codecs"]
    labels = ["UAMM-1\n(threshold, fp16)", "planner\nfp16 only", "planner\nint8+kivi4", "planner\nall codecs"]
    cols = [PAL["v1"], PAL["v2fp16"], PAL["v2i8"], PAL["v2all"]]
    panels = [("resident_vs_fp16", "resident bytes / FP16 bytes"), ("dropped_tokens", "tokens permanently dropped"),
              ("attn_rel_l2_mean", "attention-output rel. L2 error (mean)"),
              ("attn_mass_recall_mean", "attention-mass recall"),
              ("sim_fetch_ms_per_token", "modelled fetch ms / token"), ("overflow_total", "tier-budget overflow (bytes)")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (k, t) in zip(axs.ravel(), panels):
        m = [ms(e1[n], k)[0] for n in names]; s = [ms(e1[n], k)[1] for n in names]
        ax.bar(range(4), m, yerr=s, color=cols, capsize=3)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(t, fontsize=9)
        for i, v in enumerate(m):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Ablation at 1536 tokens (mean +/- std over 5 seeds)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "f05_ablation.png")


def f_context_scaling():
    e2 = load("e2_context_scaling")
    ctxs = ["512", "768", "1024", "1536", "2048", "3072"]
    series = [("v1_threshold_fp16", "UAMM-1", PAL["v1"]), ("v2_planner_int8_kivi4", "planner int8+kivi4", PAL["v2i8"]),
              ("v2_planner_all_codecs", "planner all codecs", PAL["v2all"])]
    panels = [("resident_vs_fp16", "resident / FP16"), ("dropped_tokens", "dropped tokens"),
              ("attn_rel_l2_mean", "rel. L2 error"), ("attn_mass_recall_mean", "attention-mass recall"),
              ("sim_fetch_ms_per_token", "modelled fetch ms/token"), ("overflow_total", "budget overflow (bytes)")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (k, t) in zip(axs.ravel(), panels):
        for key, lab, col in series:
            m = [ms(e2[key][c], k)[0] for c in ctxs]; s = [ms(e2[key][c], k)[1] for c in ctxs]
            ax.errorbar([int(c) for c in ctxs], m, yerr=s, marker="o", ms=3.5, label=lab, color=col, capsize=2)
        ax.set_title(t, fontsize=9); ax.set_xlabel("context (tokens)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Context-length scaling with the tier budget fixed at 168 KB (3 seeds)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "f06_context_scaling.png")


def f_budget():
    e3 = load("e3_budget")
    e8 = load("e8_attention")
    bms = np.array(e8["block_mass_sorted"])
    budgets = [4, 8, 12, 16, 24, 32, 48]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    rec = [ms(e3[str(b)], "attn_mass_recall_mean")[0] for b in budgets]
    ideal = [bms[:b].sum() for b in budgets]
    axs[0].plot(budgets, rec, "o-", color=PAL["v2all"], label="min/max bound selection (measured)")
    axs[0].plot(budgets, ideal, "s--", color="gray", label="oracle best-B blocks (avg of sampled steps, ctx 256-1536)")
    axs[0].set_xlabel("blocks fetched per step (budget B)"); axs[0].set_ylabel("attention-mass recall"); axs[0].legend(fontsize=6.5)
    axs[0].set_title("Selection quality vs budget", fontsize=9)
    axs[1].plot(budgets, [ms(e3[str(b)], "attn_rel_l2_mean")[0] for b in budgets], "o-", color=PAL["v1"])
    axs[1].set_xlabel("budget B"); axs[1].set_ylabel("rel. L2 error"); axs[1].set_title("Output error vs budget", fontsize=9)
    axs[2].plot(budgets, [ms(e3[str(b)], "sim_fetch_ms_per_token")[0] for b in budgets], "o-", color=PAL["t3"])
    axs[2].set_xlabel("budget B"); axs[2].set_ylabel("modelled fetch ms/token"); axs[2].set_title("Fetch cost vs budget", fontsize=9)
    fig.tight_layout()
    save(fig, "f07_budget_sweep.png")


def f_lambda():
    e = load("e4_lambda")
    ls = list(e.keys())
    x = [float(l) for l in ls]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (k, t) in zip(axs, [("attn_rel_l2_mean", "rel. L2 error"), ("resident_vs_fp16", "resident / FP16"),
                                ("sim_fetch_ms_per_token", "fetch ms/token"), ("flash_block_reads", "flash block reads")]):
        m = [ms(e[l], k)[0] for l in ls]; s = [ms(e[l], k)[1] for l in ls]
        ax.errorbar(range(len(ls)), m, yerr=s, marker="o", ms=4, color=PAL["v2i8"], capsize=2)
        ax.set_xticks(range(len(ls))); ax.set_xticklabels([f"{v:g}" for v in x], fontsize=7)
        ax.set_xlabel("lambda_quality"); ax.set_title(t, fontsize=9)
    fig.suptitle("Quality-penalty weight lambda: the quality / latency / memory trade-off", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "f08_lambda_sweep.png")
    # pareto
    fig, ax = plt.subplots(figsize=(6, 4.2))
    err = [ms(e[l], "attn_rel_l2_mean")[0] for l in ls]; fet = [ms(e[l], "sim_fetch_ms_per_token")[0] for l in ls]
    sc = ax.scatter(fet, err, c=range(len(ls)), cmap="viridis", s=60)
    for l, a, b in zip(ls, fet, err):
        ax.annotate(f"lambda={float(l):g}", (a, b), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("modelled fetch ms/token"); ax.set_ylabel("rel. L2 error"); ax.set_title("Error vs fetch-latency Pareto view (lambda sweep)")
    save(fig, "f09_lambda_pareto.png")


def f_stability():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    e = load("e4_hysteresis"); ks = list(e)
    ax = axs[0]
    ax.bar(range(len(ks)), [ms(e[k], "migrations")[0] for k in ks], color=PAL["t2"], label="tier migrations")
    ax.bar(range(len(ks)), [ms(e[k], "reencodes")[0] for k in ks], bottom=[ms(e[k], "migrations")[0] for k in ks],
           color=PAL["t3"], label="re-encodes")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks); ax.set_xlabel("hysteresis margin"); ax.set_title("Churn vs hysteresis margin", fontsize=9); ax.legend(fontsize=7)
    e = load("e4_residency"); ks = list(e)
    axs[1].plot(range(len(ks)), [ms(e[k], "migrations")[0] for k in ks], "o-", color=PAL["t2"], label="migrations")
    axs[1].set_ylabel("migrations")
    a2 = axs[1].twinx(); a2.plot(range(len(ks)), [ms(e[k], "sim_fetch_ms_per_token")[0] for k in ks], "s--", color=PAL["t3"], label="fetch ms")
    a2.set_ylabel("fetch ms/token"); a2.grid(False)
    axs[1].set_xticks(range(len(ks))); axs[1].set_xticklabels(ks); axs[1].set_xlabel("min residency (steps)"); axs[1].set_title("Residency: churn vs latency", fontsize=9)
    e = load("e4_interval"); ks = list(e)
    axs[2].plot(range(len(ks)), [ms(e[k], "flash_block_reads")[0] for k in ks], "o-", color=PAL["t3"], label="flash reads")
    axs[2].set_ylabel("flash block reads")
    a3 = axs[2].twinx(); a3.plot(range(len(ks)), [ms(e[k], "migrations")[0] for k in ks], "s--", color=PAL["t2"]); a3.set_ylabel("migrations"); a3.grid(False)
    axs[2].set_xticks(range(len(ks))); axs[2].set_xticklabels(ks); axs[2].set_xlabel("plan interval (steps)"); axs[2].set_title("Replan frequency", fontsize=9)
    fig.tight_layout()
    save(fig, "f10_stability_knobs.png")


def f_h2o_knobs():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    e = load("e4_decay"); ks = list(e)
    axs[0].errorbar(range(len(ks)), [ms(e[k], "attn_rel_l2_mean")[0] for k in ks], yerr=[ms(e[k], "attn_rel_l2_mean")[1] for k in ks], marker="o", color=PAL["v2all"], capsize=2)
    axs[0].set_ylabel("rel. L2 error")
    a = axs[0].twinx(); a.plot(range(len(ks)), [ms(e[k], "resident_vs_fp16")[0] for k in ks], "s--", color=PAL["t3"]); a.set_ylabel("resident / FP16"); a.grid(False)
    axs[0].set_xticks(range(len(ks))); axs[0].set_xticklabels(ks); axs[0].set_xlabel("H2O decay"); axs[0].set_title("Importance memory length", fontsize=9)
    e = load("e4_aggregation"); ks = list(e)
    axs[1].bar(range(len(ks)), [ms(e[k], "attn_rel_l2_mean")[0] for k in ks], color=PAL["v2i8"])
    axs[1].set_xticks(range(len(ks))); axs[1].set_xticklabels(ks); axs[1].set_ylabel("rel. L2 error"); axs[1].set_title("Token -> block aggregation", fontsize=9)
    e = load("e4_blocksize"); ks = list(e)
    axs[2].plot(range(len(ks)), [ms(e[k], "attn_rel_l2_mean")[0] for k in ks], "o-", color=PAL["v1"], label="error")
    axs[2].set_ylabel("rel. L2 error")
    a = axs[2].twinx(); a.plot(range(len(ks)), [ms(e[k], "migrations")[0] for k in ks], "s--", color=PAL["t2"]); a.set_ylabel("migrations"); a.grid(False)
    axs[2].set_xticks(range(len(ks))); axs[2].set_xticklabels(ks); axs[2].set_xlabel("block size (tokens)"); axs[2].set_title("Block granularity", fontsize=9)
    fig.tight_layout()
    save(fig, "f11_h2o_blocksize_knobs.png")


def f_capacity():
    e = load("e5_capacity")
    fs = ["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for pol, lab, col in (("v1", "UAMM-1", PAL["v1"]), ("v2", "UAMM-2 planner (all codecs)", PAL["v2all"])):
        for ax, (k, t) in zip(axs, [("attn_rel_l2_mean", "rel. L2 error"), ("dropped_tokens", "tokens dropped"),
                                    ("sim_fetch_ms_per_token", "modelled fetch ms/token")]):
            ax.errorbar([float(f) for f in fs], [ms(e[pol][f], k)[0] for f in fs], yerr=[ms(e[pol][f], k)[1] for f in fs],
                        marker="o", ms=4, color=col, label=lab, capsize=2)
            ax.set_xlabel("tier budget factor (1.0 = 168 KB)"); ax.set_title(t, fontsize=9)
    axs[0].legend(fontsize=7)
    fig.suptitle("Memory-budget sweep at 1536 tokens (3 seeds)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "f12_capacity_sweep.png")


def f_codecsets():
    e = load("e5b_codecsets")
    keys = list(e.keys())
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (k, t) in zip(axs, [("resident_vs_fp16", "resident / FP16"), ("attn_rel_l2_mean", "rel. L2 error"),
                                ("overflow_total", "budget overflow (bytes)")]):
        m = [ms(e[kk], k)[0] for kk in keys]
        cols = [PAL["v2i8"] if "True" in kk else PAL["v2fp16"] for kk in keys]
        ax.barh(range(len(keys)), m, color=cols)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([kk.replace("|entropy=", "  z=") for kk in keys], fontsize=7)
        ax.set_title(t, fontsize=9)
    fig.suptitle("Which codecs are enabled (blue: zlib on flash; gray: off)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "f13_codec_sets.png")


def f_timeseries():
    tr = load("e6_traces")
    v2 = tr["v2_all"]["trace"]; v1 = tr["v1"]["trace"]
    pts = [r for r in v2 if "tier_bytes" in r]
    t = [r["t"] for r in pts]
    tb = np.array([r["tier_bytes"] for r in pts]) / 1024
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4))
    ax = axs[0, 0]
    ax.stackplot(t, tb.T, labels=["T0 SRAM", "T1 DRAM hot", "T2 DRAM warm", "T3 flash"], colors=[PAL["t0"], PAL["t1"], PAL["t2"], PAL["t3"]])
    ax.plot(t, np.array([min((r["t"] + 1) * 64 * 2 * 2, 1e9) for r in pts]) / 1024, "k--", lw=1, label="FP16 KV size")
    ax.set_title("UAMM-2: bytes resident per tier over time (KiB)", fontsize=9); ax.set_xlabel("decode step"); ax.legend(fontsize=7, loc="upper left")
    ax = axs[0, 1]
    names = sorted({c for r in pts for c in r["codecs"]})
    cnt = np.array([[r["codecs"].get(c, 0) for c in names] for r in pts])
    ax.stackplot(t, cnt.T, labels=names)
    ax.set_title("UAMM-2: blocks per codec over time", fontsize=9); ax.set_xlabel("decode step"); ax.legend(fontsize=7, loc="upper left")
    ax = axs[1, 0]
    def roll(x, w=64):
        x = np.array(x, float); c = np.cumsum(np.insert(x, 0, 0)); return (c[w:] - c[:-w]) / w
    for lab, tr_, col in (("UAMM-1", v1, PAL["v1"]), ("UAMM-2 all codecs", v2, PAL["v2all"])):
        ax.plot(roll([r["fetch_ms"] for r in tr_]), color=col, label=lab)
    ax.set_title("Modelled fetch ms/token (64-step rolling mean)", fontsize=9); ax.set_xlabel("decode step"); ax.legend(fontsize=7)
    ax = axs[1, 1]
    for lab, tr_, col in (("UAMM-1", v1, PAL["v1"]), ("UAMM-2 all codecs", v2, PAL["v2all"])):
        ax.plot(roll([r["recall"] for r in tr_]), color=col, label=lab)
    ax.set_title("Attention-mass recall (64-step rolling mean)", fontsize=9); ax.set_xlabel("decode step"); ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f14_timeseries_v2.png")

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    p1 = [r for r in v1 if "dropped_tokens" in r]
    axs[0].plot([r["t"] for r in p1], [r["dropped_tokens"] for r in p1], color=PAL["v1"], label="UAMM-1")
    axs[0].plot([r["t"] for r in pts], [r["dropped_tokens"] for r in pts], color=PAL["v2all"], label="UAMM-2")
    axs[0].set_title("Cumulative tokens dropped", fontsize=9); axs[0].set_xlabel("decode step"); axs[0].legend(fontsize=7)
    axs[1].plot([r["t"] for r in p1], [r["migrations"] for r in p1], color=PAL["v1"], label="UAMM-1 migrations")
    axs[1].plot([r["t"] for r in pts], [r["migrations"] for r in pts], color=PAL["v2all"], label="UAMM-2 migrations")
    axs[1].plot([r["t"] for r in pts], [r["reencodes"] for r in pts], color=PAL["t3"], ls="--", label="UAMM-2 re-encodes")
    axs[1].set_title("Cumulative migrations / re-encodes", fontsize=9); axs[1].set_xlabel("decode step"); axs[1].legend(fontsize=7)
    axs[2].plot(roll([r["flash_reads"] for r in v1]), color=PAL["v1"], label="UAMM-1")
    axs[2].plot(roll([r["flash_reads"] for r in v2]), color=PAL["v2all"], label="UAMM-2")
    axs[2].set_title("Flash block reads per step (rolling mean)", fontsize=9); axs[2].set_xlabel("decode step"); axs[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f15_timeseries_churn.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for lab, tr_, col in (("UAMM-1", v1, PAL["v1"]), ("UAMM-2 all codecs", v2, PAL["v2all"])):
        x = np.array([r["rel_l2"] for r in tr_[64:]])
        ax.hist(np.clip(x, 0, 3), bins=60, alpha=0.55, color=col, label=f"{lab} (median {np.median(x):.2f})")
    ax.set_xlabel("per-step relative L2 error (clipped at 3)"); ax.set_ylabel("steps"); ax.legend(fontsize=7)
    ax.set_title("Distribution of per-step attention-output error")
    save(fig, "f16_error_distribution.png")


def f_attention_structure():
    e8 = load("e8_attention")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    c = np.array(e8["cum_token"])
    axs[0].plot(np.arange(1, len(c) + 1), c, color=PAL["v2i8"])
    axs[0].set_xscale("log"); axs[0].set_xlabel("top-k tokens by attention weight"); axs[0].set_ylabel("cumulative attention mass")
    axs[0].set_title("Token-level concentration", fontsize=9)
    b = np.array(e8["block_mass_sorted"])
    axs[1].bar(range(1, len(b) + 1), b, color=PAL["v2all"]); axs[1].set_xlabel("block rank"); axs[1].set_ylabel("mean attention mass")
    axs[1].set_title("Block-level mass (sorted, 16-token blocks)", fontsize=9)
    axs[2].hist(e8["entropy"], bins=15, color=PAL["t3"]); axs[2].set_xlabel("softmax entropy (nats)"); axs[2].set_title("Attention entropy per sampled step", fontsize=9)
    fig.tight_layout()
    save(fig, "f17_attention_structure.png")


def f_codec_micro():
    e7 = load("e7_codecs")
    rows = e7["codecs"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    names = ["fp16", "int8", "kivi4", "kivi2", "int8z", "kivi4z", "kivi2z"]
    for outl, col, off in ((False, PAL["v2i8"], -0.2), (True, PAL["t3"], 0.2)):
        r = {x["codec"]: x for x in rows if x["outliers"] == outl}
        axs[0].bar(np.arange(7) + off, [r[n]["ratio_vs_fp16"] for n in names], 0.4, color=col, label=f"outlier channels: {outl}")
        axs[1].bar(np.arange(7) + off, [max(r[n]["err_mean"], 1e-8) for n in names], 0.4, color=col, label=f"outliers: {outl}")
    axs[0].set_xticks(range(7)); axs[0].set_xticklabels(names, fontsize=8); axs[0].set_title("bytes / FP16 bytes (16x64 block)", fontsize=9); axs[0].legend(fontsize=7)
    axs[1].set_xticks(range(7)); axs[1].set_xticklabels(names, fontsize=8); axs[1].set_yscale("log"); axs[1].set_title("relative squared error (log)", fontsize=9)
    pc = e7["key_axis"]
    lab = [f"{p['bits']}b {'outl' if p['outliers'] else 'plain'}" for p in pc]
    axs[2].bar(np.arange(len(pc)) - 0.2, [p["per_channel"] for p in pc], 0.4, color=PAL["v2all"], label="per-channel (KIVI keys)")
    axs[2].bar(np.arange(len(pc)) + 0.2, [p["per_token"] for p in pc], 0.4, color=PAL["v1"], label="per-token")
    axs[2].set_xticks(range(len(pc))); axs[2].set_xticklabels(lab, fontsize=7); axs[2].set_yscale("log")
    axs[2].set_title("Key quantization axis: relative error (log)", fontsize=9); axs[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "f18_codec_micro.png")


def f_final_placement():
    from dataclasses import replace
    from uamm2 import Config, Simulator
    sim = Simulator(Config(), 11)
    sim.run(1536)
    bl = sim.cache.blocks
    codecs = sorted({b.codec for b in bl})
    cmap = {c: plt.cm.tab10(i) for i, c in enumerate(codecs)}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for b in bl:
        axs[0].bar(b.block_id, 1, bottom=b.tier, color=cmap[b.codec], width=0.9)
    axs[0].set_yticks([0.5, 1.5, 2.5, 3.5]); axs[0].set_yticklabels(["T0", "T1", "T2", "T3"]); axs[0].set_xlabel("block id (oldest -> newest)")
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color=cmap[c], label=c) for c in codecs], fontsize=7, ncol=3, loc="upper center")
    axs[0].set_title("Final placement map: tier (row) and codec (colour) per block", fontsize=9); axs[0].set_ylim(0, 4.9)
    axs[1].bar([b.block_id for b in bl], [b.importance for b in bl], color=PAL["v2all"]); axs[1].set_title("H2O block importance (norm.)", fontsize=9); axs[1].set_xlabel("block id")
    axs[2].bar([b.block_id for b in bl], [b.hit_ema for b in bl], color=PAL["t3"]); axs[2].set_title("selection frequency (hit_ema)", fontsize=9); axs[2].set_xlabel("block id")
    fig.tight_layout()
    save(fig, "f19_final_placement.png")
    # heatmap tier x codec
    import collections
    cnt = collections.Counter((b.tier, b.codec) for b in bl)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    M = np.array([[cnt.get((t, c), 0) for c in codecs] for t in range(4)])
    ax.imshow(M, cmap="Blues"); ax.set_xticks(range(len(codecs))); ax.set_xticklabels(codecs); ax.set_yticks(range(4)); ax.set_yticklabels(["T0", "T1", "T2", "T3"])
    for i in range(4):
        for j in range(len(codecs)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9)
    ax.grid(False); ax.set_title("Blocks per (tier, codec) at step 1536")
    save(fig, "f20_tier_codec_heatmap.png")
    return {"codecs": codecs, "matrix": M.tolist()}


# ===================================================================== hardware
def f_yosys():
    names = ["uamm_sweep_scorer", "uamm_kv_quant4", "uamm_kv_dequant4", "uamm_block_table"]
    data = {}
    for n in names:
        txt = (ROOT / "docs" / "synth" / f"{n}.stat").read_text().splitlines()
        d = {}
        for ln in txt:
            ln = ln.strip()
            parts = ln.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith("$_"):
                d[parts[1]] = int(parts[0])
            elif len(parts) == 2 and parts[0].isdigit() and parts[1] == "cells":
                d["cells"] = int(parts[0])
        data[n] = d
    json.dump(data, open(ROOT / "docs" / "synth" / "summary.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
    tot = [data[n]["cells"] for n in names]
    axs[0].bar(range(4), tot, color=[PAL["t3"], PAL["v2i8"], PAL["v2all"], PAL["t2"]])
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels([n.replace("uamm_", "") for n in names], fontsize=8)
    axs[0].set_title("Yosys generic-cell count after synth -flatten", fontsize=9)
    for i, v in enumerate(tot):
        axs[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ff = [sum(v for k, v in data[n].items() if k.startswith("$_DFF") or k.startswith("$_SDFF")) for n in names]
    comb = [data[n]["cells"] - f for n, f in zip(names, ff)]
    axs[1].bar(range(4), ff, label="flip-flops", color=PAL["t1"]); axs[1].bar(range(4), comb, bottom=ff, label="combinational", color=PAL["t3"])
    axs[1].set_xticks(range(4)); axs[1].set_xticklabels([n.replace("uamm_", "") for n in names], fontsize=8); axs[1].legend(fontsize=8)
    axs[1].set_title("Flip-flop vs combinational split", fontsize=9)
    fig.tight_layout()
    save(fig, "f21_yosys_cells.png")
    return data


def _sig(v, suffix):
    for s in v.signals:
        if s.endswith(suffix):
            return v[s]
    raise KeyError(suffix)


def _series(sig, t0, t1):
    tv = sig.tv
    pts = [(t, x) for t, x in tv]
    # value at t0
    cur = None
    for t, x in pts:
        if t <= t0:
            cur = x
        else:
            break
    out = [(t0, cur)]
    out += [(t, x) for t, x in pts if t0 < t <= t1]
    return out


def _to_int(x):
    if x is None or any(c in x.lower() for c in "xz"):
        return None
    return int(x, 2)


def draw_wave(vcd_path, signals, t0, t1, title, name, hexw=None):
    """GTKWave-style render: dark wave pane, signal-name pane on the left, time ruler on top, cursor marker."""
    import vcdvcd
    v = vcdvcd.VCDVCD(str(vcd_path), store_tvs=True)
    n = len(signals)
    BG, GRN, YEL, TXT, GRID = "#000000", "#00e000", "#ffff40", "#d8d8d8", "#2a2a2a"
    fig = plt.figure(figsize=(11, 0.34 * n + 1.5), facecolor="#1c1c1c")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 4.6], height_ratios=[0.5, n], wspace=0.0, hspace=0.0)
    axn = fig.add_subplot(gs[1, 0]); axw = fig.add_subplot(gs[1, 1]); axr = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[0, 0])
    for a in (axn, axw, axr, axh):
        a.set_facecolor(BG if a in (axw, axn) else "#1c1c1c"); a.grid(False)
        for sp in a.spines.values(): sp.set_color("#555555")
        a.set_xticks([]); a.set_yticks([])
    axh.text(0.04, 0.5, "Signals", color=TXT, fontsize=8, va="center", transform=axh.transAxes)
    axn.set_ylim(0, n); axn.set_xlim(0, 1)
    axw.set_ylim(0, n); axw.set_xlim(t0, t1)
    ticks = np.linspace(t0, t1, 9)
    for tk in ticks: axw.axvline(tk, color=GRID, lw=0.6)
    axr.set_xlim(t0, t1); axr.set_ylim(0, 1)
    for tk in ticks:
        axr.plot([tk, tk], [0, 0.35], color=TXT, lw=0.8)
        axr.text(tk, 0.45, f"{tk/1000:.0f} ns", color=TXT, fontsize=7, ha="center", va="bottom")
    cur = t0 + (t1 - t0) * 0.5
    axw.axvline(cur, color=YEL, lw=0.9); axr.text(cur, 0.02, f"Cursor {cur/1000:.0f} ns", color=YEL, fontsize=6.5, ha="center", va="bottom")
    for i, (suffix, label, kind) in enumerate(signals):
        y = n - 1 - i
        axn.text(0.04, y + 0.5, label, color=TXT, fontsize=7.6, va="center", family="monospace")
        axn.axhline(y, color="#222222", lw=0.5)
        s_ = _series(_sig(v, suffix), t0, t1)
        if kind == "bit":
            xs, ys = [], []
            for (t, x) in s_:
                xs.append(t); ys.append(y + 0.18 + 0.62 * (1 if x == "1" else 0))
            xs.append(t1); ys.append(ys[-1])
            axw.step(xs, ys, where="post", color=GRN, lw=1.1)
        else:
            for k, (t, x) in enumerate(s_):
                te = s_[k + 1][0] if k + 1 < len(s_) else t1
                d = min((te - t) * 0.12, (t1 - t0) * 0.004)
                px = [t, t + d, te - d, te, te - d, t + d, t]
                py = [y + 0.5, y + 0.8, y + 0.8, y + 0.5, y + 0.2, y + 0.2, y + 0.5]
                axw.plot(px, py, color=GRN, lw=1.0)
                val = _to_int(x)
                txt = "x" if val is None else (f"{val:x}" if kind == "hex" else str(val))
                if (te - t) > (t1 - t0) * (0.012 + 0.0042 * len(txt)):
                    axw.text((t + te) / 2, y + 0.5, txt, ha="center", va="center", fontsize=7, color=YEL, family="monospace")
    fig.suptitle(f"{Path(str(vcd_path)).name}   |   {title}", color=TXT, fontsize=8.5, x=0.01, ha="left", y=0.995)
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print("fig", name)


def f_waveforms():
    import vcdvcd
    W = ROOT / "docs" / "wave"
    P = "tb_sweep_scorer.dut."
    sc = W / "trace_sweep_scorer.vcd"
    v = vcdvcd.VCDVCD(str(sc), store_tvs=True)
    start = _sig(v, "dut.start")
    rises = [t for t, x in start.tv if x == "1"]
    # first sweep (dataset 0, pressure 0)
    t_start = rises[0]
    sigs = [("dut.clk", "clk", "bit"), ("dut.rst_n", "rst_n", "bit"), ("dut.start", "start", "bit"), ("dut.busy", "busy", "bit"),
            ("dut.rd_cnt[6:0]", "rd_cnt", "dec"), ("dut.s0_v", "s0_v", "bit"), ("dut.s0_data[24:0]", "s0_data", "hex"),
            ("dut.out_valid", "out_valid", "bit"), ("dut.out_idx[5:0]", "out_idx", "dec"), ("dut.out_score[7:0]", "out_score", "dec"),
            ("dut.out_tier[1:0]", "out_tier", "dec"), ("dut.out_evict", "out_evict", "bit"), ("dut.done", "done", "bit")]
    draw_wave(sc, sigs, t_start - 25000, t_start + 135000, "uamm_sweep_scorer: sweep start and pipeline fill (dataset 0, pressure 0)", "w01_scorer_start.png")
    # sweep end + done
    done = _sig(v, "dut.done")
    d_rises = [t for t, x in done.tv if x == "1"]
    draw_wave(sc, sigs, d_rises[0] - 90000, d_rises[0] + 40000, "uamm_sweep_scorer: last blocks, done pulse and busy deassert", "w02_scorer_done.png")
    # pressure 255 sweep: show evict
    ev = _sig(v, "dut.out_evict")
    e_rises = [t for t, x in ev.tv if x == "1"]
    sigs2 = [("dut.clk", "clk", "bit"), ("dut.pres[7:0]", "pressure (latched)", "dec"), ("dut.out_valid", "out_valid", "bit"),
             ("dut.out_idx[5:0]", "out_idx", "dec"), ("dut.out_score[7:0]", "out_score", "dec"), ("dut.out_tier[1:0]", "out_tier", "dec"),
             ("dut.out_evict", "out_evict", "bit")]
    draw_wave(sc, sigs2, e_rises[0] - 60000, e_rises[0] + 100000, "uamm_sweep_scorer: eviction asserted under memory pressure (score < 89, not protected)", "w03_scorer_evict.png")
    # mid-sweep reset
    rst = _sig(v, "dut.rst_n")
    falls = [t for t, x in rst.tv if x == "0"]
    sigs3 = [("dut.clk", "clk", "bit"), ("dut.rst_n", "rst_n", "bit"), ("dut.busy", "busy", "bit"), ("dut.rd_cnt[6:0]", "rd_cnt", "dec"),
             ("dut.s0_v", "s0_v", "bit"), ("dut.out_valid", "out_valid", "bit"), ("dut.done", "done", "bit")]
    late = [t for t in falls if t > 100000]
    tf = late[-1]
    draw_wave(sc, sigs3, tf - 100000, tf + 90000, "uamm_sweep_scorer: asynchronous reset in the middle of a sweep", "w04_scorer_reset.png")
    # codec
    cv = W / "trace_kv_codec.vcd"
    v2 = vcdvcd.VCDVCD(str(cv), store_tvs=True)
    iv = _sig(v2, "uq.in_valid")
    r = [t for t, x in iv.tv if x == "1"]
    sigsc = [("uq.clk", "clk", "bit"), ("uq.in_valid", "in_valid", "bit"), ("uq.zp[7:0]", "zp", "dec"), ("uq.inv_q12[15:0]", "inv_q12", "dec"),
             ("uq.x_flat[127:0]", "x_flat (16 lanes)", "hex"), ("uq.out_valid", "q_valid", "bit"), ("uq.q_flat[63:0]", "q_flat (16 nibbles)", "hex"),
             ("ud.step_q8[15:0]", "step_q8", "dec"), ("ud.out_valid", "x_valid", "bit"), ("ud.x_flat[127:0]", "x' (dequantized)", "hex")]
    draw_wave(cv, sigsc, r[0] - 15000, r[0] + 85000, "uamm_kv_quant4 -> uamm_kv_dequant4: two vectors through the pipeline", "w05_codec_pipeline.png")
    # block table
    bv = W / "trace_block_table.vcd"
    v3 = vcdvcd.VCDVCD(str(bv), store_tvs=True)
    we = _sig(v3, "dut.we")
    hit = _sig(v3, "dut.hit")
    wr = [t for t, x in hit.tv if x == "1"]
    wr = [wr[0] + 30000 + 20000]  # window around the first lookup that hits
    sigsb = [("dut.clk", "clk", "bit"), ("dut.we", "we", "bit"), ("dut.waddr[7:0]", "waddr", "dec"), ("dut.w_valid", "w_valid", "bit"),
             ("dut.w_tier[1:0]", "w_tier", "dec"), ("dut.w_codec[1:0]", "w_codec", "dec"), ("dut.w_phys[11:0]", "w_phys", "hex"),
             ("dut.rd_en", "rd_en", "bit"), ("dut.raddr[7:0]", "raddr", "dec"), ("dut.rd_valid", "rd_valid", "bit"), ("dut.hit", "hit", "bit"),
             ("dut.tier[1:0]", "tier", "dec"), ("dut.codec[1:0]", "codec", "dec"), ("dut.phys[11:0]", "phys", "hex")]
    draw_wave(bv, sigsb, wr[0] - 80000, wr[0] + 100000, "uamm_block_table: random update and lookup operations", "w06_block_table.png")


# ===================================================================== terminal captures
def term(text, name, title, maxlines=60, fs=7.2):
    lines = text.rstrip("\n").splitlines()[:maxlines]
    h = 0.17 * len(lines) + 1.0
    fig = plt.figure(figsize=(10.5, h), facecolor="#111111")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#111111")
    ax.text(0.012, 1 - 0.18 / h, f"$ {title}", color="#ffd866", fontsize=fs + 0.6, family="monospace", va="top", transform=ax.transAxes)
    for i, ln in enumerate(lines):
        col = "#86efac" if ("PASS" in ln or "passed" in ln or "KILLED" in ln) else ("#fca5a5" if ("FAIL" in ln or "failed" in ln) else "#e2e8f0")
        ax.text(0.012, 1 - (0.62 + i * 0.17) / h, ln[:150], color=col, fontsize=fs, family="monospace", va="top", transform=ax.transAxes)
    fig.savefig(FIG / name, dpi=170, facecolor="#111111")
    plt.close(fig)
    print("fig", name)


def f_terminal():
    cap = ROOT / "docs" / "captures"
    term((cap / "pytest_verbose.txt").read_text(), "t01_pytest.png", "python -m pytest -v tests", maxlines=78, fs=6.4)
    term((cap / "rtl_regression.txt").read_text(), "t02_rtl_regression.png", "make -C hw test", maxlines=14, fs=6.6)
    term((cap / "rtl_lint.txt").read_text(), "t03_rtl_lint.png", "make -C hw lint   (verilator --lint-only -Wall)", maxlines=24, fs=6.6)
    term((cap / "mutation.txt").read_text(), "t04_mutation.png", "RTL mutation testing (injected bugs must make a testbench FAIL)", maxlines=24, fs=7.2)
    term((cap / "rvv_inner_loop.s").read_text(), "t05_rvv_asm.png", "clang --target=riscv64 -march=rv64gcv -O2 -S ara/kv_kernels.c   (kv_qk_dot_kivi4_rvv)", maxlines=58, fs=6.6)
    term((cap / "ablate_cli.txt").read_text(), "t06_ablate_cli.png", "python -m uamm2 ablate --tokens 1536", maxlines=14, fs=6.6)
    term((cap / "ara_check.txt").read_text(), "t07_ara_check.png", "scalar C kernel vs NumPy dequantize-then-dot; RVV compile check", maxlines=14, fs=7.2)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    f_kv_growth(); f_kv_bandwidth(); f_kv_compression_effect(); f_coverage_matrix()
    f_ablation(); f_context_scaling(); f_budget(); f_lambda(); f_stability(); f_h2o_knobs(); f_capacity(); f_codecsets()
    f_timeseries(); f_attention_structure(); f_codec_micro()
    info = f_final_placement()
    json.dump(info, open(ROOT / "experiments" / "results" / "final_placement.json", "w"))
    f_yosys(); f_waveforms()
    if (ROOT / "docs" / "captures" / "pytest_verbose.txt").exists():
        f_terminal()
