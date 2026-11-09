from data import *
import numpy as np
def build(R):
    R.h1("13. Putting the Numbers in a Realistic Frame")
    R.p("Chapter 8 reports results on a small, deliberately tight configuration (168 KB for 1,536 tokens of a 64-channel cache). This chapter extrapolates those *ratios* to the scale of an actual edge deployment, using arithmetic only. These are projections from measured ratios and stated assumptions, not new measurements.", align="j")
    R.h2("13.1 Why the compression ratios transfer to other geometries")
    R.p("For a block of t tokens and d channels with value group g, the bytes of a b-bit block relative to FP16 are b/16 + 1/t + 1/g (scales and zeros are two FP16 numbers each, per channel for keys and per group for values). The expression does not depend on d. For t = 16 and g = 32 it gives 0.594 (INT8), 0.344 (4-bit) and 0.219 (2-bit), identical to the measured values in Chapter 5. Larger blocks reduce the metadata share (t = 32 gives 0.3125 for 4-bit) but, as Section 8.4.3 shows, enlarge the unit of placement.", align="j")
    R.h2("13.2 What fits on a phone-class device")
    g = bpt(32, 8, 128)
    free = 1 * 2**30
    rows = []
    for lab, r in (("FP16", 1.0), ("INT8", 0.594), ("Mixed int8/4-bit (E1: 0.371)", 0.371), ("Mixed incl. 2-bit (E1: 0.310)", 0.310), ("All 4-bit", 0.344), ("All 2-bit", 0.219)):
        rows.append([lab, f"{r:.3f}", f"{free/(g*r):,.0f}", f"{(g*r*32768)/2**30:.2f} GiB"])
    R.table(["Storage", "Bytes vs FP16", "Max tokens in 1 GiB", "KV size at 32K tokens"], rows,
            caption="Llama-3-8B-like geometry (131,072 B per token in FP16). The mixed rows use the resident ratios measured in E1 and assume the same mix at this scale.", widths=[2.4, 1.2, 1.5, 1.5])
    R.p("In the model, a 1 GiB KV budget that holds about 8,200 tokens in FP16 holds about 26,000 tokens at the 0.310 mix without dropping any: a gain of 3.2×, in line with 1/0.310. A 32K-token context shrinks from 4 GiB to about 1.2 GiB.", align="j")
    R.h2("13.3 Reads per token")
    tr = load("e6_traces")["v2_all"]["trace"]
    fr = []
    for r in tr[1024:]:
        nb = (r["t"] + 1 + 15) // 16
        fr.append(r["blocks_read"] / nb)
    frac = float(np.mean(fr))
    R.p(f"In the 1,536-token run, after step 1,024 the engine read on average {frac*100:.0f} % of the blocks per decode step (24 selected plus the protected set). If the same fraction applied at larger scale and the blocks read had the average encoding, bytes read per token would be roughly {frac:.2f} × {0.31:.2f} = {frac*0.31:.2f} of the dense FP16 read. The fraction would shrink further with longer contexts because the block budget is fixed while the cache grows, but recall would also fall (Section 8.3); the budget has to be tuned against the application's quality needs.", align="j")
    rows = []
    bw = 12.8e9
    for ctx in (8192, 32768, 131072):
        dense = g * ctx
        eff = dense * frac * 0.31 if ctx <= 8192 else dense * min(1.0, 24 * 16 / ctx + 36 / ctx) * 0.31
        rows.append([f"{ctx:,}", f"{dense/2**30:.2f} GiB", f"{bw/dense:.1f}", f"{eff/2**20:.0f} MiB", f"{bw/eff:.1f}"])
    R.table(["Context", "Dense FP16 KV read / token", "KV-only bound (tok/s, 12.8 GB/s)", "Selective + mixed codecs read / token", "KV-only bound (tok/s)"], rows,
            caption="Projected KV-read bound for the 8B-like geometry. For 8,192 tokens the measured read fraction is used; beyond that the fixed 24-block budget plus a 36-token protected set is assumed. Weight reads and compute are ignored, and the bound is not a prediction of end-to-end speed.", widths=[0.9, 1.4, 1.6, 1.6, 1.2], size=8)
    R.p("At 32K tokens the dense FP16 read bound is about 3 tokens per second on 12.8 GB/s memory; with selective reads and mixed codecs the KV-only bound is far higher, which would make the weights, not the cache, the limit again. This is the sense in which the design aims to restore the bottleneck to where it belongs. Whether recall at 24 blocks is sufficient at that length for a real model is exactly the question that real-model validation (Chapter 12, gap 1) must answer.", align="j")
    R.h2("13.4 What would convince a sceptic")
    R.table(["Claim", "Evidence in this report", "Evidence still needed"],
            [["The joint policy beats placement alone in a tight budget", "E1, E2, E5 on the synthetic trace", "Same ablation on real traces, with task accuracy"],
             ["Hysteresis and cadence control churn", "E4", "Energy measured on a device"],
             ["Hardware blocks implement the specified arithmetic", "Bit-exact RTL regression, mutation tests, lint", "Place-and-route, timing, power, integration with a bus"],
             ["The kernel is correct", "Host agreement with NumPy; RVV compiles", "Execution on Spike/Ara"],
             ["The design speeds up an edge device", "Modelled fetch time", "Measured tokens/s on target hardware"]],
            caption="Evidence available and evidence needed for each claim.", widths=[2.2, 2.2, 2.2])
    R.h2("13.5 Deployment sketch")
    R.numbered(["Pin an open 1–3B model, a runtime build and a target (FPGA with CVA6+Ara, or an Arm SoC for the first measurements).",
                "Implement the adapter and replace the cost constants with measurements.",
                "Run perplexity and long-context benchmarks at FP16, uniform 4-bit and UAMM-2, at equal memory budgets.",
                "Use results to set λ, block budget and plan interval per device class.",
                "Move the scorer and codec to hardware only if profiling shows the software planner or codec to be a bottleneck."])
