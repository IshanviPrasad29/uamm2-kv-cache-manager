from data import *

def build(R):
    R.h1("8. Simulation Results")
    R.p("All numbers in this chapter come from `experiments/run_experiments.py` (raw data in `experiments/results/*.json`) under the configuration of Chapter 7. Latency figures are model outputs under the assumed constants. Errors are measured against a dense FP32 oracle on the synthetic trace. Where we interpret a result, we say so.", align="j")

    # ------------------------------------------------------------------ E1
    R.h2("8.1 Ablation: what each ingredient contributes (E1)")
    e1 = load("e1_ablation")
    names = [("v1_threshold_fp16", "UAMM-1 threshold policy, FP16"), ("v2_planner_fp16", "Planner, FP16 only"),
             ("v2_planner_int8_kivi4", "Planner, INT8 + KIVI-4"), ("v2_planner_all_codecs", "Planner, INT8 + KIVI-4 + KIVI-2")]
    rows = []
    for k, lab in names:
        d = e1[k]
        rows.append([lab, ms(d, "resident_vs_fp16"), ms(d, "dropped_tokens", "{:.0f}"), ms(d, "attn_rel_l2_mean"), ms(d, "attn_mass_recall_mean"),
                     ms(d, "sim_fetch_ms_per_token"), ms(d, "flash_block_reads", "{:.0f}"), ms(d, "overflow_total", "{:.0f}")])
    R.table(["Variant", "Resident / FP16", "Dropped tokens", "Rel. L2 error", "Mass recall", "Fetch ms / token (model)", "Flash reads", "Overflow (B)"], rows,
            caption="Ablation at 1,536 tokens (mean ± std over 5 seeds).", widths=[1.6, 0.8, 0.7, 0.8, 0.8, 0.8, 0.7, 0.7], size=8)
    R.fig("f05_ablation.png", "Ablation results (mean ± std, 5 seeds). The FP16-only planner cannot fit the data in the budget (non-zero overflow); the UAMM-1 threshold policy fits only by dropping most tokens.", 6.4)
    R.p("**Reading the table.**", align="j", after=2)
    R.bullets([
        ("UAMM-1 fits by deleting.", "With only FP16 blocks and a 168 KB budget the threshold policy must remove on average 973 of 1,536 tokens (63 %). Attention-mass recall falls to 0.758 and the output error reaches 0.479; the reads that remain are frequently from flash (about 12,600 flash block reads)."),
        ("The planner alone cannot create space.", "With FP16 only, the planner drops nothing but cannot fit: 258 KB of overflow, i.e. the data exceeds the stated budget by that much. The lower error (0.092) is the floor due to block-sparse selection with exact data. This variant shows that placement alone is insufficient: without a codec, a budget below the data size is infeasible without deletion."),
        ("Codecs make the budget feasible.", "Adding INT8 and 4-bit codecs reduces residency to 0.371 of FP16, with no overflow and no drops. Error rises to 0.122 (vs the 0.092 floor): the price of quantisation."),
        ("2-bit codec is a knob.", "Allowing 2-bit blocks reduces residency further to 0.310 (a further 16 % saving) at an error of 0.166, i.e. 36 % more error than INT8+KIVI-4."),
        ("Latency model.", "Modelled fetch time per token falls from 0.92 ms (UAMM-1) to 0.116 ms (INT8+KIVI-4) and 0.112 ms (all codecs). Most of that difference is slower-tier traffic: flash block reads drop from about 12,600 to about 915–970, i.e. 13×. The FP16-only planner is slower than UAMM-1 (1.04 ms) because overflowing blocks must live in flash with no compression."),
    ])
    R.callout("What the ablation says", "UAMM-1's weakness is not placement, it is the lack of a way to make data smaller. A joint (tier, codec) choice turns a budget that was infeasible without data deletion into a feasible one at a cost of bounded, measured quantisation error. Every ingredient has a role: removing the codecs reintroduces overflow or deletion; removing the planner loses the control over where precision is spent.")

    # ------------------------------------------------------------------ E2
    R.h2("8.2 Scaling with context length (E2)")
    e2 = load("e2_context_scaling")
    ctxs = ["512", "768", "1024", "1536", "2048", "3072"]
    rows = []
    for c in ctxs:
        a, b, d = e2["v1_threshold_fp16"][c], e2["v2_planner_int8_kivi4"][c], e2["v2_planner_all_codecs"][c]
        rows.append([c, f"{mean(a,'attn_rel_l2_mean'):.3f}", f"{mean(d,'attn_rel_l2_mean'):.3f}", f"{mean(a,'attn_mass_recall_mean'):.3f}", f"{mean(d,'attn_mass_recall_mean'):.3f}",
                     f"{mean(a,'dropped_tokens'):.0f}", f"{mean(d,'resident_vs_fp16'):.3f}", f"{mean(a,'sim_fetch_ms_per_token'):.3f}", f"{mean(d,'sim_fetch_ms_per_token'):.3f}",
                     f"{mean(b,'overflow_total'):,.0f} / {mean(d,'overflow_total'):,.0f}"])
    R.table(["Tokens", "Err UAMM-1", "Err UAMM-2 (all)", "Recall UAMM-1", "Recall UAMM-2", "Dropped (UAMM-1)", "Resident/FP16 (UAMM-2)", "Fetch ms UAMM-1", "Fetch ms UAMM-2", "Overflow B: INT8+K4 / all"], rows,
            caption="Context scaling with the 168 KB budget fixed (mean of 3 seeds). UAMM-2 columns use all codecs unless stated.", size=7.5,
            widths=[0.5, 0.6, 0.65, 0.6, 0.65, 0.65, 0.75, 0.6, 0.6, 0.9])
    R.fig("f06_context_scaling.png", "Context scaling at a fixed budget. UAMM-1's error and drops grow quickly beyond 768 tokens; UAMM-2 degrades gradually and keeps every token until the budget is exceeded near 3,072 tokens.", 6.4)
    R.bullets([
        ("Short contexts.", "At 512 tokens the cache fits at FP16 (UAMM-1 resident ratio 1.0, error 0.0013). UAMM-2 still compresses (resident 0.538) and is 14× faster in the model (0.035 ms vs 0.473 ms) but incurs a small error (0.016). That is a policy consequence of the cost function: when latency is priced and the quality term is small, the planner lowers precision even without capacity pressure. If the application prefers exactness when memory is available, λ should be raised or compression gated on pressure (Chapter 12)."),
        ("Medium contexts.", "At 1,024 tokens UAMM-1 has dropped 496 tokens (error 0.241, recall 0.882); UAMM-2 holds all tokens with error 0.071 and recall 0.976."),
        ("Long contexts.", "At 2,048 tokens UAMM-2 (all codecs) still fits (resident 0.266, error 0.270, recall 0.913). At 3,072 tokens it overflows its budget by 28.5 KB (INT8+KIVI-4 only: 132 KB) and error is 0.515 vs 0.985 for UAMM-1, which has dropped 2,469 of 3,072 tokens. Beyond this point the fixed budget is simply too small for the data even at 2-bit; opt-in drop or a larger budget would be needed."),
        ("Recall.", "Recall falls with context in both systems but for different reasons: in UAMM-1 because tokens are missing, in UAMM-2 because the fixed block budget of 24 covers a shrinking fraction of the history (see E3)."),
    ])

    # ------------------------------------------------------------------ E3
    R.h2("8.3 How many blocks must be read? (E3)")
    e3 = load("e3_budget")
    bud = ["4", "8", "12", "16", "24", "32", "48"]
    rows = [[b, ms(e3[b], "attn_mass_recall_mean", "{:.3f}"), ms(e3[b], "attn_rel_l2_mean", "{:.3f}"), ms(e3[b], "sim_fetch_ms_per_token", "{:.3f}"),
             ms(e3[b], "resident_vs_fp16", "{:.3f}")] for b in bud]
    R.table(["Block budget B", "Mass recall", "Rel. L2 error", "Fetch ms / token", "Resident / FP16"], rows,
            caption="Effect of the attention block budget at 1,536 tokens (mean ± std, 3 seeds).", widths=[1.0, 1.3, 1.3, 1.3, 1.3])
    R.fig("f07_budget_sweep.png", "Block budget sweep. Left: recall of the min/max-bound selector versus the oracle choice of the B heaviest blocks averaged over sampled steps (indicative; the system also reads the protected blocks). Middle and right: error and fetch cost.", 6.4)
    R.p("Selecting 24 blocks (about a quarter of the roughly 96 blocks at 1,536 tokens) retains 94.5 % of the attention mass with an error of 0.159; 48 blocks reach 98.6 % at a cost that is 2.5× higher (0.278 ms vs 0.111 ms) because more blocks are fetched from slower tiers. The marginal return of a block falls quickly, as the sorted block mass (Section 8.8) predicts. Note the interaction with residency: a larger budget reads many cold blocks, which pushes the planner to keep more of them in higher precision (resident 0.376 at B=48 vs 0.310 at B=24).", align="j")

    # ------------------------------------------------------------------ E4
    R.h2("8.4 Sensitivity of the planner and the importance signal (E4)")
    R.h3("8.4.1 The quality weight λ")
    e = load("e4_lambda")
    rows = [[k, ms(e[k], "resident_vs_fp16", "{:.3f}"), ms(e[k], "attn_rel_l2_mean", "{:.3f}"), ms(e[k], "sim_fetch_ms_per_token", "{:.3f}"), ms(e[k], "flash_block_reads", "{:.0f}"), ms(e[k], "migrations", "{:.0f}")] for k in e]
    R.table(["λ", "Resident / FP16", "Rel. L2 error", "Fetch ms / token", "Flash reads", "Migrations"], rows,
            caption="Effect of the quality weight λ at 1,536 tokens (3 seeds).", widths=[0.6, 1.2, 1.2, 1.2, 1.2, 1.2])
    R.fig("f08_lambda_sweep.png", "Effect of λ on error, memory, fetch latency and flash reads.", 6.4)
    R.fig("f09_lambda_pareto.png", "Error versus fetch latency as λ varies. The knee is around λ = 2 and the default sits at it.", 4.6)
    R.p("As λ rises the planner protects quality: error falls from 0.577 at λ = 0 (maximum compression, resident 0.243) to 0.127 at λ = 32, at the price of larger residency (0.366). Latency is not monotone: up to λ = 2 fetch time is nearly flat at about 0.10–0.11 ms, then jumps to 0.20–0.25 ms at λ ≥ 4 and flash reads triple. The cause, visible in the planner's cost, is that choosing larger encodings for a block can push other blocks out of DRAM into flash, whose fixed 100 µs access cost dwarfs the transfer savings. The default λ = 2 sits at the knee: the error is within 25 % of the best seen (0.159 vs 0.127) while fetch time is less than half that of λ ≥ 4.", align="j")
    R.h3("8.4.2 Hysteresis, residency and replanning interval")
    e = load("e4_hysteresis")
    rows = [[k, ms(e[k], "migrations", "{:.0f}"), ms(e[k], "reencodes", "{:.0f}"), ms(e[k], "resident_vs_fp16", "{:.3f}"), ms(e[k], "attn_rel_l2_mean", "{:.3f}"), ms(e[k], "sim_fetch_ms_per_token", "{:.3f}")] for k in e]
    R.table(["Hysteresis margin", "Migrations", "Re-encodes", "Resident / FP16", "Rel. L2 error", "Fetch ms / token"], rows,
            caption="Hysteresis margin (3 seeds).", widths=[1.1, 1.0, 1.0, 1.2, 1.2, 1.2])
    R.fig("f10_stability_knobs.png", "Left: churn versus hysteresis margin. Middle: minimum residency. Right: planning interval.", 6.4)
    R.p("Without hysteresis (margin 0) the planner performs 1,871 migrations; with the default margin of 0.25 it performs 630, a reduction of 66 % at essentially unchanged fetch time (0.1118 vs 0.1112 ms). The trade-off is that a larger margin also suppresses precision demotions (re-encodes fall from 247 to 180), so residency grows (0.286 to 0.373) and error falls (0.208 to 0.110). Hysteresis is therefore both a churn limiter and a quality-conservative bias.", align="j")
    R.p("The minimum-residency rule has no effect below 16 steps because the planning interval is 8 (identical rows for 0 and 8): residency is only meaningful when it exceeds the interval. At 64 steps migrations fall to 470, but fetch time doubles (0.211 ms) and flash reads reach 2,384, because blocks that have become hot cannot be promoted in time. Likewise the replanning interval: planning every 2 steps gives 599 flash reads and 0.090 ms but 768 migrations, whereas every 64 steps gives 3,326 flash reads and 0.272 ms with only 345 migrations. The planner must keep up with the changes in attention focus; the cost of keeping up is the planning work itself, which is why the hardware sweep scorer exists.", align="j")
    R.h3("8.4.3 H2O decay, aggregation and block size")
    e = load("e4_decay")
    rows = [[k, ms(e[k], "resident_vs_fp16", "{:.3f}"), ms(e[k], "attn_rel_l2_mean", "{:.3f}"), ms(e[k], "sim_fetch_ms_per_token", "{:.3f}"), ms(e[k], "reencodes", "{:.0f}")] for k in e]
    R.table(["H2O decay", "Resident / FP16", "Rel. L2 error", "Fetch ms / token", "Re-encodes"], rows, caption="H2O decay (3 seeds).", widths=[1.0, 1.4, 1.4, 1.4, 1.2])
    R.fig("f11_h2o_blocksize_knobs.png", "Left: H2O decay. Middle: token-to-block aggregation. Right: block size (with the block budget scaled so the number of tokens fetched stays constant).", 6.4)
    R.p("A longer memory of attention (decay 0.995) makes the planner more conservative about compressing blocks that were important at some point: error falls from 0.202 (decay ≤ 0.9) to 0.114 and residency rises from 0.285 to 0.368, at essentially constant fetch time (0.110–0.112 ms). Decay therefore moves the system along the same memory-versus-quality axis as λ but at no latency cost in this configuration. Mean and sum aggregation are equivalent (sum is normalised by the same percentile; blocks have equal size); max aggregation gives slightly lower residency (0.301) and similar error (0.162).", align="j")
    e = load("e4_blocksize")
    rows = [[k, ms(e[k], "resident_vs_fp16", "{:.3f}"), ms(e[k], "attn_rel_l2_mean", "{:.3f}"), ms(e[k], "sim_fetch_ms_per_token", "{:.3f}"), ms(e[k], "migrations", "{:.0f}"), ms(e[k], "overflow_total", "{:.0f}")] for k in e]
    R.table(["Block size (tokens)", "Resident / FP16", "Rel. L2 error", "Fetch ms / token", "Migrations", "Overflow (B)"], rows, caption="Block size (3 seeds; block budget scaled to keep ≈384 non-protected tokens fetched).", widths=[1.2, 1.2, 1.2, 1.2, 1.0, 1.0])
    R.p("Block size is a three-way trade-off. Small blocks (8 tokens) isolate hot tokens well, giving the lowest error (0.110), but have more metadata, more migrations (1,652) and much higher fetch time (0.216 ms, since each block pays a fixed latency). Large blocks (64) cut migrations to 121 and fetch time to 0.038 ms but raise error to 0.341 and overflow the budget by 14 KB: a block must be placed and compressed as a unit, so a few hot tokens drag a whole large block into a fast tier. A block of 16 tokens is a reasonable compromise and also fits the hardware tables.", align="j")

    # ------------------------------------------------------------------ E5
    R.h2("8.5 Memory budget (E5)")
    e5 = load("e5_capacity")
    fs = ["0.5", "0.75", "1.0", "1.5", "2.0", "3.0"]
    rows = []
    for f in fs:
        a, b = e5["v1"][f], e5["v2"][f]
        rows.append([f"×{f} ({168*float(f):.0f} KB)", f"{mean(a,'resident_vs_fp16'):.3f}", f"{mean(a,'dropped_tokens'):.0f}", f"{mean(a,'attn_rel_l2_mean'):.3f}", f"{mean(a,'sim_fetch_ms_per_token'):.3f}",
                     f"{mean(b,'resident_vs_fp16'):.3f}", f"{mean(b,'attn_rel_l2_mean'):.3f}", f"{mean(b,'sim_fetch_ms_per_token'):.3f}", f"{mean(b,'overflow_total'):,.0f}"])
    R.table(["Budget", "UAMM-1 resident", "UAMM-1 dropped", "UAMM-1 error", "UAMM-1 fetch ms", "UAMM-2 resident", "UAMM-2 error", "UAMM-2 fetch ms", "UAMM-2 overflow B"], rows,
            caption="Memory-budget sweep at 1,536 tokens (mean of 3 seeds).", size=7.5, widths=[0.9, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8])
    R.fig("f12_capacity_sweep.png", "Budget sweep. UAMM-1 needs roughly three times the baseline budget to bring error below 0.09 and still drops 261 tokens; UAMM-2 reaches comparable error without drops at much smaller budgets.", 6.4)
    R.bullets([
        ("Equal-quality comparison.", "UAMM-2 at ×0.75 of the budget (126 KB) has error 0.180 and drops nothing; UAMM-1 needs ×2.0 (336 KB) to reach a similar error (0.171) and still drops 576 tokens. In the model, UAMM-2 matches UAMM-1's error with 2.7× less memory."),
        ("Infeasibility.", "At ×0.5 (84 KB) UAMM-2 overflows by 33 KB (the data do not fit even at its most compressed); UAMM-1 does not report overflow but drops 1,200 tokens and has error 0.855."),
        ("Non-monotonic quality.", "UAMM-2's error is 0.115 at ×2.0 but 0.152 at ×3.0, even though more memory is available. At ×3.0 flash is almost unused (3 reads) and the planner spends the headroom on speed (fetch 0.049 ms vs 0.088 ms) by choosing smaller encodings. This is the cost function working as designed, but it is a reminder that the optimiser, not the budget, decides the trade-off, and that a quality floor (Chapter 12) is needed to guarantee monotonic behaviour."),
    ])
    R.h3("8.5.1 Which codecs should be enabled? (E5b)")
    e5b = load("e5b_codecsets")
    labs = [("fp16|entropy=False", "FP16 only"), ("fp16+int8|entropy=False", "FP16 + INT8"), ("fp16+int8+kivi4|entropy=False", "+ KIVI-4"), ("all(+kivi2)|entropy=False", "+ KIVI-2 (all)"),
            ("kivi4_only|entropy=False", "FP16 + KIVI-4 only"), ("kivi2_only|entropy=False", "FP16 + KIVI-2 only")]
    rows = [[l, ms(e5b[k], "resident_vs_fp16", "{:.3f}"), ms(e5b[k], "attn_rel_l2_mean", "{:.3f}"), ms(e5b[k], "sim_fetch_ms_per_token", "{:.3f}"), ms(e5b[k], "overflow_total", "{:,.0f}")] for k, l in labs]
    R.table(["Enabled codecs", "Resident / FP16", "Rel. L2 error", "Fetch ms / token", "Overflow (B)"], rows, caption="Codec-set study at 1,536 tokens (3 seeds).", widths=[1.8, 1.2, 1.2, 1.2, 1.2])
    R.fig("f13_codec_sets.png", "Codec-set study. Blue bars enable entropy-coded flash variants, gray bars disable them; the two are indistinguishable on this synthetic data.", 6.4)
    R.bullets([
        ("Feasibility needs 4-bit.", "FP16 alone (overflow 257 KB) and FP16+INT8 (overflow 105 KB) cannot fit the budget; a 4-bit codec is the first that does."),
        ("KIVI-4 alone is competitive.", "FP16 + KIVI-4 reaches 0.364 resident at error 0.132 and fetch 0.112 ms, close to the full set on speed with lower error than the full set (0.159); the full set saves a further 15 % memory."),
        ("KIVI-2 alone is a poor choice.", "With only FP16 and KIVI-2 available the planner reaches 0.346 resident, but error doubles (0.308) and fetch time is 4× worse (0.475 ms, about 6,200 flash reads). The reason is structural: the tier rules force warm and cold storage to hold a 2-bit block, and without intermediate precision the planner cannot place blocks that matter more at reasonable cost."),
        ("Entropy coding gave no benefit here.", "Results with and without zlib variants are identical, which means the planner never preferred a `z` variant. On the synthetic data, zlib on already quantised random-like values saves almost nothing (Table of codec sizes in Chapter 5). Real KV tensors have more redundancy and the result may differ; this is an open question rather than a conclusion."),
    ])

    # ------------------------------------------------------------------ E6
    R.h2("8.6 Behaviour over time (E6)")
    R.p("The following figures are time series from one run (seed 11, 1,536 tokens) of UAMM-1 and UAMM-2.", align="j")
    R.fig("f14_timeseries_v2.png", "Top left: bytes resident per tier. Top right: number of blocks by codec. Bottom: rolling modelled fetch time and attention-mass recall for both systems.", 6.4)
    tr = load("e6_traces")
    v2 = tr["v2_all"]["trace"]; v1 = tr["v1"]["trace"]
    pts = [r for r in v2 if "tier_bytes" in r]
    R.p(f"The tiers fill in order: first T0 (protected blocks), then T1, then T2 and T3 as the cache outgrows the faster tiers. The codec histogram shows the ratchet in action: INT8 blocks dominate early, 4-bit blocks take over as pressure rises, and 2-bit blocks appear mainly in the last third of the run (after about 1,100 steps) when the lower tiers are full. FP16 blocks are only the protected set (about 3 blocks). UAMM-1's rolling fetch time rises to about 1.2 ms after about 400 steps and recall oscillates between 0.3 and 0.9, falling as blocks are discarded; UAMM-2's fetch time stays below 0.3 ms and its recall stays above 0.8 throughout.", align="j")
    R.fig("f15_timeseries_churn.png", "Cumulative dropped tokens, cumulative migrations and re-encodes, and flash reads per step.", 6.4)
    R.p("UAMM-1's dropped-token count rises steadily once the budget is reached and never recovers. UAMM-2's counter stays at zero. Migrations accumulate for both, with UAMM-2 also performing re-encodes (each is a decode and an encode and has a modelled cost). Flash reads for UAMM-1 are persistent and bursty; those for UAMM-2 are rare and concentrated where the query returns to an old topic.", align="j")
    R.fig("f16_error_distribution.png", "Distribution of per-step attention-output error. UAMM-1 has a heavy tail of large-error steps (those that needed tokens it had already deleted); UAMM-2 is concentrated at lower values.", 4.8)
    import numpy as np
    e1_ = np.array([r["rel_l2"] for r in v1[64:]]); e2_ = np.array([r["rel_l2"] for r in v2[64:]])
    R.p(f"Per-step error medians are {np.median(e1_):.3f} (UAMM-1) and {np.median(e2_):.3f} (UAMM-2); 95th percentiles are {np.percentile(e1_,95):.3f} and {np.percentile(e2_,95):.3f}. The heavy tail matters for users: a few very wrong steps are noticed more than a uniformly small error.", align="j")

    # ------------------------------------------------------------------ E7
    R.h2("8.7 Codec micro-benchmarks (E7)")
    R.fig("f18_codec_micro.png", "Left: measured size ratio per codec. Middle: reconstruction error (log scale). Right: per-channel versus per-token key quantisation error.", 6.4)
    ka = load("e7_codecs")["key_axis"]
    rows = [[f"{x['bits']}-bit", "with" if x["outliers"] else "without", f"{x['per_channel']:.2e}", f"{x['per_token']:.2e}", f"{x['per_token']/x['per_channel']:.1f}×"] for x in ka]
    R.table(["Bits", "Outlier channels", "Per-channel keys", "Per-token keys", "Per-token / per-channel"], rows, caption="Key quantisation axis: relative squared error of keys only (200 random 16 × 64 blocks per row).", widths=[0.8, 1.3, 1.4, 1.4, 1.5])
    R.p("Quantising keys per channel is always better, by a factor of 1.9 without outlier channels and 5.2–5.4× when four of the 64 channels are six times larger, which agrees with the design choice reported for KIVI. Error rises by roughly 25× from 4-bit to 2-bit for the same block (0.0052 → 0.130 in the codec table), which is why 2-bit is a controlled, optional tier rather than a default.", align="j")

    # ------------------------------------------------------------------ E8
    R.h2("8.8 Attention structure of the workload (E8)")
    R.fig("f17_attention_structure.png", "Left: cumulative attention mass versus number of tokens. Middle: mean attention mass of the blocks sorted by rank. Right: softmax entropy per sampled step.", 6.4)
    e8 = load("e8_attention")
    c = e8["cum_token"]; b = e8["block_mass_sorted"]
    R.p(f"On the generator, the top 1 token receives {c[0]*100:.0f} % of the attention, the top 10 receive {c[9]*100:.0f} %, the top 50 receive {c[49]*100:.0f} % and the top 100 receive {c[99]*100:.0f} %; the best block carries {b[0]*100:.0f} % of the mass and the best ten blocks {sum(b[:10])*100:.0f} %. Attention is concentrated but not extremely so (mean entropy {np.mean(e8['entropy']):.2f} nats, against a maximum of ln(256…1536) ≈ 5.5–7.3). This is a deliberately moderate setting: the heavier the concentration of real attention, the easier the problem is for any selective scheme, so this workload is not an optimistic one for block selection.", align="j")

    # ------------------------------------------------------------------ placement
    R.h2("8.9 Final placement map")
    info = load("final_placement")
    R.fig("f19_final_placement.png", "State of the cache at step 1,536: tier (row) and codec (colour) of every block, with block importance and selection frequency. Older, low-importance blocks occupy flash at 2- or 4-bit; recent blocks are in T0 at FP16.", 6.4)
    R.fig("f20_tier_codec_heatmap.png", "Blocks per (tier, codec) pair at step 1,536.", 4.2)
    M = info["matrix"]; cod = info["codecs"]
    R.p(f"The final state has 96 blocks. Tier T0 holds only FP16 blocks (protected sink and recent window). T1 holds {sum(M[1])} blocks (mainly 4-bit), T2 holds {sum(M[2])} and the flash tier T3 holds {sum(M[3])}, all at 4 or 2 bits. No block in T0 or T1 is FP16 beyond the protected set: at this budget the planner has chosen to spend bytes on keeping everything rather than on keeping any unprotected block uncompressed.", align="j")

    R.h2("8.10 Summary of the simulation findings")
    R.table(["Finding", "Evidence"],
            [["Joint (tier, codec) placement makes a tight budget feasible without deleting tokens", "E1: 0 dropped tokens vs 973; E5b: feasibility requires a 4-bit codec"],
             ["Quality is bounded and tunable", "E4: λ and decay move error between 0.11 and 0.58 along a memory axis"],
             ["Selective attention delivers most of the saving", "E3: 24 of ~96 blocks gives 94.5 % mass recall"],
             ["Hysteresis removes most churn", "E4: 1,871 → 630 migrations with the default margin"],
             ["Replanning cadence matters for flash traffic", "E4: 599 → 3,326 flash reads from interval 2 → 64"],
             ["Budget exhaustion is graceful, not catastrophic", "E2: gradual error growth; overflow reported, not hidden"],
             ["Some behaviour is non-obvious and needs a floor or a policy", "E5: non-monotonic error at 3× memory; E2: compression when memory is ample"]],
            caption="Key findings from the simulations.", widths=[3.3, 3.3])
