from data import *
def build(R):
    R.h1("7. Experimental Methodology")
    R.p("This chapter states precisely what was simulated, what was real, how quality was measured and which constants were assumed. The goal is that a reader can judge exactly what each number in Chapters 8 to 10 means.", align="j")
    R.h2("7.1 What is real and what is modelled")
    R.table(["Component", "Status in the experiments"],
            [["KV tensors, codecs, packing, entropy coding", "**Real computation.** Blocks are actually quantised, bit-packed, optionally zlib-compressed, decoded and used for attention."],
             ["Attention output and quality metrics", "**Real computation** on decoded tensors, compared with a dense FP32 attention oracle over all tokens at the same step."],
             ["Block selection (min/max bound), importance scores, planner decisions", "**Real computation** (the same code that would run on a device)."],
             ["Q, K, V stream", "**Synthetic**: produced by a parametrised trace generator, described in 6.2, not by a neural network."],
             ["Tier latencies, bandwidths, decode costs", "**Modelled constants** (6.4). The reported latency is a model output, not a measurement."],
             ["RTL behaviour", "**Real simulation** with Verilator, compared bit-exactly with Python golden models."],
             ["RVV kernel", "Scalar variant executed on the host; RVV variant compiled to RISC-V vector instructions but not executed."]],
            caption="Status of each element of the evaluation.", widths=[2.4, 4.2])
    R.h2("7.2 The synthetic trace generator")
    R.p("An LLM was not run for the experiments in this report. Instead a trace generator produces, for each decode step, a query vector and a key and value vector with the structural properties that KV-cache managers are designed around. The generator has 64 channels (`head_dim`) and exposes these features:", align="j")
    R.bullets([("Attention sinks.", "The first four tokens, and 3 % of later tokens at random, receive an extra component along a fixed “heavy” direction; the query also has a component along it. These tokens attract disproportionate attention, as heavy hitters and sinks do in real models."),
               ("Outlier key channels.", "One in sixteen channels of the keys is scaled by 6× (4 of 64 channels). This is the structure that motivates per-channel key quantisation."),
               ("Topic phases.", "Four unit topic directions are defined. Keys in each phase of 96 tokens carry that phase's topic. The query's topic follows the phase but periodically (every third phase) returns to an older topic, so that previously cold blocks become hot again and must be promoted."),
               ("Local recency.", "Part of the query is the direction of one of the 12 most recent keys, producing local attention."),
               ("Sharpness.", "The query is scaled by a factor of 4, which controls how peaked the softmax is; value vectors are independent Gaussian noise.")])
    R.p("Because the generator is designed rather than learned, its absolute numbers must not be read as what a particular LLM would give. In particular, values that are independent noise average to a small-norm vector, which inflates the *relative* L2 error of the output; the comparison of variants against each other on the same stream is the meaningful quantity. Chapter 8 includes a measurement of how concentrated the attention on this generator is (Section 8.8), so that the reader can compare it with concentration reported for real models.", align="j")
    R.h2("7.3 Metrics")
    R.table(["Metric", "Definition", "Why"],
            [["`resident_vs_fp16`", "Total bytes held across tiers divided by the FP16 size of all tokens", "Memory saving (lower is better)"],
             ["`dropped_tokens`", "Tokens permanently removed", "Irreversible information loss"],
             ["`attn_rel_l2`", "‖o_sys − o_dense‖ / ‖o_dense‖ for the attention output at each step after a warm-up of 64 steps (mean and 95th percentile)", "Output fidelity vs a dense FP32 oracle over the same (FP16-rounded) data"],
             ["`attn_mass_recall`", "Fraction of the oracle's attention probability that falls on the tokens the system read", "How much of what attention wants was available"],
             ["`sim_fetch_ms_per_token`", "Modelled time to fetch and decode all blocks read in a step (mean)", "Latency of reading the KV cache under the assumed constants"],
             ["`flash_block_reads`", "Count of block reads served from flash", "Proxy for slow-tier traffic"],
             ["`migrations`, `reencodes`", "Cumulative counts of tier moves and codec changes", "Churn (energy and bandwidth cost of management)"],
             ["`overflow_total`", "Sum over tiers of the maximum bytes by which a tier budget was exceeded", "Feasibility: non-zero means the data did not fit"]],
            caption="Metrics reported in the experiments.", widths=[1.5, 3.2, 1.9])
    R.p("**Interpretation guard.** The attention-mass recall depends only on which blocks are read and not on their precision, so it is identical for all UAMM-2 variants that use the same selection logic; the codec choice shows up in the relative L2 error. A non-zero error floor remains even with FP16 everywhere, because block-sparse attention omits some tokens; the FP16-only variant (error 0.092 at 1,536 tokens) measures that floor.", align="j")
    R.p("**Accounting rule.** `resident_vs_fp16` and the tier budgets count the bytes of each block's encoded payload, including its scales and zero points. They do **not** count the small per-block summaries that the engine keeps resident for selection (key minimum and maximum, 64 values each) or the scheduling state. Stored in FP16, the key summaries would add about 256 bytes per block, i.e. about 25 KB for the 96 blocks of a 1,536-token run, which is roughly 6 % of the FP16 cache size and 15 % of the 168 KB budget. The results below should be read with that omission in mind; Chapter 12 lists it among the open items.", align="j")
    R.h2("7.4 Assumed cost constants")
    R.p("The following constants are placeholders for an edge SoC and are not measurements. They define the latency model used by the planner and the simulator. Conclusions about *ratios* between variants depend on them; they should be replaced by values measured on the target before any absolute claim is made.", align="j")
    R.table(["Constant", "Value", "Meaning"],
            [["T0 latency / bandwidth", "0.5 µs / 100 GB/s", "Near-compute SRAM or vector register file"],
             ["T1, T2 latency / bandwidth", "2 µs / 12.8 GB/s", "LPDDR-class DRAM"],
             ["T3 latency / bandwidth", "100 µs / 1 GB/s", "Flash"],
             ["Decode cost, fp16 / int8 / kivi4 / kivi2", "0 / 0.05 / 0.12 / 0.20 ns per byte", "Software dequantisation cost model"],
             ["zlib decode cost", "1.0 ns per byte", "Additional cost for the `z` variants"],
             ["Tier capacities T0..T3", "24, 48, 32, 64 KB (total 168 KB)", "FP16 KV for 1,536 tokens is 393 KB"],
             ["Block size, v_group, sink, recent window", "16 tokens, 32 channels, 4 tokens, 32 tokens", "Geometry and protection"],
             ["Attention block budget", "24 blocks per step", "Non-protected blocks fetched"],
             ["Planner: plan interval, hysteresis, residency, λ, κ, headroom", "8 steps, 0.25, 16 steps, 2.0, 0.05, 3 %", "See Chapter 5"],
             ["H2O decay, aggregation, hit_ema", "0.95, mean, 0.9", "Importance signal"]],
            caption="Default configuration of the experiments.", widths=[2.4, 2.2, 2.0])
    R.p("λ was chosen on seeds 7–9 during development; all reported experiments use other seeds (11–15), so the default configuration is evaluated on data it was not tuned on.", align="j")
    R.h2("7.5 Experimental design")
    R.table(["ID", "Question", "Variable", "Seeds", "Tokens"],
            [["E1", "Does each ingredient matter? (ablation)", "Policy and codec set: UAMM-1 threshold; planner with FP16 only; INT8+KIVI-4; all codecs", "5", "1,536"],
             ["E2", "How does the system scale with context?", "Context 512–3,072 at a fixed budget", "3", "512–3,072"],
             ["E3", "How many blocks must be read?", "Attention block budget 4–48", "3", "1,536"],
             ["E4", "How sensitive are the planner and importance knobs?", "λ, hysteresis margin, residency, plan interval, decay, aggregation, block size", "3", "1,536"],
             ["E5", "How does memory budget change the picture?", "Tier budgets ×0.5 to ×3.0; and codec sets with and without entropy coding", "3", "1,536"],
             ["E6", "What happens over time?", "Per-step traces of tiers, codecs, churn, error", "1", "1,536"],
             ["E7", "How good are the codecs, and does the key axis matter?", "Codec and outlier presence; per-channel vs per-token keys", "200 random blocks per cell", "—"],
             ["E8", "How concentrated is attention on the generator?", "Dense attention statistics at sampled steps", "1", "1,536"]],
            caption="Experiments performed.", widths=[0.4, 2.0, 2.6, 0.9, 0.7])
    R.p("Values are reported as mean ± standard deviation across seeds. With three seeds the standard deviation is itself a rough estimate; differences smaller than about one standard deviation should not be treated as significant. The experiment code is in `experiments/run_experiments.py`, listed in the appendices, and all raw numbers are tabulated in Appendix B.", align="j")
