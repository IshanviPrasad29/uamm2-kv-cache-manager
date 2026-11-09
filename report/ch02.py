from data import *
def build(R):
    R.h2("1.7 Overview of the Work")
    R.p("We started from an existing prototype, **UAMM-1**, a unified adaptive memory manager that combined PagedAttention-style block allocation, H2O-style importance scoring and a three-tier placement policy, with a small Verilog decision engine. We reviewed the literature on KV-cache management, identified the gaps that remained, and combined the most useful ideas with three external projects into a single system, **UAMM-2**.", align="j")
    R.table(["Ingredient", "Source", "Role in UAMM-2"],
            [["Paged blocks", "PagedAttention / vLLM", "Unit of allocation, migration and tracking"],
             ["Importance from attention", "H2O (heavy hitters)", "Decayed attention scores aggregated to block importance"],
             ["Asymmetric KV quantisation", "KIVI (and KVQuant ideas)", "Per-channel keys and per-token values at 8, 4 and 2 bits"],
             ["Query-aware block selection", "Quest", "Min/max key bounds to skip blocks without fetching them"],
             ["Multi-tier placement, joint codec decisions", "InfiniGen, EvicPress, TensorRT-LLM KV compression", "(tier, codec) planner with a utility cost and hysteresis"],
             ["Hardware codec engine", "Open hardware KV-compression engines", "Reference for group quantisation and outlier handling; kept as a swappable codec slot"],
             ["Vector processor", "Ara RISC-V vector processor", "Fused attention-score kernel that reads packed 4-bit keys directly"],
             ["Portable runtime", "TVM-based portable runtimes", "Adapter seam through which the engine would be attached to a deployed model"],
             ["Original UAMM-1", "Our prior work", "Block table, tiering concept, scorer, kept as an ablation baseline"]],
            caption="The combined system at a glance.", widths=[1.6, 2.1, 2.9])
    R.p("The combined system adds four things that none of the individual sources provide together:", align="j")
    R.numbered(["A **(tier, codec) placement** for every block, so that capacity, bandwidth and precision are traded off in one decision.",
                "A **utility planner** with hysteresis that prices latency, quality risk and capacity pressure, and a **one-way precision ratchet** that prevents silent quality surprises.",
                "**Hardware**: a sweep scorer, a 4-bit codec pipeline and a block table in Verilog, each verified bit-exactly against a Python golden model and mutation-tested.",
                "**A vector kernel** for the Ara vector processor that computes attention scores directly on packed 4-bit keys without first expanding them to floating point."])
    R.p("The system was implemented as a Python reference engine (about 1,400 lines), three Verilog modules with self-checking testbenches, a C kernel with a RISC-V Vector path, and a 56-test suite. A set of eight controlled experiments produced the results in Chapters 8 to 10, including waveforms rendered from real simulation dumps of the RTL.", align="j")
    R.callout("Headline results from the experiments in this report", [
        "At 1,536 tokens with a 168 KB total cache budget (FP16 would need 393 KB), the UAMM-1 threshold policy kept 0.367 of the FP16 footprint but permanently dropped 973 of 1,536 tokens on average and retained 0.758 of the attention mass. UAMM-2 with INT8/KIVI-4/KIVI-2 codecs kept all 1,536 tokens in 0.310 of the FP16 footprint and retained 0.939 of the attention mass.",
        "Modelled fetch time per token fell from 0.923 ms (UAMM-1) to 0.112 ms (UAMM-2, all codecs) under the cost constants stated in Chapter 7.",
        "The price is quantisation error: mean relative L2 error of the attention output was 0.166 for UAMM-2 (all codecs) against 0.479 for UAMM-1, whose error is dominated by lost tokens.",
        "All RTL checks (1,088 scorer checks, 400 codec vectors, 6,000 random block-table operations) pass bit-exactly and 12 of 12 injected RTL bugs are caught."])
