from data import *
def build(R):
    R.h1("11. Research Papers, Inferences and How We Used Them")
    R.p("This chapter collects, source by source, what we concluded and how it changed the design. The detailed narrative for most entries is in Chapter 2; here they are consolidated so that the provenance of each design decision can be traced.", align="j")
    R.table(["Source", "Inference we drew", "How we used it in UAMM-2", "Not used, and why"],
            [["PagedAttention / vLLM (SOSP 2023)", "Blocks are the right unit; block tables are cheap", "Fixed 16-token blocks, block table RTL", "Prefix sharing / copy-on-write (open item)"],
             ["H2O (NeurIPS 2023)", "Attention is power-law; accumulated attention is a cheap importance signal", "Decayed token score → block importance (normalised by p95)", "Irreversible eviction as default"],
             ["Scissorhands (NeurIPS 2023)", "Importance persists, but can shift", "Workload with topic revisits; ability to promote cold blocks", "Eviction-only budgeting"],
             ["StreamingLLM (ICLR 2024)", "Sink tokens + recent window must be kept", "Protected blocks (RTL and Python)", "Dropping the middle"],
             ["SnapKV (2024)", "A small observation window can seed relevance", "Noted as a prefill-time seed (future)", "One-shot prompt compression"],
             ["FastGen / PyramidKV", "Budgets should differ by head and layer", "Structure for future (layer, head) index", "Per-head policies (open item)"],
             ["Quest (ICML 2024)", "Min/max page summaries give a query-aware bound", "Selection by `Σ max(q·kmin, q·kmax)`; tested as a true bound", "Page-level only; no token-level refinement"],
             ["InfiniGen (OSDI 2024)", "A slow tier works if the fetch set is predicted", "Multi-tier model; planner prices tier latency; prefetch listed as next step", "Layer-ahead speculation"],
             ["KIVI (ICML 2024)", "Quantise keys per channel and values per token", "Codec layer; measured 1.9–5.4× lower key error", "Residual full-precision window (we use protected T0 blocks instead)"],
             ["KVQuant (NeurIPS 2024)", "Outliers must be handled explicitly; sub-4-bit needs care", "Outlier-aware test data; hardware quantiser kept simple", "Non-uniform datatypes, pre-RoPE keys (open item)"],
             ["Palu (2024)", "Low-rank storage composes with quantisation", "Left as a future codec family", "Needs model weights"],
             ["FlexGen (ICML 2023)", "Placement across tiers can be searched; offload needs compression", "Tier + codec search, greedy rather than LP", "LP-based optimiser (open item)"],
             ["LLM in a flash", "Flash needs large, rare, sequential reads", "Flash holds only cold, entropy-capable blocks; large fixed cost per access", "Row–column bundling"],
             ["MQA / GQA / MLA", "Architecture changes shrink K/V at training time", "Engine accepts any vector size", "No model changes"],
             ["EvicPress", "Compression and eviction should be decided jointly with a utility", "Utility planner (λ-weighted error + latency + congestion)", "Our additions: hysteresis, ratchet, query-aware reads, RTL"],
             ["KV-Compress / Zipage", "Evict at the paging granularity", "Block as unit for every decision", "Variable per-head budgets"],
             ["TensorRT-LLM KV compression", "Compress when pages change tier", "Codec change is part of the same plan step as the move", "Vendor-specific kernels"],
             ["CXL-SpecKV", "A hardware cache controller can hide slow-memory latency", "RTL block table + sweep scorer as controller pieces", "CXL fabric"],
             ["Ara / Ara2", "Open RVV vector unit with AXI memory path", "Fused QK kernel; codec RTL placed on the load path", "Anything run on Ara itself (open item)"],
             ["TVM / compiler runtimes", "A runtime is an adapter target, not a design dependency", "`RuntimeAdapter` seam; skeleton with checklist", "Actual integration (open item)"],
             ["SmoothQuant", "Targets weights and activations, not the KV cache", "Removed from UAMM-1's list of KV solutions", "—"],
             ["Our UAMM-1", "Placement without precision cannot fit a tight budget", "Ablation baseline; 10 defects fixed", "Threshold policy as default"]],
            caption="Sources, inferences and effects on the design.", widths=[1.6, 2.0, 1.9, 1.2], size=7.8)
    R.h2("11.1 Cross-cutting lessons")
    R.numbered(["**Combine axes, not just methods.** The ablation shows that placement alone, codecs alone and eviction alone each leave the budget problem unsolved or solved by loss.",
                "**Measure, do not assume.** Per-block calibrated error is more reliable than a formula; advertised compression ratios are not measured ones.",
                "**Make irreversibility explicit.** The ratchet and opt-in drop make every loss visible and bounded.",
                "**Churn is a first-class cost.** Hysteresis and replanning cadence changed migrations by up to 4× and flash reads by up to 5×.",
                "**Hardware should accelerate the inner loop and be verified like software.** Golden models, boundary vectors and mutation testing found real defects (an ignored `protected` input, an untested clamp).",
                "**Keep seams.** Codec, runtime and kernel boundaries let each piece be replaced as real measurements arrive."])
