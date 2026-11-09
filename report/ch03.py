from data import *

def paper(R, title, cite, what, how, strengths, gaps, took):
    R.h3(title)
    R.p(what, bold_lead=cite, align="j")
    R.p(how, bold_lead="How it works.", align="j")
    R.p(strengths, bold_lead="What it solves.", align="j")
    R.p(gaps, bold_lead="What it leaves open for an edge device.", align="j")
    R.callout("Inference we drew", took, fill="EEF7EE")

def build(R):
    R.h1("2. What Already Exists: Research Before Our Solution")
    R.p("Before designing anything we surveyed how the community has attacked the KV-cache problem. This chapter describes the **existing algorithms individually**, as their authors presented them, not in any combination. For each one we record the mechanism, what it solves, what it leaves open for an edge device, and the inference we drew that influenced the design. Bibliographic details are in the References; recent preprints should be re-checked against their latest versions before formal submission.", align="j")

    R.h2("2.1 A taxonomy of approaches")
    R.p("KV-cache research can be organised by *which lever it pulls*. The table below lists the levers; the rest of the chapter takes them in turn.", align="j")
    R.table(["Lever", "Idea", "Representative work", "Primary saving"],
            [["Change the model", "Fewer KV heads, or a compressed latent KV", "MQA, GQA, multi-head latent attention (DeepSeek-V2)", "Bytes per token (requires training)"],
             ["Manage memory better", "Page the cache into blocks to remove fragmentation", "PagedAttention / vLLM", "Wasted capacity"],
             ["Evict tokens", "Keep only tokens judged important", "H2O, Scissorhands, StreamingLLM, SnapKV, FastGen, PyramidKV", "Capacity and bandwidth (lossy, irreversible)"],
             ["Select at read time", "Keep everything but fetch only relevant blocks", "Quest, InfiniGen", "Bandwidth"],
             ["Quantise", "Store K and V in fewer bits", "KIVI, KVQuant", "Capacity and bandwidth"],
             ["Low-rank / projection", "Store low-rank projections of K/V", "Palu", "Capacity and bandwidth"],
             ["Offload", "Spill to CPU memory or flash and prefetch", "FlexGen, InfiniGen, LLM in a flash", "Capacity"],
             ["Joint systems", "Combine compression, eviction and tiering", "EvicPress, KV-Compress, TensorRT-LLM KV-compression framework", "Several at once"],
             ["Hardware engines", "Dedicated logic for compress/decompress or cache control", "hardware KV-compression engines, CXL-SpecKV, Ara vector unit", "Energy and throughput"],
             ["Deployment runtimes", "Compile and run models portably on many devices", "TVM-based compiler runtimes", "Portability"]],
            caption="Levers used by the existing work.", widths=[1.2, 2.0, 2.3, 1.1])

    R.h2("2.2 Changing the attention architecture")
    R.h3("Multi-query attention (MQA) and grouped-query attention (GQA)")
    R.p("Shazeer's multi-query attention (2019) shares one key head and one value head across all query heads; Ainslie et al.'s grouped-query attention (2023) interpolates, sharing each KV head across a group of query heads. Both reduce the KV cache by the ratio of query heads to KV heads without changing the sequence length dimension. They are now standard in many open models (the 8-KV-head geometry used in Chapter 1 is an example).", align="j")
    R.p("**Limit.** They must be built into the model during training or fine-tuning. For a model already on the device they offer nothing, and the cache still grows linearly with context.", align="j")
    R.callout("Inference we drew", "Architecture-level savings are complementary to ours and multiply with them. Our engine treats the per-token K and V as vectors of whatever size the model produces, so a GQA or latent-attention model simply gives it smaller vectors. We did not attempt to change the model.", fill="EEF7EE")
    R.h3("Multi-head latent attention")
    R.p("DeepSeek-V2 compresses keys and values into a shared low-dimensional latent vector per token and re-expands them inside the attention computation, cutting the cache by an order of magnitude for that model family. As with GQA, the saving requires the model to be trained with this structure.", align="j")

    R.h2("2.3 Memory management: PagedAttention and vLLM")
    paper(R, "PagedAttention / vLLM", "Kwon et al., SOSP 2023.",
          "Before vLLM, serving systems reserved one contiguous slab of memory per request sized for the *maximum* sequence length, because the final length is unknown. This wasted most of the allocated memory through internal and external fragmentation.",
          "PagedAttention borrows the idea of virtual-memory paging. The KV cache of each sequence is split into fixed-size **blocks**; a per-sequence block table maps logical blocks to physical blocks scattered anywhere in memory. Blocks are allocated on demand, so waste is limited to the unfilled tail of the last block. The attention kernel gathers keys and values through the block table. Blocks can be shared between sequences (copy-on-write) for common prefixes.",
          "Near-zero fragmentation (the authors report under 4 % waste), simple allocation and the ability to share prefixes. It made memory management of the KV cache a solved engineering problem for GPU serving.",
          "PagedAttention solves *how memory is laid out*, not *how much memory is needed* or *which blocks matter*. It has no notion of importance, no compression, no tiers below the GPU/CPU pair, and it assumes every block of a sequence is read for every token.",
          "Blocks are the right unit. We adopted fixed-size blocks and a block table, and, importantly, made the block the unit that carries metadata (importance, selection frequency, codec, tier, key min/max summary). A block table is also a natural hardware structure; it became one of our three RTL modules.")

    R.h2("2.4 Token eviction")
    R.p("A family of methods observes that attention is highly concentrated, so most cached tokens contribute almost nothing to any given step, and drops them. Chapter 8 measures the same concentration on our synthetic workload.", align="j")
    paper(R, "H2O: Heavy-Hitter Oracle", "Zhang et al., NeurIPS 2023.",
          "H2O notes that attention scores follow a power law: a small set of tokens (“heavy hitters”) receives most of the accumulated attention across steps.",
          "The method keeps a fixed budget of tokens: a window of recent tokens plus the tokens with the highest accumulated attention score. When the budget is full, the lowest-scoring non-recent token is evicted. Scores are the running sum of attention weights that each token has received. Reported experiments retain quality with about 20 % of the cache on the models tested.",
          "A principled, cheap importance signal that is computed from attention weights the model already produces, and a simple eviction rule.",
          "H2O decides **only whether a token lives or dies**. There is no quantisation, so what is kept stays at full precision; there is no lower tier, so evicted tokens are gone forever; and a token that is unimportant now but important to a later question cannot be recovered. It also does not reduce bytes per kept token.",
          "We kept H2O's *signal* (decayed attention mass per token, aggregated to blocks) but changed what it controls. In UAMM-2 the signal drives **placement and precision**, which are reversible or bounded, while permanent drop is an explicit opt-in. This directly addresses the gap noted above: H2O does eviction but no quantisation.")
    paper(R, "Scissorhands", "Liu et al., NeurIPS 2023.",
          "Scissorhands builds on the “persistence of importance” hypothesis: tokens that mattered in the past tend to matter in the future.",
          "It keeps a budget of tokens, using the history of attention scores to decide which to discard, with compression performed when the budget is exceeded.",
          "It reaches large compression with a light-weight rule and no training.",
          "Like H2O it is eviction-only. The persistence assumption is also challenged by workloads whose focus shifts across topics.",
          "Persistence is a reasonable prior, but it can fail when the query returns to an old topic. Our synthetic workload therefore includes topic phases that revisit older topics, to test whether cold blocks can be promoted back.")
    paper(R, "StreamingLLM and attention sinks", "Xiao et al., ICLR 2024.",
          "StreamingLLM observes that the first few tokens of a sequence receive disproportionate attention regardless of content (“attention sinks”), and that removing them breaks the model.",
          "The method keeps the first few tokens plus a sliding window of recent tokens and evicts the rest, enabling streaming over very long inputs with a bounded cache.",
          "A trivial, fast policy that is stable for unbounded streams.",
          "The evicted middle is lost; the method does not improve the model's ability to use information from the evicted span.",
          "We protect **sink tokens** (the first four) and a **recent window** (32 tokens) unconditionally, hold them in the fastest tier at full precision, and never move or compress them. The protection is also implemented in the RTL scorer (its `protected` input).")
    paper(R, "SnapKV", "Li et al., 2024.",
          "SnapKV targets long prompts. It uses an observation window at the end of the prompt to vote on which earlier positions are relevant and compresses the prompt cache accordingly.",
          "Attention weights from the window to earlier positions are pooled, the highest-voted positions are retained per head, and the rest are dropped during prefill.",
          "Substantial prompt-cache reduction with little quality loss on long-context benchmarks.",
          "It is a one-shot decision at the end of prefill and does not adapt as generation proceeds; dropped positions cannot be recovered.",
          "Importance can be estimated from a small window rather than from all history. We retained the continuously updated H2O-style score (cheap, vectorised) rather than a one-shot vote, but noted that a prefill-time seed for the score is a natural extension (Chapter 12).")
    paper(R, "FastGen and PyramidKV", "Ge et al., 2023; Cai et al., 2024.",
          "FastGen profiles each attention head and applies a different eviction policy per head; PyramidKV allocates different budgets to different layers, giving more to lower layers where attention is broad.",
          "Both use structure in how attention behaves across heads or layers to allocate a fixed budget unevenly.",
          "Better quality for the same total budget than uniform schemes.",
          "They again decide only eviction and presume per-layer or per-head access that a flattened single-vector cache does not model.",
          "Budgets should be non-uniform across layers and heads. Our reference engine flattens K and V per token into a single vector per layer-head group; per-layer and per-head budgets are listed in Chapter 12 as unfinished work, and the block metadata structure was chosen so that a (layer, head) index can be added.")

    R.h2("2.5 Query-aware selection and retrieval")
    paper(R, "Quest", "Tang et al., ICML 2024.",
          "Quest observes that which tokens matter depends on the **current query**, so a fixed eviction rule is wasteful.",
          "For each page of the KV cache it stores the element-wise minimum and maximum of the keys. At decode time it computes, from the query and these two vectors, an upper bound on the attention score any key in the page can have, selects the pages with the highest bounds, and runs attention only on them. Nothing is evicted; everything remains stored.",
          "Large reductions in memory reads (and therefore decode latency) with little accuracy loss, without losing information.",
          "Quest reduces **bandwidth** but not **capacity**; the whole cache must still be resident, which is exactly what an edge device cannot afford.",
          "This is the technique that makes tiering practical. If the system can bound which blocks are needed before fetching them, cold blocks can live in slow memory at low precision, because most steps never touch them. We adopted min/max key summaries per block and the bound `Σ_c max(q_c·kmin_c, q_c·kmax_c)`, and our tests verify that it is a true upper bound. Quest's capacity gap is closed by our tiers and codecs.")
    paper(R, "InfiniGen", "Lee et al., OSDI 2024.",
          "InfiniGen targets the case where the KV cache does not fit in GPU memory and lives in CPU memory.",
          "It uses a lightweight speculation, computed from a skewed view of the weights, to predict at layer *i* which tokens layer *i+1* will need, and prefetches only those from CPU memory.",
          "Large speedups over naive offloading by reading only the needed entries across the slow link.",
          "It handles one slow tier, assumes a GPU/CPU split and does not change precision.",
          "A slow tier is usable if the fetch set is predicted ahead of time. Our reference engine uses bound-ranking at read time (not prefetch) and lists **prefetching** as a missing piece (Chapter 12). The tier model (SRAM, DRAM hot, DRAM warm, flash) generalises InfiniGen's two-level view.")

    R.h2("2.6 Quantisation of the KV cache")
    R.p("Weights are routinely quantised to 4 bits; the KV cache is harder because its statistics differ between keys and values and, in particular, a few key channels carry very large magnitudes.", align="j")
    paper(R, "KIVI", "Liu et al., ICML 2024.",
          "KIVI studies the element distributions of the KV cache in popular models and reports that keys have a few channels with large magnitudes (outlier channels) while values do not show this pattern.",
          "It quantises **keys per channel** (one scale and zero-point per channel across a group of tokens) and **values per token** (one scale and zero-point per token across channels), using asymmetric 2-bit integers, and keeps a small recent window in full precision. It needs no fine-tuning.",
          "About 2.6× lower peak memory (including weights) at 2 bits with small quality loss on the models tested, enabling larger batches or longer contexts.",
          "KIVI is applied uniformly. Every block gets the same precision whether it is attended a thousand times or never. It does not decide placement, and it still reads the entire quantised cache each step.",
          "The axis of quantisation matters. We implemented per-channel keys and per-token values and measured the effect directly (Chapter 8): per-channel key quantisation reduced error by 1.9× at 2 and 4 bits on outlier-free data and 5.4× at 2 bits and 5.2× at 4 bits when outlier channels are present, which supports the paper's central design choice. We then used the codec **selectively** per block, which uniform quantisation cannot do.")
    paper(R, "KVQuant", "Hooper et al., NeurIPS 2024.",
          "KVQuant pushes KV quantisation toward sub-4-bit precision for very long contexts.",
          "It combines per-channel key quantisation, quantising keys before the rotary positional embedding, non-uniform datatypes fit per layer, dense-and-sparse handling of outliers, and normalisation of the quantisation centroids.",
          "Context lengths of millions of tokens on large models with modest quality loss.",
          "The techniques require calibration data and custom kernels; they are applied uniformly and aimed at GPUs.",
          "Outliers must be handled explicitly. Our codec supports separate treatment of outlier channels (the key-axis experiment in Chapter 8), and our hardware quantiser follows the simple per-group form because outlier handling is cheaper in the surrounding logic. The more elaborate KVQuant ingredients (pre-RoPE keys, non-uniform datatypes) are listed as future work.")

    R.h2("2.7 Low-rank compression")
    paper(R, "Palu", "Chang et al., 2024.",
          "Palu observes that the projection matrices that produce K and V can be factorised so that a low-rank latent is cached instead of the full vectors.",
          "It decomposes the projection weights into low-rank factors, stores the latent, and reconstructs keys and values on the fly with a fused kernel; it adds rank-search and quantisation-friendly transformations.",
          "Compression that composes with quantisation, on a pre-trained model, without retraining from scratch.",
          "It adds reconstruction compute and requires modifying how the model is run. The latent still has to be stored for every token.",
          "Low-rank storage is orthogonal to the codecs and could be added as another codec family. It was not included because it needs access to the model's projection weights, which our reference engine does not have.")

    R.h2("2.8 Offloading and flash")
    R.p("**FlexGen** (Sheng et al., ICML 2023) offloads weights and KV cache across GPU, CPU memory and disk for throughput-oriented batch inference, using a linear-programming search over placement and compression (it also compresses weights and cache to 4 bits). **LLM in a flash** (Alizadeh et al., 2023) shows how to run models larger than DRAM from flash by minimising the number of transferred bytes and the number of reads, with windowing and row–column bundling.", align="j")
    R.callout("Inference we drew", "Flash is usable only if accesses are rare, large and sequential. That is why our flash tier stores only entropy-coded, low-precision blocks and why the planner charges a large fixed cost per flash access. Our experiments (Chapter 8) show that the planner keeps flash reads low as long as replanning is frequent and the blocks that live in flash are genuinely cold.", fill="EEF7EE")

    R.h2("2.9 Joint systems")
    R.p("A newer line of work recognises that compression, eviction and tiering interact and tries to decide them together. We reviewed the following; they influenced the design most.", align="j")
    R.table(["Work", "What it does", "What we took", "What we did differently"],
            [["EvicPress", "Jointly chooses compression level and eviction across storage tiers using a utility function", "The core idea of a joint utility, with an explicit quality term", "Added hysteresis, a precision ratchet, query-aware reads and a verified hardware scorer"],
             ["KV-Compress / Zipage (paged eviction)", "Eviction at block granularity inside a paged cache, with variable budgets per head", "Eviction and paging should share the same granularity", "Default is lossless tiering; drop is opt-in"],
             ["TensorRT-LLM KV-cache compression framework", "Applies a codec when pages move between tiers (cold-page compression)", "Compress at tier transitions, not on the hot path", "Added codec ratcheting and per-block measured error tables"]],
            caption="Joint and system-level work reviewed, and how UAMM-2 relates to each.", widths=[1.3, 1.9, 1.7, 1.7])
    R.p("None of these is, to our knowledge, a complete edge-oriented stack that includes a verified decision engine in hardware and a vector kernel for a RISC-V vector unit. That gap is where UAMM-2 sits.", align="j")

    R.h2("2.10 Hardware: engines and vector processors")
    paper(R, "Hardware KV-compression engines", "Open RTL implementations and accelerator papers.",
          "Several designs implement the compress/decompress datapath for KV data in hardware.",
          "A typical engine quantises keys and values in groups (for example 64 values per group) with separate handling of outlier values, using different bit widths for keys (4-bit) and values (2-bit), and may evict a fraction of tokens. Published headline ratios (for example 8×) are often higher than what a benchmark of the same design measures (3–4×), because scales, zero points and outlier lists consume part of the saving.",
          "A concrete, synthesisable datapath for compression, useful as an architectural reference.",
          "They contain no policy: nothing decides which blocks to compress, how far, or where to store them. Built-in eviction conflicts with a design that avoids irreversible loss by default.",
          "Advertised compression ratios must be checked against measured ones including metadata. Our own codec table (Chapter 8) reports measured ratios with scales and zero points counted. The codec interface (`encode`/`decode`/`Payload`) is the slot where such an engine's format can be plugged in.")
    paper(R, "Ara and Ara2", "Cavalcante et al., IEEE TVLSI 2020; Perotti et al., 2023.",
          "Ara is an open-source RISC-V Vector (RVV 1.0) coprocessor designed to be attached to the CVA6 application core; Ara2 extends it for multi-lane scaling.",
          "Ara executes vector instructions on 2–16 parallel lanes, each with its own slice of the vector register file, an FPU and integer units, plus a vector load/store unit that fetches data over AXI. Vector length is configurable at run time; the programming model is standard RVV.",
          "Open, synthesisable vector compute with a clean memory interface, suited to a custom SoC that includes a KV manager.",
          "Ara is a compute engine and has no concept of KV blocks, tiers or codecs. An LLM kernel for Ara must be written and benchmarked on Ara itself.",
          "We placed our **fused QK kernel** on Ara: it loads packed 4-bit key bytes with vector byte loads, widens and converts them, and accumulates with fused multiply–add, without first writing an expanded floating-point key block to memory. This saves exactly the bandwidth that a dequantise-then-dot approach would waste. The kernel compiles for `rv64gcv` and lowers to the expected vector instructions; execution on Ara or a simulator is future work (Chapter 12).")
    paper(R, "CXL-SpecKV and other cache-engine proposals", "arXiv preprints and project pages.",
          "CXL-SpecKV describes an FPGA-based cache engine that places KV data on CXL-attached memory with speculative fetch.",
          "A hardware controller predicts and prefetches KV blocks from a larger, slower memory pool.",
          "Shows that a dedicated controller can hide slow-memory latency for KV data.",
          "Targets data-centre CXL fabrics rather than an edge SoC.",
          "A hardware controller for the cache is a legitimate architecture. In UAMM-2 the per-block scoring and the block table are implemented as RTL so that the planner's inner loop does not occupy the CPU.")

    R.h2("2.11 Deployment runtime: compiler-based runtimes")
    paper(R, "TVM-based compiler runtimes", "Chen et al., OSDI 2018 (TVM).",
          "Compiler-based runtimes built on TVM are machine-learning compilation frameworks for deploying LLMs on many targets (phones, browsers, GPUs). It builds on TVM and its newer Unity design with dynamic-shape support.",
          "Models are compiled into target-specific kernels; quantised weights are supported; the runtime exposes an OpenAI-style chat API and manages a KV cache internally.",
          "Real portability: the same model can be compiled for several hardware back ends.",
          "Its KV cache is internal to the runtime. We have not verified the details of its KV-cache interface, nor that TVM can target RISC-V vector back ends such as Ara end to end.",
          "A runtime is an adapter target, not a design dependency. We therefore defined a narrow `RuntimeAdapter` interface (append token, get attended blocks, report) with a working reference adapter, and an `CompilerRuntimeAdapter` skeleton with the integration checklist. This keeps the research engine testable on its own while leaving a clear seam for deployment.")

    R.h2("2.12 Coverage matrix")
    R.p("", align="j")
    R.fig("f04_coverage_matrix.png", "Capability coverage of the related work. No single prior system covers placement, precision, query-aware selection, importance scoring, a hardware decision engine and a vector kernel together.", 6.4)
    R.h2("2.13 Summary of the gaps")
    R.numbered(["**Eviction without precision control** (H2O, Scissorhands, StreamingLLM, SnapKV): loss is irreversible and kept tokens stay at full size.",
                "**Quantisation without selection or placement** (KIVI, KVQuant): uniform precision; the whole cache is still read every token.",
                "**Selection without capacity savings** (Quest): reads fewer bytes but needs everything resident.",
                "**Offloading without compression** (FlexGen, InfiniGen): one slow tier; no joint precision decision.",
                "**Joint ideas without an edge hardware realisation** (EvicPress, TensorRT-LLM framework): no verified decision hardware, no RISC-V vector path.",
                "**Hardware engines without a policy layer** (compress engines, Ara): datapaths exist, but nothing decides what to compress and where to put it."])
    R.p("UAMM-2 is an attempt to fill the space between these gaps in one consistent design. The next chapters describe the starting point and the result.", align="j")
