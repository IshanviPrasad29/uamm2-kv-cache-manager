from data import *

def build(R):
    R.h1("1. The Problem: Running Large Language Models on Edge Processors")
    R.p("Large language models (LLMs) have moved from data-centre curiosities to components that product teams now want inside phones, laptops, vehicles, wearables, industrial controllers and embedded boards. The motivations are practical and well understood: **latency** (no network round trip), **privacy** (prompts and documents never leave the device), **availability** (the feature works offline) and **cost** (no per-token cloud bill). This report is about one specific obstacle that stands between that aspiration and a working product: the memory that a running LLM needs for its *attention history*, the **key–value (KV) cache**, and how that memory can be managed on the small, heterogeneous, power-constrained processors that live at the edge.", align="j")
    R.p("The report is organised as follows. Chapter 1 states the problem. Chapter 2 describes the algorithms that already exist for it, each on its own. Chapter 3 describes our first approach, UAMM-1, which combined two of them, and Chapter 4 sets out the gaps it left. Chapter 5 presents UAMM-2, the system built to close those gaps, and Chapter 6 explains its directory and every part of its code. Chapters 7 to 10 give the experimental setup, the simulation results, the hardware waveforms and the vector-kernel results. Chapter 11 lists the research papers we drew on, the inference taken from each and how it was used. Chapter 12 describes the problems UAMM-2 still does not address, with the solutions and implementation ideas we worked out for them. Chapter 13 places the results in a realistic frame and Chapter 14 concludes.", align="j")

    R.h2("1.1 Why LLMs on edge devices at all")
    R.p("An *edge processor* in this report means a processor that runs the model locally rather than in a data centre: a mobile application processor, an embedded RISC-V or Arm SoC, a single-board computer, a small NPU-equipped module. Typical constraints, compared with a data-centre accelerator, are shown in the table below.", align="j")
    R.table(["Resource", "Data-centre accelerator", "Edge SoC (typical range)", "Consequence"],
            [["Memory capacity", "80–192 GB HBM per accelerator", "2–16 GB shared LPDDR, often shared with the OS and other apps", "Weights and KV cache compete for the same few gigabytes"],
             ["Memory bandwidth", "2–5 TB/s (HBM)", "10–60 GB/s (LPDDR4/5)", "Two orders of magnitude less; decode is bandwidth-bound"],
             ["Compute", "Hundreds of TFLOP/s", "0.1–40 TOPS (NPU/GPU/vector unit)", "Compute is rarely the first limit during decoding"],
             ["Power / thermal", "300–700 W", "2–15 W sustained", "Every DRAM access costs energy that the battery must pay"],
             ["Storage tier", "NVMe, network storage", "eMMC / UFS / SD flash", "Flash exists but is 10–100× slower than DRAM"],
             ["Batching", "Dozens of concurrent requests", "Usually one user, one sequence", "Batching cannot amortise weight reads"]],
            caption="Order-of-magnitude comparison between a data-centre accelerator and an edge SoC. Ranges are representative, not measurements of one device.",
            widths=[1.1, 1.5, 2.0, 1.9])
    R.p("The right-hand column is the crux. Almost every technique that makes LLM serving efficient in the cloud relies on batching many requests together so that each read of the weights is shared by many sequences. An edge device typically serves one sequence at a time, so the cost of reading the model is paid by a single user. That makes the work of decoding a single token almost entirely a question of how many bytes must be moved from memory, which is why memory, not arithmetic, is the central resource on the edge.", align="j")

    R.h2("1.2 How an LLM generates text: prefill and decode")
    R.p("A decoder-only transformer produces text one token at a time. Generation has two phases with very different hardware behaviour.", align="j")
    R.bullets([("Prefill.", "The prompt (say 1,000 tokens) is processed in parallel. All prompt tokens are pushed through every layer at once. The work is dominated by matrix–matrix multiplications and is compute-bound; hardware with strong vector or matrix units handles it well."),
               ("Decode.", "Each new token is produced by one pass through the network. For that single token the processor must read essentially every weight of the model, and in every attention layer it must also read the stored keys and values of every previous token. The work is matrix–vector multiplication, which performs very little arithmetic per byte moved, and is therefore memory-bandwidth-bound.")])
    R.p("Because users experience decode as the streaming speed of the answer (tokens per second), and because the decode loop repeats for every token, the number of bytes read per token is the primary determinant of perceived speed on an edge device.", align="j")

    R.h2("1.3 What the KV cache is")
    R.p("Inside every attention layer, each token produces three vectors from its hidden state: a **query** (q), a **key** (k) and a **value** (v). Attention for the newest token computes a similarity score between its query and the key of every earlier token, converts the scores into weights with a softmax, and returns the weighted sum of the earlier tokens' values:", align="j")
    R.eq("o_t = Σ_{i ≤ t} softmax_i( q_t · k_i / √d ) · v_i")
    R.p("The keys and values of earlier tokens never change once computed (the model is causal). Recomputing them for every new token would multiply the work enormously, so inference engines store them. This store is the **KV cache**. It is the model's working memory of the conversation: every layer, every KV head, every token contributes one key vector and one value vector that must be kept for as long as the token can still be attended to.", align="j")
    R.p("The size of that memory follows a simple formula:", align="j")
    R.eq("bytes per token = 2 (K and V) × layers × KV-heads × head-dim × bytes per element")
    R.p("and the cache for one sequence is that number multiplied by the current context length. Unlike the weights, which are a fixed size, the KV cache **grows with every generated token** and, in a multi-turn session or a document-question task, may become the largest single object in memory.", align="j")

    rows = []
    for name, (L, H, D) in MODELS.items():
        b = bpt(L, H, D)
        rows.append([name, f"{L} / {H} / {D}", f"{b:,}", f"{b*4096/2**20:,.0f} MiB", f"{b*32768/2**30:,.2f} GiB", f"{b*131072/2**30:,.1f} GiB"])
    R.table(["Illustrative geometry", "layers / KV heads / head-dim", "bytes per token (FP16)", "4K context", "32K context", "128K context"], rows,
            caption="KV-cache size for four illustrative model geometries (FP16, one sequence). Geometries are representative of model families, not of any specific checkpoint.",
            widths=[1.3, 1.3, 1.2, 0.9, 0.9, 0.9])
    R.fig("f01_kv_growth.png", "KV-cache size grows linearly with context. Dashed lines show plausible free-DRAM budgets on edge devices; every geometry except the smallest crosses one of them within the context lengths that applications now request.", 5.7)

    R.h2("1.4 Why the KV cache is the bottleneck on the edge")
    R.h3("1.4.1 Capacity")
    R.p("On a phone with 8 GB of RAM, the operating system and foreground applications may leave 3–5 GB available to an on-device model. A 7–8-billion-parameter model quantised to 4 bits needs roughly 4–5 GB for its weights alone. The weights are fixed, so what remains for the KV cache shrinks to a few hundred megabytes. In the Llama-3-8B-like geometry of Table 2 that budget is exhausted at about 2–3 thousand tokens of FP16 cache, far short of the 32K–128K contexts that long-document and agentic applications need. When memory is exhausted the platform kills the process, or the engine must truncate the context, or the application must refuse the request.", align="j")
    R.h3("1.4.2 Bandwidth")
    R.p("Capacity is only half of the problem. Standard attention reads **the entire KV cache for every generated token**. Even if the cache fits, its size determines how many bytes are moved per token, and therefore the maximum speed. The next figure shows the upper bound on decode speed imposed by KV reads alone for a 12.8 GB/s memory system, ignoring weight reads and compute. For a 7B-class model with full multi-head attention the bound drops below ten tokens per second at roughly 3K tokens of context, before a single weight has been read.", align="j")
    R.fig("f02_kv_bandwidth_bound.png", "Upper bound on decode speed from KV-cache reads alone (12.8 GB/s DRAM). The curves fall as 1/context: this is the bandwidth wall that appears even when the cache fits in memory.", 5.7)
    R.h3("1.4.3 Energy")
    R.p("Each byte read from DRAM costs on the order of tens to a hundred times more energy than an arithmetic operation on that byte, and flash costs more again. A design that reads the whole cache for every token spends the battery on data movement. Reducing the bytes touched per token is therefore both a speed and a battery-life objective.", align="j")
    R.h3("1.4.4 Heterogeneous memory")
    R.p("Edge SoCs expose a *hierarchy*: a few tens of kilobytes of tightly coupled SRAM or vector register file next to the compute unit, gigabytes of DRAM behind a shared bus, and gigabytes of flash behind a controller. The hierarchy is an opportunity (cold data can be parked cheaply) and a hazard (a wrong placement decision puts hot data behind a slow link). Data-centre inference engines were not designed around this hierarchy, and most KV-cache research assumes a GPU with high-bandwidth memory and a host CPU as the only second tier.", align="j")
    R.h3("1.4.5 Why “just buy more memory” is not an answer")
    R.p("For consumer and embedded products the memory is part of the bill of materials, shares a thermal envelope with the processor and is soldered. The memory in a shipped device cannot be upgraded after the fact, whereas the context lengths that applications demand grow every year. Any solution must therefore make the same hardware serve longer contexts and do so within a fixed energy budget.", align="j")

    R.h2("1.5 Why the obvious fixes fall short")
    R.table(["Obvious fix", "What it gets you", "Why it is not sufficient alone"],
            [["Truncate the context to a fixed window", "Bounded memory and bandwidth", "Discards information the model may need; quality collapses on tasks that refer back to early content"],
             ["Store the cache in FP16 but page it to flash", "Capacity", "A flash read per token is far too slow; without knowing which blocks matter the system stalls"],
             ["Quantise the entire cache uniformly to 4 bits", "≈3× capacity and bandwidth", "Applies the same precision to important and unimportant data; still reads everything each token; 2-bit quantisation hurts quality"],
             ["Evict low-importance tokens permanently", "Capacity and bandwidth", "Irreversible: a token that becomes relevant later is gone; no precision control for what remains"],
             ["Use a bigger or smaller model", "Moves the problem", "Smaller models lose quality; the KV cache of any model still grows with context"]],
            caption="Single-technique fixes and why each leaves part of the problem open.", widths=[1.7, 1.6, 3.3])
    R.p("Each fix addresses one dimension (capacity, bandwidth, precision, placement) and ignores the others. The central observation behind this work is that these dimensions interact: the best precision for a block depends on where it is stored, how likely it is to be read, and how much room is left. A system that decides them separately will make poor joint decisions.", align="j")

    R.h2("1.6 Problem statement and design requirements")
    R.p("We therefore pose the problem as follows.", align="j")
    R.callout("Problem statement", "Given an autoregressive LLM decoding one sequence on an edge processor with a small fast memory, a larger DRAM and a slow flash tier, keep the KV cache within a fixed byte budget and reduce the bytes read per token, while keeping the attention output as close as possible to what an unconstrained FP16 dense implementation would produce, without irreversibly discarding information unless explicitly allowed.")
    R.p("From that statement we derived the requirements that shaped the design:", align="j")
    R.numbered([
        "**Unit of management is a block, not a token.** Per-token bookkeeping is too costly and defeats memory locality; paging into fixed-size blocks makes allocation, migration and hardware tables simple.",
        "**Placement and precision are decided together.** A block is placed in a (tier, codec) pair; the two choices are never made independently.",
        "**Importance must be measured, not assumed.** Block priority must come from what attention actually does, not only from age.",
        "**Reads must be selective.** The system should be able to skip fetching and decoding blocks that provably cannot matter for the current query.",
        "**Loss must be controlled and visible.** Precision may only decrease along an explicit ratchet; permanent drop is opt-in; capacity overflow is reported rather than hidden.",
        "**Decisions must be cheap enough for hardware.** The inner scoring loop must be implementable without a divider and verified bit-exactly against a software model.",
        "**The design must connect to real runtimes and real vector hardware.** The codecs must be swappable, the compute kernel must target a RISC-V vector unit, and a runtime adapter must isolate the engine from a specific inference framework.",
    ])
    R.p("The rest of the report shows how each requirement was met, how it was tested, and where it is still unmet.", align="j")
