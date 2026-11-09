from data import *
def build(R):
    R.h1("5. The Combined System: UAMM-2")
    R.p("This chapter explains the combined design. It begins with the core idea, then walks through the data model, the codecs, the importance signal, the planner, selective attention, the hardware blocks, the vector kernel and the runtime seam.", align="j")
    R.h2("5.1 The core idea: a placement is a (tier, codec) pair")
    R.p("Every block of the KV cache is described by two coordinates. The **tier** says *where* it is stored (and therefore what a read costs). The **codec** says *how many bits* it occupies (and therefore how big it is and how much error it carries). The planner picks both together. A block that is almost never attended is best stored in flash as a 2-bit entropy-coded blob. A block that is hot but large can stay in DRAM at 4 bits. Protected blocks (sinks, recent window) sit in the fastest tier at full precision. Choosing the two jointly lets the system exchange precision for capacity instead of exchanging data for nothing.", align="j")
    R.fig("d_system_architecture.png", "UAMM-2 system architecture: the decode loop, the paged cache with its tiers and codecs, the importance and planning loop, and the hardware and vector-kernel attachments.", 6.4)
    R.fig("d_separation_of_concerns.png", "Separation of concerns. The engine is split so that each part can be tested, replaced or moved to hardware independently.", 6.0)

    R.h2("5.2 Memory hierarchy and tiers")
    R.table(["Tier", "Where it models", "Capacity in experiments", "Base latency / bandwidth (assumed)", "Allowed codecs", "Intended contents"],
            [["T0", "Near-compute SRAM / vector register file", "24 KB", "0.5 µs / 100 GB/s", "any raw codec", "Sink tokens, recent window, unsealed block"],
             ["T1", "DRAM, hot", "48 KB", "2 µs / 12.8 GB/s", "any raw (non-entropy) codec", "Frequently fetched blocks; also where 2-bit blocks sit if the planner prices them there"],
             ["T2", "DRAM, warm", "32 KB", "2 µs / 12.8 GB/s", "kivi4, kivi2", "Occasionally fetched blocks"],
             ["T3", "Flash", "64 KB", "100 µs / 1 GB/s", "int8z, kivi4z, kivi2z (entropy-coded)", "Cold blocks"]],
            caption="Tiers used in the simulations. Capacities are deliberately small so that the budget is tight at a few thousand tokens; latencies and bandwidths are assumed values (Chapter 7).",
            widths=[0.4, 1.4, 0.8, 1.3, 1.3, 1.4])
    R.fig("d_memory_hierarchy.png", "The memory hierarchy used by the model, with latency and bandwidth classes.", 5.6)
    R.fig("d_tier_codec_map.png", "Which codecs are permitted in which tier. Flash only accepts entropy-coded variants; the warm DRAM tier only accepts 4- and 2-bit codecs.", 5.6)
    R.p("The tier restrictions are policy, not physics. They encode two beliefs: that holding full-precision data in a tier labelled “warm” wastes the capacity that tier exists to provide, and that entropy coding (zlib) is only worth its decode time where the access cost is already high.", align="j")

    R.h2("5.3 The block and its lifecycle")
    R.p("A block holds 16 consecutive tokens (configurable). While it is filling it is *open*: staged as raw vectors in the fastest tier and always read. When the 16th token arrives it is **sealed**: the K and V vectors are rounded to FP16 (this rounded copy is the ground truth from then on), encoded, and the following summaries are computed and kept resident because they are tiny:", align="j")
    R.bullets(["`kmin`, `kmax`: element-wise minimum and maximum of the block's keys (64 values each in the experiments);",
               "`err_table[codec]`: the measured relative squared error obtained by actually encoding and decoding this block with every codec;",
               "`size_table[codec]`: the measured byte size of the block under every codec;",
               "`importance`, `hit_ema`, `access_count`, `last_move_step`, `protected`: scheduling state."])
    R.p("Calibration by real encoding, rather than by a formula, is deliberate. The error of a codec depends strongly on the block's content (for example on outlier channels), and the planner should know what *this* block will lose, not an average. The cost is one extra encode per codec at seal time, which is a background cost in a real system.", align="j")
    R.fig("d_block_lifecycle.png", "Lifecycle of a block: open, sealed with calibration, and then moved or re-encoded by the planner; protected blocks never move.", 6.2)

    R.h2("5.4 Codecs: KIVI-style asymmetric quantisation")
    R.p("The codec layer turns a block of T tokens × d channels of keys and values into a byte payload and back. Four precision families exist: **fp16** (reference), **int8**, **kivi4** and **kivi2**; each non-FP16 family also has an entropy-coded variant with suffix `z` (zlib over the serialised blob), used on flash.", align="j")
    R.h3("5.4.1 Asymmetric affine quantisation")
    R.p("For a set of values with minimum m and maximum M, and b bits (so L = 2^b − 1 levels), the scale and zero are s = (M − m)/L and z = m. A value x is stored as q = clip(round((x − z)/s), 0, L) and reconstructed as x̂ = q·s + z. Scales and zeros are stored as FP16 (scale floored at 2^−10 to avoid division by zero). “Asymmetric” means that the zero point is the true minimum rather than zero, which uses all levels even when the data are not centred on zero.", align="j")
    R.h3("5.4.2 Keys per channel, values per token")
    R.p("Following KIVI, **keys are quantised per channel**: for each of the d channels one scale and one zero are computed across the block's tokens. **Values are quantised per token**: each token's value vector is split into groups of 32 channels and each group has its own scale and zero. Per-channel quantisation of keys isolates the few channels with large magnitude so they do not stretch the range of the others; values have no such outlier structure and are naturally dequantised token by token as attention weights multiply them.", align="j")
    R.fig("d_quant_axes.png", "Quantisation axes. Keys share parameters down a channel (across tokens); values share parameters across a group of channels within one token.", 6.0)
    R.h3("5.4.3 Packing")
    R.p("Quantised integers are packed little-endian into bytes: two 4-bit values per byte (element 0 in the low nibble) or four 2-bit values per byte. The same convention is used by the RTL nibble packer and by the RVV kernel, so a block produced by the Python codec can be consumed unchanged by the hardware and the kernel.", align="j")
    R.fig("d_packing.png", "Bit packing of 4-bit and 2-bit values into bytes. The even element is in the low bits.", 5.4)
    R.h3("5.4.4 Payload layout and measured sizes")
    R.p("The serialised payload is `[packed K][packed V][K scales][K zeros][V scales][V zeros]`. For one block of 16 tokens × 64 channels (2,048 K elements + 2,048 V elements) the measured sizes are given in the table below: metadata is not free, which is why a 4-bit block is 0.344 of FP16 rather than 0.25.", align="j")
    e7 = load("e7_codecs")["codecs"]
    rows = []
    for r in e7:
        if r["outliers"] is False:
            ro = [x for x in e7 if x["codec"] == r["codec"] and x["outliers"]][0]
            rows.append([r["codec"], f'{r["bytes"]:.0f}', f'{r["ratio_vs_fp16"]:.3f}', f'{r["err_mean"]:.2e}', f'{ro["err_mean"]:.2e}'])
    R.table(["Codec", "Bytes / block (avg)", "Ratio to FP16", "Rel. sq. error (no outlier channels)", "Rel. sq. error (with outlier channels)"], rows,
            caption="Measured size and reconstruction error per codec for a 16 × 64 block of K and V (mean over test blocks; synthetic data, see Chapter 7). The z variants reduce size slightly on flash-like data; for 4-bit and 2-bit they add a few bytes of overhead on random-like blocks.",
            widths=[0.8, 1.2, 1.0, 1.8, 1.8])
    R.h3("5.4.5 Precision ratchet")
    R.p("Re-encoding a lossy block decodes it and encodes it again, so errors compound. The cache therefore enforces a **one-way ratchet**: a block can be moved to a codec of equal or lower precision, never higher. Promoting a 4-bit block to a “fast tier” changes where it lives, not how accurate it is. The only same-precision transformations allowed are toggling the entropy wrapper, which is a lossless blob operation. This rule makes the quality effect of any sequence of planner decisions bounded and easy to reason about; it also means the planner must be cautious about early demotion (see Chapter 12).", align="j")

    R.h2("5.5 The importance signal: H2O reinterpreted")
    R.p("Each decode step produces attention weights over the tokens that were actually read. The engine updates a score per token: it multiplies all scores by a decay (0.95 by default) and then adds the new attention weight to the tokens that received it. A block's importance is the aggregate (mean by default) of its tokens' scores, divided by the 95th percentile of block scores and clipped to [0, 1]. Normalising by the 95th percentile prevents the enormous score of the attention-sink block from forcing every other block's importance toward zero. A second signal, `hit_ema`, is an exponential moving average of whether the block was selected for fetching at each step.", align="j")
    R.p("The **heat** of a block combines them: heat = ½·hit_ema + ½·importance. Importance reflects long-run attention mass; hit_ema reflects recent fetch behaviour. Using both gives the planner a stable and a responsive signal.", align="j")
    R.callout("How this differs from H2O", "In H2O the score decides which tokens are *deleted*. Here it decides where a block is *stored* and at what precision. The signal is the same; the consequence of acting on it is reversible-or-bounded instead of irreversible. Permanent drop remains available (`allow_drop`, importance below 0.02) but is off in all headline experiments.")

    R.h2("5.6 The utility planner")
    R.p("Every `plan_interval` steps (default 8) the planner re-evaluates every sealed, unprotected block. It considers every (tier, codec) candidate that the tier rules allow and whose precision is not higher than the block's current precision, and computes an expected cost in milliseconds-equivalent:", align="j")
    R.eq("cost(b, t, c) = p_access(b) · [read_ms(t, c, bytes) + decode_ms(c, bytes)] + λ · heat(b) · err_b(c) + κ · (bytes / cap_t) · (1 + 4 · fill_t)")
    R.bullets([("Latency term.", "p_access is the block's `hit_ema` (floored at 0.01); the read cost is the tier's fixed latency plus bytes over bandwidth plus decode time. A block that is rarely read pays little for living in flash."),
               ("Quality term.", "λ (default 2.0) converts measured codec error, weighted by the block's heat, into the same unit as latency. A hot block with a high-error codec is penalised; a cold block is almost free to compress."),
               ("Congestion term.", "κ (default 0.05) charges for the share of the tier that the block would occupy, scaled up as the tier fills, so that the planner does not exhaust the fast tiers early.")])
    R.p("Blocks are visited in descending heat so that the hottest blocks get first choice of fast tiers. Each takes the cheapest candidate that fits the remaining capacity (with 3 % headroom). If nothing fits, the block stays where it is and the overflow is counted. This greedy procedure is a heuristic, not an optimal knapsack solution.", align="j")
    R.h3("5.6.1 Hysteresis")
    R.p("Without damping, a block whose cost estimate fluctuates would bounce between tiers, paying migration cost each time. Two rules stop that: a block may not move again within `min_residency_steps` (default 16) of its last move, and it only moves if the best alternative is cheaper than staying by more than `hysteresis_margin` (default 25 %) of the best cost. Chapter 8 measures the effect.", align="j")
    R.fig("d_planner_flow.png", "Planner flow: candidate enumeration, cost evaluation, greedy capacity-aware selection and the hysteresis check.", 6.2)

    R.h2("5.7 Selective attention: reading less")
    R.p("At each decode step, protected blocks (sinks, recent window and the open block) are always read. Of the remaining sealed blocks, the engine reads only the `attn_block_budget` (default 24) blocks with the largest upper bound on the query–key score, computed from the resident kmin/kmax summaries:", align="j")
    R.eq("bound(b) = Σ_c max( q_c · kmin_c , q_c · kmax_c ) ≥ q · k   for every key k in b")
    R.p("The bound holds because each product q_c·k_c, for k_c between kmin_c and kmax_c, is at most the larger of the two endpoint products. A block with a low bound cannot contain a key that scores highly against this query, so it is neither fetched nor decoded. The test suite checks the bound property on random data. Selected blocks are decoded and attention is computed over the union of protected and selected tokens.", align="j")
    R.p("This is where capacity savings and bandwidth savings combine. Because skipped blocks are never touched, they can be stored in the cheapest (slowest, smallest) tier and codec without most steps ever paying for that choice; and because tier and codec are chosen on measured heat, the blocks that are touched are the ones that the system placed where they are cheap to read.", align="j")
    R.fig("d_decode_step.png", "One decode step: bound ranking, fetch of selected blocks, attention, H2O update and (every few steps) a planning pass.", 6.2)

    R.h2("5.8 Hardware: three verified RTL blocks")
    R.p("Three modules were written in synthesisable Verilog-2001. Each has a bit-exact Python golden model (`uamm2/hwmodel.py`), generated test vectors and a self-checking testbench run in Verilator. They accelerate the *inner loops* of the system; the utility planner itself remains software.", align="j")
    R.h3("5.8.1 Sweep scorer (`uamm_sweep_scorer`)")
    R.p("The scorer is the hardware heat classifier derived from UAMM-1. Block metadata is held in a small SRAM as a 25-bit word `{protected, frequency[7:0], recency[7:0], importance[7:0]}`. A `start` pulse latches an 8-bit memory-pressure value and launches a **sweep**: every block is read (one per cycle), passes through a 2-stage pipeline, and emerges as a result record `{idx, score, tier, evict}`. The score is computed without a divider:", align="j")
    R.eq("score = ( 154·imp + 64·rec + 38·frq ) >> 8      (154 + 64 + 38 = 256)")
    R.p("The tier is hot if the block is protected or score ≥ 174, warm if score ≥ 89, otherwise cold. A block is flagged for eviction only if it is **not** protected, the latched pressure is at least 243, and its score is below 89. A sweep over 64 blocks has latency NBLK + 3 cycles and completes with a `done` pulse; a `start` received while busy is ignored.", align="j")
    R.fig("d_rtl_scorer.png", "Sweep scorer micro-architecture: metadata SRAM, weighted sum, threshold comparators, protected override and eviction logic.", 6.0)
    R.h3("5.8.2 KV codec pipeline (`uamm_kv_quant4`, `uamm_kv_dequant4`)")
    R.p("The quantiser processes 16 byte-wide lanes per beat. Software or firmware supplies per-group parameters derived from the group's min and max: the zero point `zp`, a Q4.12 reciprocal `inv_q12 = round(15·4096/range)` and a Q8.8 step `step_q8 = round(range·256/15)`. Quantisation is q = clamp(((x − zp)·inv + 2048) >> 12, 0, 15) with x < zp clamped to zero; dequantisation is x′ = min(255, zp + ((q·step + 128) >> 8)). Each module has one pipeline register. The output is packed with lane i in nibble i, matching the software packing. The intended location is on the AXI stream between memory and Ara's vector load/store unit, so that (de)quantisation happens in the load path rather than in vector instructions.", align="j")
    R.fig("d_rtl_codec.png", "Codec datapath: sixteen subtract–multiply–round–clamp lanes feeding a nibble packer, and the inverse path.", 6.0)
    R.h3("5.8.3 Block table (`uamm_block_table`)")
    R.p("A direct-mapped translation table maps a logical block index (8 bits, 256 entries) to `{tier[1:0], codec[1:0], phys[11:0]}` with a separate valid bit that is cleared by reset. A lookup is a one-cycle synchronous read that returns the entry and a `hit` flag. A read in the same cycle as a write to the same address returns the **old** entry, a property fixed by design and verified by test.", align="j")
    R.fig("d_rtl_table.png", "Block-table structure: entry array, valid bits and the update and lookup ports.", 5.6)

    R.h2("5.9 The vector kernel for the Ara vector processor")
    R.p("Ara is a RISC-V Vector coprocessor. Attention scores for a block of 4-bit keys can be computed without first dequantising the keys into floating point. Because the key for token t at channel c is x = q·s_c + z_c, the dot product with a query splits into:", align="j")
    R.eq("q · k_t = Σ_c ( q_c · s_c ) · Q[t,c]  +  Σ_c q_c · z_c")
    R.p("The first term uses the integer codes Q and a query vector pre-scaled by the per-channel scales (computed once per block); the second term is a constant for the block (the **bias**). The kernel therefore folds scales and zeros into the query once, then, for each token, loads packed bytes with a vector byte load, extracts the low and high nibbles (even and odd channels), zero-extends and converts them to float, and accumulates with fused multiply–add into two pre-scaled query halves, finishing with a vector reduction. No expanded floating-point key block is ever written to memory.", align="j")
    R.fig("d_ara_integration.png", "Integration of the codec engine and the fused kernel with Ara and the memory system.", 6.2)
    R.p("A scalar version of the kernel with identical data layout runs on the host and is checked against NumPy (Chapter 10). The RVV version uses the standard intrinsics and compiles for `rv64gcv`. It has not been executed (Chapter 12).", align="j")

    R.h2("5.10 The runtime seam and the codec slot")
    R.p("A narrow `RuntimeAdapter` interface (write token, read block, migrate block, report attention importance) separates the engine from any inference framework. `ReferenceAdapter` implements it on the NumPy engine and is exercised in the tests. `RuntimeAdapter skeleton for a compiler-based runtime` is a skeleton whose docstring lists the integration steps: map to the runtime's KV pages, obtain attention weights (fused attention kernels do not normally expose them, so a row-sum output or a side kernel is needed), perform synchronised tensor movement, and confirm that the compiler can target RVV.", align="j")
    R.p("The codec layer exposes `encode(k, v, codec, v_group) → Payload` and `decode(Payload) → (k, v)`. The group quantiser and outlier handling of a hardware compression engine, or any other codec, can be wired in by implementing the same interface and registering a codec name.", align="j")
