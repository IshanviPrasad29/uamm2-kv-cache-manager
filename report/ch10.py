from data import *

def prob(R, n, title, problem, why, idea, code=None, ref=""):
    R.h3(f"Problem {n}. {title}")
    R.p(problem, bold_lead="Problem.", align="j", after=3)
    R.p(why, bold_lead="Why it matters.", align="j", after=3)
    R.p(idea, bold_lead="Solution and implementation idea.", align="j", after=3)
    if code: R.code(code, size=7.8)
    if ref: R.p(ref, bold_lead="Basis.", align="j", after=8)

def build(R):
    R.h1("12. Problems UAMM-2 Does Not Yet Address, and Our Solutions")
    R.p("UAMM-2 closes the main gaps of UAMM-1, but several technical problems remain, most of them about quantisation and the precision that attention actually requires. This chapter states each problem, explains why it matters, and gives the solution and the implementation we worked out. The ideas were developed by reasoning about the mathematics of attention and about the structure of the existing code; they indicate where each change would be made in the project of Chapter 6.", align="j")

    R.h2("12.1 Problems of quantisation and required precision")
    prob(R, 1, "Key error is amplified by the softmax; value error is not",
         "UAMM-2 treats keys and values with the same bit width. The two enter attention differently.",
         "A key error Δk changes the logit by δ = q·Δk, and the attention weight of that token is scaled by about e^δ. Large queries make the softmax sharp, so a small δ can change which token dominates. A value error ΔV changes the output only linearly, by Σ wᵢΔvᵢ. Keys therefore need more precision than values for the same output error. On our workload the 2-bit codec has a squared error of 0.130 per block, which is tolerable for values but can reorder attention among keys that have close logits.",
         "Use separate bit widths for keys and values (a pair such as K4/V2) and let the planner choose the pair. Implementation: extend the codec name to carry two widths; `encode` receives `bits_k` and `bits_v`; `calibrate` measures a *logit-error* metric for keys, E_k = max over sampled queries of |q·(k − k̂)|, in addition to the value error, and the planner prices the two separately.",
         "# codecs.py (sketch)\ndef encode(k, v, bits_k, bits_v, v_group):\n    qk, sk, zk = _asym_quant(k, bits_k, axis=0)          # per-channel keys\n    qv, sv, zv = _asym_quant(v_grouped, bits_v, axis=2)   # per-token values\n    return pack(qk, bits_k) + pack(qv, bits_v) + meta\n\n# planner cost: separate quality terms\ncost = p*read_ms + lam_k*heat*err_logit_k[c] + lam_v*heat*err_v[c] + congestion",
         "KIVI quantises keys and values along different axes; hardware compression engines commonly give keys 4 bits and values 2 bits.")
    prob(R, 2, "No guaranteed bound on the logit error of a compressed block",
         "The planner uses a calibrated average error. It does not know whether, for the current query, a compressed block could change the ranking of the top tokens.",
         "With per-channel scales s_c, the dequantisation error of channel c is at most s_c/2. For a query q the logit error of any token in the block is bounded by |δ| ≤ ½ Σ_c |q_c|·s_c. This is cheap (one dot product with the scale vector) and is exactly the kind of quantity the selection step already computes from the min/max summaries.",
         "At selection time compute B_err(b) = ½ Σ_c |q_c| s_{b,c} for every candidate block. If B_err(b) exceeds a tolerance τ times the logit gap to the block that would be ranked next, promote the block (decode it from a higher-precision copy or from FP16 in the fast tier) or exclude it from the approximate path. Implementation: store `scale_k` (already in the payload) as a resident per-block vector, add `err_bound(q, block)` to `attention.py`, and call it inside `select_blocks`.",
         "def err_bound(q, scale_k):          # scale_k: per-channel key scales of the block\n    return 0.5 * float(np.abs(q) @ scale_k)\n\n# in select_blocks: keep bound ranking, add a risk flag\nrisky = [b for b in chosen if err_bound(q, b.scale_k) > tau]",
         "Follows from the affine quantiser of KIVI and the bound-based selection of Quest.")
    prob(R, 3, "Outlier tokens, not only outlier channels",
         "Per-channel key quantisation handles channels with large magnitude. A single token with an unusually large key or value inside a 16-token block still stretches the min/max range of every channel it touches.",
         "In a block of 16 tokens one outlier can enlarge the range for all channels by a factor of several, so the other 15 tokens use only a few quantisation levels. Heavy-hitter and sink tokens are exactly the tokens most likely to be large and most important to keep accurate.",
         "Keep a small FP16 side buffer per block for the top-m tokens by key norm (m = 1 or 2), exclude them from the min/max computation, and store their indices. The kernel then adds their exact contribution. Implementation: in `_asym_quant`, find the indices of the m largest rows, replace them with the channel median before computing min/max, and append `(index, fp16 key, fp16 value)` records to the payload; `decode` writes the exact rows back. The overhead is m × 2d × 2 bytes per block (256 B for m = 1, d = 64).",
         "idx = np.argsort(-np.linalg.norm(k, axis=1))[:m]\nk_clip = k.copy(); k_clip[idx] = np.median(k, axis=0)\nq, s, z = _asym_quant(k_clip, bits, axis=0)\npayload += pack_sidecar(idx, k[idx], v[idx])   # exact rows",
         "KVQuant keeps outliers in a separate sparse structure (dense-and-sparse quantisation).")
    prob(R, 4, "Errors compound when a block is re-encoded repeatedly",
         "Re-encoding decodes the current payload and encodes it again. The stored error table is measured from the FP16 original, so it does not reflect earlier lossy steps.",
         "A block that moves INT8 → 4-bit → 2-bit has a true error larger than the table's 2-bit entry. The planner then underestimates the cost of its own past decisions. With independent errors e₁, e₂ the accumulated error is approximately 1 − (1 − e₁)(1 − e₂).",
         "Keep an accumulated error per block and use it in the quality term and in the ratchet check; limit the chain length (for example, at most two lossy re-encodes); and for cold blocks keep the FP16 original in flash so that re-encoding always starts from it. Implementation: add `b.err_acc` to `KVBlock`, update it in `reencode` as err_acc = 1 − (1 − err_acc)(1 − err_table[new]), and replace `b.err_table[c]` in the planner by max(err_table[c], err_acc_after_move).",
         "b.err_acc = 1 - (1 - b.err_acc) * (1 - b.err_table[codec])      # in reencode()\nquality_term = lam * heat * max(b.err_table[c], 1 - (1 - b.err_acc) * (1 - b.err_table[c]))",
         "The ratchet and calibration in Chapter 5; KVQuant calibrates quantisation error before use.")
    prob(R, 5, "Per-block min/max ranges drift with block size and content",
         "Asymmetric quantisation uses the exact minimum and maximum of a block. Small blocks have noisy ranges and any outlier sets the step size for the whole block.",
         "Block size 16 gave good results in our sweep, yet a block of 8 tokens had the lowest error (0.110) and a block of 64 the highest (0.341), showing the dependence of error on the range estimate.",
         "Use clipped ranges: compute the range from the 2nd and 98th percentile of each channel and clamp the extremes to the end codes; select the clipping percentile per block by minimising the calibrated error (the calibration pass already encodes each block, so trying two clipping levels adds one more encode). Combine with Problem 3 so that clipped values are handled by the sidecar.",
         None, "Clipping and percentile calibration are standard in post-training quantisation practice and are used in KV-cache quantisers.")
    prob(R, 6, "Precision is chosen by heat alone, not by sensitivity",
         "The planner raises or lowers precision according to block heat (attention received). But two blocks with equal heat can differ in how much their error changes the output.",
         "The output error caused by a block is roughly proportional to its attention weight times the norm of the error in its values and to the logit change of its keys. A block with high heat but a very flat value distribution tolerates more error than a block with the same heat and large values.",
         "Define a sensitivity score S(b) = w̄_b · (‖V_b‖_F · e_v(c) + ‖q‖ · e_k(c)), where w̄_b is the mean attention weight on the block, and let the quality term use S(b) instead of heat × err. Implementation: `h2o.py` already supplies w̄_b through the token scores; `paged_cache.py` can store ‖V_b‖ at sealing; the planner multiplies.",
         "S = b.mean_attn * (b.v_norm * err_v[c] + q_norm * err_k[c])\ncost = p*read_ms + lam * S + congestion",
         "Sensitivity-based allocation is the principle behind mixed-precision quantisation schemes.")
    prob(R, 7, "Positional encoding mixes channels before quantisation",
         "If keys are stored after rotary position embedding (RoPE), each channel pair is rotated by a position-dependent angle, which changes the per-channel statistics and makes the quantisation error position dependent.",
         "Rotation mixes the large-magnitude channels into their pair partner, so outlier channels are no longer isolated and per-channel quantisation loses part of its advantage.",
         "Store keys before RoPE and apply the rotation inside the attention kernel: since (R_p k)·q = k·(R_pᵀ q), rotate the query by the inverse of each token's rotation instead of the stored key. For a block of 16 tokens this costs 16 rotated copies of q (16 × d multiply-adds), small compared with reading the keys. Implementation: `kv_qk_prepare` is called per token position, with cosine and sine tables for the block's positions.",
         None, "KVQuant quantises keys before the rotary embedding for this reason.")

    R.h2("12.2 Problems of selection, placement and planning")
    prob(R, 8, "No prefetch: returning to an old topic is served from flash on demand",
         "Blocks that were cold become hot when the query returns to an earlier topic; the first reads must wait for the slow level.",
         "With replanning every 64 steps the experiments show 3,326 flash reads and 0.272 ms per token, against 599 reads and 0.090 ms with replanning every 2 steps. Replanning more often costs migrations (768 against 345).",
         "Use the bound ranking that selection already computes. After choosing the top B blocks, take ranks B+1 … B+P (for example P = 8) as predicted next reads and promote any that are in flash into DRAM during the current step, overlapped with the attention computation. Implementation: a `prefetch(q, candidates)` function in `attention.py` returns block identifiers; the simulator calls `cache.reencode` and a tier move through the adapter asynchronously; the cost model charges the transfer to a background channel.",
         "ranked = np.argsort(-bound)\nread     = ranked[:B]\nprefetch = [b for b in ranked[B:B+P] if tier[b] == FLASH]   # promote ahead of need",
         "InfiniGen prefetches the entries predicted to be needed from the slow tier.")
    prob(R, 9, "One flattened vector per token: no layers, heads or grouped-query structure",
         "The engine stores one vector per token. Real models hold layers × KV heads independent streams, each with its own attention pattern.",
         "An 8B-class geometry has 32 layers × 8 KV heads = 256 streams. A single global fetch budget of 24 blocks starves heads that need broad attention and wastes reads on heads whose attention is sharp.",
         "Add (layer, head) to the block identity and index the block table by (layer, head, block). Give each stream a budget B_{l,h} from its measured attention entropy H_{l,h}: B_{l,h} ∝ exp(H_{l,h}) normalised to the total budget. Implementation: `KVBlock` gains `layer` and `head`; `H2OImportance` keeps one score array per stream; `select_blocks` is called per stream with its own budget; the RTL block table widens its address by log₂(L·H) bits.",
         None, "PyramidKV assigns layer-dependent budgets; FastGen uses head-dependent policies.")
    prob(R, 10, "The planner is greedy",
         "Blocks are visited in heat order and each takes its cheapest feasible slot. A fast level can be filled by moderately hot blocks before a slightly hotter block is considered.",
         "The budget sweep shows non-monotone behaviour: relative error 0.115 at twice the budget but 0.152 at three times, a sign that the heuristic is not always near the best assignment.",
         "Formulate the assignment as a multiple-choice knapsack: choose one (tier, codec) per block to minimise Σ cost(b, t, c) subject to Σ bytes in tier t ≤ cap_t. Solve it offline on recorded traces with an integer-programming solver to measure the gap; if the gap is material use Lagrangian relaxation (a price per tier, adjusted until capacities are met), which is cheap enough to run online and maps to the `capacity_price` term that already exists.",
         "for it in range(20):\n    choice = {b: argmin_tc(cost(b,t,c) + price[t]*bytes(b,c)) for b in blocks}\n    for t in tiers:\n        price[t] = max(0, price[t] + step * (used(choice,t) - cap[t]))   # dual update",
         "FlexGen formulates placement across memory levels as a linear program.")
    prob(R, 11, "No quality floor: the planner may lose accuracy it did not need to lose",
         "The cost function trades accuracy for latency everywhere, even when memory is plentiful.",
         "At 512 tokens the cache fits at FP16 (error 0.0013 under UAMM-1), yet UAMM-2 compresses to 0.54 of FP16 and the error becomes 0.016. At a budget three times larger the error rises from 0.115 to 0.152 because the spare room is spent on speed.",
         "Gate compression on pressure and add a per-request error budget. A block is demoted only if its tier's fill exceeds a threshold φ (for example 0.85) or if the saving exceeds a margin. The request carries E_max; the planner maintains Σ_b heat(b)·err_acc(b) ≤ E_max and refuses a demotion that would break it. Implementation: add `pressure(tier)` to `tiers.py`, a `min_fill_to_demote` parameter to `Config`, and the running sum to the planner.",
         "if used[t] / cap[t] < cfg.min_fill_to_demote and codec_rank(c) > b.rank:\n    continue                                   # no pressure: keep current precision\nif err_sum + heat * new_err > cfg.error_budget:\n    continue",
         "KIVI keeps a recent window at full precision for the same reason.")
    prob(R, 12, "Selection summaries are not charged to memory",
         "Each block keeps minimum and maximum key vectors resident for selection, but their bytes are not counted in any tier budget.",
         "For 64 channels in FP16 the summaries occupy 256 B per block, about 25 KB for 96 blocks, about 15 % of the 168 KB budget used in the experiments. For d = 128 the summaries grow to 512 B per block while a 2-bit block is about 1.8 KB, so they still add roughly 28 % to the 2-bit footprint.",
         "Charge the summaries to the fast level; store them in 8 bits using the same affine quantiser, with floor/ceil rounding so the bound remains valid; or keep one summary per group of 8 channels and use the looser bound. Measure the effect on selection recall. Implementation: `kmin`/`kmax` become `int8` arrays with the rule kmin = floor, kmax = ceil on the quantised grid.",
         None, "Quest stores two vectors per page and discusses this overhead.")
    prob(R, 13, "No graceful fallback when the budget is exhausted",
         "When even 2-bit blocks do not fit, the planner reports overflow and does nothing more.",
         "At 3,072 tokens the all-codec system overflows by 28.5 KB (INT8 and 4-bit only: 132 KB).",
         "When overflow persists after demotion, evict the lowest-importance tokens inside the least important blocks (token level, not block level), with an explicit counter and an importance threshold, and with the block table recording the surviving token mask. Implementation: a `compact(block, keep_mask)` function re-encodes a block with fewer tokens; `allow_drop` becomes a three-valued policy (never, block, token).",
         None, "H2O evicts by accumulated attention; KV-Compress evicts at the paging granularity.")
    prob(R, 14, "No sharing between sequences",
         "Every sequence owns its blocks, so a common prompt is stored once per sequence.",
         "An 800-token system prompt shared by four sessions occupies four times the memory of one copy, which is serious when the whole budget is a few hundred megabytes.",
         "Reference-count blocks and use copy-on-write in the block table. Shared prefix blocks are identified by a hash of their token ids; they are stored at the lowest precision tolerated by the most demanding sharer (the maximum of the sharers' rank floors). Implementation: `KVBlock.refcount`, a dictionary from prefix hash to block id, and a copy on first write to a shared block.",
         None, "PagedAttention shares blocks across sequences with copy-on-write.")

    R.h2("12.3 Problems of systems and hardware")
    prob(R, 15, "Migration is not synchronised with attention",
         "In the simulator a block moves instantaneously. In a real system an attention step may be reading a block while it is being re-encoded.",
         "A block re-encoded from 4-bit to 2-bit while a step reads it would return mixed data and corrupt that step's output.",
         "Add a version counter and a busy flag to every block-table entry. Migration writes the new payload into a second buffer, then flips the entry (tier, codec, physical address, version + 1) in one table update. An attention step records the version it started with and re-reads if the version changed. This is the same atomic-update idea as double buffering and fits the existing block-table RTL, which already returns the old entry when a read and a write coincide.",
         None, "FlexGen overlaps data movement with computation; the block table is the natural place for per-block state.")
    prob(R, 16, "Flash writes and wear",
         "Every demotion into flash writes a block. Flash has limited write endurance and large erase units.",
         "A planner that moves blocks back and forth would wear the device. The one-way precision rule already stops promoting precision, but tier moves between DRAM and flash can still alternate.",
         "Count writes per block and per flash page, add a write cost to the move cost in `tiers.py` (proportional to the wear fraction), and require a larger hysteresis margin for moves into flash. Write blocks to flash in log-structured fashion so that a block is written once per demotion.",
         None, "LLM in a flash stresses reading flash in few, large, sequential accesses; the same pattern suits writes.")
    prob(R, 17, "Hardware blocks do not yet match the software policy",
         "(a) The codec RTL quantises 8-bit data, whereas the software codecs start from FP16. (b) The scorer implements the original weighted score, not the utility cost. (c) The designs have not been taken through placement and routing.",
         "The quantiser alone is about 11,300 generic gates, mostly sixteen 8 × 16-bit multipliers; the table and scorer memories appear as flip-flops until mapped to RAM.",
         "(a) Add a front stage that converts FP16 to a group-normalised 8-bit value, (x − min)/(max − min) × 255, using the group min and max supplied by firmware. (b) Implement the utility cost in fixed point with lookup tables for the read-time and error terms, and verify it against the Python planner with the same golden-model and mutation method used for the current scorer. (c) Map the memories to block RAM and replace the reciprocal multiplier with a shift when the range is rounded up to a power of two, or reduce the number of lanes.",
         None, "Hardware KV-compression engines use group quantisers of the same structure.")
    R.h2("12.4 Order of work")
    R.table(["Step", "Goal", "Problems"],
            [["1", "Precision control: separate key/value widths, logit-error bound, sidecar, error tracking, sensitivity", "1, 2, 3, 4, 5, 6, 7"],
             ["2", "Policy: quality floor, prefetch, per-layer budgets, summary accounting, better optimiser", "8, 9, 10, 11, 12"],
             ["3", "Capacity limits and sharing: token-level fallback, copy-on-write", "13, 14"],
             ["4", "Systems and hardware: atomic migration, flash wear, FP16 front stage, utility scorer", "15, 16, 17"]],
            caption="Suggested order of implementation.", widths=[0.6, 4.4, 1.7])
