from data import *
def build(R):
    R.h1("4. Gaps in UAMM-1")
    R.p("Combining paging and importance scoring was a first step, but UAMM-1 inherited the limits of the algorithms it combined and added defects of its own. This chapter identifies each gap, first at the level of ideas and then at the level of implementation, and gives the evidence for its consequence.", align="j")
    R.h2("4.1 Gaps at the level of ideas")
    R.table(["Gap", "Explanation", "Consequence"],
            [["No quantisation", "H2O-style importance only decides where a block lives or whether it is deleted. Kept data stays at full 16-bit precision. Quantisation methods such as KIVI, which would shrink kept data, were not part of the system.", "When the budget is smaller than the FP16 data, the only remedy is deletion"],
             ["Placement without precision", "Importance and placement were linked, but placement and precision were not. A block could not be stored smaller in a slower level.", "Slow levels hold full-size blocks and are quickly exhausted"],
             ["No selective reading", "All retained blocks were assumed to be read at every step. There was no query-aware rule, as in Quest, for skipping blocks that cannot matter.", "Bandwidth stays proportional to the retained cache"],
             ["No measure of codec loss", "Because no codec existed, nothing estimated how much error a smaller format would cause for a given block.", "No principled trade-off between size and accuracy"],
             ["Irreversible, silent loss", "Deletion happened whenever the lowest level was full, without a flag or an opt-in.", "Output quality degraded with no visible signal"],
             ["Isolated hardware", "The decision engine scored one block at a time and had no memory of block metadata or a translation table.", "The software could not hand a whole table to the hardware"]],
            caption="Gaps at the level of ideas.", widths=[1.4, 3.3, 2.0])
    R.h2("4.2 Defects in the implementation")
    R.table(["Defect", "Why it matters", "Response in UAMM-2"],
            [["The eviction decision was computed but never acted on, and data was deleted silently when the lowest level was full", "Output changes without notice", "Drop is explicit and opt-in; otherwise overflow is reported as a number"],
             ["Importance was clamped at 1.0 although H2O scores are sums that grow without bound", "Heavy hitters and sinks saturate; the signal stops discriminating", "Scores decay each step; block importance is normalised by the 95th percentile"],
             ["Recency was constant or unused", "A quarter of the score carried no information", "Recency enters through an exponential average of how often attention fetched the block"],
             ["No capacity-aware cascade and no hysteresis", "Blocks oscillate between levels; each move costs energy", "Greedy capacity-aware assignment, minimum residency and a relative-improvement margin"],
             ["Latency charged per token, migrations free", "Cost of moving data was invisible", "Cost per block per level and codec; migrations accounted separately"],
             ["O(N) Python loop per token for H2O", "Does not scale with context", "One vectorised decay per step"],
             ["Division by 100 in Verilog", "Dividers are large and slow", "Integer weights 154/64/38 (sum 256) and a shift"],
             ["Verilog and Python disagreed on protected blocks", "Hardware and software make different decisions", "RTL honours the protected flag; bit-exact golden model"],
             ["Three test vectors, no boundary cases", "Little confidence in the RTL", "17 sweeps with threshold edges, 400 codec vectors, 6,000 random table operations, mutation testing"],
             ["SmoothQuant listed as a KV-cache solution", "It targets weights and activations", "Replaced by KV-specific KIVI-style codecs"]],
            caption="Implementation defects found on review and the response to each.", widths=[2.3, 1.9, 2.5])
    R.h2("4.3 Evidence: what the gaps cost")
    e1 = load("e1_ablation")
    a = e1["v1_threshold_fp16"]; b = e1["v2_planner_all_codecs"]
    R.p(f"The experiments of Chapter 8 measure the consequence. With a total budget of 168 KB for 1,536 tokens (FP16 would need 393 KB), the UAMM-1 policy keeps {mean(a,'resident_vs_fp16'):.2f} of the FP16 footprint only by permanently deleting {mean(a,'dropped_tokens'):.0f} of the 1,536 tokens on average. The remaining blocks then retain only {mean(a,'attn_mass_recall_mean'):.2f} of the attention mass, the mean relative error of the attention output is {mean(a,'attn_rel_l2_mean'):.2f}, and about {mean(a,'flash_block_reads'):,.0f} block reads are served from flash. A system with codecs, by contrast, keeps every token at {mean(b,'resident_vs_fp16'):.2f} of the FP16 footprint with {mean(b,'attn_mass_recall_mean'):.2f} recall. The gap is therefore not in the placement rule but in the absence of a way to make data smaller and of a way to read less.", align="j")
    R.callout("Summary of what was left unresolved", ["1. No quantisation, so a tight budget forces deletion.", "2. Placement and precision decided separately (placement only).", "3. No query-aware selection of what to read.", "4. No measured error per block, hence no principled quality control.", "5. Silent, irreversible loss and several implementation defects.", "6. Hardware that was neither connected to a block table nor adequately verified."])
