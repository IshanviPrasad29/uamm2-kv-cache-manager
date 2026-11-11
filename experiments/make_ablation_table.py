"""Writes docs/ablation_results.txt from the raw E1 results (mean over the 5 seeds 11-15)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
e1 = json.loads((ROOT / "experiments" / "results" / "e1_ablation.json").read_text())
cols = [("resident_vs_fp16", "res/fp16"), ("resident_with_summaries_vs_fp16", "res+summ"), ("dropped_tokens", "dropped"),
        ("attn_rel_l2_mean", "l2_mean"), ("attn_rel_l2_p95", "l2_p95"), ("attn_mass_recall_mean", "recall"),
        ("sim_fetch_ms_per_token", "fetch_ms*"), ("migrations", "migr"), ("flash_block_reads", "flash_rd"),
        ("overflow_total", "overflow_B")]
w = max(len(n) for n in e1) + 2
lines = ["Ablation and baselines, 1536 tokens, mean over seeds 11-15 (lambda was tuned on seeds 7-9)",
         "variant".ljust(w) + "".join(c[1].rjust(12) for c in cols)]
for n, rs in e1.items():
    lines.append(n.ljust(w) + "".join(f"{np.mean([r[k] for r in rs]):.4g}".rjust(12) for k, _ in cols))
lines.append("* fetch_ms is simulated with ASSUMED latency constants on a SYNTHETIC workload; overflow_B>0 = does not fit the tier budget.")
lines.append("res+summ counts the per-block key min/max summaries (not charged to the tiers in the main experiments).")
(ROOT / "docs" / "ablation_results.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
