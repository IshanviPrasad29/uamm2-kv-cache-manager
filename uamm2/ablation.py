"""Ablation: original UAMM-1 threshold policy vs UAMM-2 planner with progressively more codecs."""
from dataclasses import replace
from typing import Dict

from .config import Config
from .simulator import Simulator


def variants(base: Config) -> Dict[str, Config]:
    return {
        "v1_threshold_fp16": replace(base, policy="threshold", codecs=("fp16",),
                                     use_entropy_coding=False),
        "v2_planner_fp16": replace(base, policy="utility", codecs=("fp16",),
                                   use_entropy_coding=False),
        "v2_planner_int8_kivi4": replace(base, policy="utility",
                                         codecs=("fp16", "int8", "kivi4"),
                                         use_entropy_coding=False),
        "v2_planner_all_codecs": replace(base, policy="utility",
                                         codecs=("fp16", "int8", "kivi4", "kivi2"),
                                         use_entropy_coding=True),
        "v2_all_codecs_with_drop": replace(base, policy="utility", allow_drop=True,
                                           codecs=("fp16", "int8", "kivi4", "kivi2"),
                                           use_entropy_coding=True),
    }


def run_ablation(n_tokens: int, base: Config = None, seed: int = 7) -> Dict[str, Dict]:
    base = base or Config()
    return {name: Simulator(cfg, seed).run(n_tokens) for name, cfg in variants(base).items()}


def format_table(results: Dict[str, Dict]) -> str:
    cols = [("resident_vs_fp16", "resident/fp16"), ("dropped_tokens", "dropped_tok"),
            ("attn_rel_l2_mean", "rel_l2_mean"), ("attn_rel_l2_p95", "rel_l2_p95"),
            ("attn_mass_recall_mean", "mass_recall"), ("sim_fetch_ms_per_token", "fetch_ms/tok*"),
            ("migrations", "migrations"), ("flash_block_reads", "flash_reads"),
            ("overflow_total", "overflow_B")]
    w = max(len(n) for n in results) + 2
    lines = ["variant".ljust(w) + "".join(c[1].rjust(15) for c in cols)]
    for name, r in results.items():
        r = dict(r, overflow_total=sum(r["capacity_overflow_bytes_max"].values()))
        lines.append(name.ljust(w) + "".join(str(r[c[0]]).rjust(15) for c in cols))
    lines.append("* simulated with assumed latency constants; synthetic workload.\noverflow_B > 0 means the variant could not fit the data in the tier budget (infeasible without dropping).")
    return "\n".join(lines)
