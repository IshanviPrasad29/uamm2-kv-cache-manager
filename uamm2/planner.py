"""Placement planners.

UtilityPlanner (UAMM-2): chooses a (tier, codec) PAIR per block by minimizing expected cost

    cost(b,t,c) = p_access(b) * [read_ms(t,c,bytes) + decode_ms(c,bytes)]     (latency if fetched)
                + lambda * heat(b) * rel_err(b,c)                              (quality risk)
                + capacity_price * (bytes / cap_t) * (1 + 4 * fill_t)          (congestion)

greedily in descending "heat" order under per-tier byte capacities. Hysteresis (minimum residency and
a relative improvement margin) stops promote/demote thrashing. Precision is a one-way ratchet
(see codecs.py), so a block can only move to a codec of equal or lower precision.
This is a greedy heuristic, not an optimal knapsack solution.

ThresholdPlanner reproduces the original UAMM-1 policy (0.60/0.25/0.15 score, fixed thresholds,
fp16 only, cascade demotion, drop when nothing fits) for ablation.
"""
from typing import Dict, List, Tuple

from .codecs import CODEC_RANK, codec_rank, parse_codec
from .config import Config
from .paged_cache import KVBlock, all_codec_names
from .tiers import DROP, FLASH, read_cost_ms

Plan = Dict[int, Tuple[int, str]]   # block_id -> (tier, codec) ; tier == DROP means evict


def codec_allowed(tier: int, codec: str) -> bool:
    family, entropy = parse_codec(codec)
    if entropy and tier != FLASH:
        return False
    if tier == 2 and CODEC_RANK[family] < CODEC_RANK["kivi4"]:
        return False
    return True


def _fixed_usage(cfg: Config, blocks: List[KVBlock]) -> List[int]:
    used = [0] * len(cfg.tiers)
    for b in blocks:
        if not b.dropped and (b.protected or not b.sealed):
            used[b.tier] += b.nbytes(cfg.head_dim)
    return used


class UtilityPlanner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cands = [(t, c) for t in range(len(cfg.tiers)) for c in all_codec_names(cfg)
                      if codec_allowed(t, c)]

    @staticmethod
    def heat(b: KVBlock) -> float:
        return 0.5 * b.hit_ema + 0.5 * b.importance

    def _cost(self, b, t, c, nbytes, used, cap) -> float:
        cfg = self.cfg
        p = max(b.hit_ema, 0.01)
        fill = used[t] / max(cap[t], 1.0)
        return (p * read_cost_ms(cfg, t, c, nbytes)
                + cfg.lambda_quality * self.heat(b) * b.err_table[c]
                + cfg.capacity_price * (nbytes / cfg.tiers[t].capacity_bytes) * (1.0 + 4.0 * fill))

    def plan(self, blocks: List[KVBlock], step: int) -> Plan:
        cfg = self.cfg
        cap = [s.capacity_bytes * (1.0 - cfg.capacity_headroom) for s in cfg.tiers]
        used = _fixed_usage(cfg, blocks)
        movable = [b for b in blocks if b.sealed and not b.protected and not b.dropped]
        movable.sort(key=lambda b: -self.heat(b))
        plan: Plan = {}
        for b in movable:
            best, best_cost, cur, cur_cost = None, float("inf"), None, float("inf")
            for t, c in self.cands:
                if codec_rank(c) < b.rank:
                    continue
                nbytes = b.payload.nbytes if c == b.codec else b.size_table[c]
                if used[t] + nbytes > cap[t]:
                    continue
                cost = self._cost(b, t, c, nbytes, used, cap)
                if cost < best_cost:
                    best, best_cost = (t, c, nbytes), cost
                if (t, c) == (b.tier, b.codec):
                    cur, cur_cost = (t, c, nbytes), cost
            choice = best
            if cur is not None and best is not None:
                too_soon = (step - b.last_move_step) < cfg.min_residency_steps
                close = cur_cost <= best_cost + cfg.hysteresis_margin * max(best_cost, 1e-4)
                if too_soon or close:
                    choice = cur
            if choice is None:
                if cfg.allow_drop and b.importance < cfg.drop_importance:
                    plan[b.block_id] = (DROP, b.codec)
                else:      # nothing fits: stay put (counted as overflow by the simulator)
                    used[b.tier] += b.nbytes(cfg.head_dim)
                continue
            t, c, nbytes = choice
            used[t] += nbytes
            plan[b.block_id] = (t, c)
        return plan


class ThresholdPlanner:
    """Original UAMM-1 policy (fp16 only)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def plan(self, blocks: List[KVBlock], step: int) -> Plan:
        cfg = self.cfg
        cap = [s.capacity_bytes for s in cfg.tiers]
        used = _fixed_usage(cfg, blocks)
        movable = [b for b in blocks if b.sealed and not b.protected and not b.dropped]
        if not movable:
            return {}
        max_age = max(1, max(step - b.last_access for b in movable))
        max_cnt = max(1, max(b.access_count for b in movable))

        def score(b):
            rec = 1.0 - (step - b.last_access) / max_age
            return cfg.alpha * b.importance + cfg.beta * rec + cfg.gamma * b.access_count / max_cnt

        plan: Plan = {}
        for b in sorted(movable, key=lambda x: -score(x)):
            s = score(b)
            target = 1 if s >= cfg.t1_threshold else (2 if s >= cfg.t2_threshold else 3)
            if cfg.threshold_fill_fast:
                target = 1                 # capacity-aware variant: best-scoring blocks take the fastest tier with room
            nbytes = b.payload.nbytes
            placed = False
            for t in range(target, len(cfg.tiers)):
                if used[t] + nbytes <= cap[t]:
                    used[t] += nbytes
                    plan[b.block_id] = (t, b.codec)
                    placed = True
                    break
            if not placed:
                plan[b.block_id] = (DROP, b.codec)    # UAMM-1 behaviour: data loss when full
        return plan


class StaticPlanner:
    """Baseline without a planner: every sealed, unprotected block is re-encoded once to one fixed codec
    (cfg.static_codec) and placed, hottest (by H2O importance) first, in the fastest tier that still has room.
    There is no cost model, no heat/error trade-off and no hysteresis. Blocks that fit nowhere stay where
    they are and are reported as overflow (nothing is dropped), as in UtilityPlanner."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def plan(self, blocks: List[KVBlock], step: int) -> Plan:
        cfg = self.cfg
        cap = [s.capacity_bytes * (1.0 - cfg.capacity_headroom) for s in cfg.tiers]
        used = _fixed_usage(cfg, blocks)
        movable = [b for b in blocks if b.sealed and not b.protected and not b.dropped]
        movable.sort(key=lambda b: -b.importance)
        plan: Plan = {}
        for b in movable:
            c = cfg.static_codec if codec_rank(cfg.static_codec) >= b.rank else b.codec
            nbytes = b.payload.nbytes if c == b.codec else b.size_table[c]
            for t in range(1, len(cfg.tiers)):
                if codec_allowed(t, c) and used[t] + nbytes <= cap[t]:
                    used[t] += nbytes
                    plan[b.block_id] = (t, c)
                    break
            else:
                used[b.tier] += b.nbytes(cfg.head_dim)
        return plan


def make_planner(cfg: Config):
    if cfg.policy == "utility":
        return UtilityPlanner(cfg)
    if cfg.policy == "static":
        return StaticPlanner(cfg)
    return ThresholdPlanner(cfg)
