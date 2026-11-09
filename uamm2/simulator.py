"""End-to-end UAMM-2 reference simulator.

Real: KV tensors are quantized/packed/entropy-coded, attention runs on the decoded tensors, and
quality is measured against a dense FP32 oracle. Simulated (assumed constants): tier latencies,
bandwidths and decode costs. No LLM is run; the q/k/v stream is synthetic (see workload.py).
"""
from collections import Counter
from typing import Dict

import numpy as np

from .attention import dense_oracle, select_blocks, sparse_attention
from .config import Config
from .h2o import H2OImportance
from .paged_cache import PagedKVCache
from .planner import Plan, make_planner
from .tiers import DROP, move_cost_ms, read_cost_ms, usage, decode_cost_ms
from .workload import SyntheticWorkload


class Simulator:
    def __init__(self, cfg: Config, seed: int = 7):
        self.cfg = cfg
        self.cache = PagedKVCache(cfg)
        self.h2o = H2OImportance(cfg)
        self.planner = make_planner(cfg)
        self.work = SyntheticWorkload(cfg.head_dim, seed)
        self.stats = Counter()
        self.migration_ms = 0.0
        self.overflow = [0] * len(cfg.tiers)
        self.plan_log = []
        self.trace = []          # per-step records (filled by run)

    # ------------------------------------------------------------------
    def _update_protection(self, t: int) -> None:
        cfg = self.cfg
        lo = t + 1 - cfg.recent_window_tokens
        for b in self.cache.blocks:
            b.protected = (b.start < cfg.sink_tokens) or (b.end > lo) or (not b.sealed)

    def _apply_plan(self, plan: Plan, step: int) -> None:
        cfg, cache = self.cfg, self.cache
        for bid, (tier, codec) in plan.items():
            b = cache.blocks[bid]
            if tier == DROP:
                cache.drop(b)
                self.stats["drops"] += 1
                self.stats["dropped_tokens"] += b.n_tokens
                continue
            moved = False
            if codec != b.codec:
                before = b.payload.nbytes
                cache.reencode(b, codec)
                self.migration_ms += (decode_cost_ms(cfg, b.codec, before) +
                                      decode_cost_ms(cfg, b.codec, b.payload.nbytes))
                self.stats["reencodes"] += 1
                moved = True
            if tier != b.tier:
                self.migration_ms += move_cost_ms(cfg, b.tier, tier, b.payload.nbytes)
                b.tier = tier
                self.stats["migrations"] += 1
                moved = True
            if moved:
                b.last_move_step = step
        used = usage(cfg, cache.blocks)
        for t, u in enumerate(used):
            self.overflow[t] = max(self.overflow[t], max(0, u - cfg.tiers[t].capacity_bytes))

    # ------------------------------------------------------------------
    def run(self, n_tokens: int) -> Dict:
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        cfg, cache, d = self.cfg, self.cache, self.cfg.head_dim
        K_or = np.zeros((n_tokens, d), np.float32)
        V_or = np.zeros((n_tokens, d), np.float32)
        rel_l2, cosine, recall, fetch_ms, flash_reads = [], [], [], [], 0
        warmup = 2 * cfg.recent_window_tokens

        for t in range(n_tokens):
            q, k, v = self.work.step(t)
            k = k.astype(np.float16).astype(np.float32)
            v = v.astype(np.float16).astype(np.float32)
            K_or[t], V_or[t] = k, v
            cache.append_token(k, v)
            self._update_protection(t)

            prot = [b for b in cache.blocks if b.protected and not b.dropped]
            cands = [b for b in cache.blocks if b.sealed and not b.protected and not b.dropped]
            chosen = select_blocks(q, cands, cfg.attn_block_budget)
            read = sorted(prot + chosen, key=lambda b: b.block_id)
            res = sparse_attention(q, cache, read)
            ref, w_or = dense_oracle(q, K_or[:t + 1], V_or[:t + 1])

            step_rec = {"t": t,
                        "fetch_ms": sum(read_cost_ms(cfg, b.tier, b.codec, b.nbytes(d)) for b in read),
                        "recall": float(w_or[res.token_idx].sum()),
                        "rel_l2": float(np.linalg.norm(res.out - ref) / (np.linalg.norm(ref) + 1e-12)),
                        "blocks_read": len(read),
                        "flash_reads": sum(1 for b in read if b.tier == len(cfg.tiers) - 1)}
            self.trace.append(step_rec)
            if t >= warmup:
                rel_l2.append(float(np.linalg.norm(res.out - ref) / (np.linalg.norm(ref) + 1e-12)))
                cosine.append(float(res.out @ ref / (np.linalg.norm(res.out) * np.linalg.norm(ref) + 1e-12)))
                recall.append(float(w_or[res.token_idx].sum()))
                fetch_ms.append(sum(read_cost_ms(cfg, b.tier, b.codec, b.nbytes(d)) for b in read))
                flash_reads += sum(1 for b in read if b.tier == len(cfg.tiers) - 1)

            h = cfg.hit_ema
            chosen_ids = {b.block_id for b in chosen}
            for b in cands:
                if b.block_id in chosen_ids:
                    b.hit_ema = h * b.hit_ema + (1 - h)
                    b.last_access, b.access_count = t, b.access_count + 1
                else:
                    b.hit_ema *= h
            for b in prot:
                b.last_access, b.access_count = t, b.access_count + 1
            self.h2o.update(res.token_idx, res.weights, t + 1)

            if (t + 1) % cfg.plan_interval == 0 and any(b.sealed for b in cache.blocks):
                self.h2o.refresh(cache.blocks)
                plan = self.planner.plan(cache.blocks, t)
                self._apply_plan(plan, t)
                step_rec["tier_bytes"] = usage(cfg, cache.blocks)
                step_rec["codecs"] = dict(Counter(b.codec for b in cache.blocks if not b.dropped))
                step_rec["migrations"] = self.stats["migrations"]
                step_rec["reencodes"] = self.stats["reencodes"]
                step_rec["dropped_tokens"] = self.stats["dropped_tokens"]

        return self._summary(n_tokens, rel_l2, cosine, recall, fetch_ms, flash_reads)

    # ------------------------------------------------------------------
    def _summary(self, n_tokens, rel_l2, cosine, recall, fetch_ms, flash_reads) -> Dict:
        cfg, d = self.cfg, self.cfg.head_dim
        blocks = self.cache.blocks
        live = [b for b in blocks if not b.dropped]
        used = usage(cfg, blocks)
        resident = sum(used)
        fp16_total = n_tokens * d * 2 * 2
        tier_blocks = Counter(b.tier for b in live)
        codecs = Counter(b.codec for b in live)
        p95 = float(np.percentile(rel_l2, 95)) if rel_l2 else 0.0
        return {
            "policy": cfg.policy,
            "tokens": n_tokens,
            "blocks": len(blocks),
            "fp16_kv_bytes": fp16_total,
            "resident_bytes": resident,
            "resident_vs_fp16": round(resident / fp16_total, 4),
            "tier_bytes": {cfg.tiers[i].name: used[i] for i in range(len(used))},
            "tier_blocks": {cfg.tiers[i].name: tier_blocks.get(i, 0) for i in range(len(used))},
            "codec_blocks": dict(sorted(codecs.items())),
            "attn_rel_l2_mean": round(float(np.mean(rel_l2)), 5) if rel_l2 else None,
            "attn_rel_l2_p95": round(p95, 5),
            "attn_cosine_mean": round(float(np.mean(cosine)), 6) if cosine else None,
            "attn_mass_recall_mean": round(float(np.mean(recall)), 5) if recall else None,
            "sim_fetch_ms_per_token": round(float(np.mean(fetch_ms)), 5) if fetch_ms else None,
            "flash_block_reads": flash_reads,
            "migrations": self.stats["migrations"],
            "reencodes": self.stats["reencodes"],
            "migration_ms_total_sim": round(self.migration_ms, 3),
            "dropped_blocks": self.stats["drops"],
            "dropped_tokens": self.stats["dropped_tokens"],
            "capacity_overflow_bytes_max": {cfg.tiers[i].name: self.overflow[i]
                                            for i in range(len(used))},
            "note": "synthetic workload + assumed latency constants; not a hardware measurement",
        }
