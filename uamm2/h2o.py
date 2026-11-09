"""H2O-style heavy-hitter importance, fed by attention weights the engine already computed.

Token score = decayed running sum of attention weight received. Only tokens that were actually
attended (their block was fetched) receive weight; everything decays each step. Block importance
aggregates token scores (mean|max|sum) and is normalized by the 95th percentile of block scores,
so the (huge) attention-sink score does not squash every other block toward zero.

Cost: one vectorized decay over N tokens per step (O(N) in numpy, not a Python loop).
Note: this signal drives PLACEMENT (lossless tiering). Permanent token dropping, as in the original
H2O paper, is a separate opt-in (Config.allow_drop).
"""
from typing import List

import numpy as np

from .config import Config
from .paged_cache import KVBlock


class H2OImportance:
    def __init__(self, cfg: Config, capacity: int = 4096):
        self.cfg = cfg
        self.tok = np.zeros(capacity, np.float64)

    def _ensure(self, n: int) -> None:
        if n > len(self.tok):
            self.tok = np.concatenate([self.tok, np.zeros(max(n, 2 * len(self.tok)) - len(self.tok))])

    def update(self, token_idx: np.ndarray, weights: np.ndarray, n_tokens: int) -> None:
        self._ensure(n_tokens)
        self.tok[:n_tokens] *= self.cfg.h2o_decay
        np.add.at(self.tok, token_idx, np.maximum(weights, 0.0))

    def refresh(self, blocks: List[KVBlock]) -> None:
        if not blocks:
            return
        bs = self.cfg.block_size
        n = blocks[-1].end
        nb = len(blocks)
        grid = np.zeros(nb * bs)
        grid[:n] = self.tok[:n]
        grid = grid.reshape(nb, bs)
        counts = np.array([b.n_tokens for b in blocks], dtype=np.float64)
        agg_mode = self.cfg.h2o_aggregation
        if agg_mode == "sum":
            agg = grid.sum(axis=1)
        elif agg_mode == "max":
            agg = grid.max(axis=1)
        else:
            agg = grid.sum(axis=1) / np.maximum(counts, 1)
        scale = float(np.percentile(agg, 95)) + 1e-12
        norm = np.clip(agg / scale, 0.0, 1.0)
        for b, x in zip(blocks, norm):
            b.importance = float(x)
