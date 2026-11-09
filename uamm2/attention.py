"""Block-sparse attention over the tiered cache + dense FP32 oracle for quality measurement."""
from dataclasses import dataclass
from typing import List

import numpy as np

from .paged_cache import KVBlock, PagedKVCache


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class AttnResult:
    out: np.ndarray
    token_idx: np.ndarray
    weights: np.ndarray
    read_blocks: List[KVBlock]


def select_blocks(q: np.ndarray, candidates: List[KVBlock], budget: int) -> List[KVBlock]:
    """Query-aware selection using per-block key min/max bounds.

    bound(b) = sum_c max(q_c*kmin_c, q_c*kmax_c) is an upper bound on q.k over the block, so blocks
    that cannot contain a high-scoring key are skipped without fetching (or decoding) them.
    """
    if budget >= len(candidates):
        return list(candidates)
    kmin = np.stack([b.kmin for b in candidates])
    kmax = np.stack([b.kmax for b in candidates])
    bound = np.maximum(q * kmin, q * kmax).sum(axis=1)
    top = np.argsort(-bound, kind="stable")[:budget]
    return [candidates[i] for i in sorted(top)]


def sparse_attention(q: np.ndarray, cache: PagedKVCache, read_blocks: List[KVBlock]) -> AttnResult:
    d = q.shape[0]
    ks, vs, idx = [], [], []
    for b in read_blocks:
        k, v = cache.get_kv(b)
        ks.append(k)
        vs.append(v)
        idx.append(np.arange(b.start, b.start + b.n_tokens))
    K, V, tok = np.concatenate(ks), np.concatenate(vs), np.concatenate(idx)
    w = softmax(K @ q / np.sqrt(d))
    return AttnResult(w @ V, tok, w, read_blocks)


def dense_oracle(q: np.ndarray, K: np.ndarray, V: np.ndarray):
    w = softmax(K @ q / np.sqrt(q.shape[0]))
    return w @ V, w
