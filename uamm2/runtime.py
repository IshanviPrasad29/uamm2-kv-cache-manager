"""Runtime adapter layer: the seam between UAMM-2 and a real inference engine.

ReferenceAdapter is fully functional against the NumPy reference engine in this package.
CompilerRuntimeAdapter is a documented SKELETON: it is NOT connected to a compiler-based runtime, and nothing in it has been
run against a real runtime build. What I verified about a compiler-based runtime (from its docs): it compiles models via
Apache TVM Unity, targets CUDA/Vulkan/Metal GPUs plus ARM/x86 CPUs through LLVM, supports dynamic
shapes and low-bit weight quantization, and exposes an OpenAI-style API. I did NOT verify how its
paged KV cache is exposed or whether a RISC-V/RVV (Ara) target works end to end; both need
validation against a pinned TVM version before this adapter can be filled in.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from .config import Config
from .paged_cache import PagedKVCache


class RuntimeAdapter(ABC):
    @abstractmethod
    def write_token(self, k: np.ndarray, v: np.ndarray) -> int:
        """Append one token's K/V; return its block id."""

    @abstractmethod
    def read_block(self, block_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (K, V) float32 for a block, decoding whatever codec it is stored in."""

    @abstractmethod
    def migrate_block(self, block_id: int, tier: int, codec: str) -> None:
        """Move/re-encode a block. Real runtimes must synchronize with in-flight attention."""

    @abstractmethod
    def attention_importance(self) -> Dict[int, float]:
        """Return {token_id: attention_weight} for the last step, if the backend can expose it."""


class ReferenceAdapter(RuntimeAdapter):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache = PagedKVCache(cfg)
        self._last: Dict[int, float] = {}

    def write_token(self, k, v) -> int:
        b, _ = self.cache.append_token(k, v)
        return b.block_id

    def read_block(self, block_id: int):
        return self.cache.get_kv(self.cache.blocks[block_id])

    def migrate_block(self, block_id: int, tier: int, codec: str) -> None:
        b = self.cache.blocks[block_id]
        if not b.sealed:
            raise ValueError("cannot migrate the open block")
        if not 0 <= tier < len(self.cfg.tiers):
            raise ValueError("bad tier")
        if codec != b.codec:
            self.cache.reencode(b, codec)
        b.tier = tier

    def attention_importance(self) -> Dict[int, float]:
        return dict(self._last)

    def record_attention(self, token_idx, weights) -> None:
        self._last = {int(t): float(w) for t, w in zip(token_idx, weights)}


class CompilerRuntimeAdapter(RuntimeAdapter):
    """SKELETON - see module docstring. Integration checklist (all unverified):

    1. write_token / read_block: map to the engine's paged KV cache pages. Requires reading the
       pinned TVM version's KV-cache API; storage there is engine-owned tensors, not NumPy.
    2. attention_importance: fused attention kernels normally do not expose softmax weights.
       Either add a row-sum output to the attention kernel or accumulate scores in a side kernel
       (on Ara: a vfredusum loop over the already-computed logits).
    3. migrate_block: needs real, synchronized tensor movement and a codec kernel (see ara/).
    4. Target: confirm TVM's LLVM path can emit RVV 1.0 code for the model's kernels, and what
       runtime (bare-metal / Linux on CVA6) hosts the TVM runtime.
    """

    def _todo(self):
        raise NotImplementedError("CompilerRuntimeAdapter is a skeleton; see its docstring for the checklist")

    def write_token(self, k, v):
        self._todo()

    def read_block(self, block_id):
        self._todo()

    def migrate_block(self, block_id, tier, codec):
        self._todo()

    def attention_importance(self):
        self._todo()
