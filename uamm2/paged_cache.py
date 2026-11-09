"""Paged KV cache holding REAL tensors (as encoded payloads), not just metadata."""
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .codecs import Payload, calibrate, codec_rank, decode, encode, parse_codec
from .config import Config


def all_codec_names(cfg: Config) -> List[str]:
    names: List[str] = []
    for fam in cfg.codecs:
        names.append(fam)
        if cfg.use_entropy_coding and fam != "fp16":
            names.append(fam + "z")
    return names


@dataclass
class KVBlock:
    block_id: int
    start: int
    n_tokens: int = 0
    sealed: bool = False
    k_stage: Optional[List[np.ndarray]] = field(default_factory=list)   # open block only
    v_stage: Optional[List[np.ndarray]] = field(default_factory=list)
    payload: Optional[Payload] = None
    tier: int = 0
    codec: str = "fp16"
    kmin: Optional[np.ndarray] = None       # key summaries, kept resident (tiny)
    kmax: Optional[np.ndarray] = None
    err_table: Dict[str, float] = field(default_factory=dict)
    size_table: Dict[str, int] = field(default_factory=dict)
    importance: float = 0.0                 # H2O-derived, normalized to [0,1]
    hit_ema: float = 1.0                    # how often attention actually fetched this block
    last_access: int = 0
    access_count: int = 0
    last_move_step: int = 0
    protected: bool = False
    dropped: bool = False

    @property
    def end(self) -> int:
        return self.start + self.n_tokens

    @property
    def rank(self) -> int:
        return codec_rank(self.codec)

    def nbytes(self, d: int) -> int:
        if self.dropped:
            return 0
        if self.payload is not None:
            return self.payload.nbytes
        return self.n_tokens * d * 2 * 2     # open block staged as fp16-equivalent


class PagedKVCache:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.blocks: List[KVBlock] = []
        self._codec_names = all_codec_names(cfg)

    @property
    def n_tokens(self) -> int:
        return sum(b.n_tokens for b in self.blocks)

    def append_token(self, k: np.ndarray, v: np.ndarray) -> Tuple[KVBlock, bool]:
        """Append one token. Returns (block, sealed_now)."""
        cfg = self.cfg
        if not self.blocks or self.blocks[-1].sealed:
            self.blocks.append(KVBlock(block_id=len(self.blocks), start=self.n_tokens))
        b = self.blocks[-1]
        b.k_stage.append(np.asarray(k, np.float32))
        b.v_stage.append(np.asarray(v, np.float32))
        b.n_tokens += 1
        sealed = False
        if b.n_tokens == cfg.block_size:
            self._seal(b)
            sealed = True
        return b, sealed

    def _seal(self, b: KVBlock) -> None:
        cfg = self.cfg
        k = np.stack(b.k_stage).astype(np.float32)
        v = np.stack(b.v_stage).astype(np.float32)
        payload = encode(k, v, "fp16", cfg.v_group)
        k16, v16 = decode(payload)           # fp16-rounded originals = ground truth from here on
        b.kmin, b.kmax = k16.min(axis=0), k16.max(axis=0)
        b.err_table, b.size_table = calibrate(k16, v16, self._codec_names, cfg.v_group)
        b.payload, b.sealed = payload, True
        b.k_stage = b.v_stage = None
        b.tier, b.codec = 0, "fp16"

    def get_kv(self, b: KVBlock) -> Tuple[np.ndarray, np.ndarray]:
        if b.dropped:
            raise ValueError("block was dropped")
        if not b.sealed:
            return np.stack(b.k_stage), np.stack(b.v_stage)
        return decode(b.payload)

    def reencode(self, b: KVBlock, codec: str) -> None:
        """Change codec. Decodes the current (possibly lossy) payload first, so error compounds."""
        new_family, new_entropy = parse_codec(codec)
        old_family, old_entropy = parse_codec(b.codec)
        if new_family == old_family:
            # same precision: only the entropy wrapper changes -> lossless blob transform
            raw = zlib.decompress(b.payload.blob) if old_entropy else b.payload.blob
            blob = zlib.compress(raw, 6) if new_entropy else raw
            b.payload = Payload(codec, b.payload.n_tokens, b.payload.d, b.payload.v_group, blob)
        else:
            k, v = decode(b.payload)
            b.payload = encode(k, v, codec, self.cfg.v_group)
        b.codec = codec

    def drop(self, b: KVBlock) -> None:
        b.dropped, b.payload = True, None
