"""Tier accounting and (assumed) cost model. No real memory movement happens here."""
from typing import List

from .codecs import parse_codec
from .config import Config
from .paged_cache import KVBlock

NUM_TIERS = 4
FLASH = 3
DROP = -1


def usage(cfg: Config, blocks: List[KVBlock]) -> List[int]:
    used = [0] * len(cfg.tiers)
    for b in blocks:
        if not b.dropped:
            used[b.tier] += b.nbytes(cfg.head_dim)
    return used


def decode_cost_ms(cfg: Config, codec: str, nbytes: int) -> float:
    family, entropy = parse_codec(codec)
    ns = cfg.decode_ns_per_byte[family] + (cfg.zlib_ns_per_byte if entropy else 0.0)
    return nbytes * ns * 1e-6


def read_cost_ms(cfg: Config, tier: int, codec: str, nbytes: int) -> float:
    spec = cfg.tiers[tier]
    xfer_ms = nbytes / (spec.bandwidth_gbps * 1e9) * 1e3
    return spec.base_latency_ms + xfer_ms + decode_cost_ms(cfg, codec, nbytes)


def move_cost_ms(cfg: Config, src: int, dst: int, nbytes: int) -> float:
    a, b = cfg.tiers[src], cfg.tiers[dst]
    return (a.base_latency_ms + b.base_latency_ms +
            nbytes / (a.bandwidth_gbps * 1e9) * 1e3 + nbytes / (b.bandwidth_gbps * 1e9) * 1e3)
