"""Configuration for UAMM-2.

All latency / bandwidth / decode-cost numbers below are ASSUMED placeholders for an
edge SoC. They are not measurements. Replace them with numbers measured on your
target (Ara FPGA build, LPDDR, flash) before drawing any performance conclusion.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class TierSpec:
    name: str
    capacity_bytes: int
    base_latency_ms: float      # fixed cost per block access
    bandwidth_gbps: float       # GB/s


def _default_tiers() -> Tuple[TierSpec, ...]:
    return (
        TierSpec("T0_sram_vrf", 24 * 1024, 0.0005, 100.0),   # near-compute, protected set
        TierSpec("T1_dram_hot", 48 * 1024, 0.0020, 12.8),    # DRAM, light codecs
        TierSpec("T2_dram_warm", 32 * 1024, 0.0020, 12.8),   # DRAM, 4/2-bit codecs only
        TierSpec("T3_flash", 64 * 1024, 0.1000, 1.0),        # flash, entropy-coded
    )


@dataclass
class Config:
    # ---- geometry -------------------------------------------------------
    head_dim: int = 64              # channels of the flattened per-token K (and V) vector
    block_size: int = 16            # tokens per KV block (paged)
    v_group: int = 32               # channel group size for per-token value quantization
    sink_tokens: int = 4            # attention-sink tokens (always protected)
    recent_window_tokens: int = 32  # recent window (always protected)

    # ---- sparse attention (block selection) -----------------------------
    attn_block_budget: int = 24     # max non-protected blocks fetched per decode step

    # ---- H2O-style importance -------------------------------------------
    h2o_decay: float = 0.95
    h2o_aggregation: str = "mean"   # mean | max | sum

    # ---- planner --------------------------------------------------------
    policy: str = "utility"         # "utility" (UAMM-2) | "threshold" (original UAMM-1)
    plan_interval: int = 8          # replan every N decode steps
    hysteresis_margin: float = 0.25
    min_residency_steps: int = 16
    lambda_quality: float = 2.0     # ms-equivalent penalty per unit (heat * rel_err); tuned on synthetic seeds 7-9
    capacity_price: float = 0.05    # ms-equivalent congestion price (see planner)
    capacity_headroom: float = 0.03
    hit_ema: float = 0.9            # EMA factor for per-block selection frequency
    allow_drop: bool = False        # H2O-style permanent eviction (lossy), off by default
    drop_importance: float = 0.02

    # ---- codecs ---------------------------------------------------------
    codecs: Tuple[str, ...] = ("fp16", "int8", "kivi4", "kivi2")  # enabled families
    use_entropy_coding: bool = True                               # zlib variants on flash

    # ---- original UAMM-1 threshold policy (kept for ablation) -----------
    alpha: float = 0.60
    beta: float = 0.25
    gamma: float = 0.15
    t1_threshold: float = 0.68
    t2_threshold: float = 0.35

    # ---- cost model (ASSUMED) -------------------------------------------
    tiers: Tuple[TierSpec, ...] = field(default_factory=_default_tiers)
    decode_ns_per_byte: Dict[str, float] = field(default_factory=lambda: {
        "fp16": 0.0, "int8": 0.05, "kivi4": 0.12, "kivi2": 0.20})
    zlib_ns_per_byte: float = 1.0

    def __post_init__(self):
        if self.head_dim % self.v_group != 0:
            raise ValueError("head_dim must be divisible by v_group")
        if self.h2o_aggregation not in ("mean", "max", "sum"):
            raise ValueError("h2o_aggregation must be mean|max|sum")
        if self.policy not in ("utility", "threshold"):
            raise ValueError("policy must be utility|threshold")
        if not 0.0 <= self.h2o_decay <= 1.0:
            raise ValueError("h2o_decay must be in [0,1]")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
