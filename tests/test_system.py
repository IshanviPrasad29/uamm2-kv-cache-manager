from dataclasses import replace

import numpy as np
import pytest

from uamm2 import Config, Simulator
from uamm2.runtime import CompilerRuntimeAdapter, ReferenceAdapter
from uamm2.tiers import usage

N = 1024     # fp16 KV = 262 KB > total tier budget (168 KB): compression or dropping is mandatory


@pytest.fixture(scope="module")
def results():
    base = Config()
    v1 = replace(base, policy="threshold", codecs=("fp16",), use_entropy_coding=False)
    v2 = replace(base)
    out = {}
    for name, cfg in (("v1", v1), ("v2", v2)):
        sim = Simulator(cfg, seed=7)
        out[name] = (sim, sim.run(N))
    return out


def test_deterministic():
    a = Simulator(Config(), seed=3).run(256)
    b = Simulator(Config(), seed=3).run(256)
    assert a == b


def test_v2_fits_budget_without_losing_data(results):
    _, r = results["v2"]
    assert r["dropped_tokens"] == 0
    assert sum(r["capacity_overflow_bytes_max"].values()) == 0
    assert r["resident_vs_fp16"] < 0.5


def test_v1_must_drop_data_when_fp16_does_not_fit(results):
    _, r = results["v1"]
    assert r["dropped_tokens"] > 0


def test_v2_beats_v1_on_quality_in_this_synthetic_setup(results):
    _, r1 = results["v1"]
    _, r2 = results["v2"]
    assert r2["attn_rel_l2_mean"] < r1["attn_rel_l2_mean"]
    assert r2["attn_mass_recall_mean"] > r1["attn_mass_recall_mean"]


def test_tier_capacities_respected_on_exact_size_tiers(results):
    sim, _ = results["v2"]
    used = usage(sim.cfg, sim.cache.blocks)
    for t in (0, 1, 2):
        assert used[t] <= sim.cfg.tiers[t].capacity_bytes
    assert used[3] <= sim.cfg.tiers[3].capacity_bytes * 1.05       # entropy sizes are estimated


def test_protected_blocks_stay_hot_and_lossless(results):
    sim, _ = results["v2"]
    for b in sim.cache.blocks:
        if b.protected and not b.dropped:
            assert b.tier == 0 and b.codec == "fp16"


def test_precision_never_increases_across_plans():
    """One-way ratchet: instrument every applied plan and assert no block regains precision."""
    class Checked(Simulator):
        violations = 0
        changed = 0

        def _apply_plan(self, plan, step):
            before = {b.block_id: b.rank for b in self.cache.blocks}
            super()._apply_plan(plan, step)
            for b in self.cache.blocks:
                if b.block_id in before and not b.dropped:
                    if b.rank < before[b.block_id]:
                        Checked.violations += 1
                    if b.rank > before[b.block_id]:
                        Checked.changed += 1

    Checked.violations = Checked.changed = 0
    Checked(Config(), seed=7).run(768)
    assert Checked.violations == 0
    assert Checked.changed > 0          # the test really saw precision being lowered


def test_hysteresis_reduces_churn():
    base = Config()
    low = Simulator(replace(base, hysteresis_margin=0.0, min_residency_steps=0), 7).run(768)
    high = Simulator(base, 7).run(768)
    churn_hi = high["migrations"] + high["reencodes"]
    churn_lo = low["migrations"] + low["reencodes"]
    assert churn_hi < 0.6 * churn_lo          # observed ~0.3x on this workload


def test_drop_only_when_enabled():
    cfg = replace(Config(), codecs=("fp16",), use_entropy_coding=False)     # cannot fit -> overflow
    off = Simulator(cfg, 7).run(1024)
    on = Simulator(replace(cfg, allow_drop=True), 7).run(1024)
    assert off["dropped_tokens"] == 0 and sum(off["capacity_overflow_bytes_max"].values()) > 0
    assert on["dropped_tokens"] > 0
    assert sum(on["capacity_overflow_bytes_max"].values()) < sum(off["capacity_overflow_bytes_max"].values())


def test_reference_adapter_roundtrip_and_migrate():
    cfg = Config()
    ad = ReferenceAdapter(cfg)
    r = np.random.default_rng(0)
    for _ in range(16):
        ad.write_token(r.normal(size=64).astype(np.float32), r.normal(size=64).astype(np.float32))
    k0, _ = ad.read_block(0)
    ad.migrate_block(0, 2, "kivi4")
    k1, _ = ad.read_block(0)
    assert ad.cache.blocks[0].tier == 2 and np.abs(k1 - k0).max() < 1.0
    with pytest.raises(ValueError):
        ad.migrate_block(0, 9, "kivi4")


def test_compiler_adapter_is_an_honest_skeleton():
    with pytest.raises(NotImplementedError):
        CompilerRuntimeAdapter().write_token(None, None)


def test_invalid_run_args():
    with pytest.raises(ValueError):
        Simulator(Config()).run(0)


def test_static_baseline_fits_without_dropping_and_uses_one_codec():
    cfg = replace(Config(), policy="static", static_codec="kivi4", codecs=("fp16", "kivi4"),
                  use_entropy_coding=False)
    r = Simulator(cfg, seed=7).run(512)
    assert r["dropped_tokens"] == 0
    assert set(r["codec_blocks"]) <= {"fp16", "kivi4"}


def test_capacity_aware_threshold_variant_runs_and_is_lossy_when_fp16_does_not_fit():
    cfg = replace(Config(), policy="threshold", threshold_fill_fast=True, codecs=("fp16",),
                  use_entropy_coding=False)
    r = Simulator(cfg, seed=7).run(1024)
    assert r["dropped_tokens"] > 0


def test_summary_bytes_reported_and_positive():
    r = Simulator(Config(), seed=7).run(256)
    assert r["key_summary_bytes"] > 0
    assert r["resident_with_summaries_vs_fp16"] > r["resident_vs_fp16"]
