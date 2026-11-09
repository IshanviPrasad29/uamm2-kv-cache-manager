import numpy as np

from uamm2 import Config
from uamm2.attention import dense_oracle, select_blocks, sparse_attention
from uamm2.codecs import decode
from uamm2.h2o import H2OImportance
from uamm2.paged_cache import PagedKVCache


def _fill(cfg, n, seed=0):
    r = np.random.default_rng(seed)
    cache = PagedKVCache(cfg)
    K = r.normal(size=(n, cfg.head_dim)).astype(np.float32)
    V = r.normal(size=(n, cfg.head_dim)).astype(np.float32)
    for k, v in zip(K, V):
        cache.append_token(k, v)
    return cache, K.astype(np.float16).astype(np.float32), V.astype(np.float16).astype(np.float32)


def test_block_sealing_and_partial_tail():
    cfg = Config()
    cache, K, V = _fill(cfg, 40)
    assert [b.sealed for b in cache.blocks] == [True, True, False]
    assert [b.n_tokens for b in cache.blocks] == [16, 16, 8]
    assert cache.n_tokens == 40
    k0, v0 = cache.get_kv(cache.blocks[0])
    assert np.allclose(k0, K[:16]) and np.allclose(v0, V[:16])


def test_reencode_ratchet_and_entropy_switch_lossless():
    cfg = Config()
    cache, _, _ = _fill(cfg, 16)
    b = cache.blocks[0]
    cache.reencode(b, "kivi4")
    k_a, v_a = decode(b.payload)
    cache.reencode(b, "kivi4z")             # same family -> lossless blob transform
    k_b, v_b = decode(b.payload)
    assert np.array_equal(k_a, k_b) and np.array_equal(v_a, v_b)
    cache.reencode(b, "kivi4")
    assert b.payload.codec == "kivi4"


def test_selection_bound_is_a_true_upper_bound():
    cfg = Config()
    cache, K, _ = _fill(cfg, 160, seed=2)
    r = np.random.default_rng(9)
    for _ in range(20):
        q = r.normal(size=cfg.head_dim).astype(np.float32)
        for b in cache.blocks:
            bound = np.maximum(q * b.kmin, q * b.kmax).sum()
            true_max = (K[b.start:b.end] @ q).max()
            assert bound + 1e-3 >= true_max


def test_sparse_attention_equals_dense_when_everything_is_read_fp16():
    cfg = Config()
    cache, K, V = _fill(cfg, 96, seed=5)
    q = np.random.default_rng(1).normal(size=cfg.head_dim).astype(np.float32)
    res = sparse_attention(q, cache, cache.blocks)
    ref, _ = dense_oracle(q, K, V)
    assert np.allclose(res.out, ref, atol=1e-4)
    assert abs(res.weights.sum() - 1.0) < 1e-5


def test_select_blocks_budget_and_dense_fallback():
    cfg = Config()
    cache, _, _ = _fill(cfg, 160)
    q = np.ones(cfg.head_dim, np.float32)
    cands = [b for b in cache.blocks if b.sealed]
    assert len(select_blocks(q, cands, 3)) == 3
    assert len(select_blocks(q, cands, 100)) == len(cands)


def test_h2o_heavy_hitter_ranks_higher_and_importance_normalized():
    cfg = Config()
    cache, _, _ = _fill(cfg, 128)
    h = H2OImportance(cfg)
    r = np.random.default_rng(0)
    for step in range(60):
        idx = np.arange(128)
        w = r.random(128) * 0.01
        w[20] += 0.5                       # token 20 (block 1) is a heavy hitter
        h.update(idx, w / w.sum(), 128)
    h.refresh(cache.blocks)
    imps = [b.importance for b in cache.blocks]
    assert all(0.0 <= x <= 1.0 for x in imps)
    assert int(np.argmax(imps)) == 1


def test_h2o_decay_forgets():
    cfg = Config(h2o_decay=0.5)
    h = H2OImportance(cfg)
    h.update(np.array([3]), np.array([1.0]), 8)
    for _ in range(10):
        h.update(np.array([0]), np.array([0.0]), 8)
    assert h.tok[3] < 1e-2


def test_config_validation():
    import pytest
    with pytest.raises(ValueError):
        Config(head_dim=48, v_group=32)
    with pytest.raises(ValueError):
        Config(policy="nope")
    with pytest.raises(ValueError):
        Config(h2o_decay=1.5)
