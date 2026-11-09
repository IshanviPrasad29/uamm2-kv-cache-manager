import numpy as np
import pytest

from uamm2.codecs import (CODEC_RANK, calibrate, codec_rank, decode, encode, pack_bits,
                          parse_codec, perchannel_key_roundtrip, pertoken_key_roundtrip,
                          rel_error, unpack_bits)


def _kv(t=16, d=64, seed=0, outliers=True):
    r = np.random.default_rng(seed)
    scale = np.ones(d)
    if outliers:
        scale[r.choice(d, 4, replace=False)] = 6.0
    return (r.normal(size=(t, d)) * scale).astype(np.float32), r.normal(size=(t, d)).astype(np.float32)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("count", [1, 7, 64, 1000])
def test_pack_unpack_roundtrip(bits, count):
    r = np.random.default_rng(bits * 1000 + count)
    q = r.integers(0, 1 << bits, count).astype(np.uint8)
    assert np.array_equal(unpack_bits(pack_bits(q, bits), bits, count), q)


def test_pack_nibble_order_low_first():
    assert pack_bits(np.array([0x3, 0xA], np.uint8), 4) == bytes([0xA3])   # element 0 -> low nibble


def test_fp16_roundtrip_is_fp16_exact():
    k, v = _kv()
    k2, v2 = decode(encode(k, v, "fp16", 32))
    assert np.array_equal(k2, k.astype(np.float16).astype(np.float32))
    assert np.array_equal(v2, v.astype(np.float16).astype(np.float32))


def test_error_and_size_ordering():
    k, v = _kv()
    errs, sizes = calibrate(k, v, ["fp16", "int8", "kivi4", "kivi2"], 32)
    assert errs["fp16"] < errs["int8"] < errs["kivi4"] < errs["kivi2"]
    assert sizes["fp16"] > sizes["int8"] > sizes["kivi4"] > sizes["kivi2"]
    assert sizes["kivi4"] / sizes["fp16"] < 0.36
    assert sizes["kivi2"] / sizes["fp16"] < 0.25
    assert errs["int8"] < 1e-3 and errs["kivi4"] < 0.05


@pytest.mark.parametrize("fam", ["int8", "kivi4", "kivi2"])
def test_entropy_variant_is_lossless_wrt_base(fam):
    k, v = _kv(seed=3)
    a = decode(encode(k, v, fam, 32))
    b = decode(encode(k, v, fam + "z", 32))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_per_channel_keys_beat_per_token_on_outlier_channels():
    k, _ = _kv(seed=4, outliers=True)
    e_ch = float(((perchannel_key_roundtrip(k, 4) - k) ** 2).sum() / (k ** 2).sum())
    e_tok = float(((pertoken_key_roundtrip(k, 4) - k) ** 2).sum() / (k ** 2).sum())
    assert e_ch < e_tok


def test_constant_input_does_not_blow_up():
    k = np.full((16, 64), 3.0, np.float32)
    v = np.full((16, 64), -2.0, np.float32)
    for name in ("int8", "kivi4", "kivi2"):
        k2, v2 = decode(encode(k, v, name, 32))
        assert np.isfinite(k2).all() and np.abs(k2 - k).max() < 0.01


def test_partial_block_shapes():
    k, v = _kv(t=5)
    k2, v2 = decode(encode(k, v, "kivi4z", 32))
    assert k2.shape == (5, 64) and v2.shape == (5, 64)


def test_bad_inputs():
    with pytest.raises(ValueError):
        parse_codec("fp16z")
    with pytest.raises(ValueError):
        parse_codec("bogus")
    k, v = _kv()
    with pytest.raises(ValueError):
        encode(k, v[:, :32], "int8", 32)
    with pytest.raises(ValueError):
        encode(k, v, "int8", 48)       # d not divisible by v_group


def test_rank_ordering():
    assert codec_rank("fp16") < codec_rank("int8") < codec_rank("kivi4z") <= CODEC_RANK["kivi2"]
