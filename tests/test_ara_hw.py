import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import find_verilator
from uamm2 import hwmodel
from uamm2.ara import AraKernels, rvv_compile_check
from uamm2.codecs import decode, encode

HW = Path(__file__).resolve().parent.parent / "hw"


# ------------------------------------------------------------- Ara kernel
@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
@pytest.mark.parametrize("codec", ["kivi4", "kivi4z"])
@pytest.mark.parametrize("t,d", [(16, 64), (5, 64), (16, 32), (1, 8)])
def test_c_kernel_matches_numpy_dequant_dot(codec, t, d):
    r = np.random.default_rng(t * d)
    k = (r.normal(size=(t, d)) * r.choice([0.5, 6.0], size=d)).astype(np.float32)
    v = r.normal(size=(t, d)).astype(np.float32)
    q = r.normal(size=d).astype(np.float32)
    p = encode(k, v, codec, min(32, d))
    ref = decode(p)[0] @ q
    got = AraKernels().qk_scores_kivi4(q, p)
    assert np.allclose(got, ref, atol=1e-4 * max(1.0, np.abs(ref).max()))


def test_c_kernel_rejects_other_codecs():
    k = np.zeros((16, 64), np.float32)
    with pytest.raises(ValueError):
        AraKernels().qk_scores_kivi4(np.zeros(64, np.float32), encode(k, k, "int8", 32))


@pytest.mark.skipif(shutil.which("clang") is None, reason="clang not available")
def test_rvv_path_compiles_for_riscv64():
    try:
        obj = rvv_compile_check()
    except subprocess.CalledProcessError:
        pytest.skip("this clang cannot target riscv64 with the V extension")
    assert obj.stat().st_size > 0        # compile check only: not executed


# ------------------------------------------------------------- golden models
def test_golden_scorer_properties():
    assert hwmodel.scorer_model(255, 255, 255, 0, 0)[0] == 255
    assert hwmodel.scorer_model(0, 0, 0, 0, 255) == (0, 2, 1)
    assert hwmodel.scorer_model(0, 0, 0, 1, 255) == (0, 0, 0)       # protected: hot, never evicted
    # weights sum to 256 so the score never exceeds 255
    assert (255 * (hwmodel.W_IMP + hwmodel.W_REC + hwmodel.W_FRQ)) >> 8 == 255
    # boundary vectors really sit on both sides of each threshold
    scores = {hwmodel.scorer_model(i, r, f, 0, 0)[0] for i, r, f, _ in hwmodel.scorer_datasets()[0]}
    for s in (88, 89, 173, 174):
        assert s in scores


def test_golden_codec_error_is_bounded():
    checked = 0
    for zp, inv, step, x, q, deq in hwmodel.codec_vectors():
        rng = max(x) - min(x)
        if rng == 0 or zp != min(x):          # skip constant groups and deliberately mis-set zp vectors
            continue
        checked += 1
        assert max(abs(a - b) for a, b in zip(x, deq)) <= rng / 15 / 2 + 2
    assert checked >= 100


# ------------------------------------------------------------- RTL regression
@pytest.mark.skipif(find_verilator() is None, reason="verilator not installed")
def test_rtl_regression_verilator(tmp_path):
    vbin = find_verilator()
    build = tmp_path / "build"
    cmd = ["make", "-C", str(HW), "test", f"VERILATOR={vbin}", f"BUILD={build}"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out[-3000:]
    for tag in ("TB_SWEEP_SCORER PASS", "TB_KV_CODEC PASS", "TB_BLOCK_TABLE PASS"):
        assert tag in out, out[-3000:]
