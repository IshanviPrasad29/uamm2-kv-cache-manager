"""Python binding for the Ara/RVV KV kernels (ara/kv_kernels.c).

On a host machine this builds the *scalar* path of the C kernel with gcc (the RVV path needs a
RISC-V toolchain and Ara/Spike/FPGA to execute). The scalar and RVV paths share the same data layout
and algorithm, so the host test validates layout/packing/algebra; it does not validate the RVV
instruction sequence by execution.
"""
import ctypes
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from .codecs import FAMILY_BITS, Payload, parse_codec

ARA_DIR = Path(__file__).resolve().parent.parent / "ara"


def build_host_lib(out_dir: Optional[str] = None) -> Path:
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="uamm2_ara_"))
    so = out_dir / "libkv_kernels_host.so"
    subprocess.run(["gcc", "-O2", "-Wall", "-shared", "-fPIC", str(ARA_DIR / "kv_kernels.c"),
                    "-o", str(so)], check=True)
    return so


def rvv_compile_check(out_dir: Optional[str] = None) -> Path:
    """Compile the RVV path with clang for riscv64 (object file only; cannot be executed here)."""
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="uamm2_rvv_"))
    obj = out_dir / "kv_kernels_rvv.o"
    subprocess.run(["clang", "--target=riscv64-unknown-elf", "-march=rv64gcv", "-mabi=lp64d", "-O2",
                    "-Wall", "-ffreestanding", "-nostdlib", "-c", str(ARA_DIR / "kv_kernels.c"),
                    "-o", str(obj)], check=True)
    return obj


class AraKernels:
    def __init__(self, lib_path: Optional[str] = None):
        self.lib = ctypes.CDLL(str(lib_path or build_host_lib()))
        fp = ctypes.POINTER(ctypes.c_float)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        self.lib.kv_qk_prepare.restype = ctypes.c_float
        self.lib.kv_qk_prepare.argtypes = [fp, fp, fp, ctypes.c_int, fp, fp]
        self.lib.kv_qk_dot_kivi4.restype = None
        self.lib.kv_qk_dot_kivi4.argtypes = [fp, fp, ctypes.c_float, u8, ctypes.c_int, ctypes.c_int, fp]

    @staticmethod
    def _f(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def qk_scores_kivi4(self, q: np.ndarray, payload: Payload) -> np.ndarray:
        """Scores S[t] = q . K[t] computed directly from a kivi4/kivi4z payload's packed keys."""
        family, entropy = parse_codec(payload.codec)
        if family != "kivi4":
            raise ValueError("payload must be kivi4 or kivi4z")
        blob = zlib.decompress(payload.blob) if entropy else payload.blob
        t, d = payload.n_tokens, payload.d
        if d % 2:
            raise ValueError("d must be even")
        nk = (t * d * FAMILY_BITS[family]) // 8
        off = nk + nk                                     # skip packed K and packed V
        sk = np.frombuffer(blob[off:off + 2 * d], "<f2").astype(np.float32)
        zk = np.frombuffer(blob[off + 2 * d:off + 4 * d], "<f2").astype(np.float32)
        packed = np.frombuffer(blob[:nk], np.uint8).copy()
        q = np.ascontiguousarray(q, np.float32)
        qe = np.empty(d // 2, np.float32)
        qo = np.empty(d // 2, np.float32)
        out = np.empty(t, np.float32)
        bias = self.lib.kv_qk_prepare(self._f(q), self._f(sk), self._f(zk), d, self._f(qe), self._f(qo))
        self.lib.kv_qk_dot_kivi4(self._f(qe), self._f(qo), ctypes.c_float(bias),
                                 packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), t, d,
                                 self._f(out))
        return out
