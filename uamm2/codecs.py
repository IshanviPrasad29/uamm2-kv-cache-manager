"""KIVI-style KV block codecs (real quantization, real byte payloads).

Per block of T tokens with d channels:
  * keys   : asymmetric, quantized PER-CHANNEL (min/max over the block's tokens)
  * values : asymmetric, quantized PER-TOKEN in channel groups of `v_group`
This follows the asymmetric scheme described in the KIVI paper. Scales/zeros are
stored as fp16. Codec names:
  fp16 | int8 | kivi4 | kivi2   and entropy-coded variants  int8z | kivi4z | kivi2z
The "z" variants run zlib over the serialized blob (used on the flash tier).
Precision is a one-way ratchet: re-encoding a lossy block can only lower precision.
"""
import zlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

CODEC_RANK = {"fp16": 0, "int8": 1, "kivi4": 2, "kivi2": 3}
FAMILY_BITS = {"fp16": 16, "int8": 8, "kivi4": 4, "kivi2": 2}


def parse_codec(name: str) -> Tuple[str, bool]:
    """Return (family, entropy_coded)."""
    entropy = name.endswith("z")
    family = name[:-1] if entropy else name
    if family not in FAMILY_BITS or (entropy and family == "fp16"):
        raise ValueError(f"unknown codec {name!r}")
    return family, entropy


def codec_rank(name: str) -> int:
    return CODEC_RANK[parse_codec(name)[0]]


@dataclass
class Payload:
    codec: str
    n_tokens: int
    d: int
    v_group: int
    blob: bytes

    @property
    def nbytes(self) -> int:
        return len(self.blob)


# ---------------------------------------------------------------- primitives
def _asym_quant(x: np.ndarray, bits: int, axis: int):
    levels = (1 << bits) - 1
    mn = x.min(axis=axis, keepdims=True)
    mx = x.max(axis=axis, keepdims=True)
    zero = mn.astype(np.float16)
    scale = np.maximum((mx - mn) / levels, 2.0 ** -10).astype(np.float16)
    q = np.clip(np.rint((x - zero.astype(np.float32)) / scale.astype(np.float32)), 0, levels)
    return q.astype(np.uint8), scale, zero


def _asym_dequant(q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32) + zero.astype(np.float32)


def pack_bits(q: np.ndarray, bits: int) -> bytes:
    """Pack uint8 values (< 2**bits) little-endian within each byte (element 0 = low bits)."""
    flat = q.reshape(-1).astype(np.uint8)
    if bits == 8:
        return flat.tobytes()
    per = 8 // bits
    pad = (-len(flat)) % per
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    m = flat.reshape(-1, per)
    out = np.zeros(m.shape[0], np.uint8)
    for i in range(per):
        out |= (m[:, i] << (bits * i)).astype(np.uint8)
    return out.tobytes()


def unpack_bits(buf: bytes, bits: int, count: int) -> np.ndarray:
    arr = np.frombuffer(buf, np.uint8)
    if bits == 8:
        return arr[:count].copy()
    per = 8 // bits
    mask = (1 << bits) - 1
    out = np.empty((len(arr), per), np.uint8)
    for i in range(per):
        out[:, i] = (arr >> (bits * i)) & mask
    return out.reshape(-1)[:count]


def _packed_len(count: int, bits: int) -> int:
    return count if bits == 8 else -(-count * bits // 8)


# ------------------------------------------------------------------ encode
def encode(k: np.ndarray, v: np.ndarray, codec: str, v_group: int) -> Payload:
    t, d = k.shape
    if v.shape != (t, d):
        raise ValueError("k and v must have the same shape (T, d)")
    if d % v_group:
        raise ValueError("d must be divisible by v_group")
    family, entropy = parse_codec(codec)
    if family == "fp16":
        blob = k.astype("<f2").tobytes() + v.astype("<f2").tobytes()
    else:
        bits = FAMILY_BITS[family]
        qk, sk, zk = _asym_quant(k.astype(np.float32), bits, axis=0)               # per-channel
        v3 = v.astype(np.float32).reshape(t, d // v_group, v_group)
        qv, sv, zv = _asym_quant(v3, bits, axis=2)                                  # per-token
        blob = (pack_bits(qk, bits) + pack_bits(qv, bits) +
                sk.tobytes() + zk.tobytes() + sv.tobytes() + zv.tobytes())
    if entropy:
        blob = zlib.compress(blob, 6)
    return Payload(codec, t, d, v_group, blob)


# ------------------------------------------------------------------ decode
def decode(p: Payload) -> Tuple[np.ndarray, np.ndarray]:
    family, entropy = parse_codec(p.codec)
    blob = zlib.decompress(p.blob) if entropy else p.blob
    t, d, g = p.n_tokens, p.d, p.v_group
    if family == "fp16":
        n = t * d * 2
        k = np.frombuffer(blob[:n], "<f2").astype(np.float32).reshape(t, d)
        v = np.frombuffer(blob[n:2 * n], "<f2").astype(np.float32).reshape(t, d)
        return k, v
    bits = FAMILY_BITS[family]
    nk = _packed_len(t * d, bits)
    off = 0
    qk = unpack_bits(blob[off:off + nk], bits, t * d).reshape(t, d); off += nk
    qv = unpack_bits(blob[off:off + nk], bits, t * d).reshape(t, d // g, g); off += nk
    ng = t * (d // g)
    sk = np.frombuffer(blob[off:off + 2 * d], "<f2").reshape(1, d); off += 2 * d
    zk = np.frombuffer(blob[off:off + 2 * d], "<f2").reshape(1, d); off += 2 * d
    sv = np.frombuffer(blob[off:off + 2 * ng], "<f2").reshape(t, d // g, 1); off += 2 * ng
    zv = np.frombuffer(blob[off:off + 2 * ng], "<f2").reshape(t, d // g, 1); off += 2 * ng
    k = _asym_dequant(qk, sk, zk)
    v = _asym_dequant(qv, sv, zv).reshape(t, d)
    return k, v


def rel_error(k, v, k2, v2) -> float:
    num = float(((k - k2) ** 2).sum() + ((v - v2) ** 2).sum())
    den = float((k ** 2).sum() + (v ** 2).sum()) + 1e-12
    return num / den


def calibrate(k: np.ndarray, v: np.ndarray, codec_names, v_group: int):
    """Encode `k,v` with every codec once; return (err_table, size_table).

    Done when a block is sealed (originals available), like a background calibration pass.
    """
    errs: Dict[str, float] = {}
    sizes: Dict[str, int] = {}
    for name in codec_names:
        p = encode(k, v, name, v_group)
        k2, v2 = decode(p)
        errs[name] = rel_error(k, v, k2, v2)
        sizes[name] = p.nbytes
    return errs, sizes


def pertoken_key_roundtrip(k: np.ndarray, bits: int) -> np.ndarray:
    """Per-TOKEN key quantization (the alternative KIVI argues against); used in tests."""
    q, s, z = _asym_quant(k.astype(np.float32), bits, axis=1)
    return _asym_dequant(q, s, z)


def perchannel_key_roundtrip(k: np.ndarray, bits: int) -> np.ndarray:
    q, s, z = _asym_quant(k.astype(np.float32), bits, axis=0)
    return _asym_dequant(q, s, z)
