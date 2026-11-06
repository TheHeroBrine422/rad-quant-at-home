# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""exl3dec -- an EXL3 trellis tensor back to its weight, on the CPU with numpy (no torch, no GPU).

    W (k = in, n = out) = diag(suh) . Hb . W_hat . Hb . diag(svh)        (exllamav3's order: had over k, x suh,
                                                                         had over n, x svh; Hb = 128-block Hadamard)
W_hat: 16x16 tiles, each a tail-biting ring of 16*K bits in 32-bit little-endian words read MSB first. Ring position
i reads D(i) = KA + bit (i mod 16) of MASK new bits (integer K: KA = K, MASK = 0; fractional K = KA + n/16:
MASK = frac_mask(n)); its state is the 16-bit window ending at S(i) = sum of D up to i; a 0-bit step (sub-1 K,
qtf2) reads the codebook at state ^ salt(i). Position i = 8t + j sits at tile row (k) 2(t%4) + {0,1,8,9}[j%4],
column (n) t/4 + 8(j/4). Transcribed from exllamav3 (quant/frac.cu, codebook.cuh), our fork's i8 / i8c codebooks
(quantize/exl3-patches 0003-0007) and v3/e3/exl3frac2/qtf2.cuh (the salt), and checked against the quantizer's own
dequantized weights (TESTS.md)."""
import math
import numpy as np

MUL1 = 0x83DCD12D


def frac_mask(n):
    return sum(1 << i for i in range(16) if ((i + 1) * n) // 16 - (i * n) // 16)


def ka_mask(K):
    """(KA, MASK) of bitrate K (16K must be an even integer for fractional K)"""
    if float(K).is_integer(): return int(K), 0
    ka = int(K); n = round((K - ka) * 16)
    assert abs((K - ka) * 16 - n) < 1e-9 and n % 2 == 0, K
    return ka, frac_mask(n)


def _f16(bits): return np.array([bits], dtype=np.uint16).view(np.float16)[0]


def codebook_lut(cb):
    """the fp16 value of every 16-bit state, as float32 [65536]"""
    s = np.arange(65536, dtype=np.uint64)
    x = (s * MUL1) & 0xFFFFFFFF
    bs = sum(((x >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    k_inv, k_bias = float(_f16(0x1EEE)), float(_f16(0xC931))
    v = (1024.0 + bs) * k_inv + k_bias                      # exact in f64: one rounding to f16, as __hfma
    v = v.astype(np.float16)
    if cb == "mul1": return v.astype(np.float32)
    s4 = np.float32(4.0) * np.float32(_f16(0x1EEE)); inv = np.float32(1.0) / s4
    if cb == "i8":
        q = np.clip(np.rint(v.astype(np.float32) * inv), np.float32(-127), np.float32(127))
        return (q * s4).astype(np.float16).astype(np.float32)
    if cb == "i8c":                                         # #78: C = 0.02, SD = 1.0546900 (exl3frac2.load)
        C, SD, sd = np.float32(0.02), np.float32(1.0546900), np.float32(36.950135)
        x = (v.astype(np.float32) * inv) * (np.float32(1.0) / sd)
        a = (C * x) * x
        y = np.array([math.fma(float(p), float(q), float(q)) if hasattr(math, "fma") else float(p) * float(q) + float(q)
                      for p, q in zip(a, x)], dtype=np.float64).astype(np.float32) * (sd / SD)
        q = np.clip(np.rint(y), np.float32(-127), np.float32(127))
        return (q * s4).astype(np.float16).astype(np.float32)
    if cb == "i4":
        st, iv = np.float32(2.6) / np.float32(7.5), np.float32(7.5) / np.float32(2.6)
        q = np.clip(np.rint(v.astype(np.float32) * iv - np.float32(0.5)), -8, 7) + np.float32(0.5)
        return (q * st).astype(np.float16).astype(np.float32)
    raise ValueError(f"codebook {cb!r} not supported by exl3dec")


_LUT = {}


def lut(cb):
    if cb not in _LUT: _LUT[cb] = codebook_lut(cb)
    return _LUT[cb]


def ring_plan(K):
    """per ring position: first word, shift, salt (0 unless a 0-bit step), and the tile element it lands on"""
    ka, mask = ka_mask(K)
    D = np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)])
    S = np.cumsum(D)
    nw = int(round(8 * K)); R = nw * 32
    assert S[-1] == R, (K, S[-1], R)
    start = (S - 16) % R
    wi, o = start >> 5, start & 31
    salt = np.where(D == 0, (0x9E37 * ((np.arange(256) & 15) + 1)) & 0xFFFF, 0).astype(np.uint64)
    t, j = np.arange(256) >> 3, np.arange(256) & 7
    r = (t & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1); c = (t >> 2) + 8 * ((j >> 2) & 1)
    return nw, wi, (wi + 1) % nw, (48 - o).astype(np.uint64), salt, r * 16 + c


def decode_what(trellis, K, cb, tn_sel=None):
    """trellis int16 [kt, nt, 16K] -> W_hat float32 [16 kt, 16 len(tn_sel)] (all tile columns by default)"""
    kt, nt, words = trellis.shape
    assert words == round(16 * K), (trellis.shape, K)
    nw, w0, w1, sh, salt, elem = ring_plan(K)
    tab = lut(cb)
    tsel = np.arange(nt) if tn_sel is None else np.asarray(tn_sel)
    out = np.empty((kt * 16, len(tsel) * 16), dtype=np.float32)
    u16 = trellis.view(np.uint16)
    inv = np.argsort(elem)                                  # tile element -> ring position
    step = max(1, (1 << 21) // max(1, len(tsel) * 256))
    for a in range(0, kt, step):
        b = min(kt, a + step)
        t = u16[a:b][:, tsel].reshape(-1, words)
        w = (t[:, 0::2].astype(np.uint64) | (t[:, 1::2].astype(np.uint64) << np.uint64(16)))
        v = ((w[:, w0] << np.uint64(32)) | w[:, w1]) >> sh
        st = (v & np.uint64(0xFFFF)) ^ salt
        vals = tab[st.astype(np.int64)][:, inv]             # tiles x (16 x 16 row-major, r = k)
        out[a * 16:b * 16] = vals.reshape(b - a, len(tsel), 16, 16).transpose(0, 2, 1, 3).reshape((b - a) * 16, len(tsel) * 16)
    return out


def had128():
    i = np.arange(128)
    pc = np.vectorize(lambda x: bin(x).count("1"))(i[:, None] & i[None, :])
    return (np.where(pc % 2, -1.0, 1.0) / math.sqrt(128)).astype(np.float32)


def decode(t, K, cb, n_blocks=None):
    """t: {"trellis", "suh", "svh"} numpy arrays -> W float32 [k, n] (or only the 128-column blocks n_blocks of n)"""
    tr, suh, svh = t["trellis"], t["suh"].astype(np.float32), t["svh"].astype(np.float32)
    kt, nt, _ = tr.shape
    k, n = kt * 16, nt * 16
    nb = np.arange(n // 128) if n_blocks is None else np.asarray(n_blocks)
    tn = (nb[:, None] * 8 + np.arange(8)[None, :]).reshape(-1)
    W = decode_what(tr, K, cb, tn)                           # [k, 128 len(nb)]
    H = had128()
    W = (H @ W.reshape(k // 128, 128, -1)).reshape(k, -1)    # had over k
    W *= suh[:, None]
    W = (W.reshape(k, -1, 128) @ H).reshape(k, -1)           # had over n
    cols = (nb[:, None] * 128 + np.arange(128)[None, :]).reshape(-1)
    W *= svh[cols][None, :]
    return W, cols
