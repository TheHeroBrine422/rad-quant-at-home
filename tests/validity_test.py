#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""validity_test.py -- checker.check.validity (the format rule of docs/VERIFY.md, shared by the checker and the scorer):
finite scales, at most 1% exact zeros (an honest refit can underflow one f16 scale: a V3 K0.5 q_proj, 11-08), |x| <= 1e3,
the mul1 marker, the trellis width of K. Seconds, stdlib + numpy."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checker.check import validity
from checker import exl3dec

res = []


def ok(name, cond, detail=""):
    res.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def t(suh=None, svh=None, words=40, marker=exl3dec.MUL1):
    s = np.full(5120, 0.02, np.float16) if suh is None else suh
    v = np.full(1024, 0.01, np.float16) if svh is None else svh
    return {"suh": s, "svh": v, "trellis": np.zeros((320, 64, words), np.int16), "mul1": np.array(np.uint32(marker).view(np.int32))}


ok("clean", validity("k", t(), 2.5) is None)
s = np.full(5120, 0.02, np.float16); s[4426] = 0
ok("one zero suh (refit underflow) accepted", validity("k", t(suh=s), 2.5) is None)
s = np.full(5120, 0.02, np.float16); s[:52] = 0
ok("52 zeros (> 1%) refused", "zero scales" in (validity("k", t(suh=s), 2.5) or ""), validity("k", t(suh=s), 2.5))
v = np.full(1024, 0.01, np.float16); v[:11] = 0
ok("11 of 1024 svh zeros refused", "zero scales" in (validity("k", t(svh=v), 2.5) or ""))
v = np.full(1024, 0.01, np.float16); v[:10] = 0
ok("10 of 1024 svh zeros accepted", validity("k", t(svh=v), 2.5) is None)
s = np.full(5120, 0.02, np.float16); s[3] = np.nan
ok("NaN refused", "not finite" in (validity("k", t(suh=s), 2.5) or ""))
s = np.full(5120, 0.02, np.float16); s[3] = 2000
ok("huge refused", "above" in (validity("k", t(suh=s), 2.5) or ""))
ok("marker refused", "marker" in (validity("k", t(marker=0x1234), 2.5) or ""))
ok("width refused", "words per tile" in (validity("k", t(words=32), 2.5) or ""))
print(f"\n{sum(res)}/{len(res)} checks passed")
sys.exit(0 if all(res) else 1)
