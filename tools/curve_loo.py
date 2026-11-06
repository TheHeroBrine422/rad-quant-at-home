#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""curve_loo.py PROXY.jsonl ... -- how well a tensor's error at one K is predicted from its errors at other K (the
score band of docs/VERIFY.md). Reads ourq proxy logs (one per level; key, K, proxy_err), and prints per tensor family:
the rate-distortion slope (median d log rel / dK), the leave-one-K-out residual of a local quadratic fit and of linear
interpolation between the two neighbours (SD, p99, max of |residual| in log rel), and the spread of the family shape
when extrapolating from the lowest K. Residuals are log(actual / predicted): 0.01 = 1 %."""
import collections, json, math, sys
import numpy as np

rel = collections.defaultdict(dict)
for p in sys.argv[1:]:
    for l in open(p):
        if l.strip():
            r = json.loads(l); rel[r["key"]][float(r["K"])] = math.log(r["proxy_err"])
fam = lambda k: k.rsplit(".", 1)[-1]
Ks = sorted({k for v in rel.values() for k in v})
print(f"{len(rel)} tensors, K {Ks}")
quad, intp, slope, ext = (collections.defaultdict(list) for _ in range(4))
for key, v in rel.items():
    ks = sorted(v)
    for a, b in zip(ks, ks[1:]): slope[fam(key)].append(-(v[b] - v[a]) / (b - a))
    for K in ks:
        o = [x for x in ks if x != K]
        if len(o) >= 3:
            c = np.polyfit(o, [v[x] for x in o], 2); quad[fam(key)].append(v[K] - np.polyval(c, K))
        lo, hi = [x for x in o if x < K], [x for x in o if x > K]
        if lo and hi:
            a, b = max(lo), min(hi); intp[fam(key)].append(v[K] - (v[a] + (v[b] - v[a]) * (K - a) / (b - a)))
    k0 = ks[0]
    for K in ks[1:]: ext[(fam(key), K - k0)].append(v[K] - v[k0])
st = lambda x: f"sd {np.std(x):.4f} p99 {np.percentile(np.abs(x), 99):.4f} max {np.max(np.abs(x)):.4f}" if len(x) else "-"
print(f"{'family':12s} {'slope/bit':>9s}  {'leave-one-K-out quadratic':34s}  {'neighbour interpolation':34s}  n")
for f in sorted(slope):
    print(f"{f:12s} {np.median(slope[f]):9.3f}  {st(quad[f]):34s}  {st(intp[f]):34s}  {len(quad[f])}")
print("extrapolation from the lowest K with the family's median shape: SD of the residual by distance")
for f in sorted(slope):
    row = []
    for d in sorted({d for (g, d) in ext if g == f}):
        x = np.array(ext[(f, d)])
        if len(x) > 3: row.append(f"{d:g}b {np.std(x - np.median(x)):.4f}")
    print(f"  {f:12s} " + "  ".join(row))
