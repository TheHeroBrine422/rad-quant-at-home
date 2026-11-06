#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""seed_spread.py DIR... -- the honest spread of the quantizer's error over repeats (ourq output dirs of the same
settings and K, e.g. --seed-offset 0..4): per tensor the range and SD of log(proxy_err) over the dirs, its size n
(in x out, from the shard) and SD x sqrt(n), the size-scaled noise constant of docs/VERIFY.md (a quant's error is a
sum over many independent tiles, so its relative noise falls like 1 / sqrt(n))."""
import collections, glob, json, math, os, struct, sys
import numpy as np

rel, n = collections.defaultdict(list), {}
for d in sys.argv[1:]:
    for l in open(os.path.join(d, "proxy.jsonl")):
        if l.strip(): r = json.loads(l); rel[(float(r["K"]), r["key"])].append(r["proxy_err"])
    for f in glob.glob(os.path.join(d, "q-*.safetensors")):
        with open(f, "rb") as fh: h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in h.items():
            if k.endswith(".trellis"): n[k[:-len(".trellis")]] = v["shape"][0] * v["shape"][1] * 256
c = []
for (K, k), v in sorted(rel.items()):
    if len(v) < 2: continue
    lv = np.log(v); sd = lv.std(ddof=1); c.append(sd * math.sqrt(n[k]))
    print(f"K{K:g} {k.rsplit('.', 1)[-1]:12s} L{k.split('.')[3]:>2s} n {n[k] / 1e6:7.2f}M  seeds {len(v)}  range {lv.max() - lv.min():.4f}  "
          f"sd {sd:.4f}  sd*sqrt(n) {c[-1]:6.1f}")
if c: print(f"sd*sqrt(n): median {np.median(c):.1f}, max {max(c):.1f} over {len(c)} tensors")
