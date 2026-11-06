#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""checker_tamper.py RESULTDIR PARAMS.json DATAROOT SCRATCH -- does the checker catch bad results? Copies an accepted
quant_layer result and tampers with it: codes of one tensor changed (1% of its trellis words), the per-row claim
shrunk 10%, the logged proxy shrunk 5%, a tensor removed, a wrong settings stamp; each must
FAIL, the untouched copy must PASS (full and sampled)."""
import json, os, shutil, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checker import check

res, pfile, data, scratch = sys.argv[1:5]
p = json.load(open(pfile)); L = p["layer"]; shard = f"q-L{L:02d}.safetensors"
dp = lambda rel: os.path.join(data, rel)
out = []


def copy(name):
    d = os.path.join(scratch, name)
    if os.path.exists(d): shutil.rmtree(d)
    shutil.copytree(res, d); return d


def run(name, d, expect, rows_frac=1.0, seed=None):
    rep = check.check_job("quant_layer", p, d, dp, rows_frac=rows_frac, seed=seed)
    good = rep["pass"] == expect
    out.append((name, good, rep["summary"]))
    print(f"[{'PASS' if good else 'FAIL'}] {name}: checker says {'pass' if rep['pass'] else 'fail'} ({rep['summary'][:150]})", flush=True)


def tensor_span(path, key):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    a, b = h[key]["data_offsets"]; return 8 + n + a, 8 + n + b


run("untouched, full", res, True)
run("untouched, sampled 25%", res, True, 0.25, 1)
key = json.loads(open(os.path.join(res, "proxy.jsonl")).readline())["key"]

d = copy("codes")
a, b = tensor_span(os.path.join(d, shard), f"{key}.trellis")
with open(os.path.join(d, shard), "r+b") as f:
    f.seek(a); buf = bytearray(f.read(b - a)); w = np.frombuffer(buf, dtype=np.uint16)
    rng = np.random.default_rng(0); i = rng.choice(len(w), max(1, len(w) // 100), replace=False)
    w2 = w.copy(); w2[i] ^= rng.integers(1, 65535, len(i), dtype=np.uint16)
    f.seek(a); f.write(w2.tobytes())
run("1% of one tensor's codes changed", d, False)
run("1% of one tensor's codes changed, sampled 25%", d, False, 0.25, 2)

d = copy("claims")
z = dict(np.load(os.path.join(d, "rowerr.npz"))); z[f"{key}.err"] = z[f"{key}.err"] * 0.9
np.savez(os.path.join(d, "rowerr.npz"), **z)
run("per-row claim 10% low", d, False, 0.25, 3)

d = copy("log")
rows = [json.loads(l) for l in open(os.path.join(d, "proxy.jsonl"))]; rows[0]["proxy_err"] *= 0.95
open(os.path.join(d, "proxy.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
run("logged proxy 5% low", d, False)

d = copy("missing")
import safetensors.numpy as stn                                                      # only the test needs it
t = stn.load_file(os.path.join(d, shard)); meta = check.ST(os.path.join(d, shard)).hdr.get("__metadata__")
t.pop(f"{key}.svh"); stn.save_file(t, os.path.join(d, shard), metadata=meta)
run("a tensor missing from the shard", d, False)

d = copy("stamp")
j = json.load(open(os.path.join(d, "job.json"))); j["settings_stamp"] = "0" * 16; json.dump(j, open(os.path.join(d, "job.json"), "w"))
run("wrong settings stamp", d, False)

n_ok = sum(x[1] for x in out)
print(f"{n_ok}/{len(out)} tamper checks as expected")
json.dump(out, open(os.path.join(scratch, "tamper.json"), "w"), indent=1)
sys.exit(0 if n_ok == len(out) else 1)
