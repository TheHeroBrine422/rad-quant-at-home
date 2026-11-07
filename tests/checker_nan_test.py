#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""checker_nan_test.py -- the replicate-mode checker (checker/check.py) on the synthetic layer of tests/score_test.py,
no GPU: an honest result passes in full and sampled mode; a shard with a NaN scale, a per-row claim with a NaN, and a
claim whose rows are all 0/0 must each FAIL (a NaN comparison used to pass, issue #3); an honest quant_tensor group-0
result passes against its own shard and group weights file (issue #14)."""
import json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tests"))
import score_test as T                                       # its helpers only (no run())
from checker import check, exl3dec

RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def honest_log_and_claims(C0, p):
    """proxy_err per tensor = the exact decoded rel (what an honest ourq logs), and per output row the claim
    (err_j, ref_j) with sum(err) / sum(ref) == rel: e_j^T H e_j and w_j^T H w_j over the row's column of E / W"""
    wts = check.ST(f"{T.ROOT}/weights/L00.safetensors")
    logged, claims = {}, {}
    for hname, keys in T.PLAN:
        H = check.hessian(f"{T.ROOT}/hessians/{hname}.safetensors")
        for key in keys:
            Wq, _ = exl3dec.decode(C0[key], p["K"], p["codebook"])
            Wt = np.ascontiguousarray(wts.get(f"{key}.weight").T)
            E = Wq - Wt
            err = np.einsum("ij,ij->j", E, H @ E, dtype=np.float64); ref = np.einsum("ij,ij->j", Wt, H @ Wt, dtype=np.float64)
            logged[key] = float(err.sum() / ref.sum()); claims[key] = (err, ref)
    return logged, claims


def result(name, C, p, logged, claims=None):
    d = f"{T.W}/{name}"
    T.make_result(d, C, p)
    with open(f"{d}/proxy.jsonl", "w") as f:
        for key in C: f.write(json.dumps({"key": key, "K": p["K"], "codebook": p["codebook"], "scaled_hessian": True,
                                          "proxy_err": logged[key]}) + "\n")
    if claims is not None:
        np.savez(f"{d}/rowerr.npz", **{f"{k}.err": e for k, (e, _) in claims.items()}, **{f"{k}.ref": r for k, (_, r) in claims.items()})
    return d


def run():
    C0 = T.setup()
    p = dict(T.P, verify="replicate")
    logged, claims = honest_log_and_claims(C0, p)
    k0 = T.KEYS[0]

    rep = check.check_job("quant_layer", p, result("honest", C0, p, logged), T.dp)
    ok("honest result, full: PASS", rep["pass"], rep["summary"])
    rep = check.check_job("quant_layer", p, result("honest-claims", C0, p, logged, claims), T.dp, rows_frac=0.25, seed=1)
    ok("honest result with per-row claims, sampled 25%: PASS", rep["pass"], rep["summary"])

    C = {k: {x: v.copy() for x, v in c.items()} for k, c in C0.items()}
    C[k0]["suh"][0] = np.float16("nan")
    rep = check.check_job("quant_layer", p, result("nan-scale", C, p, logged), T.dp)
    ok("a NaN scale in one tensor, full: FAIL", not rep["pass"] and rep["summary"].startswith("structure:"), rep["summary"])

    bad = dict(claims); e, r = claims[k0]; bad[k0] = (np.full_like(e, np.nan), r)
    rep = check.check_job("quant_layer", p, result("nan-claim", C0, p, logged, bad), T.dp, rows_frac=0.25, seed=1)
    ok("a NaN per-row claim, sampled 25%: FAIL", not rep["pass"] and k0 in rep.get("bad", []), rep["summary"])

    bad = dict(claims); bad[k0] = (np.zeros_like(e), np.zeros_like(r))
    rep = check.check_job("quant_layer", p, result("zero-claim", C0, p, logged, bad), T.dp, rows_frac=0.25, seed=1)
    ok("an all-zero per-row claim (0/0), sampled 25%: FAIL", not rep["pass"] and k0 in rep.get("bad", []), rep["summary"])

    # quant_tensor: group 0 of the layer, its own weights file and shard name (issue #14)
    g0 = T.PLAN[0]; gk = g0[1]; pg = dict(p, group=0)
    wts_all = check.ST(f"{T.ROOT}/weights/L00.safetensors")
    T.write_st(f"{T.ROOT}/weights/L00-g0.safetensors", {f"{k}.weight": ("BF16", T.bf16(wts_all.get(f"{k}.weight"))) for k in gk})
    d = result("tensor-g0", {k: C0[k] for k in gk}, pg, logged)
    os.replace(f"{d}/q-L00.safetensors", f"{d}/q-L00-g0.safetensors")
    rep = check.check_job("quant_tensor", pg, d, T.dp)
    ok("quant_tensor group 0, honest, full: PASS", rep["pass"], rep["summary"])

    n_ok = sum(1 for _, c in RESULTS if c)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    shutil.rmtree(T.W, ignore_errors=True)
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
