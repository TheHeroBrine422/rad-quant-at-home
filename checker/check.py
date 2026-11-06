#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""check.py -- verify a returned quant_layer / quant_head result without trusting the worker (CPU, numpy only).

  check.py job --type quant_layer --params P.json --dir RESULTDIR --data DATAROOT [--rows-frac F] [--tol T]
  check.py ourq --dir OURQ_OUT --layer L --K K --codebook CB --ckpt CKPT --hess HESSDIR [--wq] [--rows-frac F]

1. structure: the shard holds exactly {key}.trellis / suh / svh / mul1 of every linear in the layer plan, with the
   dtypes and shapes the bf16 weights imply; proxy.jsonl covers each tensor once with the job's K / codebook /
   scaled-Hessian setting; job.json carries the job's settings stamp.
2. proxy recompute: each tensor decoded from its codes (exl3dec), rel = tr(dW H dW^T) / tr(W H W^T) against OUR
   input Hessian and OUR bf16 weight, compared with the proxy_err the worker logged (exllamav3's post-refit error,
   which is the same quantity against the undamped Hessian): |rel / logged - 1| <= tol. --rows-frac F < 1 scores a
   random subset of the 128-wide output blocks (the checker's seed, unknown to the worker) and compares them with
   the worker's per-row claim (rowerr.npz: ourq --save-wq's err / ref per output row) on exactly those blocks; the
   claim's whole-tensor total must also match the log. ~4x cheaper at F = 0.25.
--wq (ourq mode): also compare the decode with ourq's own dequantized weights in OUT/wq (decoder validation)."""
import argparse, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checker import exl3dec
from common.util import st_header

TOL = 2e-3            # default |rel / logged - 1| (TESTS.md: measured decode-vs-log spread, x ~10)
_DT = {"F32": np.float32, "F16": np.float16, "I16": np.int16, "I32": np.int32, "BF16": np.uint16, "I64": np.int64}


class ST:
    """a safetensors file as numpy memmaps"""
    def __init__(self, path):
        self.path = path; self.hdr = st_header(path)
        with open(path, "rb") as f: self.base = 8 + struct.unpack("<Q", f.read(8))[0]
    def keys(self): return [k for k in self.hdr if k != "__metadata__"]
    def get(self, k):
        e = self.hdr[k]; dt = _DT[e["dtype"]]; a, b = e["data_offsets"]
        x = np.memmap(self.path, dtype=dt, mode="r", offset=self.base + a, shape=tuple(e["shape"]) or (1,))
        if e["dtype"] == "BF16": return (np.asarray(x).astype(np.uint32) << 16).view(np.float32)
        return np.asarray(x) if e["shape"] else np.asarray(x).reshape(())


def hessian(path):
    """packed upper triangle (hessian_upper, row-major) -> symmetric float32 [k, k], blockwise (no index arrays)"""
    f = ST(path); k = int(f.hdr["__metadata__"]["k"]); up = f.get("hessian_upper")
    H = np.empty((k, k), dtype=np.float32); off = 0
    for r in range(k):
        H[r, r:] = up[off:off + k - r]; off += k - r
    B = 1024
    for i in range(0, k, B):
        blk = H[i:i + B, i:i + B]; H[i:i + B, i:i + B] = np.triu(blk) + np.triu(blk, 1).T
        for j in range(i + B, k, B): H[j:j + B, i:i + B] = H[i:i + B, j:j + B].T
    return H


def rel_err(Wt, Wq, H):
    """tr(E^T H E) / tr(W^T H W) over the columns given (E = Wq - W, both [k, n'])"""
    E = Wq - Wt
    num = float(np.einsum("ij,ij->", E, H @ E, dtype=np.float64))
    den = float(np.einsum("ij,ij->", Wt, H @ Wt, dtype=np.float64))
    return num, den


def check_tensors(plan, shard, get_weight, hess_path, K, cb, logged, rows_frac=1.0, seed=None, wq_dir=None, claims=None):
    """per tensor: decode, recompute rel on all (or a random subset of) 128-wide output blocks, compare with the
    worker's claims: the logged proxy_err (whole tensor) and, given claims {key: (err, ref) per output row}, the claim
    on exactly the sampled blocks plus the claim's whole-tensor total against the log"""
    rng = np.random.default_rng(seed)
    out = []
    for hname, keys in plan:
        H = None
        for key in keys:
            t0 = time.time()
            t = {x: shard.get(f"{key}.{x}") for x in ("trellis", "suh", "svh")}
            n = t["svh"].shape[0]; nb = n // 128
            sel = None if rows_frac >= 1 else np.sort(rng.choice(nb, max(1, int(round(nb * rows_frac))), replace=False))
            Wq, cols = exl3dec.decode(t, K, cb, sel)
            t1 = time.time()
            Wt = np.ascontiguousarray(get_weight(key)[cols].T)          # checkpoint (out, in) -> [k, n']
            if H is None: H = hessian(hess_path(hname))
            num, den = rel_err(Wt, Wq, H)
            rec = num / den
            r = {"key": key, "rel": rec, "logged": logged.get(key), "cols": len(cols), "n": n,
                 "decode_s": round(t1 - t0, 2), "s": round(time.time() - t0, 2)}
            if r["logged"] and sel is None: r["ratio"] = rec / r["logged"]
            c = (claims or {}).get(key)
            if c is not None:
                err, ref = np.asarray(c[0], np.float64), np.asarray(c[1], np.float64)
                if err.shape != (n,) or ref.shape != (n,): r["claim_bad_shape"] = True
                else:
                    r["claim_sel_ratio"] = rec / (err[cols].sum() / ref[cols].sum())
                    if r["logged"]: r["claim_log_ratio"] = (err.sum() / ref.sum()) / r["logged"]
            if wq_dir:
                wq = ST(os.path.join(wq_dir, f"{key}.safetensors")).get("weight").astype(np.float32)[cols].T
                d = Wq - wq
                r["wq_rel_fro"] = float(np.linalg.norm(d) / np.linalg.norm(wq)); r["wq_max_abs"] = float(np.abs(d).max())
                r["wq_rel_proxy"] = rel_err(Wt, wq, H)[0] / den
            out.append(r)
            print(json.dumps(r), flush=True)
        del H
    return out


def verdict(rows, tol):
    """every tensor needs a comparison: the full-tensor ratio, or (sampled) the claim on the sampled blocks; every
    comparison present must be within tol"""
    bad, worst = [], 0.0
    for r in rows:
        rs = [r[k] for k in ("ratio", "claim_sel_ratio", "claim_log_ratio") if k in r]
        if r.get("claim_bad_shape") or not ("ratio" in r or "claim_sel_ratio" in r): bad.append(r["key"]); continue
        w = max(abs(x - 1) for x in rs); worst = max(worst, w)
        if w > tol: bad.append(r["key"])
    return not bad, bad, worst


def load_claims(path):
    if not os.path.exists(path): return None
    z = np.load(path, allow_pickle=False)
    return {k[:-4]: (z[k], z[k[:-4] + ".ref"]) for k in z.files if k.endswith(".err")}


def check_job(jtype, p, res_dir, data_path, rows_frac=1.0, tol=None, seed=None):
    """the server-side check of one accepted job (admin.py check); returns a report dict"""
    import jobs
    from jobs import quant_common as Q
    tol = TOL if tol is None else tol
    m = jobs.get(jtype)
    plan = m.plan(p)
    shard_name = f"q-L{p['layer']:02d}.safetensors" if jtype == "quant_layer" else "q-head.safetensors"
    wfile = m.wfile(p) if jtype == "quant_layer" else "head.safetensors"
    rep = {"pass": False, "summary": ""}
    try:
        Q.check_shard(p, plan, os.path.join(res_dir, shard_name), data_path(f"{Q.root(p)}/weights/{wfile}"))
        rows, meta = Q.check_log(p, plan, os.path.join(res_dir, "proxy.jsonl"), os.path.join(res_dir, "job.json"))
    except Exception as e:
        rep["summary"] = f"structure: {e}"; return rep
    wts = ST(data_path(f"{Q.root(p)}/weights/{wfile}"))
    t0 = time.time()
    res = check_tensors(plan, ST(os.path.join(res_dir, shard_name)), lambda k: wts.get(f"{k}.weight"),
                        lambda h: data_path(f"{Q.root(p)}/hessians/{h}.safetensors"), p["K"], p["codebook"],
                        {r["key"]: r["proxy_err"] for r in rows}, rows_frac, seed,
                        claims=load_claims(os.path.join(res_dir, "rowerr.npz")))
    ok, bad, worst = verdict(res, tol)
    rep.update({"pass": ok, "bad": bad, "worst": worst, "tensors": res, "ourq_stamp": meta.get("ourq_stamp"),
                "seconds": round(time.time() - t0, 1),
                "summary": f"{len(res)} tensors, worst |rel/logged-1| {worst:.2e} (tol {tol:.0e}), {time.time() - t0:.0f}s"})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="mode", required=True)
    x = sp.add_parser("job"); x.add_argument("--type", default="quant_layer"); x.add_argument("--params", required=True)
    x.add_argument("--dir", required=True); x.add_argument("--data", required=True)
    x = sp.add_parser("ourq"); x.add_argument("--dir", required=True); x.add_argument("--layer", type=int, required=True)
    x.add_argument("--K", type=float, required=True); x.add_argument("--codebook", required=True)
    x.add_argument("--ckpt", required=True); x.add_argument("--hess", required=True); x.add_argument("--wq", action="store_true")
    x.add_argument("--keys", help="comma list: only these tensors (substring match)")
    x.add_argument("--claims", help="a rowerr.npz (jobs/rowerr_tool.py) to check against (default DIR/rowerr.npz)")
    for x in sp.choices.values():
        x.add_argument("--rows-frac", type=float, default=1.0); x.add_argument("--tol", type=float, default=TOL)
        x.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.mode == "job":
        rep = check_job(a.type, json.load(open(a.params)), a.dir, lambda rel: os.path.join(a.data, rel), a.rows_frac, a.tol, a.seed)
        print(("PASS " if rep["pass"] else "FAIL ") + rep["summary"]); sys.exit(0 if rep["pass"] else 1)
    from jobs import qwen
    K = int(a.K) if a.K.is_integer() else a.K
    plan = qwen.layer_plan("qwen3_5", a.layer)
    if a.keys: plan = [(h, [k for k in ks if any(s in k for s in a.keys.split(","))]) for h, ks in plan]
    idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    files = {}
    def gw(k):
        f = idx[f"{k}.weight"]
        if f not in files: files[f] = ST(os.path.join(a.ckpt, f))
        return files[f].get(f"{k}.weight")
    logged = {}
    for l in open(os.path.join(a.dir, "proxy.jsonl")):
        r = json.loads(l); logged[r["key"]] = r["proxy_err"]                  # the last entry of a key wins
    t0 = time.time()
    claims = load_claims(a.claims or os.path.join(a.dir, "rowerr.npz"))   # (not ourq's wq/*.rowerr.npz: see jobs/rowerr_tool.py)
    res = check_tensors(plan, ST(os.path.join(a.dir, f"q-L{a.layer:02d}.safetensors")), gw,
                        lambda h: os.path.join(a.hess, f"{h}.safetensors"), K, a.codebook, logged, a.rows_frac, a.seed,
                        os.path.join(a.dir, "wq") if a.wq else None, claims)
    ok, bad, worst = verdict(res, a.tol)
    print(f"{'PASS' if ok else 'FAIL'} {len(res)} tensors, worst |rel/logged-1| {worst:.2e} (tol {a.tol:.0e}), {time.time() - t0:.1f}s"
          + (f", bad {bad}" if bad else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
