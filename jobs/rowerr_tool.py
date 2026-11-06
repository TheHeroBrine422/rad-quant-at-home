#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rowerr_tool.py CKPT HESS WQ OUT.npz DEVICE KEY:HESSNAME ... -- the per-output-row claim a quant job uploads:
err[r] = (w - q)_r^T H (w - q)_r and ref[r] = w_r^T H w_r (H = the raw input Hessian, w = the bf16 weight, q = ourq
--save-wq's dequantized weight), saved as <key>.err / <key>.ref (float64). The checker compares it with its own decode
on random output blocks.

Computed here rather than taken from ourq's own wq/<key>.rowerr.npz, which (until proxy2b/exp/w04-blockgain/
rowerr.patch is merged) is computed on the weight quantize_exl3 leaves regularized in place: err ~ ref.
GPU (torch) with a CPU fallback on OOM."""
import json, os, struct, sys
import numpy as np, torch
from safetensors import safe_open

ck, hs, wq, out, dev = sys.argv[1:6]
dev = torch.device("cuda", int(dev)) if torch.cuda.is_available() else torch.device("cpu")
idx = json.load(open(os.path.join(ck, "model.safetensors.index.json")))["weight_map"]


def hessian(name, d):
    p = os.path.join(hs, f"{name}.safetensors")
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; k = int(json.loads(fh.read(n))["__metadata__"]["k"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper").to(d)
    H = torch.zeros(k, k, dtype=torch.float32, device=d)
    iu = torch.triu_indices(k, k, device=d)
    H[iu[0], iu[1]] = up; del up, iu
    return H + H.triu(1).T


def rows(w, q, H):
    err, ref = [], []
    for c in range(0, w.shape[1], 4096):
        W = w[:, c:c + 4096]; E = W - q[:, c:c + 4096]
        err.append(((H @ E) * E).sum(0).double().cpu()); ref.append(((H @ W) * W).sum(0).double().cpu())
    return torch.cat(err).numpy(), torch.cat(ref).numpy()


res, H, Hn = {}, None, None
for kh in sys.argv[6:]:
    key, hname = kh.split(":")
    with safe_open(os.path.join(ck, idx[f"{key}.weight"]), "pt", device="cpu") as f: w = f.get_tensor(f"{key}.weight")
    with safe_open(os.path.join(wq, f"{key}.safetensors"), "pt", device="cpu") as f: q = f.get_tensor("weight")
    try:
        if Hn != hname: H = None; torch.cuda.empty_cache(); H = hessian(hname, dev); Hn = hname
        e, r = rows(w.to(H.device).float().T, q.to(H.device).float().T, H)
    except torch.OutOfMemoryError:
        H = None; torch.cuda.empty_cache(); H = hessian(hname, "cpu"); Hn = hname
        e, r = rows(w.float().T, q.float().T, H)
    res[f"{key}.err"], res[f"{key}.ref"] = e, r
    print(f"{key}: err/ref {e.sum() / r.sum():.4e}", flush=True)
np.savez(out + ".part.npz", **res); os.replace(out + ".part.npz", out)
