#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""datapack_check.py DATA MODEL@REV [--hin-sha F] [--hout-sha F] [--hout-snap S] [--ckpt DIR [--deep]] [--settings STAMP]
-- check a model's datapack for the V3 ladder (0.6; stdlib only) before jobs are generated from its manifest.

  manifest     DATA/manifest.json lists every file under models/MODEL@REV with its current size and mtime (a file
               changed after `admin.py manifest` would be served with a stale hash; run `manifest` again)
  hessians     hessians/*.safetensors in the manifest = the names of --hin-sha (sha256sum format, e.g. an integrity
               scan's list), every sha256 equal
  hout         hout/<snap>/*.safetensors = the names of --hout-sha, every sha256 equal
  weights      every trunk layer's group files L<NN>-g<g>.safetensors (admin.py datapack --groups) are in the manifest
               and hold exactly the group's keys (jobs/qwen.py layer_plan) with the checkpoint's dtype and shape (--ckpt);
               --deep: every tensor's bytes hash equal to the checkpoint's (reads both, ~2x the weights)
  inputs       every group's Hessian and every trunk key's H_out file exist in the manifest
  settings     models/MODEL@REV/rxq/*.toml are listed (--settings STAMP: that one must be there)
Last line: "DATAPACK OK ..." or "DATAPACK FAIL: ..."; exit 0 / 1."""
import argparse, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common.util import st_header   # noqa: E402
from jobs import qwen                # noqa: E402


def shalist(path):
    """sha256sum output -> {basename: sha256}"""
    out = {}
    for l in open(path):
        if l.strip():
            h, n = l.split(None, 1); out[os.path.basename(n.strip().lstrip("*"))] = h
    return out


def tensor_sha(path, hdr, name):
    """sha256 of one tensor's bytes in a safetensors file (hdr = its st_header)"""
    a, b = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        base = 8 + int.from_bytes(f.read(8), "little"); f.seek(base + a); left = b - a; h = hashlib.sha256()
        while left:
            c = f.read(min(left, 1 << 24)); h.update(c); left -= len(c)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data"); ap.add_argument("model_rev")
    ap.add_argument("--hin-sha"); ap.add_argument("--hout-sha"); ap.add_argument("--hout-snap", default="bn1920")
    ap.add_argument("--ckpt"); ap.add_argument("--deep", action="store_true"); ap.add_argument("--settings")
    ap.add_argument("--arch", default="qwen3_5"); ap.add_argument("--layers", type=int, help="default: the datapack's config.json")
    a = ap.parse_args(argv)
    root = f"models/{a.model_rev}"; R = os.path.join(a.data, root)
    bad, info = [], []
    man = json.load(open(os.path.join(a.data, "manifest.json")))["files"]
    mine = {k: v for k, v in man.items() if k.startswith(root + "/")}

    # manifest: complete and current
    on_disk = []
    for dp, dn, fn in os.walk(R, followlinks=True):
        dn[:] = [d for d in dn if not d.startswith(".")]
        on_disk += [os.path.relpath(os.path.join(dp, f), a.data) for f in fn if not f.startswith(".") and not f.endswith(".part")]
    miss = sorted(set(on_disk) - set(mine)); gone = sorted(set(mine) - set(on_disk))
    if miss: bad.append(f"manifest: {len(miss)} files not listed, e.g. {miss[:2]} (run `admin.py manifest`)")
    if gone: bad.append(f"manifest: {len(gone)} listed files are gone, e.g. {gone[:2]}")
    stale = [k for k in set(on_disk) & set(mine)
             if (lambda st: st.st_size != mine[k]["size"] or st.st_mtime != mine[k].get("mtime"))(os.stat(os.path.join(a.data, k)))]
    if stale: bad.append(f"manifest: {len(stale)} files changed since it was built, e.g. {sorted(stale)[:2]} (run `admin.py manifest`)")
    info.append(f"{len(mine)} files, {sum(v['size'] for v in mine.values()) / 1e9:.1f} GB")

    def cmp(sub, ref_path, what):
        got = {k[len(sub) + 1:]: v["sha256"] for k, v in mine.items() if k.startswith(sub + "/") and k.endswith(".safetensors")
               and "/" not in k[len(sub) + 1:]}
        ref = shalist(ref_path)
        if set(got) != set(ref):
            bad.append(f"{what}: names differ from {os.path.basename(ref_path)}: only in the datapack {sorted(set(got) - set(ref))[:3]}, "
                       f"only in the list {sorted(set(ref) - set(got))[:3]}")
        diff = sorted(n for n in set(got) & set(ref) if got[n] != ref[n])
        if diff: bad.append(f"{what}: {len(diff)} sha256 differ from {os.path.basename(ref_path)}, e.g. {diff[:3]}")
        info.append(f"{what} {len(got)} files, {len(set(got) & set(ref)) - len(diff)} sha256 = {os.path.basename(ref_path)}")
        return got
    hin = cmp(f"{root}/hessians", a.hin_sha, "H_in") if a.hin_sha else None
    hout = cmp(f"{root}/hout/{a.hout_snap}", a.hout_sha, "H_out") if a.hout_sha else None

    # weights and the per-group inputs
    cfg = json.load(open(os.path.join(R, "config.json")))
    nl = a.layers or (cfg.get("text_config") or cfg)["num_hidden_layers"]
    wm = hdrs = None
    if a.ckpt:
        wm = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]; hdrs = {}
    ng = nt = deep = 0
    for l in range(nl):
        for g, (h, ks) in enumerate(qwen.layer_plan(a.arch, l)):
            rel = f"{root}/weights/L{l:02d}-g{g}.safetensors"; ng += 1
            if rel not in mine: bad.append(f"weights: {rel} missing"); continue
            if f"{root}/hessians/{h}.safetensors" not in mine: bad.append(f"inputs: L{l} g{g}: Hessian {h} missing")
            if hout is not None:
                for k in ks:
                    if f"{k}.safetensors" not in hout: bad.append(f"inputs: H_out of {k} missing")
            p = os.path.join(a.data, rel); hd = st_header(p)
            names = [n for n in hd if n != "__metadata__"]
            if sorted(names) != sorted(f"{k}.weight" for k in ks): bad.append(f"weights: {rel} holds {names[:3]}, not group {g}'s keys"); continue
            nt += len(names)
            if wm is not None:
                for n in names:
                    f = wm[n]
                    if f not in hdrs: hdrs[f] = st_header(os.path.join(a.ckpt, f))
                    e = hdrs[f][n]
                    if (hd[n]["dtype"], hd[n]["shape"]) != (e["dtype"], e["shape"]):
                        bad.append(f"weights: {n} {hd[n]['dtype']} {hd[n]['shape']} != checkpoint {e['dtype']} {e['shape']}")
                    elif a.deep:
                        if tensor_sha(p, hd, n) != tensor_sha(os.path.join(a.ckpt, f), hdrs[f], n): bad.append(f"weights: {n} bytes differ")
                        else: deep += 1
    info.append(f"weights {ng} group files, {nt} tensors" + (" = checkpoint dtype / shape" if wm is not None else "")
                + (f", {deep} byte-identical" if a.deep else ""))
    st = sorted(k.rsplit("/", 1)[1][:-5] for k in mine if re.fullmatch(re.escape(root) + r"/rxq/[0-9a-f]{16}\.toml", k))
    info.append(f"settings {st}")
    if a.settings and a.settings not in st: bad.append(f"settings: {a.settings}.toml not in {root}/rxq (admin.py rxq-settings, then manifest)")
    mj = os.path.join(R, "model.json")
    if os.path.exists(mj): info.append(f"n_params {json.load(open(mj))['n_params']:,}")
    else: bad.append("model.json missing")
    for x in info: print(x)
    for b in bad[:30]: print("FAIL", b)
    if len(bad) > 30: print(f"... {len(bad) - 30} more")
    print(f"DATAPACK {'OK' if not bad else 'FAIL'}: {a.model_rev}, " + ("; ".join(info[:3]) if not bad else f"{len(bad)} problem(s)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
