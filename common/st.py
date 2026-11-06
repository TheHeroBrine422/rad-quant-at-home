# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""safetensors without torch: copy named tensors byte for byte from checkpoint shards into one new file; merge quant
shards (a layer's quant_tensor groups -> one q-LNN.safetensors) or take a subset of one."""
import json, os, struct
from .util import st_header


def copy_tensors(ckpt_dir, weight_map, names, out, metadata=None):
    """names (checkpoint keys) -> one safetensors file at out (.part, then renamed); bytes copied unchanged"""
    src, hdrs = {}, {}
    for n in names:
        f = weight_map[n]
        if f not in hdrs: hdrs[f] = st_header(os.path.join(ckpt_dir, f))
        src[n] = (f, hdrs[f][n])
    hdr, off = {}, 0
    if metadata: hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for n in names:                                      # keep the caller's order
        e = src[n][1]; ln = e["data_offsets"][1] - e["data_offsets"][0]
        hdr[n] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + ln]}; off += ln
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for n in names:
            f, e = src[n]
            base = 8 + struct.unpack("<Q", open(os.path.join(ckpt_dir, f), "rb").read(8))[0]
            a, b = e["data_offsets"]
            with open(os.path.join(ckpt_dir, f), "rb") as fi:
                fi.seek(base + a); left = b - a
                while left:
                    chunk = fi.read(min(left, 1 << 24)); fo.write(chunk); left -= len(chunk)
    os.replace(out + ".part", out)


def _write(out, items, metadata):
    """items: [(name, dtype, shape, path, abs offset, length)] -> out (.part, renamed)"""
    hdr, off = {}, 0
    if metadata: hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for n, dt, sh, _, _, ln in items: hdr[n] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + ln]}; off += ln
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for _, _, _, path, a, ln in items:
            with open(path, "rb") as fi:
                fi.seek(a); left = ln
                while left:
                    chunk = fi.read(min(left, 1 << 24)); fo.write(chunk); left -= len(chunk)
    os.replace(out + ".part", out)


def _items(path, names=None):
    h = st_header(path); base = 8 + struct.unpack("<Q", open(path, "rb").read(8))[0]
    keys = [k for k in h if k != "__metadata__"] if names is None else names
    return [(k, h[k]["dtype"], h[k]["shape"], path, base + h[k]["data_offsets"][0],
             h[k]["data_offsets"][1] - h[k]["data_offsets"][0]) for k in keys], h.get("__metadata__") or {}


def merge_shards(paths, out):
    """several quant shards (disjoint tensors) -> one; metadata: the first's, with every part's ourq_stamp listed"""
    items, meta, stamps = [], {}, []
    for p in paths:
        it, m = _items(p); items += it; meta = meta or dict(m)
        if m.get("ourq_stamp") and m["ourq_stamp"] not in stamps: stamps.append(m["ourq_stamp"])
    names = [i[0] for i in items]
    if len(set(names)) != len(names): raise ValueError("merge_shards: a tensor in two parts")
    if len(stamps) > 1: meta["ourq_stamps"] = ",".join(stamps)
    _write(out, sorted(items, key=lambda i: i[0]), meta)


def subset_shard(path, names, out):
    items, meta = _items(path, names)
    _write(out, items, meta)
