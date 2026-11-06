#!/usr/bin/env python3
"""hfrange.py REPO OUTDIR KEY... -- fetch single tensors of a HF safetensors checkpoint by HTTP range (no full
download): reads model.safetensors.index.json, each shard's header (2 small range requests), then each tensor's bytes.
Writes OUTDIR/part.safetensors (all requested tensors), model.safetensors.index.json (pointing at it) and config.json."""
import json, os, struct, sys, urllib.request

repo, out, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
base = f"https://huggingface.co/{repo}/resolve/main/"
UA = {"User-Agent": "hfrange/0.1"}


def get(name, a=None, b=None):
    h = dict(UA)
    if a is not None: h["Range"] = f"bytes={a}-{b}"
    with urllib.request.urlopen(urllib.request.Request(base + name, headers=h), timeout=120) as r: return r.read()


os.makedirs(out, exist_ok=True)
idx = json.loads(get("model.safetensors.index.json"))["weight_map"]
open(os.path.join(out, "config.json"), "wb").write(get("config.json"))
hdrs, tens = {}, {}
for k in keys:
    f = idx[k]
    if f not in hdrs:
        n = struct.unpack("<Q", get(f, 0, 7))[0]
        hdrs[f] = (json.loads(get(f, 8, 8 + n - 1)), 8 + n)
    h, off = hdrs[f]; e = h[k]; a, b = e["data_offsets"]
    data = get(f, off + a, off + b - 1)
    assert len(data) == b - a, (k, len(data), b - a)
    tens[k] = (e["dtype"], e["shape"], data)
    print(f"{k} {e['dtype']} {e['shape']} {len(data) / 2**20:.1f} MiB", flush=True)
hdr, pos = {}, 0
for k, (dt, sh, d) in tens.items(): hdr[k] = {"dtype": dt, "shape": sh, "data_offsets": [pos, pos + len(d)]}; pos += len(d)
hb = json.dumps(hdr).encode(); hb += b" " * (-len(hb) % 8)
p = os.path.join(out, "part.safetensors")
with open(p + ".part", "wb") as fh:
    fh.write(struct.pack("<Q", len(hb))); fh.write(hb)
    for k, (_, _, d) in tens.items(): fh.write(d)
os.replace(p + ".part", p)
json.dump({"metadata": {}, "weight_map": {k: "part.safetensors" for k in tens}}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print("done", p)
