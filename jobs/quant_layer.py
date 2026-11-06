# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""quant_layer: one trunk layer of a ladder level with quantize/ourq.py --layers L-L.

Inputs: that layer's bf16 linears (weights/LNN.safetensors, cut from the checkpoint by server/admin.py datapack),
its four input-Hessian files, the model's config.json (ourq's template) and, with settings.hout, that snapshot's
H_out files for the layer. Outputs: q-LNN.safetensors (ourq's shard, unchanged), proxy.jsonl (ourq's per-tensor log),
stamp.json (ourq's code stamp record), job.json (settings stamp, ourq stamp, image, GPU, time), ourq.log; with
"verify": "replicate" (the default) also rowerr.npz, the per-row claim the checker samples.
Requirements (27B, measured on an RTX 4070 capped at 7.2 GB, TESTS.md section 11): every tensor but down_proj peaks at
<= 4.2 GB of VRAM and ~2 GB host RAM; down_proj's scale refit needs > 7.2 GB, so on an 8 GB card it falls back to the
CPU (+~50 s, ~10 GB host RAM), and at K6 its trellis search needs a small cuBLAS workspace (set by the job on cards
below 12 GB). A layer job: 8 GB VRAM up to K6 (measured; 12 GB above), 12 GB RAM."""
import os
from . import qwen, quant_common as Q

NAME, VERSION, IMPLEMENTED = "quant_layer", 3, True
SCHEMA = Q.schema(with_layer=True)
check_extra = Q.check_extra
# a 27B layer: ~0.4 G parameters; 32 GB cards for the 27B (notes); the 0.8B runs in far less
LAYER_PARAMS_MAX = 420_000_000
TOL, compare = Q.TOL, Q.compare


def requires(p, info=None):
    return {"gpu": True, "cards": 1, "ram_gb": 12, "vram_gb": Q.down_vram_gb(p)} if Q.big(info) else {"gpu": True, "cards": 1, "ram_gb": 8, "vram_gb": 4}


def lumens(p): return 3.0                               # ~2.5 min of an R9700 per 27B layer (proxies count the same)
def wfile(p): return f"L{p['layer']:02d}.safetensors"
def shard(p): return f"q-L{p['layer']:02d}.safetensors"
def plan(p): return qwen.layer_plan(p["arch"], p["layer"])
def inputs(p): return Q.inputs_for(p, plan(p), wfile(p))
def outputs(p): return {shard(p): Q.shard_cap(p, LAYER_PARAMS_MAX), **Q.SMALL}
def required_outputs(p): return Q.required(p, shard(p))
def upload_bytes(p, sizes): return Q.upload_bytes(p, sizes, wfile(p), LAYER_PARAMS_MAX)


def run(p, ctx):
    L = p["layer"]
    return Q.run_ourq(p, ctx, plan(p), wfile(p), ["--layers", f"{L}-{L}"], shard(p))


def accept(p, files, result, data_path):
    Q.check_shard(p, plan(p), files[shard(p)], data_path(f"{Q.root(p)}/weights/{wfile(p)}"))
    Q.check_log(p, plan(p), files["proxy.jsonl"], files["job.json"])
