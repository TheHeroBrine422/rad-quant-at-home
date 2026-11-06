# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""quant_head: the LM head of a ladder level with quantize/ourq.py --only-head (~22 GB host RAM peak at the 27B: the
server leases it only to workers that declared ram_gb >= 48, and only while that much of it is free). Inputs: weights/head.safetensors (lm_head.weight), the
lm_head.input Hessian, config.json. Outputs as quant_layer with q-head.safetensors."""
from . import qwen, quant_common as Q

NAME, VERSION, IMPLEMENTED = "quant_head", 3, True
SCHEMA = Q.schema(with_layer=False)
check_extra = Q.check_extra
HEAD_PARAMS_MAX = 1_300_000_000          # 248k x 5120


TOL, compare = Q.TOL, Q.compare


def requires(p, info=None):
    return {"gpu": True, "cards": 1, "ram_gb": 48, "vram_gb": 32} if Q.big(info) else {"gpu": True, "cards": 1, "ram_gb": 16, "vram_gb": 8}


def lumens(p): return 15.0                              # ~10-20 min per 27B head
def plan(p): return qwen.head_plan(p["arch"])
def inputs(p): return Q.inputs_for(p, plan(p), "head.safetensors")
def outputs(p): return {"q-head.safetensors": Q.shard_cap(p, HEAD_PARAMS_MAX), **Q.SMALL}
def shard(p): return "q-head.safetensors"
def wfile(p): return "head.safetensors"
def required_outputs(p): return Q.required(p, shard(p))
def upload_bytes(p, sizes): return Q.upload_bytes(p, sizes, wfile(p), HEAD_PARAMS_MAX)


def run(p, ctx):
    return Q.run_ourq(p, ctx, plan(p), "head.safetensors", ["--only-head"], "q-head.safetensors")


def accept(p, files, result, data_path):
    Q.check_shard(p, plan(p), files["q-head.safetensors"], data_path(f"{Q.root(p)}/weights/head.safetensors"))
    Q.check_log(p, plan(p), files["proxy.jsonl"], files["job.json"])
