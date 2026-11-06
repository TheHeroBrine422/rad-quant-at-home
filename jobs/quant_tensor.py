# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""quant_tensor (0.3): one Hessian group of a trunk layer -- the linears that share one input Hessian, quantized together
(GDN layers: in_proj_qkv + in_proj_z | out_proj | gate_proj + up_proj | down_proj; full-attention layers: q / k / v |
o_proj | gate + up | down) -- with quantize/ourq.py --layers L-L --only <the group's keys>. ourq's seeds go by position in
the layer and the first tensor of a group prepares the shared Hessian, so a whole group gives exactly the codes of
the full-layer run (ourq.py --only); a single tensor of a group would not, which is why the unit is the group.

The small-card job (TESTS.md section 10, 27B on an RTX 4070 capped at 7.2 GB): groups 0-2 need <= 4.2 GB of VRAM and
~2 GB host RAM; group 3 (down_proj) needs > 7.2 GB for its GPU scale refit, so 8 GB cards run that refit on the CPU
(~10 GB host RAM, +~50 s). Inputs: the group's weights (weights/LNN-gG.safetensors, cut by admin.py datapack
--groups) and the group's Hessian. Outputs as quant_layer, the
shard named q-LNN-gG.safetensors (admin.py export merges a layer's groups into q-LNN.safetensors)."""
from . import qwen, quant_common as Q

NAME, VERSION, IMPLEMENTED = "quant_tensor", 1, True
SCHEMA = Q.schema(with_layer=True)
SCHEMA["properties"]["group"] = {"type": "integer", "minimum": 0, "maximum": 7}
SCHEMA["required"] = SCHEMA["required"] + ["group"]
TOL, compare = Q.TOL, Q.compare
GROUP_PARAMS_MAX = 210_000_000                         # gate + up of the 27B: 2 x 17408 x 5120


def check_extra(p):
    Q.check_extra(p)
    if p["group"] >= len(qwen.layer_plan(p["arch"], p["layer"])): raise ValueError(f"layer {p['layer']} has no group {p['group']}")


def requires(p, info=None):
    if not Q.big(info): return {"gpu": True, "cards": 1, "ram_gb": 4, "vram_gb": 4}
    down = qwen.keys(plan(p))[-1].endswith("down_proj")
    return {"gpu": True, "cards": 1, "ram_gb": 12 if down else 4, "vram_gb": Q.down_vram_gb(p) if down else 6}


def lumens(p):                                           # a 27B layer is 3; its down_proj group is ~half the work
    return 1.5 if qwen.keys(plan(p))[-1].endswith("down_proj") else 0.5


def wfile(p): return f"L{p['layer']:02d}-g{p['group']}.safetensors"      # admin.py datapack --groups
def shard(p): return f"q-L{p['layer']:02d}-g{p['group']}.safetensors"
def plan(p): return [qwen.layer_plan(p["arch"], p["layer"])[p["group"]]]
def inputs(p): return Q.inputs_for(p, plan(p), wfile(p))
def outputs(p): return {shard(p): Q.shard_cap(p, GROUP_PARAMS_MAX), **Q.SMALL}
def required_outputs(p): return Q.required(p, shard(p))


def run(p, ctx):
    L = p["layer"]
    import os
    only = os.path.join(ctx.stage_dir, "only.txt")
    return Q.run_ourq(p, ctx, plan(p), wfile(p), ["--layers", f"{L}-{L}", "--only", only], shard(p),
                      ourq_shard=f"q-L{L:02d}.safetensors")


def accept(p, files, result, data_path):
    Q.check_shard(p, plan(p), files[shard(p)], data_path(f"{Q.root(p)}/weights/{wfile(p)}"))
    Q.check_log(p, plan(p), files["proxy.jsonl"], files["job.json"])
