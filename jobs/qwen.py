# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Model layouts the quantization jobs know (one function per architecture; the same plan as quantize/ourq.py)."""
L = "model.language_model.layers"


def layer_plan(arch, l):
    """[(input-Hessian name, [linear keys sharing it])] of layer l, in ourq.py's order"""
    if arch != "qwen3_5": raise ValueError(f"unknown arch {arch!r}")
    p = f"{L}.{l}"
    full = l % 4 == 3
    return [(f"{p}.input", [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"] if full else
                           [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"]),
            (f"{p}.self_attn.o_proj.input" if full else f"{p}.linear_attn.out_proj.input",
             [f"{p}.self_attn.o_proj" if full else f"{p}.linear_attn.out_proj"]),
            (f"{p}.post_attention.input", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]),
            (f"{p}.mlp.down_proj.input", [f"{p}.mlp.down_proj"])]


def head_plan(arch):
    if arch != "qwen3_5": raise ValueError(f"unknown arch {arch!r}")
    return [("lm_head.input", ["lm_head"])]


def keys(plan):
    return [k for _, ks in plan for k in ks]
