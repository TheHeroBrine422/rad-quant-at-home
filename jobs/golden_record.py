# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""golden_record (big-box tier, trusted donors only): the bf16 27B reference rows for one chunk of a public corpus, in
radiance's --kld-record --kld-top layout (kld.json, tokens.i32, top.bin, rows.f32, gen.u8 if the recorder writes it).

Two recorders, the job's choice:
  tp2      radiance --model models/<m>@<rev>/bf16.rad --tp 2 --kld-record --kld-corpus CHUNK --kld-top N: a gang
           job on a worker with 2 free 32 GB cards
  staged   eval/gold27/mkgold27.py (one card, the model in N stages of layers, bf16 scratch on the donor's disk) from
           the HF checkpoint (models/<m>@<rev>/hf/)
The corpus chunk (corpora/<corpus>/<chunk>.jsonl) and its marker (corpus.json) must be public (golden_common).

Server-side check at completion (golden_common.check_golden): the layout's sizes, the documents and sources and
radiance's corpus hash against our chunk, and on ~256 random rows: the head sorted and finite, head + tail = 1,
the true next token's log p = -nll. Spot checks re-record the chunk on a trusted worker: identical tokens, per-row
nll within TOL, the same argmax on >= 98% of sampled rows (another kernel path moves the nll by ~1e-2: gold27)."""
import json, os, re, shutil, time
from common.util import write_json_atomic
from . import golden_common as G
from .quant_common import ID

NAME, VERSION, IMPLEMENTED, TRUSTED_ONLY = "golden_record", 1, True, True
SCHEMA = {"type": "object", "required": ["model", "revision", "corpus", "chunk", "method", "top", "max_model_len",
                                         "max_positions"], "properties": {
    "model": {"type": "string", "pattern": ID}, "revision": {"type": "string", "pattern": ID},
    "corpus": {"type": "string", "pattern": ID}, "chunk": {"type": "string", "pattern": ID},
    "method": {"type": "string", "enum": ["tp2", "staged"]},
    "stages": {"type": "integer", "minimum": 1, "maximum": 32, "description": "staged: layer groups (3 on a 32 GB card)"},
    "top": {"type": "integer", "minimum": 16, "maximum": 4096},
    "max_model_len": {"type": "integer", "minimum": 256, "maximum": 262144},
    "max_positions": {"type": "integer", "minimum": 1, "maximum": 20_000_000}}}
TOL = {"nll_mae": 0.02, "top1_agree": 0.98}
HF_FILE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"


def requires(p, info=None):
    if p["method"] == "tp2": return {"gpu": True, "cards": 2, "vram_gb": 32, "ram_gb": 64}
    return {"gpu": True, "cards": 1, "vram_gb": 32, "ram_gb": 48}


def lumens(p): return round(p["max_positions"] / 50_000 * (2 if p["method"] == "tp2" else 1), 3)  # ~20 card-min per 1M


def root(p): return f"models/{p['model']}@{p['revision']}"
def chunk(p): return f"corpora/{p['corpus']}/{p['chunk']}.jsonl"


def inputs(p):
    c = [chunk(p), f"corpora/{p['corpus']}/corpus.json"]
    if p["method"] == "tp2": return [f"{root(p)}/bf16.rad"] + c, []
    return [f"{root(p)}/hf/config.json", f"{root(p)}/hf/model.safetensors.index.json"] + c, [re.escape(f"{root(p)}/hf/") + HF_FILE]


def outputs(p):
    n, T = p["max_positions"], p["top"]
    return {"kld.json": 16 << 20, "tokens.i32": 4 * n + (64 << 20), "gen.u8": n, "top.bin": 8 * T * n, "rows.f32": 12 * n,
            "job.json": 64 << 10}


def required_outputs(p): return {"kld.json", "tokens.i32", "top.bin", "rows.f32", "job.json"}


def run(p, ctx):
    t0 = time.time()
    ok, why = G.marker_ok(json.load(open(ctx.input(f"corpora/{p['corpus']}/corpus.json"))))
    if not ok: raise RuntimeError(f"corpus {p['corpus']}: {why}: refusing it")
    s = ctx.stage_dir
    if os.path.exists(s): shutil.rmtree(s)
    os.makedirs(s)
    cp = os.path.join(s, "chunk.jsonl"); os.symlink(ctx.input(chunk(p)), cp)
    g = os.path.join(s, "golden")
    log = os.path.join(s, "recorder.log")
    if p["method"] == "tp2":
        argv = [ctx.cfg.radiance, "--model", ctx.input(f"{root(p)}/bf16.rad"), "--tp", "2", "--kv-cache-dtype", "bf16",
                "--max-model-len", str(p["max_model_len"]), "--max-num-seqs", "1", "--host-headroom-mib",
                str(ctx.cfg.headroom_mib or 16384), "--kld-record", g, "--kld-corpus", cp, "--kld-top", str(p["top"])]
    else:
        hf = os.path.join(s, "hf"); os.makedirs(hf)
        pre = f"{root(p)}/hf/"
        for n in ctx.input_names:
            if n.startswith(pre): os.symlink(ctx.input(n), os.path.join(hf, n[len(pre):]))
        idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))
        miss = sorted({f for f in idx["weight_map"].values() if not os.path.exists(os.path.join(hf, f))})
        if miss: raise RuntimeError(f"checkpoint shards missing from the job's inputs: {miss[:3]}")
        argv = [ctx.cfg.python, ctx.cfg.mkgold27, hf, cp, g, "--scratch", os.path.join(s, "scratch"), "--stages",
                str(p.get("stages", 3)), "--scratch-dtype", "bf16", "--model-name", f"{p['model']}@{p['revision']}"]
    rc = ctx.run_cmd(argv, log=log)
    if rc != 0:
        raise RuntimeError(f"recorder exit {rc}: {open(log, 'rb').read()[-1500:].decode(errors='replace')}")
    for f in G.GOLDEN_FILES + ("gen.u8",):
        if os.path.exists(os.path.join(g, f)): os.replace(os.path.join(g, f), os.path.join(ctx.out_dir, f))
    summ = G.check_golden({f: os.path.join(ctx.out_dir, f) for f in G.GOLDEN_FILES + ("gen.u8",) if os.path.exists(os.path.join(ctx.out_dir, f))},
                          top=p["top"], max_positions=p["max_positions"], corpus=cp, sample=64)
    shutil.rmtree(s, ignore_errors=True)
    meta = {"type": NAME, "version": VERSION, "method": p["method"], "image": ctx.cfg.image, "gpu": ctx.cfg.gpu_name,
            "devices": ctx.devices, "seconds": round(time.time() - t0, 1), "rows": summ["rows"], "docs": summ["docs"]}
    write_json_atomic(os.path.join(ctx.out_dir, "job.json"), meta)
    return {k: meta[k] for k in ("rows", "docs", "seconds")}


def accept(p, files, result, data_path):
    G.check_golden({f: files[f] for f in G.GOLDEN_FILES + ("gen.u8",) if f in files}, top=p["top"],
                   max_positions=p["max_positions"], corpus=data_path(chunk(p)))


def compare(p, a, b, tol=None):
    return G.compare_goldens(a, b, dict(TOL, **(tol or {})))
