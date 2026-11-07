# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eval_mix: score one candidate mix of the search against a public golden.

A mix is a base ladder level plus a list of (weight, level) pairs: the weights it names come from those levels, every
other weight from the base. The donor keeps the levels' pieces in its content-addressed input cache
(models/<m>@<rev>/rad/<level>/skeleton.radh + t/<weight>.radw, cut by admin.py ladder-add; common/radfile.py), so a
candidate downloads only the pieces it does not hold yet, assembles the container locally (radmix.cpp's layout, byte
for byte; reflinks on XFS / btrfs, else a copy), and scores it with radiance --kld-ref against the golden it also
caches (goldens/<g>/, marked public: jobs/golden_common.py). Outputs: rows.f32 (radiance's --kld-out FILE.rows:
[rows, 4] f32 {kl, nll_q, nll_p, top1}, reference row order), kld.json (radiance's report), job.json (the assembled
container's sha256 and size, image, GPU, timings).

Server-side check at completion (stdlib, < 1 s per 1M rows): the nll_p column equals the golden's own NLL column bit
for bit (right golden, every row scored, rows in order), every KL finite and >= 0 (to rounding), top1 in {0, 1}, and
the report's mean equal to the rows' mean, and the result's kld_mean / container sha256 equal to the files'. 0.5 (#4):
the mean is checked against the search's predicted KL for the mix (plausible(): far worse is refused, far better is
suspect); spot checks (server/verify.py) re-score a few random documents of the golden on a trusted worker (params
docs / sub_rows: the worker cuts a golden of just those documents, sub_golden) and compare position by position and
per document (compare_rows); a candidate the search adopts, or a suspect one, is re-run in full. The container sha256
must be identical (assembly is deterministic)."""
import json, math, os, re, time
from common import radfile as R
from common.util import write_json_atomic
from . import golden_common as G
from .quant_common import ID

NAME, VERSION, IMPLEMENTED = "eval_mix", 2, True     # v2 (0.5): docs / sub_rows (spot checks of a few documents)
PAIR = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string", "pattern": R.WEIGHT_RE.pattern}}
SCHEMA = {"type": "object", "required": ["model", "revision", "base", "mix", "golden", "kv", "max_model_len", "bytes",
                                         "n_params", "rows"], "properties": {
    "model": {"type": "string", "pattern": ID}, "revision": {"type": "string", "pattern": ID},
    "base": {"type": "string", "pattern": ID, "description": "the ladder level every weight not in mix comes from"},
    "mix": {"type": "array", "maxItems": 8192, "items": PAIR, "description": "[weight, level], sorted by weight"},
    "act": {"type": "array", "maxItems": 8192, "items": PAIR, "description": "[weight, f16|a8t|a8g<N>], sorted"},
    "golden": {"type": "string", "pattern": ID}, "kv": {"type": "string", "enum": ["bf16"]},
    "max_model_len": {"type": "integer", "minimum": 256, "maximum": 262144},
    "bytes": {"type": "integer", "minimum": 1, "maximum": 1 << 42, "description": "the assembled container's size"},
    "n_params": {"type": "integer", "minimum": 1, "maximum": 1 << 42, "description": "the model's parameter count (bpw)"},
    "rows": {"type": "integer", "minimum": 1, "maximum": 50_000_000, "description": "the golden's scored positions"},
    "docs": {"type": "array", "minItems": 1, "maxItems": 64, "description": "a spot check: score only these documents "
             "[index, row0, rows] of the golden (trusted workers only)", "items": {
                 "type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 0}}},
    "sub_rows": {"type": "integer", "minimum": 1, "maximum": 50_000_000, "description": "with docs: their rows"}}}
TOL = {"row_atol": 1e-3, "row_rtol": 0.05, "rows_frac": 0.995, "mean_rel": 2e-3, "top1_flip": 0.01, "doc_mean_rel": 5e-3}
# plausibility vs the search's prediction (#4): a mean KL above worse x pred + abs is refused at completion (an honest
# surprise of the knapsack stays far inside x3), one below better x pred is accepted but suspect (the search hopes to
# beat its prediction, by a few to ~20 %, not by half) and re-run in full on a trusted worker before it counts. A batch
# may narrow or widen the band only inside these limits
PLAUSIBLE = {"worse": 3.0, "better": 0.5, "abs": 1e-3}
PLAUSIBLE_LIMITS = {"worse": (1.5, 10.0), "better": (0.1, 0.9)}
HEADROOM_MIB = 8192


def check_extra(p):
    ws = [w for w, _ in p["mix"]]
    if ws != sorted(set(ws)) or len(ws) != len(set(ws)): raise ValueError("mix: weights must be sorted and unique")
    for w, lv in p["mix"]:
        if not re.fullmatch(ID, lv): raise ValueError(f"mix: bad level {lv!r}")
        if lv == p["base"]: raise ValueError(f"mix: {w} names the base level (leave it out)")
    acts = p.get("act", [])
    if [w for w, _ in acts] != sorted({w for w, _ in acts}): raise ValueError("act: weights must be sorted and unique")
    for w, m in acts:
        if not R.act_ok(m): raise ValueError(f"act: {w}: {m!r} is not f16, a8t or a8g<N>")
    if ("docs" in p) != ("sub_rows" in p): raise ValueError("docs and sub_rows go together")
    if "docs" in p:
        idx = [d[0] for d in p["docs"]]
        if idx != sorted(set(idx)): raise ValueError("docs: sorted and unique")
        if sum(d[2] for d in p["docs"]) != p["sub_rows"] or any(d[1] + d[2] > p["rows"] or d[2] < 1 for d in p["docs"]):
            raise ValueError("docs: rows do not add up")


def bpw(p): return p["bytes"] * 8 / p["n_params"]


def n_rows(p): return p.get("sub_rows", p["rows"])


def requires(p, info=None):
    """by size (user, 2026-11-05): up to 4.5 bpw fits a 16 GB card, up to 7 bpw a 32 GB card; above that TP2 (a gang
    job on a 2-card worker)"""
    b = bpw(p)
    if b <= 4.5: return {"gpu": True, "cards": 1, "vram_gb": 16, "ram_gb": 16}
    if b <= 7.0: return {"gpu": True, "cards": 1, "vram_gb": 32, "ram_gb": 24}
    return {"gpu": True, "cards": 2, "vram_gb": 32, "ram_gb": 32}


def lumens(p): return round(max(0.5, n_rows(p) / 40_000) * requires(p)["cards"], 3)   # ~2.5 min per 100k positions


def root(p): return f"models/{p['model']}@{p['revision']}/rad"
def piece(p, level, w): return f"{root(p)}/{level}/t/{w}.radw"
def skeleton(p): return f"{root(p)}/{p['base']}/skeleton.radh"
def gfile(p, f): return f"goldens/{p['golden']}/{f}"


def inputs(p):
    req = [skeleton(p)] + [piece(p, lv, w) for w, lv in p["mix"]] + [gfile(p, f) for f in G.GOLDEN_FILES + ("golden.json",)]
    opt = [re.escape(f"{root(p)}/{p['base']}/t/") + R.WEIGHT_RE.pattern + r"\.radw", re.escape(gfile(p, "gen.u8"))]
    return req, opt


def outputs(p): return {"rows.f32": 16 * n_rows(p), "kld.json": 4 << 20, "job.json": 64 << 10}
def required_outputs(p): return {"rows.f32", "kld.json", "job.json"}
def upload_bytes(p, sizes): return 16 * n_rows(p) + (256 << 10)


def run(p, ctx):
    t0 = time.time()
    ok, why = G.marker_ok(json.load(open(ctx.input(gfile(p, "golden.json")))))
    if not ok: raise RuntimeError(f"golden {p['golden']}: {why}: refusing it")
    gm = json.load(open(ctx.input(gfile(p, "kld.json"))))
    if gm.get("rows") != p["rows"]: raise RuntimeError(f"golden has {gm.get('rows')} rows, the job says {p['rows']}")
    # 1. the container, from the cached pieces
    sk = R.Rad(ctx.input(skeleton(p)), skeleton=True)
    mix = dict(p["mix"])
    paths = {}
    for e in sk.entries:
        w = e["name_s"]; rel = piece(p, mix.get(w, p["base"]), w)
        if rel not in ctx.input_names: raise RuntimeError(f"no piece for {w} ({rel})")
        paths[w] = ctx.input(rel)
    os.makedirs(ctx.stage_dir, exist_ok=True)
    model = os.path.join(ctx.stage_dir, "mix.rad")
    try:
        info = R.assemble(model, ctx.input(skeleton(p)), paths, dict(p.get("act", [])), clone=True)
        if info["bytes"] != p["bytes"]: raise RuntimeError(f"assembled {info['bytes']} bytes, the job says {p['bytes']}")
        t1 = time.time()
        # 2. the golden, staged under radiance's file names (a spot check: only its documents, cut out)
        gd = os.path.join(ctx.stage_dir, "golden"); os.makedirs(gd)
        if "docs" in p:
            if G.sub_golden({f: ctx.input(gfile(p, f)) for f in G.GOLDEN_FILES}, gd, p["docs"]) != p["sub_rows"]:
                raise RuntimeError("the documents' rows differ from the job's sub_rows")
        else:
            for f in G.GOLDEN_FILES + ("gen.u8",):
                if gfile(p, f) in ctx.input_names: os.symlink(ctx.input(gfile(p, f)), os.path.join(gd, f))
        # 3. radiance --kld-ref (argv from validated values only; no shell)
        rep = os.path.join(ctx.out_dir, "kld.json")
        argv = [ctx.cfg.radiance, "--model", model, "--tp", str(len(ctx.devices) or 1), "--kv-cache-dtype", p["kv"],
                "--max-model-len", str(p["max_model_len"]), "--max-num-seqs", "1", "--host-headroom-mib",
                str(ctx.cfg.headroom_mib or HEADROOM_MIB), "--kld-ref", gd, "--kld-out", rep]
        rc = ctx.run_cmd(argv, log=os.path.join(ctx.stage_dir, "radiance.log"))
        if rc != 0:
            tail = open(os.path.join(ctx.stage_dir, "radiance.log"), "rb").read()[-1500:].decode(errors="replace")
            raise RuntimeError(f"radiance exit {rc}: {tail}")
        os.replace(rep + ".rows", os.path.join(ctx.out_dir, "rows.f32"))
    finally:
        if os.path.exists(model): os.remove(model)
        if os.path.exists(model + ".part"): os.remove(model + ".part")
    r = json.load(open(rep))
    meta = {"type": NAME, "version": VERSION, "container_sha256": info["sha256"], "bytes": info["bytes"],
            "bpw": round(bpw(p), 4), "cloned": info["cloned"], "copied": info["copied"], "image": ctx.cfg.image,
            "gpu": ctx.cfg.gpu_name, "devices": ctx.devices, "assemble_s": round(t1 - t0, 2),
            "eval_s": round(time.time() - t1, 2), "kld_mean": r["all"]["kld"]["mean"]}
    write_json_atomic(os.path.join(ctx.out_dir, "job.json"), meta)
    return {k: meta[k] for k in ("container_sha256", "bytes", "kld_mean", "assemble_s", "eval_s")}


def accept(p, files, result, data_path):
    """the checks at completion; returns {"kld_mean"} (the rows' mean) for the plausibility check"""
    n = n_rows(p)
    if os.path.getsize(files["rows.f32"]) != 16 * n: raise ValueError("rows.f32: wrong size")
    A = G.f32(files["rows.f32"])
    gm = json.load(open(data_path(gfile(p, "kld.json"))))
    g = G.f32(data_path(gfile(p, "rows.f32")))
    w = 3 if "top" in gm else 2
    if len(g) != w * p["rows"]: raise ValueError("golden size differs from the job's rows")
    if "docs" in p:
        if not G.spans_ok(gm, p["docs"]): raise ValueError("docs do not match the golden")
        g = G.take_rows(g, p["docs"], w)
    if A[2::4] != g[0::w]: raise ValueError("nll_p column is not the golden's: another golden, missing or reordered rows")
    kl, nq, t1 = A[0::4], A[1::4], A[3::4]
    if not all(math.isfinite(x) and x >= -1e-4 for x in kl): raise ValueError("a KL is negative or not finite")
    if not all(math.isfinite(x) for x in nq): raise ValueError("an nll_q is not finite")
    if any(x not in (0.0, 1.0) for x in t1): raise ValueError("top1 not in {0, 1}")
    rep = json.load(open(files["kld.json"]))
    mean = math.fsum(kl) / n
    rm = rep["all"]["kld"]["mean"]
    close = lambda x: isinstance(x, (int, float)) and abs(x - mean) <= 1e-5 * max(abs(mean), 1e-6) + 1e-9
    if rep["all"]["positions"] != n or not close(rm):
        raise ValueError(f"report (mean {rm}, {rep['all']['positions']} positions) does not match the rows ({mean})")
    meta = json.load(open(files["job.json"]))
    if meta.get("bytes") != p["bytes"] or not re.fullmatch("[0-9a-f]{64}", str(meta.get("container_sha256"))):
        raise ValueError("job.json: container size / sha256 missing or wrong")
    # what the search reads (batch status): the result's own numbers must be the files' (#4)
    if not close(result.get("kld_mean")) or result.get("container_sha256") != meta["container_sha256"]:
        raise ValueError("result: kld_mean / container_sha256 differ from the rows / job.json")
    return {"kld_mean": mean}


def plausible(pred, info):
    """(verdict, detail) of a result's mean KL against the search's prediction for the mix (#4): reject (far worse:
    refused), suspect (far better: accepted, re-run in full on a trusted worker before it counts) or pass"""
    band = {k: pred.get(k, v) for k, v in PLAUSIBLE.items()}
    kld, want = info["kld_mean"], pred["kld"]
    det = {"pred_kld": want, "kld_mean": kld, "ratio": kld / want if want > 0 else None, "band": band}
    if kld > band["worse"] * want + band["abs"]: return "reject", det
    if kld < band["better"] * want: return "suspect", det
    return "pass", det


def band_of(spec_band):
    """a batch's band override, kept inside PLAUSIBLE_LIMITS"""
    out = {}
    for k, (lo, hi) in PLAUSIBLE_LIMITS.items():
        if spec_band and k in spec_band: out[k] = min(hi, max(lo, float(spec_band[k])))
    return out


def spot_params(p, meta, k, rng):
    """the params of a spot check that re-scores k random documents of the golden (kld.json meta)"""
    spans = G.doc_spans(meta, rng.sample(range(len(meta["docs"])), min(k, len(meta["docs"]))))
    return dict(p, docs=spans, sub_rows=sum(s[2] for s in spans))


def compare(p, a, b, tol=None, pb=None):
    """a donor's result dir a against a trusted re-run's b (pb: its params; with docs, a spot check of those
    documents: the donor's rows of them, row by row and per document)"""
    t = dict(TOL, **(tol or {}))
    sa, sb = (json.load(open(os.path.join(d, "job.json"))).get("container_sha256") for d in (a, b))
    if sa != sb: return False, {"error": "assembled containers differ", "donor": sa, "trusted": sb}
    return G.compare_rows(os.path.join(a, "rows.f32"), os.path.join(b, "rows.f32"), t, (pb or {}).get("docs"))
