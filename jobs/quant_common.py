# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Shared by quant_layer, quant_tensor and quant_head: parameters, the settings stamp, the staging of inputs into the
layout quantize/ourq.py reads (RXM_CKPT / RXM_WORK / RXM_TEMPLATE), the ourq command line, and the shard structure check.

verify (0.3, VERSION 3): "replicate" (default: the 0.2 path, ourq --save-wq + the per-row claim rowerr.npz for the
checker, spot-check re-runs) or "score" (docs/VERIFY.md: the server scores the codes themselves; no --save-wq, no
rowerr step, so less time, VRAM and host RAM). Not part of the settings stamp: it changes what is uploaded, not the
quantization."""
import ast, json, os, re, shutil, struct, time, zipfile
from common.util import stamp, st_header, write_json_atomic
from . import qwen

ID = r"[a-z0-9][a-z0-9._-]{0,63}"
SETTINGS = {"type": "object", "required": ["sigma_reg", "sigma_reg_in", "sigma_reg_out", "scaled_hessian", "qtf2",
                                           "hout", "seed_offset"], "properties": {
    "sigma_reg": {"type": "number", "minimum": 0, "maximum": 1},
    "sigma_reg_in": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "sigma_reg_out": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "scaled_hessian": {"type": "boolean"},
    "qtf2": {"type": "boolean"},
    "hout": {"type": ["string", "null"], "pattern": ID, "description": "H_out snapshot id (models/<m>@<rev>/hout/<id>/)"},
    "seed_offset": {"type": "integer", "minimum": 0, "maximum": 1000}}}


def score_mode(p): return p.get("verify") == "score"


def schema(with_layer):
    props = {"model": {"type": "string", "pattern": ID}, "revision": {"type": "string", "pattern": ID},
             "arch": {"type": "string", "enum": ["qwen3_5"]},
             "K": {"type": "number", "minimum": 0.125, "maximum": 8, "multipleOf": 0.125},
             "codebook": {"type": "string", "enum": ["mul1", "i8", "i4", "i8c"]},
             "level": {"type": "string", "pattern": ID},
             "settings": SETTINGS, "stamp": {"type": "string", "pattern": "[0-9a-f]{16}"},
             "verify": {"type": "string", "enum": ["replicate", "score"]}}
    req = ["model", "revision", "arch", "K", "codebook", "level", "settings", "stamp"]
    if with_layer:
        props["layer"] = {"type": "integer", "minimum": 0, "maximum": 255}; req.append("layer")
    return {"type": "object", "required": req, "properties": props}


def settings_stamp(p):
    """one per ladder level: what the quantizer is asked to do, layer excluded"""
    return stamp({k: p[k] for k in ("model", "revision", "arch", "K", "codebook", "settings")})


def check_extra(p):
    if p["stamp"] != settings_stamp(p): raise ValueError("settings stamp does not match the parameters")
    K = p["K"]
    if K < 1 and not p["settings"]["qtf2"]: raise ValueError("K < 1 needs qtf2")
    if K != int(K) and (K * 16) % 2: raise ValueError("fractional K must be KA + n/16 with n even")
    if p["settings"]["qtf2"] and p["codebook"] not in ("mul1", "i8", "i8c"): raise ValueError("qtf2 needs mul1, i8 or i8c")


def root(p): return f"models/{p['model']}@{p['revision']}"


# what a job needs, by model size (models/<m>@<rev>/model.json from admin.py datapack); the 27B class measured on
# the R9700 / notes: a trunk layer ~24 GB host RAM on a 32 GB card, the LM head ~22 GB peak (48 GB declared: it is
# tight on a 32 GB desktop next to everything else); the 0.8B / 2B proxies run in a few GB
BIG_PARAMS = 8e9


def big(info): return bool(info) and info.get("n_params", 0) >= BIG_PARAMS


# 27B down_proj (17408 x 5120) on a card capped at 7.2 GB (TESTS.md section 11): fits with the scale refit on the CPU
# (OURQ_REFIT_CPU_FALLBACK) and, at K6, a small cuBLAS workspace (without it cuBLAS fails at 7.17 GB reserved); measured
# K2-K6, so 8 GB cards get it up to K6
SMALL_CARD_MAX_K = 6
SMALL_CARD_GB = 12                                      # a worker below this (declared or capped) gets the small-card env


def down_vram_gb(p): return 8 if float(p["K"]) <= SMALL_CARD_MAX_K else 12


TOL = {"proxy_rel": 0.015}      # spot check: |donor proxy_err / trusted re-run's - 1| per tensor (seed spread ~0.5%)


def read_proxy(d):
    out = {}
    for l in open(os.path.join(d, "proxy.jsonl")):
        if l.strip(): r = json.loads(l); out[r["key"]] = float(r["proxy_err"])          # the last entry of a key wins
    return out


def compare(p, a, b, tol=None):
    """a donor's result (a) against a trusted re-run (b): every tensor's logged proxy error within tol. The checker
    (checker/check.py) has already tied each log to its own codes, so this compares the codes' quality."""
    t = dict(TOL, **(tol or {}))
    pa, pb = read_proxy(a), read_proxy(b)
    if set(pa) != set(pb): return False, {"error": "tensor sets differ"}
    rat = {k: pa[k] / pb[k] if pb[k] > 0 else float("inf") for k in pa}
    worst = max(rat, key=lambda k: abs(rat[k] - 1))
    return abs(rat[worst] - 1) <= t["proxy_rel"], {"worst": worst, "ratio": rat[worst], "tol": t["proxy_rel"],
                                                   "ratios": {k: round(v, 5) for k, v in rat.items()}}


def inputs_for(p, plan, wfile):
    r = root(p)
    req = [f"{r}/config.json", f"{r}/weights/{wfile}"] + [f"{r}/hessians/{h}.safetensors" for h, _ in plan]
    opt = []
    if p["settings"]["hout"]:
        alt = "|".join(re.escape(k) for k in qwen.keys(plan))
        opt.append(re.escape(f"{r}/hout/{p['settings']['hout']}/") + f"(?:{alt})\\.safetensors")
    return req, opt


def shard_cap(p, n_params):
    return 32 * 2 ** 20 + int(n_params * p["K"] / 8 * 1.05)


def upload_bytes(p, sizes, wrel, n_params_max):
    """expected upload: the shard (the bf16 weight file's parameters at K bits, plus scales and headers), proxy /
    stamp / job / log files, and in replicate mode the per-row claim"""
    w = sizes.get(f"{root(p)}/weights/{wrel}")
    n = w // 2 if w else n_params_max
    return int(n * float(p["K"]) / 8 * 1.02) + (1 << 20) + (0 if score_mode(p) else 4 << 20)


LOG_CAP = 1 << 20
SMALL = {"proxy.jsonl": 256 << 10, "stamp.json": 1 << 20, "job.json": 64 << 10, "ourq.log": LOG_CAP, "rowerr.npz": 16 << 20}


def required(p, shard):
    """the outputs a result must have: score mode needs no per-row claim"""
    return {shard, "proxy.jsonl", "stamp.json", "job.json"} | (set() if score_mode(p) else {"rowerr.npz"})


def stage(p, ctx, plan, wfile):
    """symlink the cached inputs into a fixed layout (fixed paths keep ourq's code stamp the same for every job)"""
    r = root(p); s = ctx.stage_dir
    if os.path.exists(s): shutil.rmtree(s)
    ck, hs, ho = f"{s}/ckpt", f"{s}/work/challenge/hessians/calib", f"{s}/hout"
    for d in (ck, hs, ho, f"{s}/cwd"): os.makedirs(d)
    os.symlink(ctx.input(f"{r}/weights/{wfile}"), f"{ck}/{wfile}")
    wm = {f"{k}.weight": wfile for k in qwen.keys(plan)}
    with open(f"{ck}/model.safetensors.index.json", "w") as f: json.dump({"metadata": {}, "weight_map": wm}, f)
    for h, _ in plan: os.symlink(ctx.input(f"{r}/hessians/{h}.safetensors"), f"{hs}/{h}.safetensors")
    os.symlink(ctx.input(f"{r}/config.json"), f"{s}/template.json")
    nh = 0
    if p["settings"]["hout"]:
        pre = f"{r}/hout/{p['settings']['hout']}/"
        for n in ctx.input_names:
            if n.startswith(pre): os.symlink(ctx.input(n), f"{ho}/{n[len(pre):]}"); nh += 1
    return {"ckpt": ck, "work": f"{s}/work", "template": f"{s}/template.json", "hout": ho if nh else None, "cwd": f"{s}/cwd"}


def ourq_argv(p, cfg, out, st, which, device=0):
    S = p["settings"]
    K = p["K"]
    pre = [cfg.python] + ([cfg.capwrap, repr(float(cfg.vram_cap_gb))] if cfg.vram_cap_gb else [])
    a = pre + [cfg.ourq, "-K", str(int(K)) if float(K).is_integer() else repr(float(K)),
         "-o", out, "--device", str(device), "--codebook", p["codebook"]] + ([] if score_mode(p) else ["--save-wq"]) + which
    if S["qtf2"] or p["codebook"] != "mul1" or p["K"] != int(p["K"]):
        a += ["--frac-ext", cfg.exl3_src]
    if S["qtf2"]: a.append("--qtf2")
    if S["scaled_hessian"]: a.append("--scaled-hessian")
    a += ["--sigma-reg", repr(float(S["sigma_reg"]))]
    if S["sigma_reg_in"] is not None: a += ["--sigma-reg-in", repr(float(S["sigma_reg_in"]))]
    if st["hout"]:
        a += ["--hout", st["hout"]]
        if S["sigma_reg_out"] is not None: a += ["--sigma-reg-out", repr(float(S["sigma_reg_out"]))]
    if S["seed_offset"]: a += ["--seed-offset", str(S["seed_offset"])]
    return a


def small_card(cfg):
    """a worker with less than SMALL_CARD_GB per card (declared --vram-gb, or capped by --vram-cap-gb)"""
    v = [x for x in ((getattr(cfg, "caps", None) or {}).get("vram_gb"), getattr(cfg, "vram_cap_gb", None)) if x]
    return bool(v) and min(v) < SMALL_CARD_GB


def run_ourq(p, ctx, plan, wfile, which, shard_name, ourq_shard=None):
    """ourq.py on the staged inputs; ourq_shard: the file ourq writes when it differs from the job's output name
    (quant_tensor: ourq writes q-LNN.safetensors, the job uploads q-LNN-gG.safetensors)"""
    t0 = time.time()
    st = stage(p, ctx, plan, wfile)
    out = ctx.out_dir
    shard = os.path.join(out, shard_name)
    if "--only" in which:                      # quant_tensor: the group's keys, in the stage dir (fixed path: code stamp)
        with open(which[which.index("--only") + 1], "w") as f: f.write("\n".join(qwen.keys(plan)) + "\n")
    argv = ourq_argv(p, ctx.cfg, out, st, which, ctx.device)
    done = os.path.exists(shard) and (score_mode(p) or os.path.exists(os.path.join(out, "rowerr.npz")))
    if not done:
        for f in os.listdir(out):              # a killed earlier try: start clean (ourq appends to proxy.jsonl)
            q = os.path.join(out, f)
            shutil.rmtree(q) if os.path.isdir(q) else os.remove(q)
        env = dict(ctx.cfg.env)
        # OURQ_REFIT_CPU_FALLBACK: a scale refit that does not fit the card runs on the CPU (27B down_proj needs a
        # second 1.1 GiB copy of its 17408^2 Hessian: > 7.2 GB in all; on the CPU it takes ~1 min and ~10 GB host RAM)
        env.update({"RXM_CKPT": st["ckpt"], "RXM_WORK": st["work"], "RXM_TEMPLATE": st["template"], "OURQ_REFIT_CPU_FALLBACK": "1"})
        if small_card(ctx.cfg): env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # (ignored on ROCm)
        rc = ctx.run_cmd(argv, env=env, cwd=st["cwd"], log=os.path.join(out, "ourq.log"))
        if rc != 0: raise RuntimeError(f"ourq.py exit {rc} (see ourq.log)")
        if ourq_shard and ourq_shard != shard_name: os.replace(os.path.join(out, ourq_shard), shard)
        t1 = time.time()
        if not score_mode(p): pack_rowerr(ctx, plan, st, out)
        t_rowerr = time.time() - t1 if not score_mode(p) else None
    else: t_rowerr = None
    hdr = st_header(shard)
    ls = re.findall(r"^layer \d+: (\d+)s", open(os.path.join(out, "ourq.log")).read(), re.M)
    meta = {"type": ctx.jtype, "version": ctx.jversion, "settings_stamp": p["stamp"], "level": p["level"], "verify": p.get("verify", "replicate"),
            "ourq_stamp": (hdr.get("__metadata__") or {}).get("ourq_stamp"), "image": ctx.cfg.image,
            "gpu": ctx.cfg.gpu_name, "seconds": round(time.time() - t0, 1),
            "ourq_layer_s": int(ls[-1]) if ls else None, "rowerr_s": None if t_rowerr is None else round(t_rowerr, 1), "argv": argv[argv.index(ctx.cfg.ourq):]}
    write_json_atomic(os.path.join(out, "job.json"), meta)
    for f in ("model.safetensors.index.json", "config.json"):        # the server rebuilds them on export
        if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
    lg = os.path.join(out, "ourq.log")
    if os.path.getsize(lg) > LOG_CAP:
        with open(lg, "rb") as f: f.seek(-LOG_CAP + 4096, 2); tail = f.read()
        with open(lg, "wb") as f: f.write(b"[truncated]\n" + tail)
    return {k: meta[k] for k in ("ourq_stamp", "seconds", "ourq_layer_s", "rowerr_s")}


def pack_rowerr(ctx, plan, st, out):
    """the per-output-row claim (rowerr.npz, jobs/rowerr_tool.py) from ourq --save-wq's dequantized weights; the
    weights themselves stay local and are deleted"""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rowerr_tool.py")
    pre = [ctx.cfg.python] + ([ctx.cfg.capwrap, repr(float(ctx.cfg.vram_cap_gb))] if ctx.cfg.vram_cap_gb else [])
    argv = pre + [tool, st["ckpt"], f"{st['work']}/challenge/hessians/calib", os.path.join(out, "wq"),
            os.path.join(out, "rowerr.npz"), str(ctx.device)] + [f"{k}:{h}" for h, ks in plan for k in ks]
    rc = ctx.run_cmd(argv, env=dict(ctx.cfg.env), cwd=st["cwd"], log=os.path.join(out, "ourq.log"))
    if rc != 0: raise RuntimeError(f"rowerr_tool exit {rc} (see ourq.log)")
    shutil.rmtree(os.path.join(out, "wq"))


def expected_tensors(p, plan, weights_hdr):
    """{tensor name: (dtype, shape)} a shard must hold exactly"""
    K = p["K"]; words = int(round(16 * K))
    exp = {}
    for k in qwen.keys(plan):
        n, kin = weights_hdr[f"{k}.weight"]["shape"]          # checkpoint (out, in)
        exp[f"{k}.trellis"] = ("I16", [kin // 16, n // 16, words])
        exp[f"{k}.suh"] = ("F16", [kin]); exp[f"{k}.svh"] = ("F16", [n])
        exp[f"{k}.mul1"] = ("I32", [])
    return exp


def check_shard(p, plan, shard, weights_file):
    """structure of a returned shard against the bf16 weights it was made from; returns the shard header"""
    hdr = st_header(shard)
    exp = expected_tensors(p, plan, st_header(weights_file))
    got = {k: (v["dtype"], v["shape"]) for k, v in hdr.items() if k != "__metadata__"}
    if set(got) != set(exp):
        raise ValueError(f"shard keys: missing {sorted(set(exp) - set(got))[:3]}, extra {sorted(set(got) - set(exp))[:3]}")
    for k, (dt, sh) in exp.items():
        if tuple(got[k][0:1]) != (dt,) or list(got[k][1]) != sh: raise ValueError(f"{k}: {got[k]} != {(dt, sh)}")
    if not (hdr.get("__metadata__") or {}).get("ourq_stamp"): raise ValueError("shard has no ourq_stamp")
    return hdr


def check_log(p, plan, proxy_path, job_path):
    """the proxy log covers every tensor once with the job's settings; job.json carries the job's stamp"""
    rows = [json.loads(l) for l in open(proxy_path) if l.strip()]
    keys = qwen.keys(plan)
    if sorted(r["key"] for r in rows) != sorted(keys): raise ValueError("proxy.jsonl: keys differ from the layer plan")
    for r in rows:
        if float(r["K"]) != float(p["K"]) or r["codebook"] != p["codebook"] or bool(r["scaled_hessian"]) != p["settings"]["scaled_hessian"]:
            raise ValueError(f"proxy.jsonl {r['key']}: settings differ from the job")
        if not (isinstance(r["proxy_err"], float) and 0 <= r["proxy_err"] < 10): raise ValueError(f"{r['key']}: bad proxy_err")
    meta = json.load(open(job_path))
    if meta.get("settings_stamp") != p["stamp"]: raise ValueError("job.json: settings stamp differs")
    return rows, meta

def check_claims(plan, hdr, path):
    """the per-row claim (rowerr.npz) is an uncompressed zip (np.savez) of float arrays <key>.err / <key>.ref, one pair
    per tensor of the plan, each as long as the tensor has output rows. Read from the zip directory and the .npy headers
    only, with the stdlib: no array is loaded (a member could declare any size) and nothing malformed reaches np.load
    in admin.py check"""
    keys = qwen.keys(plan)
    n = {f"{k}.{x}.npy": hdr[f"{k}.svh"]["shape"][0] for k in keys for x in ("err", "ref")}
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist(); names = {i.filename for i in infos}
            if len(names) != len(infos) or names != set(n):
                raise ValueError(f"rowerr.npz: members differ from the plan (missing {sorted(set(n) - names)[:3]}, extra {sorted(names - set(n))[:3]})")
            for i in infos:
                m = n[i.filename]
                if i.compress_type != zipfile.ZIP_STORED: raise ValueError(f"rowerr.npz {i.filename}: compressed, want stored")
                if i.file_size > 128 + 8 * m + 64: raise ValueError(f"rowerr.npz {i.filename}: {i.file_size} bytes, too large for {m} rows")
                with zf.open(i) as fh:
                    head = fh.read(10)
                    if head[:6] != b"\x93NUMPY" or head[6] not in (1, 2, 3): raise ValueError(f"rowerr.npz {i.filename}: not a .npy")
                    if head[6] == 1: hl = struct.unpack("<H", head[8:10])[0]
                    else: hl = struct.unpack("<I", head[8:10] + fh.read(2))[0]
                    if hl > 4096: raise ValueError(f"rowerr.npz {i.filename}: header too long")
                    d = ast.literal_eval(fh.read(hl).decode("latin1"))
                if not isinstance(d, dict) or d.get("descr") not in ("<f8", "<f4") or d.get("fortran_order") is not False or d.get("shape") != (m,):
                    raise ValueError(f"rowerr.npz {i.filename}: header {d!r:.120}, want shape ({m},) <f8/<f4")
    except (zipfile.BadZipFile, KeyError, SyntaxError, struct.error, UnicodeDecodeError, IndexError, OSError) as e:
        raise ValueError(f"rowerr.npz: not a valid npz ({type(e).__name__})")
