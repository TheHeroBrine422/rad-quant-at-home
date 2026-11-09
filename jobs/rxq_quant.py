# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rxq_quant (ladder branch, 2026-11-08): one Hessian group (or one tensor) of a trunk layer at one ladder level with
rad-exl3-mix's V3 quantizer pipeline (`python -m rxq job SPEC -o OUT`, rad-exl3-mix pipeline/), the settings the V3
freeze pins (pipeline/config/v3.toml: H_out at every level, tile CD on attn_qg, sub-1 damping, i4 + qtf2, C9 alpha on the
writers, ...). The jobs are generated from the settings file by `python -m rxq.ladder jobs` (rad-exl3-mix) and enqueued
with `admin.py enqueue-jobs`; this module only checks and runs them.

Parameters: model, revision, arch, level, K, codebook (i8 / i4 / mul1), layer, group, keys (the tensors: the group's,
or one of them, in the group's order), settings {rxq: the settings stamp (rxq stamp), seed_offset}, stamp (this
queue's settings stamp of the params, as quant_*), config (the settings file's datapack relpath:
models/M@R/rxq/<settings stamp>.toml), hout (H_out snapshot id, or null where H_out does not apply), c9 (the keys whose
job writes C9 scales: plain writers with alpha != 0), req {vram_gb, ram_gb, est_s} (from rxq.ladder's measured
table: what the scheduler leases on; est_s = R9700 seconds, the lumens), pins {relpath: sha256}, verify (score).
Inputs: content-addressed. pins names every input with its sha256 and must be exactly the inputs the parameters imply
(settings file, the group's weights LNN-gG, the group's Hessian, each key's H_out file); the worker refuses a lease
whose input hashes differ from the pins, and the server refuses a result whose job.json does not record them.
Outputs: t-<key>.safetensors per key (trellis / suh / svh / mul1 marker, rxq metadata), c9-<key>.safetensors for the
c9 keys, log.jsonl (rxq's per-tensor record), job.json (rxq's, plus this job's input hashes and argv), rxq.log,
tilecd.jsonl (attn_qg). Verification by score (docs/VERIFY.md): codes() gives the scorer the decoded inputs.
Assembly: admin.py export --level L OUT links the accepted job dirs as OUT/jobs/<id> and runs (or prints)
`python -m rxq assemble OUT/level OUT/jobs/*`.
Worker: --rxq PIPELINE_DIR (default: rad-exl3-mix/pipeline next to --ourq), --exl3-src, --python. TEST ONLY: --env
RXQ_CPU_STANDIN=1 runs on the CPU with rxq's stand-in kernels; the server refuses such results unless it was started
with RQAH_ALLOW_STANDIN=1."""
import json, os, re, shutil, time

from common.util import write_json_atomic, st_header
from . import qwen, quant_common as Q

NAME, VERSION, IMPLEMENTED = "rxq_quant", 1, True
TOL = {"proxy_rel": 0.015}
HEX64 = {"type": "string", "pattern": "[0-9a-f]{64}"}
KEY = {"type": "string", "pattern": r"model\.language_model\.layers\.\d{1,3}\.[a-z_]+\.[a-z_]+"}
SCHEMA = {"type": "object", "required": ["model", "revision", "arch", "level", "K", "codebook", "layer", "group", "keys",
                                         "settings", "stamp", "config", "hout", "c9", "req", "pins"],
          "properties": {
              "model": {"type": "string", "pattern": Q.ID}, "revision": {"type": "string", "pattern": Q.ID},
              "arch": {"type": "string", "enum": ["qwen3_5"]}, "level": {"type": "string", "pattern": Q.ID},
              "K": {"type": "number", "minimum": 0.125, "maximum": 8, "multipleOf": 0.125},
              "codebook": {"type": "string", "enum": ["mul1", "i8", "i4"]},
              "layer": {"type": "integer", "minimum": 0, "maximum": 255}, "group": {"type": "integer", "minimum": 0, "maximum": 7},
              "keys": {"type": "array", "items": KEY, "minItems": 1, "maxItems": 4},
              "settings": {"type": "object", "required": ["rxq", "seed_offset"], "properties": {
                  "rxq": {"type": "string", "pattern": "[0-9a-f]{16}"},
                  "seed_offset": {"type": "integer", "minimum": 0, "maximum": 1000}}},
              "stamp": {"type": "string", "pattern": "[0-9a-f]{16}"},
              "config": {"type": "string", "pattern": r"models/[a-z0-9][a-z0-9._-]{0,63}@[a-z0-9][a-z0-9._-]{0,63}/rxq/[0-9a-f]{16}\.toml"},
              "hout": {"type": ["string", "null"], "pattern": Q.ID},
              "c9": {"type": "array", "items": KEY, "maxItems": 4},
              "req": {"type": "object", "required": ["vram_gb", "ram_gb", "est_s"], "properties": {
                  "vram_gb": {"type": "number", "minimum": 1, "maximum": 96}, "ram_gb": {"type": "number", "minimum": 1, "maximum": 128},
                  "est_s": {"type": "number", "minimum": 0, "maximum": 86400}}},
              "pins": {"type": "object", "additionalProperties": HEX64},
              "verify": {"type": "string", "enum": ["score"]}}}
GROUP_PARAMS_MAX = 210_000_000                         # gate + up of the 27B: 2 x 17408 x 5120
WRITERS = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj")


def plan(p):
    """[(the group's Hessian name, the job's keys)] (the scorer's / structure check's plan)"""
    h, ks = qwen.layer_plan(p["arch"], p["layer"])[p["group"]]
    return [(h, list(p["keys"]))]


def root(p): return Q.root(p)
def wfile(p): return f"L{p['layer']:02d}-g{p['group']}.safetensors"
def tfile(key): return f"t-{key}.safetensors"
def c9file(key): return f"c9-{key}.safetensors"


def expected_inputs(p):
    r = root(p); (h, ks), = plan(p)
    names = [p["config"], f"{r}/weights/{wfile(p)}", f"{r}/hessians/{h}.safetensors"]
    if p["hout"]: names += [f"{r}/hout/{p['hout']}/{k}.safetensors" for k in ks]
    return names


def settings_stamp(p): return Q.settings_stamp(p)


def check_extra(p):
    if p["stamp"] != settings_stamp(p): raise ValueError("settings stamp does not match the parameters")
    plan_ = qwen.layer_plan(p["arch"], p["layer"])
    if p["group"] >= len(plan_): raise ValueError(f"layer {p['layer']} has no group {p['group']}")
    grp = plan_[p["group"]][1]
    if len(set(p["keys"])) != len(p["keys"]) or [k for k in grp if k in p["keys"]] != p["keys"]:
        raise ValueError(f"keys must be members of group {p['group']} of layer {p['layer']}, in its order")
    if not set(p["c9"]) <= set(p["keys"]) or any(not k.endswith(WRITERS) for k in p["c9"]):
        raise ValueError("c9: writer keys of this job only")
    if not p["config"].startswith(root(p) + "/rxq/") or not p["config"].endswith(f"/{p['settings']['rxq']}.toml"):
        raise ValueError("config: models/M@R/rxq/<settings.rxq>.toml of this model")
    if sorted(p["pins"]) != sorted(expected_inputs(p)): raise ValueError("pins: exactly the inputs the parameters imply")
    K = p["K"]
    if K != int(K) and (K * 16) % 2: raise ValueError("fractional K must be KA + n/16 with n even")
    if p["codebook"] == "i4" and K > 3: raise ValueError("i4 levels stop at K3")


def requires(p, info=None):
    return {"gpu": True, "cards": 1, "vram_gb": float(p["req"]["vram_gb"]), "ram_gb": float(p["req"]["ram_gb"])}


def lumens(p): return max(0.05, float(p["req"]["est_s"]) / 60.0)     # R9700 card-minutes


def inputs(p): return sorted(p["pins"]), []


def outputs(p):
    o = {tfile(k): Q.shard_cap(p, GROUP_PARAMS_MAX) for k in p["keys"]}
    o.update({c9file(k): 1 << 20 for k in p["c9"]})
    o.update({"log.jsonl": 1 << 20, "job.json": 256 << 10, "rxq.log": Q.LOG_CAP, "tilecd.jsonl": 64 << 10})
    return o


def required_outputs(p):
    return {tfile(k) for k in p["keys"]} | {c9file(k) for k in p["c9"]} | {"log.jsonl", "job.json"}


def upload_bytes(p, sizes):
    w = sizes.get(f"{root(p)}/weights/{wfile(p)}")
    n = w // 2 if w else GROUP_PARAMS_MAX
    if w: n = n * len(p["keys"]) // len(qwen.layer_plan(p["arch"], p["layer"])[p["group"]][1])
    return int(n * float(p["K"]) / 8 * 1.02) + (1 << 20)


# ---------------------------------------------------------------------------------------------------- worker side
def rxq_dir(cfg):
    d = getattr(cfg, "rxq", None)
    if not d and getattr(cfg, "ourq", None): d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(cfg.ourq))), "pipeline")
    if not d or not os.path.isdir(os.path.join(d, "rxq")): raise RuntimeError("rxq_quant: the worker needs --rxq (rad-exl3-mix pipeline/)")
    return os.path.abspath(d)


def stage(p, ctx):
    """symlink the cached inputs into the layout rxq's job spec names (fixed paths)"""
    r = root(p); s = ctx.stage_dir
    if os.path.exists(s): shutil.rmtree(s)
    ck, hs, ho = f"{s}/ckpt", f"{s}/hessians", f"{s}/hout"
    for d in (ck, hs, ho): os.makedirs(d)
    os.symlink(ctx.input(f"{r}/weights/{wfile(p)}"), f"{ck}/{wfile(p)}")
    grp = qwen.layer_plan(p["arch"], p["layer"])[p["group"]][1]
    with open(f"{ck}/model.safetensors.index.json", "w") as f: json.dump({"metadata": {}, "weight_map": {f"{k}.weight": wfile(p) for k in grp}}, f)
    (h, ks), = plan(p)
    os.symlink(ctx.input(f"{r}/hessians/{h}.safetensors"), f"{hs}/{h}.safetensors")
    if p["hout"]:
        for k in ks: os.symlink(ctx.input(f"{r}/hout/{p['hout']}/{k}.safetensors"), f"{ho}/{k}.safetensors")
    os.symlink(ctx.input(p["config"]), f"{s}/settings.toml")
    spec = {"type": "rxq.quant", "version": 1, "level": p["level"], "K": int(p["K"]) if float(p["K"]).is_integer() else p["K"],
            "codebook": p["codebook"], "seed_offset": p["settings"]["seed_offset"], "keys": list(p["keys"]),
            "settings_stamp": p["settings"]["rxq"], "config": f"{s}/settings.toml",
            "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho if p["hout"] else None}}
    write_json_atomic(f"{s}/spec.json", spec)
    return f"{s}/spec.json"


def input_hashes(ctx):
    return dict(getattr(ctx, "input_sha", None) or {})


def run(p, ctx):
    t0 = time.time()
    got = input_hashes(ctx)
    bad = [n for n, h in p["pins"].items() if got.get(n) != h]
    if bad: raise RuntimeError(f"inputs differ from the job's pins (content-addressed): {bad[:3]}")
    spec = stage(p, ctx)
    out = ctx.out_dir; cfg = ctx.cfg
    rd = rxq_dir(cfg)
    standin = (cfg.env or {}).get("RXQ_CPU_STANDIN") == "1" or os.environ.get("RXQ_CPU_STANDIN") == "1"
    argv = [cfg.python, "-m", "rxq", "job", spec, "-o", out, "--device", "cpu" if standin else str(ctx.device)]
    done = all(os.path.exists(os.path.join(out, tfile(k))) for k in p["keys"]) and os.path.exists(os.path.join(out, "job.json"))
    if not done:                        # rxq skips finished tensors itself (each written .part -> rename); keep the ones
        logged = set()                  # an earlier try finished AND logged (a kill between the two leaves a tensor unlogged)
        lp = os.path.join(out, "log.jsonl")
        if os.path.exists(lp):
            rows = []
            for l in open(lp):
                try: r = json.loads(l)
                except ValueError: break                     # a torn last line
                if r.get("key") in p["keys"] and os.path.exists(os.path.join(out, tfile(r["key"]))): rows.append(l if l.endswith("\n") else l + "\n"); logged.add(r["key"])
            with open(lp + ".part", "w") as f: f.writelines(rows)
            os.replace(lp + ".part", lp)
        for k in p["keys"]:
            if k not in logged:
                for f in (tfile(k), c9file(k)):
                    if os.path.exists(os.path.join(out, f)): os.remove(os.path.join(out, f))
        fork = os.path.dirname(os.path.dirname(os.path.abspath(cfg.exl3_src))) if cfg.exl3_src else None
        env = {"PYTHONPATH": os.pathsep.join(x for x in (rd, fork, os.environ.get("PYTHONPATH")) if x),
               "RXQ_RXM": os.path.dirname(rd), "RXQ_REFIT_CPU_FALLBACK": "1"}
        if cfg.exl3_src: env["RXQ_EXL3_SRC"] = cfg.exl3_src
        if cfg.vram_cap_gb: env["VRAM_CAP_GB"] = repr(float(cfg.vram_cap_gb))
        if Q.small_card(cfg): env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        rc = ctx.run_cmd(argv, env=env, cwd=ctx.stage_dir, log=os.path.join(out, "rxq.log"))
        if rc != 0: raise RuntimeError(f"rxq job exit {rc} (see rxq.log)")
    jp = os.path.join(out, "job.json")
    jr = json.load(open(jp))
    jr["rqah"] = {"type": ctx.jtype, "version": ctx.jversion, "stamp": p["stamp"], "level": p["level"], "inputs": got,
                  "argv": argv[1:], "image": cfg.image, "gpu": cfg.gpu_name, "seconds": round(time.time() - t0, 1)}
    write_json_atomic(jp, jr)
    lg = os.path.join(out, "rxq.log")
    if os.path.exists(lg) and os.path.getsize(lg) > Q.LOG_CAP:
        with open(lg, "rb") as f: f.seek(-Q.LOG_CAP + 4096, 2); tail = f.read()
        with open(lg, "wb") as f: f.write(b"[truncated]\n" + tail)
    return {"code_stamp": jr.get("code_stamp"), "kernels": jr.get("kernels"), "seconds": jr["rqah"]["seconds"],
            "rxq_seconds": jr.get("seconds")}


# ---------------------------------------------------------------------------------------------------- server side
def expected_tensors(p, weights_hdr):
    return Q.expected_tensors(p, plan(p), weights_hdr)


def check_tensor_file(p, key, path, weights_hdr):
    """one t-<key> file: exactly the four tensors of its shape at K, the job's settings in its metadata"""
    hdr = st_header(path)
    exp = {n: v for n, v in expected_tensors(p, weights_hdr).items() if n.startswith(key + ".")}
    got = {k: (v["dtype"], v["shape"]) for k, v in hdr.items() if k != "__metadata__"}
    if set(got) != set(exp): raise ValueError(f"{tfile(key)}: tensors {sorted(got)[:4]} != {sorted(exp)[:4]}")
    for n, (dt, sh) in exp.items():
        if got[n][0] != dt or list(got[n][1]) != sh: raise ValueError(f"{n}: {got[n]} != {(dt, sh)}")
    m = hdr.get("__metadata__") or {}
    if m.get("rxq_settings") != p["settings"]["rxq"]: raise ValueError(f"{tfile(key)}: settings {m.get('rxq_settings')} != the job's")
    if m.get("key") != key or m.get("codebook") != p["codebook"] or float(m.get("K", "nan")) != float(p["K"]) or m.get("level") != p["level"]:
        raise ValueError(f"{tfile(key)}: metadata (key / K / codebook / level) differs from the job")
    if not re.fullmatch(r"[0-9a-f]{16}", m.get("rxq_code", "")): raise ValueError(f"{tfile(key)}: no code stamp")
    if m.get("rxq_kernels", "gpu") != "gpu" and os.environ.get("RQAH_ALLOW_STANDIN") != "1":
        raise ValueError(f"{tfile(key)}: made with the {m['rxq_kernels']} kernels (test only)")
    if (key in p["c9"]) != ("c9" in m): raise ValueError(f"{tfile(key)}: C9 {'missing' if key in p['c9'] else 'unexpected'}")
    return hdr


def check_c9(p, key, path, weights_hdr):
    hdr = st_header(path)
    n, k = weights_hdr[f"{key}.weight"]["shape"]
    got = {x: (v["dtype"], v["shape"]) for x, v in hdr.items() if x != "__metadata__"}
    if set(got) != {"c", "r"} or list(got["c"][1]) != [n] or list(got["r"][1]) != [k]:
        raise ValueError(f"{c9file(key)}: tensors {got}, want c [{n}] and r [{k}]")
    m = hdr.get("__metadata__") or {}
    if m.get("rxq_settings") != p["settings"]["rxq"] or m.get("key") != key: raise ValueError(f"{c9file(key)}: metadata differs")


def check_log(p, log_path, job_path):
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    last = {r["key"]: r for r in rows}
    if sorted(last) != sorted(p["keys"]): raise ValueError("log.jsonl: keys differ from the job")
    for k, r in last.items():
        if float(r["K"]) != float(p["K"]) or r["codebook"] != p["codebook"] or r["rxq"]["settings"] != p["settings"]["rxq"]:
            raise ValueError(f"log.jsonl {k}: settings differ from the job")
        if not (isinstance(r["proxy_err"], float) and 0 <= r["proxy_err"] < 10): raise ValueError(f"{k}: bad proxy_err")
        if bool(r.get("hout")) != bool(p["hout"]): raise ValueError(f"{k}: H_out {r.get('hout')} vs the job's {p['hout']}")
        if (k in p["c9"]) != ("c9" in r): raise ValueError(f"log.jsonl {k}: C9 record {'missing' if k in p['c9'] else 'unexpected'}")
    meta = json.load(open(job_path))
    if meta.get("settings_stamp") != p["settings"]["rxq"]: raise ValueError("job.json: settings stamp differs")
    rq = meta.get("rqah") or {}
    if rq.get("stamp") != p["stamp"]: raise ValueError("job.json: this job's stamp missing or different")
    if rq.get("inputs") != p["pins"]: raise ValueError("job.json: the inputs it ran on are not the pinned ones")
    return last, meta


def accept(p, files, result, data_path):
    miss = required_outputs(p) - set(files)
    if miss: raise ValueError(f"outputs missing: {sorted(miss)[:3]}")
    whdr = st_header(data_path(f"{root(p)}/weights/{wfile(p)}"))
    for k in p["keys"]: check_tensor_file(p, k, files[tfile(k)], whdr)
    for k in p["c9"]: check_c9(p, k, files[c9file(k)], whdr)
    check_log(p, files["log.jsonl"], files["job.json"])


class _Codes:
    """the job's t-files as one getter of "<key>.<tensor>" (the scorer's view of a shard)"""
    def __init__(self, files):
        from checker.check import ST
        self.st = {k: ST(f) for k, f in files.items()}
    def get(self, name):
        key, x = name.rsplit(".", 1)
        return self.st[key].get(name)


def codes(p, res_dir, weights_file):
    """structure-checked access to a result's codes for server/score.py"""
    whdr = st_header(weights_file)
    files = {k: os.path.join(res_dir, tfile(k)) for k in p["keys"]}
    for k, f in files.items(): check_tensor_file(p, k, f, whdr)
    return _Codes(files)


def read_proxy(d):
    out = {}
    for l in open(os.path.join(d, "log.jsonl")):
        if l.strip(): r = json.loads(l); out[r["key"]] = float(r["proxy_err"])
    return out


def compare(p, a, b, tol=None):
    t = dict(TOL, **(tol or {}))
    pa, pb = read_proxy(a), read_proxy(b)
    if set(pa) != set(pb): return False, {"error": "tensor sets differ"}
    rat = {k: pa[k] / pb[k] if pb[k] > 0 else float("inf") for k in pa}
    worst = max(rat, key=lambda k: abs(rat[k] - 1))
    return abs(rat[worst] - 1) <= t["proxy_rel"], {"worst": worst, "ratio": rat[worst], "tol": t["proxy_rel"]}
