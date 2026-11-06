# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""noop: sleep, read its inputs, write a deterministic file. For tests of the queue, the cache, the uploads, the
scheduler (declared cards / RAM / VRAM: a GPU-free stand-in for GPU and gang jobs) and the spot checks.

v2 (0.2): cards / ram_gb / vram_gb (what the scheduler must find free), proc (run the work in a child process, as
GPU jobs do, and report the device list it was given). Test hook: a worker started with
--env RQAH_TEST_NOOP_SALT=x writes salted bytes of the right size (accepted, but a spot check catches it)."""
import hashlib, json, os, sys, time

NAME, VERSION, IMPLEMENTED = "noop", 2, True
SCHEMA = {"type": "object", "required": ["seconds", "size", "seed"], "properties": {
    "seconds": {"type": "number", "minimum": 0, "maximum": 3600},
    "size": {"type": "integer", "minimum": 0, "maximum": 64 << 20},
    "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 31},
    "inputs": {"type": "array", "maxItems": 16, "items": {"type": "string", "pattern": r"test/[A-Za-z0-9][A-Za-z0-9._-]{0,63}"}},
    "fail": {"type": "boolean"},
    "tag": {"type": "string", "pattern": r"[A-Za-z0-9._-]{0,64}"},
    "cards": {"type": "integer", "minimum": 0, "maximum": 16},
    "ram_gb": {"type": "number", "minimum": 0, "maximum": 4096},
    "vram_gb": {"type": "number", "minimum": 0, "maximum": 1024},
    "proc": {"type": "boolean"}}}
TOL = {}


def requires(p, info=None):
    c = p.get("cards", 0)
    return {"gpu": c > 0, "cards": c, "ram_gb": p.get("ram_gb", 0), "vram_gb": p.get("vram_gb", 0)}


def lumens(p): return round(0.1 + p["seconds"] / 60, 3)
def inputs(p): return list(p.get("inputs", [])), []
def outputs(p): return {"out.bin": p["size"]}
def required_outputs(p): return {"out.bin"}


def content(seed, size, salt=""):
    out, i = bytearray(), 0
    while len(out) < size:
        out += hashlib.sha256(f"{salt}{seed}:{i}".encode()).digest(); i += 1
    return bytes(out[:size])


CHILD = "import json, os, sys; json.dump({k: v for k, v in os.environ.items() if k.endswith('VISIBLE_DEVICES')}, open(sys.argv[1], 'w'))"


def run(p, ctx):
    h = hashlib.sha256()
    for n in inputs(p)[0]:
        with open(ctx.input(n), "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    seen = None
    if p.get("proc"):
        os.makedirs(ctx.stage_dir, exist_ok=True)
        fp = os.path.join(ctx.stage_dir, "env.json")
        rc = ctx.run_cmd([sys.executable, "-I", "-c", CHILD, fp])
        if rc: raise RuntimeError(f"child exit {rc}")
        seen = json.load(open(fp))
    t_end = time.time() + p["seconds"]
    while time.time() < t_end:
        if ctx.cancelled(): raise InterruptedError("cancelled")
        time.sleep(min(0.2, max(0.0, t_end - time.time())))
    if p.get("fail"): raise RuntimeError("noop: asked to fail")
    path = os.path.join(ctx.out_dir, "out.bin")
    with open(path + ".part", "wb") as f: f.write(content(p["seed"], p["size"], ctx.cfg.env.get("RQAH_TEST_NOOP_SALT", "")))
    os.replace(path + ".part", path)
    return {"inputs_sha256": h.hexdigest(), "slept": p["seconds"], "devices": ctx.devices, "child_env": seen}


def accept(p, files, result, data_path=None):
    if os.path.getsize(files["out.bin"]) != p["size"]: raise ValueError("out.bin: wrong size")


def compare(p, a, b, tol=None):
    sa, sb = [hashlib.sha256(open(os.path.join(d, "out.bin"), "rb").read()).hexdigest() for d in (a, b)]
    return sa == sb, {"donor_sha256": sa[:16], "trusted_sha256": sb[:16]}
