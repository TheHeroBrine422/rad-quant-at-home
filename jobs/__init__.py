# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The built-in job types. This registry is the whole list of things a worker can be asked to do: the server sends a
type name and a parameter object, never code. A worker refuses a type it does not know, parameters that do not match
the type's schema exactly, and input files outside what the type allows for those parameters.

Every job module defines
    NAME, VERSION, SCHEMA       the type, its version (bumped on any change of meaning), its parameter schema
    ACCEPTS                     (optional, 0.5) older job versions whose results this code still accepts and checks
                                (e.g. (2,) when version 3 only added a parameter with a default): queued and running
                                jobs of those versions keep going to the workers that declare them. Default: none, so
                                after a bump the old version's queued jobs wait (admin.py versions, retire --migrate)
    IMPLEMENTED                 False: the interface exists, the worker refuses to run it
    TRUSTED_ONLY                True: leased only to workers the operator flagged trusted (big-box jobs)
    requires(p, info)           what one job needs: {"gpu", "cards", "vram_gb" (per card), "ram_gb" (host)}; info is
                                the model's datapack record (models/<m>@<rev>/model.json: n_params, ...) or None. The
                                server leases a job only to a worker whose declared and currently free cards / RAM
                                cover it; cards > 1 is a gang job (all its cards on one worker)
    lumens(p)                   points for one verified job: its reference cost in R9700 card-minutes
    inputs(p)                   (required relpaths, [regex for optional relpaths]) under the server's data root
    outputs(p)                  {file name: max bytes}: the only files a worker uploads and the server accepts
    upload_bytes(p, sizes)      (0.4) what one job is expected to upload, in bytes (sizes: {input relpath: bytes});
                                the scheduler charges it against a worker's upload budget (expected_upload())
    required_outputs(p)         the subset that must be present at completion
    run(p, ctx) -> dict         the work; ctx is worker.worker.JobContext; returns a small JSON result
    accept(p, files, result, data_path)   cheap structural check on the server at completion (files: name -> path);
                                raises ValueError to reject
    compare(p, a, b, tol)       spot check: a donor's accepted result dir a against a trusted re-run's b ->
                                (ok, detail dict); tol: {name: value} overrides of the type's TOL"""
import re
from common.schema import validate, SchemaError
from . import noop, quant_layer, quant_tensor, quant_head, eval_mix, golden_record, rxq_quant

REGISTRY = {m.NAME: m for m in (noop, quant_layer, quant_tensor, quant_head, eval_mix, golden_record, rxq_quant)}
# the quantization job types: verified by score / the checker, exported per level, probation-eligible in score mode
QUANT = ("quant_layer", "quant_tensor", "quant_head", "rxq_quant")


class Rejected(ValueError):
    """a job this worker / server will not run: unknown type, bad parameters, bad inputs"""


def served(m):
    """the job versions of type m whose results this server code accepts: its VERSION and ACCEPTS"""
    return {m.VERSION, *getattr(m, "ACCEPTS", ())}


def get(jtype):
    m = REGISTRY.get(jtype) if isinstance(jtype, str) else None
    if m is None: raise Rejected(f"unknown job type {jtype!r}")
    return m


def requirements(m, p, info=None):
    """a type's requires(), normalised: gpu, cards (1 for a GPU job unless it says more), vram_gb, ram_gb"""
    r = dict(m.requires(p, info))
    r["gpu"] = bool(r.get("gpu")); r["cards"] = int(r.get("cards", 1 if r["gpu"] else 0))
    r["vram_gb"] = float(r.get("vram_gb", 0)); r["ram_gb"] = float(r.get("ram_gb", 0))
    if getattr(m, "TRUSTED_ONLY", False): r["trusted"] = True
    return r


def expected_upload(m, p, sizes=None):
    """bytes one job is expected to upload: the type's upload_bytes(p, sizes), else the sum of its output caps (an
    upper bound)"""
    f = getattr(m, "upload_bytes", None)
    if f is not None:
        try: return max(0, int(f(p, sizes or {})))
        except (KeyError, ValueError, TypeError): pass
    return sum(m.outputs(p).values())


def check_params(jtype, params):
    m = get(jtype)
    try:
        validate(params, m.SCHEMA)
        if hasattr(m, "check_extra"): m.check_extra(params)
    except (SchemaError, ValueError) as e:
        raise Rejected(f"{jtype}: {e}")
    return m


def check_inputs(jtype, params, names):
    """the server's input list for a job must be exactly the type's required files plus optional ones it allows"""
    m = get(jtype)
    req, opt = m.inputs(params)
    names = list(names)
    if len(set(names)) != len(names): raise Rejected("duplicate input")
    missing = [r for r in req if r not in names]
    if missing: raise Rejected(f"inputs missing {missing[:3]}")
    rx = [re.compile(x) for x in opt]
    rs = set(req)
    for n in names:
        if n in rs: continue
        if not any(x.fullmatch(n) for x in rx): raise Rejected(f"input {n!r} not allowed for {jtype}")
    return m
