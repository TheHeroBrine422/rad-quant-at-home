# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The lease filter, shared by Queue.lease (what to hand out now) and the donor API (how many queued jobs each of a
donor's workers could run): stdlib only."""
import json
import jobs


def norm_req(r):
    r = dict(r); r["gpu"] = bool(r.get("gpu")); r["cards"] = int(r.get("cards", 1 if r["gpu"] else 0))
    r["ram_gb"] = float(r.get("ram_gb", 0)); r["vram_gb"] = float(r.get("vram_gb", 0))
    return r


def job_filter(wr, caps):
    """f(job row) -> its normalised requirements if worker row wr with declared caps may ever run it, else None (the
    free-cards / free-RAM part is the caller's)"""
    types = set(caps.get("job_types") or ()) & {n for n, m in jobs.REGISTRY.items() if m.IMPLEMENTED}
    vers = caps.get("versions", {})
    disk = caps.get("disk_gb", 0) * 1e9
    cards = caps.get("cards", 1 if caps.get("gpu") else 0)

    def eligible(j):
        if j["type"] not in types or vers.get(j["type"], j["version"]) != j["version"]: return None
        if j["exclude_worker"] == wr["id"]: return None
        if j["needs"] == "verifier" and not wr["verifier"]: return None
        if j["needs"] == "trusted" and not (wr["trusted"] or wr["verifier"]): return None
        r = norm_req(json.loads(j["requires"]))
        if r["gpu"] and not caps.get("gpu"): return None
        if r["cards"] > cards or r["vram_gb"] > caps.get("vram_gb", 0) or r["ram_gb"] > caps.get("ram_gb", 0): return None
        if j["input_bytes"] > disk: return None
        return r
    return eligible
