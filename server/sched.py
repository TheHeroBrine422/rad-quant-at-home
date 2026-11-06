# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The lease filter, shared by Queue.lease (what to hand out now) and the donor API (how many queued jobs each of a
donor's workers could run), and the 0.4 ordering rules (docs/SCHED.md): stdlib only.

Order of the queue: priority classes (jobs.priority, highest first) are strict: nothing inside the scheduler moves a
job ahead of a higher class except a hard constraint (the worker cannot run the higher job now: cards / RAM busy,
or its data budget is too small), and a busy worker that keeps passing over a higher-class job it could run is
drained for it once that job has waited --class-wait-s. Inside a class: whole levels first (the level whose first
job was enqueued earliest, so a level becomes usable as early as possible), and inside a level the jobs whose inputs
the worker already caches; a worker with a download budget instead takes the job with the least to download first."""
import fnmatch, json
import jobs

# enqueue-level's default priority by the granularity of K (user 2026-11-06): coarse ladder steps first
GRANULARITY_PRIORITY = ((1, 40), (2, 30), (4, 20), (8, 10), (16, 5))


def level_priority(K, boost=0):
    """integer K 40, halves 30, quarters 20, eighths 10, sixteenths 5, anything finer 0; plus boost"""
    K = float(K)
    for d, pr in GRANULARITY_PRIORITY:
        if abs(K * d - round(K * d)) < 1e-9: return pr + boost
    return boost


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


class Transfers:
    """what a job would move over a worker's link: download = its inputs whose sha256 the worker's cache report does
    not list, upload = the type's expected upload (memoised per job; reset when the manifest changes)"""
    def __init__(self, man, memo):
        self.man, self.memo = man, memo

    def up(self, j):
        u = self.memo.get(j["id"])
        if u is None:
            names = json.loads(j["inputs"])
            sizes = {n: self.man[n]["size"] for n in names if n in self.man}
            try: u = jobs.expected_upload(jobs.get(j["type"]), json.loads(j["params"]), sizes)
            except Exception: u = 0
            self.memo[j["id"]] = u
        return u

    def down(self, j, cached):
        """(bytes to download, bytes already cached)"""
        down = have = 0
        for n in json.loads(j["inputs"]):
            m = self.man.get(n)
            if m is None: continue
            if m["sha256"][:16] in cached: have += m["size"]
            else: down += m["size"]
        return down, have


def fits_budget(budget, down, up):
    """budget: the worker's declaration ({down, up}: bytes left, None = unlimited) or None"""
    if not budget: return True
    bd, bu = budget.get("down"), budget.get("up")
    return (bd is None or down <= bd) and (bu is None or up <= bu)


def order_key(budgeted):
    """sort key inside one priority class; c: {j, down, cached, rank} (rank: the level's first job id)"""
    if budgeted: return lambda c: (c["down"], c["rank"], c["j"]["id"])
    return lambda c: (c["rank"], -c["cached"], c["j"]["id"])


def level_ranks(db, levels):
    """{level: id of its first job} (any state: a level keeps its place when jobs of it are requeued)"""
    levels = sorted({l for l in levels if l is not None})
    out = {}
    for i in range(0, len(levels), 500):
        part = levels[i:i + 500]
        for r in db.execute(f"SELECT level, MIN(id) FROM jobs WHERE level IN ({','.join('?' * len(part))}) GROUP BY level", part):
            out[r[0]] = r[1]
    return out


def match_levels(levels, pattern):
    """admin.py reprioritize: levels matching a shell pattern (fnmatch: *, ?, [..])"""
    return [l for l in levels if l is not None and fnmatch.fnmatchcase(l, pattern)]
