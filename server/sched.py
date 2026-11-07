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

QUANT = ("quant_layer", "quant_tensor", "quant_head")
# probation (workers registered without an invite, docs/VERIFY.md "Identity"): the defaults of server.py's
# --probation-pass / --eval-unlock
PROBATION_PASS, EVAL_UNLOCK = 3, 5


def passed_quant(q, wid):
    """a worker's quant results the scorer or checker passed and that earned lumens (a duplicate earns none);
    q(sql, args) -> rows"""
    return q("SELECT COUNT(*) FROM jobs j WHERE j.accepted_by=? AND j.verify_of IS NULL AND j.state='accepted' AND "
             "j.check_state='pass' AND j.type IN ('quant_layer','quant_tensor','quant_head') AND "
             "(SELECT COALESCE(SUM(lumens), 0) FROM points p WHERE p.job_id=j.id AND p.worker_id=j.accepted_by) > 1e-9",
             (wid,))[0][0]


def tier(wr, passed, probation_pass=PROBATION_PASS, eval_unlock=EVAL_UNLOCK):
    """what a worker may lease: "full" (registered with an invite or by the operator, or unlocked), "quant" (on
    probation with probation_pass passed results: score-mode quant jobs only, any number at once) or "probation"
    (score-mode quant jobs only, one at a time). eval_mix and every type whose check takes our hardware (replicate
    quant, golden_record) unlock at eval_unlock passed results. A quarantined probation worker never unlocks."""
    p = wr["probation"] if "probation" in wr.keys() else 0
    if not p or wr["verifier"]: return "full"
    if p >= 2 or wr["quarantined"] or passed < probation_pass: return "probation"
    return "quant" if passed < eval_unlock else "full"


def score_quant(j):
    """a quant job verified by the server's own scorer (no card time of ours)"""
    if j["type"] not in QUANT: return False
    try: return json.loads(j["params"]).get("verify") == "score"
    except (ValueError, AttributeError): return False


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


def job_filter(wr, caps, failed=frozenset(), groups=frozenset(), tier_="full"):
    """f(job row) -> its normalised requirements if worker row wr with declared caps may ever run it, else None (the
    free-cards / free-RAM part is the caller's). failed: ids of queued jobs this worker already failed, rejected or let
    expire since their last requeue (one attempt per worker per job). groups: best-of-N groups (jobs.grp) of which the
    worker's donor already holds or delivered a copy (#11: one copy of a group per donor). tier_: tier() of the
    worker; below "full" only score-mode quant jobs"""
    types = set(caps.get("job_types") or ()) & {n for n, m in jobs.REGISTRY.items() if m.IMPLEMENTED}
    vers = caps.get("versions", {})
    disk = caps.get("disk_gb", 0) * 1e9
    cards = caps.get("cards", 1 if caps.get("gpu") else 0)

    def eligible(j):
        if j["type"] not in types or vers.get(j["type"], j["version"]) != j["version"]: return None
        if j["exclude_worker"] == wr["id"]: return None
        if j["id"] in failed: return None
        if tier_ != "full" and not score_quant(j): return None
        if groups and j["grp"] in groups: return None
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


def donor_groups(db, wr):
    """jobs.grp of the copies the worker's donor (any of its workers) holds a lease on or delivered the accepted result
    of; verifiers (our own boxes) are not limited"""
    if wr["verifier"]: return frozenset()
    return frozenset(r[0] for r in db.execute(
        "SELECT DISTINCT j.grp FROM leases l JOIN jobs j ON j.id=l.job_id JOIN workers w ON w.id=l.worker_id "
        "WHERE w.donor_id=? AND j.grp IS NOT NULL AND (l.state='active' OR (l.state='done' AND j.state='accepted' "
        "AND j.accepted_lease=l.id))", (wr["donor_id"],)))


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
