# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verification, quarantine and points ("lumens") -- stdlib only, shared by server.py and admin.py.

Spot checks: when a donor's result is accepted, it is drawn for a re-run with probability `rate` (default 4%: the
3-5% band), and always for a donor's first `first` accepted job of each type. A drawn job's verify_state is
"pending" and a copy of it (same type, parameters and inputs; verify_of = the original) is queued for verifier
workers only (our own hardware, flag `verifier`), never the original donor. When the copy is accepted, the type's
compare() checks the donor's result against it (quant: proxy error per tensor; eval_mix: the per-position rows and
the container hash; golden_record: tokens, per-row nll, argmax) and the original becomes pass or fail. Verifiers'
own results are never drawn.

A failed spot check (or a failed checker run on a quant result, admin.py check) quarantines the donor: no more
leases, its active leases are cancelled, and every accepted result of it that no re-run confirmed is invalidated:
points revoked, results moved aside, the job queued again for anyone else.

Points: a job's lumens (its type's lumens(p): reference R9700 card-minutes, +10% if it came back within twice that)
are granted to the worker whose result was accepted, once the result is verified: quant types need the checker's
pass, and a drawn job its spot check's pass; the other types are verified by their structural check at completion.
Verifier workers earn nothing (our machines); timed-out or superseded leases earn nothing.

0.3: quant jobs whose params say "verify": "score" are verified by score instead (server/score.py, docs/VERIFY.md):
no replication spot checks; the scorer decides check_state (pass / fail / provisional), a provisional one is re-run
once on a verifier as an anchor (verify_state untouched), a pass settles points (not for a duplicate of another
donor's codes), a fail quarantines. A quarantine leaves score-verified results alone: their quality is proven by
their own codes, whoever made them."""
import json, os, secrets, shutil, time
import jobs
from server.score import is_score

QUANT = ("quant_layer", "quant_head", "quant_tensor")


class Policy:
    def __init__(self, rate=0.04, first=1, tol=None):
        self.rate, self.first, self.tol = rate, first, tol or {}


def _draw(rate):
    return secrets.randbelow(1_000_000) < int(rate * 1_000_000)


def worker(db, wid): return db.execute("SELECT * FROM workers WHERE id=?", (wid,)).fetchone() if wid else None


def net_points(db, job_id, wid):
    return db.execute("SELECT COALESCE(SUM(lumens), 0) FROM points WHERE job_id=? AND worker_id=?", (job_id, wid)).fetchone()[0]


def settle(S, db, job_id):
    """grant the job's lumens if its result is accepted and verified and nothing has been granted for it yet"""
    j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if j is None or j["state"] != "accepted" or j["verify_of"] or not j["accepted_by"]: return 0
    w = worker(db, j["accepted_by"])
    if w is None or w["verifier"] or w["quarantined"]: return 0
    if j["type"] in QUANT and j["check_state"] != "pass": return 0
    if j["verify_state"] not in (None, "pass"): return 0
    if j["check_state"] == "pass" and json.loads(j["check_detail"] or "{}").get("duplicate_of"): return 0
    if net_points(db, job_id, w["id"]) > 0: return 0
    m = jobs.get(j["type"]); p = json.loads(j["params"])
    lum = float(m.lumens(p))
    l = db.execute("SELECT issued FROM leases WHERE id=?", (j["accepted_lease"],)).fetchone()
    quick = l is not None and j["accepted_at"] - l["issued"] <= 2 * 60 * lum
    lum = round(lum * (1.1 if quick else 1.0), 4)
    db.execute("INSERT INTO points(t, worker_id, job_id, lumens, kind, note) VALUES(?,?,?,?,?,?)",
               (time.time(), w["id"], job_id, lum, "grant", "quick-return bonus" if quick else None))
    S.event(db, "points", job_id, w["id"], detail={"lumens": lum})
    return lum


def after_accept(S, results, job_id, policy):
    """called after a completion was accepted (outside the accept transaction): settle a re-run's comparison, or draw
    the job for a spot check, then settle points"""
    with S.tx() as db:
        j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if j is None or j["state"] != "accepted": return None
        orig = db.execute("SELECT * FROM jobs WHERE id=?", (j["verify_of"],)).fetchone() if j["verify_of"] else None
        if is_score(json.loads(j["params"])): return None       # scored by server/score.py (scorer thread / admin score)
        if orig is None:
            w = worker(db, j["accepted_by"])
            m = jobs.get(j["type"])
            if w is not None and not w["verifier"] and hasattr(m, "compare") and j["verify_state"] is None:
                n_before = db.execute("SELECT COUNT(*) FROM jobs WHERE accepted_by=? AND type=? AND verify_of IS NULL AND "
                                      "(accepted_at<? OR (accepted_at=? AND id<?))",
                                      (w["id"], j["type"], j["accepted_at"], j["accepted_at"], job_id)).fetchone()[0]
                if n_before < policy.first or _draw(policy.rate):
                    enqueue_rerun(S, db, j, why="first" if n_before < policy.first else "drawn")
            settle(S, db, job_id)
            return None
    # a re-run came back: compare outside the transaction (reads result files)
    if orig["state"] != "accepted":
        with S.tx() as db:
            db.execute("UPDATE jobs SET verify_state=NULL WHERE id=? AND verify_state='pending'", (orig["id"],))
            S.event(db, "spot_moot", orig["id"], detail={"rerun": job_id, "orig_state": orig["state"]})
        return None
    return compare_and_record(S, results, orig, j, policy)


def enqueue_rerun(S, db, j, why="drawn", mark=True):
    """the same job for verifier workers only; mark: the original's verify_state becomes pending (spot checks; a
    score-mode anchor leaves it alone, its scorer decides)"""
    rid = S.enqueue(db, j["type"], j["version"], json.loads(j["params"]), json.loads(j["inputs"]), json.loads(j["requires"]),
                    j["input_bytes"], level=None, ukey=None, priority=j["priority"] + 10, max_attempts=j["max_attempts"],
                    needs="verifier", verify_of=j["id"], exclude_worker=j["accepted_by"])
    if mark: db.execute("UPDATE jobs SET verify_state='pending', verify_detail=NULL WHERE id=?", (j["id"],))
    S.event(db, "spot_draw", j["id"], j["accepted_by"], detail={"rerun": rid, "why": why})
    return rid


def compare_and_record(S, results, orig, rerun, policy):
    m = jobs.get(orig["type"])
    try:
        ok, detail = m.compare(json.loads(orig["params"]), os.path.join(results, str(orig["id"])),
                               os.path.join(results, str(rerun["id"])), policy.tol.get(orig["type"]))
    except Exception as e:
        ok, detail = None, {"error": f"compare failed: {e!r}"[:500]}
    with S.tx() as db:
        st = "error" if ok is None else "pass" if ok else "fail"
        db.execute("UPDATE jobs SET verify_state=?, verify_detail=? WHERE id=?", (st, json.dumps(detail, default=str)[:20000], orig["id"]))
        S.event(db, f"spot_{st}", orig["id"], orig["accepted_by"], detail={"rerun": rerun["id"], **{k: v for k, v in detail.items() if k != "ratios"}})
        if ok is False:
            quarantine(S, db, results, orig["accepted_by"], f"spot check of job {orig['id']} failed (re-run {rerun['id']})")
        else:
            settle(S, db, orig["id"])
    return st


def invalidate(S, db, results, j, why, exclude=None):
    """an accepted result we no longer trust: points revoked, files moved aside, the job queued again"""
    if j["accepted_by"]:
        n = net_points(db, j["id"], j["accepted_by"])
        if n > 0:
            db.execute("INSERT INTO points(t, worker_id, job_id, lumens, kind, note) VALUES(?,?,?,?,?,?)",
                       (time.time(), j["accepted_by"], j["id"], -n, "revoke", why[:200]))
    d = os.path.join(results, str(j["id"])) if results else None
    if d and os.path.exists(d): os.replace(d, d + f".invalid-{int(time.time())}")
    db.execute("UPDATE scores SET state='void' WHERE job_id=? AND state<>'fail'", (j["id"],))     # no longer a reference
    db.execute("UPDATE jobs SET state='queued', attempts=0, accepted_lease=NULL, accepted_by=NULL, accepted_at=NULL, "
               "result=NULL, files=NULL, check_state=NULL, check_detail=NULL, verify_state=NULL, exclude_worker=?, "
               "error=?, requeued=?, updated=? WHERE id=?", (exclude, why[:2000], time.time(), time.time(), j["id"]))
    for r in db.execute("SELECT id FROM jobs WHERE verify_of=? AND state IN ('queued','leased')", (j["id"],)).fetchall():
        db.execute("UPDATE leases SET state='cancelled', ended=? WHERE job_id=? AND state='active'", (time.time(), r["id"]))
        db.execute("UPDATE jobs SET state='cancelled', lease_id=NULL, lease_owner=NULL, lease_expires=NULL WHERE id=?", (r["id"],))
    S.event(db, "invalidate", j["id"], detail={"why": why[:300]})


def quarantine(S, db, results, wid, reason):
    """no more leases for this worker; its active leases cancelled; its unconfirmed accepted results invalidated"""
    w = worker(db, wid)
    if w is None: return 0
    db.execute("UPDATE workers SET quarantined=1, quarantine_reason=? WHERE id=?", (reason[:500], wid))
    now = time.time()
    # its spool dirs: Queue.sweep_spool (the reaper)
    for l in db.execute("SELECT * FROM leases WHERE worker_id=? AND state='active'", (wid,)).fetchall():
        db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (now, l["id"]))
        db.execute("UPDATE jobs SET state='queued', lease_id=NULL, lease_owner=NULL, lease_expires=NULL, exclude_worker=?, "
                   "updated=? WHERE id=? AND lease_id=?", (wid, now, l["job_id"], l["id"]))
    n = 0
    for j in db.execute("SELECT * FROM jobs WHERE accepted_by=? AND state='accepted' AND verify_of IS NULL AND "
                        "(verify_state IS NULL OR verify_state<>'pass')", (wid,)).fetchall():
        if is_score(json.loads(j["params"])) and j["check_state"] == "pass": continue     # proven by its own codes
        invalidate(S, db, results, j, f"donor {wid} quarantined: {reason}", exclude=wid); n += 1
    S.event(db, "quarantine", worker_id=wid, detail={"reason": reason[:300], "invalidated": n})
    return n


def standings(S):
    """per worker: lumens, verified jobs, name (the leaderboard; names are what donors registered with)"""
    net = {}
    for r in S.q("SELECT worker_id, job_id, SUM(lumens) n FROM points GROUP BY worker_id, job_id"):
        if r["n"] > 1e-9: net.setdefault(r["worker_id"], []).append(r["n"])
    rows = [{"id": w["id"], "name": w["name"], "quarantined": w["quarantined"], "verifier": w["verifier"],
             "lumens": round(sum(net.get(w["id"], [])), 4), "jobs": len(net.get(w["id"], []))}
            for w in S.q("SELECT * FROM workers")]
    return sorted(rows, key=lambda r: (-r["lumens"], r["id"]))
