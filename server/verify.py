# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verification, quarantine and points ("lumens") -- stdlib only, shared by server.py and admin.py.

Spot checks: when a donor's result is accepted, it is drawn for a re-run with probability `rate` (default 4%: the
3-5% band; eval_mix: --eval-spot-rate, default 8%), and always for a donor's first `first` accepted job of each type.
0.5 (#4): an eval_mix spot check re-scores a few random documents of the golden (Policy.docs), not the whole golden,
and its rows must match the donor's rows of those documents. An eval_mix result far better than the search predicted
(check_state 'suspect'), or one the search adopts (server/batch.py adopt), is re-run in full on a verifier
(trust_state); a suspect one earns nothing until that passes. A quarantine keeps only the eval_mix results a full
re-run confirmed. A drawn job's verify_state is
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
no replication spot checks; the scorer decides check_state (pass / fail / provisional / suspect), a provisional or
suspect one is re-run once on a verifier as an anchor (verify_state untouched), a pass settles points (not for a duplicate of another
donor's codes), a fail quarantines. A quarantine leaves score-verified results alone: their quality is proven by
their own codes, whoever made them."""
import json, os, secrets, shutil, time
import jobs
from server.score import is_score

QUANT = jobs.QUANT


class Policy:
    """rate / first: the spot-check draw; rates: per-type rates instead of rate (eval_mix: --eval-spot-rate); docs: how
    many documents an eval_mix spot check re-scores; data: the data root (an eval_mix spot check reads the golden's
    document list; without it the whole job is re-run, as before 0.5)"""
    def __init__(self, rate=0.04, first=1, tol=None, rates=None, docs=3, data=None):
        self.rate, self.first, self.tol = rate, first, tol or {}
        self.rates, self.docs, self.data = rates or {}, docs, data


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
    if j["trust_state"] == "fail" or (j["check_state"] == "suspect" and j["trust_state"] != "pass"): return 0   # #4
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
            if w is not None and not w["verifier"] and j["check_state"] == "suspect" and j["trust_state"] is None:
                enqueue_trusted(S, db, j, "suspect")          # #4: far better than predicted: a full re-run first
            elif w is not None and not w["verifier"] and hasattr(m, "compare") and j["verify_state"] is None:
                n_before = db.execute("SELECT COUNT(*) FROM jobs WHERE accepted_by=? AND type=? AND verify_of IS NULL AND "
                                      "(accepted_at<? OR (accepted_at=? AND id<?))",
                                      (w["id"], j["type"], j["accepted_at"], j["accepted_at"], job_id)).fetchone()[0]
                if n_before < policy.first or _draw(policy.rates.get(j["type"], policy.rate)):
                    why = "first" if n_before < policy.first else "drawn"
                    sp = spot_params(m, j, policy)
                    enqueue_rerun(S, db, j, why=why, params=sp)
            settle(S, db, job_id)
            return None
    # a re-run came back: compare outside the transaction (reads result files)
    if orig["state"] != "accepted":
        with S.tx() as db:
            db.execute("UPDATE jobs SET verify_state=NULL WHERE id=? AND verify_state='pending'", (orig["id"],))
            db.execute("UPDATE jobs SET trust_state=NULL WHERE id=? AND trust_state='pending'", (orig["id"],))
            S.event(db, "spot_moot", orig["id"], detail={"rerun": job_id, "orig_state": orig["state"]})
        return None
    return compare_and_record(S, results, orig, j, policy)


def spot_params(m, j, policy):
    """an eval_mix spot check re-scores policy.docs random documents of the golden (#4); None = the whole job"""
    if not (hasattr(m, "spot_params") and policy.data and policy.docs): return None
    p = json.loads(j["params"])
    try: meta = json.load(open(os.path.join(policy.data, m.gfile(p, "kld.json"))))
    except (OSError, ValueError): return None
    return m.spot_params(p, meta, policy.docs, secrets.SystemRandom())


def enqueue_rerun(S, db, j, why="drawn", mark=True, params=None, state="verify_state"):
    """the same job (or params: a part of it) for verifier workers only; mark: the original's `state` becomes pending
    (verify_state: spot checks; trust_state: a full trusted re-run of an eval_mix result, #4; a score-mode anchor
    leaves both alone, its scorer decides)"""
    p = params or json.loads(j["params"])
    req = json.loads(j["requires"])
    if params is not None:
        try: req = jobs.requirements(jobs.get(j["type"]), p)
        except Exception: pass
    rid = S.enqueue(db, j["type"], j["version"], p, json.loads(j["inputs"]), req,
                    j["input_bytes"], level=None, ukey=None, priority=j["priority"] + 10, max_attempts=j["max_attempts"],
                    needs="verifier", verify_of=j["id"], exclude_worker=j["accepted_by"])
    if mark: db.execute(f"UPDATE jobs SET {state}='pending'{', verify_detail=NULL' if state == 'verify_state' else ''} WHERE id=?",
                        (j["id"],))
    S.event(db, "spot_draw" if state == "verify_state" else "trust_draw", j["id"], j["accepted_by"],
            detail={"rerun": rid, "why": why, **({"docs": [d[0] for d in p["docs"]]} if "docs" in p else {})})
    return rid


def enqueue_trusted(S, db, j, why):
    """#4: a full re-run of an eval_mix result on a verifier (the search adopts it, or it is far better than
    predicted); none if one is pending or passed, or a verifier made the result"""
    w = worker(db, j["accepted_by"])
    if j["state"] != "accepted" or j["trust_state"] in ("pending", "pass") or (w is not None and w["verifier"]): return None
    return enqueue_rerun(S, db, j, why=why, state="trust_state")


def compare_and_record(S, results, orig, rerun, policy):
    m = jobs.get(orig["type"])
    pb = json.loads(rerun["params"])
    # an eval_mix re-run of the whole job is the trusted re-run (#4, trust_state); one of a few documents a spot check
    full_trust = orig["type"] == "eval_mix" and "docs" not in pb and orig["trust_state"] == "pending"
    try:
        args = (json.loads(orig["params"]), os.path.join(results, str(orig["id"])), os.path.join(results, str(rerun["id"])),
                policy.tol.get(orig["type"]))
        ok, detail = m.compare(*args, pb=pb) if orig["type"] == "eval_mix" else m.compare(*args)
    except Exception as e:
        ok, detail = None, {"error": f"compare failed: {e!r}"[:500]}
    with S.tx() as db:
        st = "error" if ok is None else "pass" if ok else "fail"
        if full_trust:
            db.execute("UPDATE jobs SET trust_state=? WHERE id=?", (st, orig["id"]))
            if orig["verify_state"] == "pending" and not db.execute(
                    "SELECT 1 FROM jobs WHERE verify_of=? AND id<>? AND state IN ('queued','leased')", (orig["id"], rerun["id"])).fetchone():
                db.execute("UPDATE jobs SET verify_state=?, verify_detail=? WHERE id=?", (st, json.dumps(detail, default=str)[:20000], orig["id"]))
        else:
            db.execute("UPDATE jobs SET verify_state=?, verify_detail=? WHERE id=?", (st, json.dumps(detail, default=str)[:20000], orig["id"]))
        kind = "trust" if full_trust else "spot"
        S.event(db, f"{kind}_{st}", orig["id"], orig["accepted_by"], detail={"rerun": rerun["id"], **{k: v for k, v in detail.items() if k != "ratios"}})
        if ok is False:
            w = worker(db, orig["accepted_by"])
            quarantine(S, db, results, orig["accepted_by"], f"{'trusted re-run' if full_trust else 'spot check'} of job "
                       f"{orig['id']} failed (re-run {rerun['id']}); donor {w['donor_id'] if w else '?'}: admin.py exclude-donor "
                       "drops all its donations")
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
               "result=NULL, files=NULL, check_state=NULL, check_detail=NULL, verify_state=NULL, trust_state=NULL, "
               "exclude_worker=?, error=?, requeued=?, updated=? WHERE id=?", (exclude, why[:2000], time.time(), time.time(), j["id"]))
    for r in db.execute("SELECT id FROM jobs WHERE verify_of=? AND state IN ('queued','leased')", (j["id"],)).fetchall():
        db.execute("UPDATE leases SET state='cancelled', ended=? WHERE job_id=? AND state='active'", (time.time(), r["id"]))
        db.execute("UPDATE jobs SET state='cancelled', lease_id=NULL, lease_owner=NULL, lease_expires=NULL WHERE id=?", (r["id"],))
    S.event(db, "invalidate", j["id"], detail={"why": why[:300]})


def quarantine(S, db, results, wid, reason):
    """no more leases for this worker; its active leases cancelled; its unconfirmed accepted results invalidated"""
    w = worker(db, wid)
    if w is None: return 0
    # a probation worker (registered without an invite) that is quarantined never unlocks (probation 2)
    db.execute("UPDATE workers SET quarantined=1, quarantine_reason=?, probation=CASE WHEN probation>0 THEN 2 ELSE 0 END "
               "WHERE id=?", (reason[:500], wid))
    now = time.time()
    # its spool dirs: Queue.sweep_spool (the reaper)
    for l in db.execute("SELECT * FROM leases WHERE worker_id=? AND state='active'", (wid,)).fetchall():
        db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (now, l["id"]))
        db.execute("UPDATE jobs SET state='queued', lease_id=NULL, lease_owner=NULL, lease_expires=NULL, exclude_worker=?, "
                   "updated=? WHERE id=? AND lease_id=?", (wid, now, l["job_id"], l["id"]))
    n = 0
    for j in db.execute("SELECT * FROM jobs WHERE accepted_by=? AND state='accepted' AND verify_of IS NULL", (wid,)).fetchall():
        if is_score(json.loads(j["params"])) and j["check_state"] == "pass": continue     # proven by its own codes
        if j["type"] == "eval_mix":
            if j["trust_state"] == "pass": continue             # re-run in full; a spot check of a few documents is not proof
        elif j["verify_state"] == "pass": continue
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
