# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Verify quant results by score (docs/VERIFY.md): a quant job whose params say "verify": "score" is accepted when its
codes are valid for the declared format and K and their Hessian-weighted error, recomputed here from the codes with the
exact decode against OUR bf16 weights and OUR input Hessians, lies in the band expected for that tensor at that K.
Nothing the worker reports is trusted (its proxy.jsonl is informational). Numpy only (the server's CPU, or admin.py
score on any box with the data root).

Per tensor:
  rel = tr(dW H dW^T) / tr(W H W^T)          dW = decode(codes) - W, H the layer's input Hessian (as checker/check.py)
The band is one-sided (a lower error is never a cheat: the server computes it, and finding good codes is the work):
  log(rel) <= log(center) + tol
with center / tol from earlier scored rows of the same tensor in the same SERIES (model, revision, arch, codebook,
settings without seed_offset: the ladder's levels differ only in K), best basis first:
  repeat   rows at the same K (honest repeats, best-of-N copies, the anchor re-run): center = their minimum
           (trusted rows) / their upper median (every passed row, #12)
  interp   rows on both sides: linear in log(rel) between the nearest lower and upper K
  extrap   rows on one side: the nearest, moved along the codebook's rate-distortion slope (-d log rel / dK = SLOPE)
with tol = TOL[basis] (extrap: + TOL["extrap_per_bit"] x |dK|; FAMILY: k_proj / v_proj's own table) + Z x NOISE /
sqrt(n), the honest seed noise of a tensor of n = in x out weights. Two bounds when trusted rows
exist (our verifiers / imported references): one from every passed row, one from trusted rows only (so a chain of
slightly-worse donor results cannot drift the curve). The trusted one is the authority: failing it fails; failing only
the all bound is suspect (no points, an anchor re-run at that K decides) or, with trusted rows at that very K, a pass
(#12: one better-than-honest result must not fail every honest one after it). A tensor with no row at all is
provisional: the job is re-run once on a verifier (an anchor) and decided against it.

Also per tensor: the codes' sha256 (trellis + suh + svh). Codes identical to a published / imported reference ->
replay, fail. Identical to an earlier donor's row of the same tensor and K -> duplicate: accepted (it is a valid result)
but earns nothing (the first submission of given codes earns).

Re-checks: every new passed row re-evaluates the passed rows of its tensor that had only a wider basis (extrap or
provisional); a row now out of its band fails (the donor is quarantined, its unconfirmed results requeued)."""
import hashlib, json, math, os, time
import numpy as np

import jobs
from common.util import stamp
from jobs import quant_common as Q
from checker import exl3dec
from checker.check import ST, hessian, validity

# tolerances in log rel: a base per basis (the curve's own scatter, 27B levels) + Z x the honest seed noise of the
# tensor, NOISE / sqrt(in x out) (a quant's error is a sum over many independent tiles); docs/VERIFY.md
TOL = {"repeat": 0.02, "interp": 0.04, "extrap": 0.04, "extrap_per_bit": 0.03}
# k_proj / v_proj scatter more around the curve (27B levels: residuals ~5x the others, extrapolation SD up to ~6 % / bit)
FAMILY = {f: {"repeat": 0.02, "interp": 0.08, "extrap": 0.08, "extrap_per_bit": 0.12} for f in ("k_proj", "v_proj")}
# seed noise SD x sqrt(n): the layer-input group (lighter damping) ~5x the rest (0.8B: max 26 vs 6.3; TESTS.md sec. 10)
NOISE = {f: 30.0 for f in ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")}
NOISE_DEFAULT, Z = 8.0, 4.0
SLOPE = {"mul1": 1.37, "i8": 1.37, "i8c": 1.37, "i4": 1.37}   # -d log rel / dK; 27B i8 levels: 1.358-1.370 (2 ln 2 = 1.386)

def is_score(p): return isinstance(p, dict) and p.get("verify") == "score"


def series(p):
    """the curve a result is judged on: everything that shapes the quantizer's error except K (and the seed)"""
    s = {k: v for k, v in p["settings"].items() if k != "seed_offset"}
    return stamp({"model": p["model"], "revision": p["revision"], "arch": p["arch"], "codebook": p["codebook"], "settings": s})


def group(p, jtype):
    """one job's identity across best-of-N copies (seed_offset differs): series, K and the layer / head"""
    return f"{series(p)}:K{float(p['K'])}:{p.get('layer', 'head') if jtype != 'quant_tensor' else str(p['layer']) + '/' + str(p['group'])}"


def family(key): return key.rsplit(".", 1)[-1]


def tol(key, basis, n=None):
    """the tolerance of one basis for a tensor of n = in x out weights (the noise term once, not per bit)"""
    t = FAMILY.get(family(key), TOL)[basis]
    if n and basis != "extrap_per_bit": t += Z * NOISE.get(family(key), NOISE_DEFAULT) / math.sqrt(n)
    return t


# ------------------------------------------------------------------ the band
def bound(rows, K, codebook, key, n=None, center="min"):
    """(log center, tol, basis, refs) from scored rows [(K, rel)] of one tensor of n weights, or None. center: how the
    rows at one K are summarised, "min" (trusted rows: our own runs) or "median" (every passed row: the upper median,
    so one result better than the honest quantizer does not move the band for everyone else, #12)"""
    if not rows: return None
    s = SLOPE.get(codebook, 1.37)
    at = {}
    for k, r in rows:
        if r and r > 0: at.setdefault(k, []).append(r)
    if not at: return None
    best = {k: min(v) if center == "min" else sorted(v)[len(v) // 2] for k, v in at.items()}
    if K in best: return math.log(best[K]), tol(key, "repeat", n), "repeat", len(best)
    lo = [k for k in best if k < K]; hi = [k for k in best if k > K]
    if lo and hi:
        a, b = max(lo), min(hi); la, lb = math.log(best[a]), math.log(best[b])
        return la + (lb - la) * (K - a) / (b - a), tol(key, "interp", n), "interp", len(best)
    k0 = max(lo) if lo else min(hi)
    return math.log(best[k0]) - s * (K - k0), tol(key, "extrap", n) + tol(key, "extrap_per_bit") * abs(K - k0), "extrap", len(best)


def judge(rel, b):
    """(ok, margin) for one rel against one bound: margin = log(rel) - (center + tol); ok when <= 0"""
    if b is None: return None, None
    if not (rel and rel > 0 and math.isfinite(rel)): return False, math.inf
    mg = math.log(rel) - (b[0] + b[1])
    return mg <= 0, mg


def ref_rows(db, ser, key, exclude_job=None, trusted_only=False):
    q = "SELECT K, rel FROM scores WHERE series=? AND key=? AND state='pass'"
    a = [ser, key]
    if exclude_job is not None: q += " AND (job_id IS NULL OR job_id<>?)"; a.append(exclude_job)
    if trusted_only: q += " AND kind='trusted'"
    return [(float(r[0]), float(r[1])) for r in db.execute(q, a).fetchall()]


def decide(db, ser, key, K, codebook, rel, n=None, exclude_job=None):
    """the verdict of one tensor's rel: (state pass|fail|provisional|suspect, basis, bound value, detail).
    Two bounds: "all" (every passed row, upper median per K) and "trusted" (our rows only, minimum per K). Without a
    trusted bound the all bound decides. With one, the trusted bound is the authority (#12): failing it fails; passing
    it while failing the all bound is no fail: a pass when the trusted rows are at this very K (repeat basis), else
    "suspect" (no points; an anchor re-run at this K decides). A rel more than the repeat tolerance below the trusted
    center is marked outlier_better (an event for the operator: better than our own quantizer)."""
    ba = bound(ref_rows(db, ser, key, exclude_job), K, codebook, key, n, center="median")
    bt = bound(ref_rows(db, ser, key, exclude_job, trusted_only=True), K, codebook, key, n)
    if ba is None and bt is None: return "provisional", None, None, {}
    det, ok = {}, {}
    for name, b in (("all", ba), ("trusted", bt)):
        if b is None: continue
        o, mg = judge(rel, b)
        det[name] = {"basis": b[2], "center": math.exp(b[0]), "tol": round(b[1], 4), "margin": round(mg, 4) if math.isfinite(mg) else "inf", "refs": b[3]}
        ok[name] = o
    if bt is None: st = "pass" if ok["all"] else "fail"
    elif not ok["trusted"]: st = "fail"
    elif ok.get("all", True): st = "pass"
    else:
        det["all_only_fail"] = True
        st = "pass" if bt[2] == "repeat" else "suspect"
    if bt is not None and rel and rel > 0 and math.log(rel) < bt[0] - tol(key, "repeat", n): det["outlier_better"] = True
    b = ba or bt
    return st, b[2], math.exp(b[0] + b[1]), det


# ------------------------------------------------------------------ the decode
def codes_sha(t):
    h = hashlib.sha256()
    for x in ("trellis", "suh", "svh"): h.update(np.ascontiguousarray(t[x]).tobytes())
    return h.hexdigest()


def den_of(S, model_key, key, Wt, H):
    """tr(W H W^T): the same for every result of the tensor, computed once (half the GEMM work of a score)"""
    r = S.q("SELECT den FROM score_den WHERE model=? AND key=?", (model_key, key))
    if r: return float(r[0][0])
    den = float(np.einsum("ij,ij->", Wt, H @ Wt, dtype=np.float64))
    with S.tx() as db: db.execute("INSERT OR IGNORE INTO score_den(model, key, den) VALUES(?,?,?)", (model_key, key, den))
    return den


def score_files(S, jtype, p, res_dir, data_path, log=None):
    """decode every tensor of a result and compute rel against our weights / Hessians -> (rows, error)"""
    m = jobs.get(jtype); plan = m.plan(p)
    shard = os.path.join(res_dir, m.shard(p))
    wf = data_path(f"{Q.root(p)}/weights/{m.wfile(p)}")
    try: Q.check_shard(p, plan, shard, wf)
    except Exception as e: return None, f"structure: {e}"
    sh, wts = ST(shard), ST(wf)
    mk = f"{p['model']}@{p['revision']}"
    out = []
    for hname, keys in plan:
        H = None
        for key in keys:
            t0 = time.time()
            t = {x: sh.get(f"{key}.{x}") for x in ("trellis", "suh", "svh", "mul1")}
            bad = validity(key, t, float(p["K"]))
            if bad: return None, f"format: {bad}"
            Wq, cols = exl3dec.decode(t, float(p["K"]), p["codebook"])
            Wt = np.ascontiguousarray(wts.get(f"{key}.weight").T)       # checkpoint (out, in) -> [k, n]
            if H is None: H = hessian(data_path(f"{Q.root(p)}/hessians/{hname}.safetensors"))
            den = den_of(S, mk, key, Wt, H)
            E = Wq - Wt; del Wq
            num = float(np.einsum("ij,ij->", E, H @ E, dtype=np.float64)); del E
            rel = num / den if den > 0 else math.inf
            if not math.isfinite(rel): return None, f"format: {key}: decoded weight not finite"
            out.append({"key": key, "rel": rel, "num": num, "den": den, "sha": codes_sha(t), "n": int(Wt.size),
                        "s": round(time.time() - t0, 2)})
            if log: log(f"  {key}: rel {rel:.4e} ({time.time() - t0:.1f}s)")
        del H
    return out, None


# ------------------------------------------------------------------ jobs
def worker_kind(w):
    return "trusted" if w is not None and w["verifier"] else "donor"


def score_job(S, results, data_path, job_id, policy=None, log=None):
    """score one accepted score-mode quant result; store rows, set check_state, then act (points, anchor, quarantine)"""
    from server import verify as V
    j = S.q("SELECT * FROM jobs WHERE id=?", (job_id,))[0]
    p = json.loads(j["params"]); res_dir = os.path.join(results, str(job_id))
    w = S.q("SELECT * FROM workers WHERE id=?", (j["accepted_by"],)) if j["accepted_by"] else []
    w = w[0] if w else None
    kind = worker_kind(w)
    t0 = time.time()
    rows, err = score_files(S, j["type"], p, res_dir, data_path, log)
    ser, K, grp = series(p), float(p["K"]), group(p, j["type"])
    det = {"series": ser, "K": K, "group": grp, "kind": kind, "seconds": None, "tensors": []}
    with S.tx() as db:
        j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if j["state"] != "accepted" or j["check_state"] not in (None, "queued"): return j["check_state"]
        if err:
            st = "fail"; det["error"] = err
        else:
            states, dup = [], None
            for r in rows:
                same = db.execute("SELECT * FROM scores WHERE codes_sha=? AND key=? AND (job_id IS NULL OR job_id<>?) "
                                  "AND state<>'void' ORDER BY id LIMIT 1", (r["sha"], r["key"], job_id)).fetchone()
                tr = {"key": r["key"], "rel": r["rel"], "s": r["s"]}
                if same is not None and same["kind"] == "published":
                    s_, basis, bv, d = "fail", "replay", None, {"replay_of": same["note"] or same["id"]}
                else:
                    s_, basis, bv, d = decide(db, ser, r["key"], K, p["codebook"], r["rel"], r["n"], exclude_job=job_id)
                    if kind == "trusted":                   # our own hardware: a reference, whatever the band says
                        if s_ != "pass": d["trusted_outlier"] = s_
                        s_ = "pass"
                    elif same is not None and same["kind"] == "donor":    # the first submission of given codes earns
                        dup = same["job_id"]; d["duplicate_of"] = same["job_id"]
                tr.update({"state": s_, "basis": basis, "bound": bv, **d})
                det["tensors"].append(tr); states.append(s_)
                r.update(state=s_, basis=basis, bound=bv)
            st = agg(states)
            if dup is not None and st == "pass": det["duplicate_of"] = dup
            for r in rows:
                rs = r["state"] if st != "fail" else ("fail" if r["state"] == "fail" else "void")
                db.execute("INSERT INTO scores(job_id, grp, series, key, K, rel, num, den, codes_sha, worker_id, donor_id, kind, "
                           "state, basis, bound, t, n) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (job_id, grp, ser, r["key"], K, r["rel"], r["num"], r["den"], r["sha"], w["id"] if w else None,
                            w["donor_id"] if w else None, kind, rs, r["basis"], r["bound"], time.time(), r["n"]))
            det["total_rel"] = sum(r["num"] for r in rows) / sum(r["den"] for r in rows)
        det["seconds"] = round(time.time() - t0, 1)
        db.execute("UPDATE jobs SET check_state=?, check_detail=? WHERE id=?", (st, json.dumps(det, default=str)[:60000], job_id))
        S.event(db, f"score_{st}", job_id, j["accepted_by"], detail={k: det.get(k) for k in ("K", "kind", "total_rel", "error", "duplicate_of")})
        out = [{"key": t["key"], "rel": t["rel"], "trusted_center": t["trusted"]["center"]} for t in det["tensors"]
               if t.get("outlier_better")]
        if out and kind == "donor":                         # better than our own quantizer by more than the noise: look
            S.event(db, "score_outlier", job_id, j["accepted_by"], detail={"tensors": out[:8]})
        orig = db.execute("SELECT * FROM jobs WHERE id=?", (j["verify_of"],)).fetchone() if j["verify_of"] else None
        if st == "pass" and orig is None: V.settle(S, db, job_id)
        elif st == "fail" and kind == "donor" and orig is None:
            V.quarantine(S, db, results, j["accepted_by"], f"score check of job {job_id} failed: " + (err or _worst(det)))
        elif st == "fail":
            S.event(db, "score_trusted_fail", job_id, j["accepted_by"], detail={"error": err})     # look: our own hardware
        elif st in ("provisional", "suspect"):              # no reference yet / out of the all band only: an anchor decides
            if not db.execute("SELECT 1 FROM jobs WHERE verify_of=? AND state IN ('queued','leased','accepted')", (job_id,)).fetchone():
                V.enqueue_rerun(S, db, j, why="anchor" if st == "provisional" else "suspect", mark=False)
    if st == "pass": reevaluate(S, results, ser, [r["key"] for r in rows or []])
    return st


def agg(states):
    """a job's state from its tensors': any fail fails, then provisional, then suspect"""
    for st in ("fail", "provisional", "suspect"):
        if st in states: return st
    return "pass"


def _worst(det):
    bad = [t for t in det.get("tensors", []) if t.get("state") == "fail"]
    if not bad: return "?"
    t = bad[0]
    if "replay_of" in t: return f"{t['key']}: codes of a published result"
    a = t.get("trusted") or t.get("all") or {}
    return f"{t['key']}: rel {t['rel']:.4e} vs {a.get('basis')} center {a.get('center', 0):.4e} (+{a.get('tol')} log), margin {a.get('margin')}"


def reevaluate(S, results, ser, keys):
    """after a new reference: decide provisional jobs of these tensors, and re-check passed rows that had a wider basis"""
    from server import verify as V
    with S.tx() as db:
        jids = set()
        for key in keys:
            for r in db.execute("SELECT * FROM scores WHERE series=? AND key=? AND kind='donor' AND "
                                "(state IN ('provisional','suspect') OR (state='pass' AND basis IN ('extrap','interp')))", (ser, key)).fetchall():
                jids.add(r["job_id"])
        for jid in sorted(x for x in jids if x is not None):
            j = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            if j is None or j["state"] != "accepted" or j["check_state"] not in ("provisional", "suspect", "pass"): continue
            p = json.loads(j["params"]); det = json.loads(j["check_detail"] or "{}")
            rows = db.execute("SELECT * FROM scores WHERE job_id=?", (jid,)).fetchall()
            states, changed = [], False
            for r in rows:
                if r["state"] not in ("provisional", "suspect", "pass"): states.append(r["state"]); continue
                if r["state"] == "pass" and r["basis"] == "repeat": states.append("pass"); continue
                s_, basis, bv, d = decide(db, ser, r["key"], float(r["K"]), p["codebook"], r["rel"], r["n"], exclude_job=jid)
                # still no reference, or out of the all band only while the trusted band holds and no anchor at this K
                # has come back: unchanged (a passed row is never downgraded to suspect)
                if s_ in ("provisional", "suspect"): states.append(r["state"]); continue
                if (s_, basis) != (r["state"], r["basis"]):
                    db.execute("UPDATE scores SET state=?, basis=?, bound=? WHERE id=?", (s_, basis, bv, r["id"])); changed = True
                    for t in det.get("tensors", []):
                        if t["key"] == r["key"]: t.update({"state": s_, "basis": basis, "bound": bv, **d, "rechecked": time.time()})
                states.append(s_)
            if not changed: continue
            st = agg(states)
            if st == j["check_state"]:
                db.execute("UPDATE jobs SET check_detail=? WHERE id=?", (json.dumps(det, default=str)[:60000], jid)); continue
            db.execute("UPDATE jobs SET check_state=?, check_detail=? WHERE id=?", (st, json.dumps(det, default=str)[:60000], jid))
            S.event(db, f"score_recheck_{st}", jid, j["accepted_by"])
            if st == "pass": V.settle(S, db, jid)
            elif st == "fail":
                db.execute("UPDATE scores SET state=CASE WHEN state='fail' THEN 'fail' ELSE 'void' END WHERE job_id=?", (jid,))
                w = V.worker(db, j["accepted_by"])
                if w is not None and not w["verifier"]:
                    V.quarantine(S, db, results, j["accepted_by"], f"score re-check of job {jid} failed: {_worst(det)}")


def pending(S, level=None):
    q = "SELECT id, params FROM jobs WHERE state='accepted' AND type IN ('quant_layer','quant_head','quant_tensor') AND check_state IS NULL"
    a = []
    if level: q += " AND level=?"; a.append(level)
    return [r["id"] for r in S.q(q + " ORDER BY accepted_at, id", a) if is_score(json.loads(r["params"]))]


def run_pending(S, results, data_path, level=None, log=None):
    """score every accepted score-mode quant result not scored yet (oldest first); returns {state: count}"""
    n = {}
    for jid in pending(S, level):
        try: st = score_job(S, results, data_path, jid, log=log)
        except Exception as e:
            st = "error"
            with S.tx() as db:
                db.execute("UPDATE jobs SET check_state='error', check_detail=? WHERE id=? AND check_state IS NULL",
                           (json.dumps({"error": f"{e!r}"[:500]}), jid))
                S.event(db, "score_error", jid, detail={"error": f"{e!r}"[:300]})
        n[st] = n.get(st, 0) + 1
        if log: log(f"job {jid}: {st}")
    return n


def import_reference(S, data_path, jtype, p, res_dir, kind="trusted", note=None, log=None):
    """our own result (an ourq output dir laid out as a job's result: the shard; proxy.jsonl not needed) as reference rows
    (kind trusted, state pass), or a released model's shard as a replay guard (kind published, state guard: its codes
    only; never a reference for the band, its settings are another series)"""
    rows, err = score_files(S, jtype, p, res_dir, data_path, log)
    if err: raise ValueError(err)
    ser, K = series(p), float(p["K"])
    with S.tx() as db:
        for r in rows:
            db.execute("INSERT INTO scores(job_id, grp, series, key, K, rel, num, den, codes_sha, kind, state, basis, t, note, n) "
                       "VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (group(p, jtype), ser, r["key"], K, r["rel"], r["num"], r["den"],
                                                                   r["sha"], kind, "pass" if kind == "trusted" else "guard",
                                                                   "import", time.time(), note, r["n"]))
    return rows


def best_of_group(S, rows):
    """export: among accepted copies of one job group (best of N), the passed one with the lowest total error"""
    ok = [r for r in rows if r["check_state"] == "pass"]
    if not ok: return None
    return min(ok, key=lambda r: (json.loads(r["check_detail"] or "{}").get("total_rel", math.inf), r["id"]))
