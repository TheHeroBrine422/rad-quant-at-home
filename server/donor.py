# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""What a donor may see and set about their own workers (stdlib only; server.py's /api/v0/me..., used by the donor
CLI client/rqah.py). Authenticated with any one worker token; the answer covers every worker of that worker's
donor (store.py: donors).

  GET  /api/v0/me                  profile, workers (caps as declared, flags, tier, running jobs, lumens), totals, queue
                                   depth (and how many queued jobs each worker could run)
  GET  /api/v0/me/jobs?limit=N&worker=ID&active=1     leases of the donor's workers, newest first, each with its
                                   verification state (VERIFICATION below) and lumens
  GET  /api/v0/me/points?limit=N   lumens per worker and type, the ledger (grants / revokes), what is still pending,
                                   the leaderboard rank (opted-in donors only)
  GET  /api/v0/me/events?after=ID&limit=N   the server's audit log for the donor's workers (allow-listed fields,
                                   scrubbed of paths / addresses / tokens)
  POST /api/v0/me/profile  {display_name?, leaderboard?}   the public leaderboard entry (opt-in)

The leaderboard credits by display name (board()): every opted-in donor showing the same name adds to one row. The
name is a label for credit only: trust, scoring, spot checks, quarantine and invalidation are per worker (token).
`POST /api/v0/me/link` was removed with this (410, LINK_GONE).

Nothing here reveals another donor's workers, any token or invite, job inputs or server paths."""
import json, re, time, unicodedata
import jobs
from server import sched
from server.verify import QUANT

QUANT_SET = set(QUANT)
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
LABEL_TYPES = {"quant_layer": "quantize a layer", "quant_tensor": "quantize part of a layer", "quant_head": "quantize the LM head", "eval_mix": "score a candidate mix",
               "golden_record": "record reference outputs", "noop": "test job"}
VERIFICATION = ("running", "verified", "accepted", "awaiting check", "spot-check pending", "failed check",
                "failed spot check", "voided", "superseded", "expired", "abandoned", "failed", "rejected", "cancelled")

# what must never reach a donor-facing answer or the public page
SCRUB = [(re.compile(r"\b(?:rq|ri)[A-Za-z0-9_-]{20,}"), "<token>"),
         (re.compile(r"(?i)bearer\s+\S+"), "Bearer <token>"),
         (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<hex>"),
         (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<ip>"),
         (re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){3,7}[0-9a-f]{1,4}\b|\b(?:[0-9a-f]{1,4}:)+:(?:[0-9a-f]{1,4})?\b"), "<ip>"),
         (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
         (re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://\S+"), "<url>"),
         (re.compile(r"(?<![\w.<])~?/(?:[\w.@+-]+/)*[\w.@+-]+/?"), "<path>")]


def scrub(s, n=300):
    s = str(s if s is not None else "")
    for rx, rep in SCRUB: s = rx.sub(rep, s)
    return s[:n]


# a display name: printable, no markup-ish or address-like content (the page escapes anyway; this keeps it clean)
NAME_BAD = re.compile(r"(?i)(?:\b(?:\d{1,3}\.){3}\d{1,3}\b|@|://|\b(?:rq|ri)[A-Za-z0-9_-]{16,}|[0-9a-f]{24,}|[<>&\"'`\\])")


def clean_display_name(s):
    """-> the name to store, or raises ValueError (1..32 printable characters, no addresses, tokens or markup)"""
    if not isinstance(s, str): raise ValueError("display name: a string")
    s = " ".join(unicodedata.normalize("NFC", s).split())
    if not 1 <= len(s) <= 32: raise ValueError("display name: 1..32 characters")
    if any(unicodedata.category(c)[0] == "C" for c in s): raise ValueError("display name: no control or format characters")
    if NAME_BAD.search(s): raise ValueError("display name: no addresses, e-mails, URLs, tokens or <>&\"'`\\ characters")
    return s


def public_name(s):
    """a stored display name as the public page may show it (None: hide); stored names were checked, older rows or
    operator edits may not have been"""
    try: return clean_display_name(s)
    except ValueError: return None


def label(jtype, params, level=None):
    """a short public description of a job: no paths, only validated identifiers"""
    p = params if isinstance(params, dict) else {}
    ident = lambda x: x if isinstance(x, str) and ID_RE.fullmatch(x) else "?"
    if jtype == "quant_layer": return f"{ident(p.get('level', level))} · layer {int(p['layer']) if isinstance(p.get('layer'), int) else '?'}"
    if jtype == "quant_tensor":
        n = lambda x: int(x) if isinstance(x, int) and not isinstance(x, bool) else "?"
        return f"{ident(p.get('level', level))} · layer {n(p.get('layer'))} part {n(p.get('group'))}"
    if jtype == "quant_head": return f"{ident(p.get('level', level))} · LM head"
    if jtype == "eval_mix": return f"search · mix on {ident(p.get('base'))}"
    if jtype == "golden_record": return f"golden · {ident(p.get('corpus'))} {ident(p.get('chunk'))}"
    if jtype == "noop": return "test job"
    return ident(jtype)


def donor_workers(S, w):
    if w["donor_id"] is None: return [w]
    return S.q("SELECT * FROM workers WHERE donor_id=? ORDER BY id", (w["donor_id"],))


def _donor(S, w):
    r = S.q("SELECT * FROM donors WHERE id=?", (w["donor_id"],)) if w["donor_id"] is not None else []
    return r[0] if r else None


def _in(ids): return "(" + ",".join(str(int(i)) for i in ids) + ")"


def net_by_job(S, wids):
    """{(worker, job): net lumens}"""
    return {(r["worker_id"], r["job_id"]): r["n"] for r in
            S.q(f"SELECT worker_id, job_id, SUM(lumens) n FROM points WHERE worker_id IN {_in(wids)} GROUP BY worker_id, job_id")}


def name_key(name):
    """the leaderboard's grouping key of a checked display name: case-insensitive"""
    return unicodedata.normalize("NFC", name.casefold())


def hidden_names(S):
    """keys of the names the operator hid (admin.py donor ID --hide on any donor showing that name)"""
    return {name_key(n) for n in (public_name(d["display_name"]) for d in S.q("SELECT display_name FROM donors WHERE hidden=1")) if n}


def board(S):
    """the public leaderboard, one row per public display name (case-insensitive): the lumens of every non-verifier,
    not quarantined worker whose donor opted in under that name; verified jobs = (worker, job) pairs with net lumens
    > 0. A quarantined worker's lumens count nowhere, and nothing else changes for the name. A name the operator hid
    on any donor is off the board (its donors count with the others). Rows show the name's most common spelling
    among their donors. -> (ranked rows, (others' count of donors, others' lumens))"""
    ws = {r["id"]: r for r in S.q("SELECT id, donor_id, verifier, quarantined, last_seen FROM workers")}
    ds = {r["id"]: r for r in S.q("SELECT * FROM donors")}
    per = {}
    for r in S.q("SELECT worker_id, job_id, SUM(lumens) n FROM points GROUP BY worker_id, job_id"):
        w = ws.get(r["worker_id"])
        if w is None or w["verifier"] or w["quarantined"] or r["n"] <= 1e-9: continue
        d = per.setdefault(w["donor_id"] if w["donor_id"] is not None else -w["id"], {"lumens": 0.0, "jobs": 0})
        d["lumens"] += r["n"]; d["jobs"] += 1
    last = {}
    for w in ws.values():
        if w["donor_id"] is not None and not w["quarantined"]: last[w["donor_id"]] = max(last.get(w["donor_id"], 0), w["last_seen"] or 0)
    hid = hidden_names(S)
    named, others = {}, [0, 0.0]
    for did, v in per.items():
        d = ds.get(did); name = public_name(d["display_name"]) if d else None
        key = name_key(name) if name else None
        if d and d["leaderboard"] and not d["hidden"] and key and key not in hid:
            g = named.setdefault(key, {"donors": [], "spell": {}, "lumens": 0.0, "jobs": 0, "last_seen": None})
            g["donors"].append(did); g["spell"][name] = g["spell"].get(name, 0) + 1
            g["lumens"] += v["lumens"]; g["jobs"] += v["jobs"]
            g["last_seen"] = max(g["last_seen"] or 0, last.get(did) or 0) or None
        else:
            others[0] += 1; others[1] += v["lumens"]
    rows = []
    for key, g in named.items():
        name = min(g["spell"], key=lambda n: (-g["spell"][n], n))       # most donors; ties: the first in code-point order
        rows.append({"name": name, "key": key, "donors": sorted(g["donors"]), "lumens": round(g["lumens"], 2), "jobs": g["jobs"],
                     "last_seen": g["last_seen"]})
    rows.sort(key=lambda r: (-r["lumens"], -r["jobs"], r["key"]))
    for i, r in enumerate(rows): r["rank"] = i + 1
    return rows, (others[0], round(others[1], 2))


def queue_depth(S):
    out = {}
    for r in S.q("SELECT type, state, (needs='verifier') rerun, COUNT(*) n FROM jobs WHERE state IN ('queued','leased') "
                 "GROUP BY type, state, rerun"):
        t = out.setdefault(r["type"], {"queued": 0, "running": 0, "reruns": 0})
        if r["rerun"]: t["reruns"] += r["n"]
        else: t["queued" if r["state"] == "queued" else "running"] += r["n"]
    return out


def verification(j, l, net, wr):
    """one lease's state as its donor sees it (VERIFICATION)"""
    if l["state"] == "active": return "running"
    if l["state"] != "done": return l["state"]
    if j is None or j["accepted_lease"] != l["id"]: return "voided"
    if net > 1e-9: return "verified"
    if wr is not None and wr["quarantined"]: return "voided"
    if j["verify_state"] == "pending": return "spot-check pending"
    if j["verify_state"] == "fail": return "failed spot check"
    if j["check_state"] == "fail": return "failed check"
    if j["type"] in QUANT_SET and j["check_state"] != "pass": return "awaiting check"
    return "accepted"


def budget_view(caps):
    """a worker's declared data budget as its donor sees it (numbers only), None if it declared none"""
    b = caps.get("budget") if isinstance(caps, dict) else None
    if not isinstance(b, dict): return None
    num = lambda v: int(v) if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    return {"down": num(b.get("down")), "up": num(b.get("up")),
            "spent": [d for d in (b.get("spent") or []) if d in ("down", "up")], "reset": num(b.get("reset"))}


def budget_paused(caps):
    """the worker declared a spent data budget (upload: nothing new until the reset; download: only jobs whose inputs
    it caches)"""
    b = budget_view(caps)
    return bool(b and b["spent"])


def _expected(j):
    try: return round(float(jobs.get(j["type"]).lumens(json.loads(j["params"]))), 4)
    except Exception: return None


def tier_of(S, x):
    """(sched.tier of worker row x, its passed quant results or None when it was never on probation)"""
    if not x["probation"] or x["verifier"]: return "full", None
    n = sched.passed_quant(S.q, x["id"])
    return sched.tier(x, n, *_thresholds(S)), n


def _thresholds(S): return getattr(S, "probation", (sched.PROBATION_PASS, sched.EVAL_UNLOCK))


def me(S, w):
    ws = donor_workers(S, w); wids = [x["id"] for x in ws]; d = _donor(S, w); pp, eu = _thresholds(S)
    net = net_by_job(S, wids)
    run = {}
    for r in S.q(f"SELECT l.worker_id, j.id, j.type FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.state='active' "
                 f"AND l.worker_id IN {_in(wids)} ORDER BY l.issued"):
        run.setdefault(r["worker_id"], []).append({"job": r["id"], "type": r["type"]})
    pend = S.q(f"SELECT * FROM jobs WHERE state='accepted' AND accepted_by IN {_in(wids)} AND verify_of IS NULL")
    pend = [j for j in pend if net.get((j["accepted_by"], j["id"]), 0) <= 1e-9 and not j["verify_state"] == "fail"
            and not j["check_state"] == "fail"]
    queued = S.q("SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC, id LIMIT 5000")
    workers = []
    for x in ws:
        caps = json.loads(x["caps"] or "{}")
        lum = [v for (wi, _), v in net.items() if wi == x["id"] and v > 1e-9]
        tr, passed = tier_of(S, x)
        f = sched.job_filter(x, caps, tier_=tr) if caps.get("job_types") else None
        workers.append({"id": x["id"], "name": x["name"], "registered": x["created"], "last_seen": x["last_seen"],
                        "gpu_name": scrub(caps.get("gpu_name", ""), 64), "cards": caps.get("cards"), "vram_gb": caps.get("vram_gb"),
                        "ram_gb": caps.get("ram_gb"), "disk_gb": caps.get("disk_gb"), "job_types": caps.get("job_types", []),
                        "image": scrub(caps.get("image", ""), 128), "budget": budget_view(caps),
                        "cached_inputs": (caps.get("cache") or {}).get("n") if isinstance(caps.get("cache"), dict) else None,
                        "trusted": bool(x["trusted"]), "verifier": bool(x["verifier"]), "quarantined": bool(x["quarantined"]),
                        "disabled": bool(x["disabled"]),
                        "tier": {"tier": tr, "passed": passed, "probation_pass": pp, "eval_unlock": eu} if x["probation"] else {"tier": tr},
                        "quarantine_reason": scrub(x["quarantine_reason"]) if x["quarantined"] else None,
                        "running": run.get(x["id"], []), "lumens": round(sum(lum), 4), "verified_jobs": len(lum),
                        "could_run": sum(1 for j in queued if f(j) is not None) if f else None})
    rows, _ = board(S)
    mine = [r for r in rows if d is not None and d["id"] in r["donors"]]
    name = public_name(d["display_name"]) if d else None
    return {"server_time": time.time(), "worker": {"id": w["id"], "name": w["name"]},
            "donor": {"id": d["id"] if d else None, "display_name": d["display_name"] if d else None,
                      "leaderboard": bool(d and d["leaderboard"]), "hidden": bool(d and d["hidden"]),
                      "name_hidden": bool(name and name_key(name) in hidden_names(S)),
                      "rank": mine[0]["rank"] if mine else None, "on_board": len(rows),
                      "board_lumens": mine[0]["lumens"] if mine else None},
            "workers": workers, "lumens": round(sum(x["lumens"] for x in workers), 4),
            "verified_jobs": sum(x["verified_jobs"] for x in workers),
            "pending": {"jobs": len(pend), "lumens": round(sum(_expected(j) or 0 for j in pend), 4)},
            "queue": {"by_type": queue_depth(S)}}


def my_jobs(S, w, limit=50, worker=None, active=False):
    ws = {x["id"]: x for x in donor_workers(S, w)}
    wids = [worker] if worker in ws else list(ws)
    if worker is not None and worker not in ws: return {"jobs": [], "error": "not one of your workers"}
    net = net_by_job(S, wids)
    rows = S.q(f"SELECT l.id lid, l.worker_id, l.issued, l.ended, l.state lstate, l.note, j.* FROM leases l JOIN jobs j "
               f"ON j.id=l.job_id WHERE l.worker_id IN {_in(wids)}" + (" AND l.state='active'" if active else "") +
               " ORDER BY l.issued DESC LIMIT ?", (max(1, min(int(limit), 1000)),))
    out = []
    for r in rows:
        l = {"id": r["lid"], "state": r["lstate"]}
        n = net.get((r["worker_id"], r["id"]), 0)
        v = verification(r, l, n, ws[r["worker_id"]])
        out.append({"job": r["id"], "type": r["type"], "label": label(r["type"], json.loads(r["params"]), r["level"]),
                    "worker": r["worker_id"], "worker_name": ws[r["worker_id"]]["name"], "lease": r["lid"][:8],
                    "issued": r["issued"], "ended": r["ended"], "lease_state": r["lstate"], "job_state": r["state"],
                    "verification": v, "lumens": round(n, 4) if v != "running" else 0,
                    "lumens_expected": _expected(r) if v in ("running", "accepted", "awaiting check", "spot-check pending") else None,
                    "note": scrub(r["note"]) if r["note"] else None, "rerun": bool(r["verify_of"])})
    return {"jobs": out}


def my_points(S, w, limit=50):
    m = me(S, w); wids = [x["id"] for x in m["workers"]]
    by_type = {}
    for r in S.q(f"SELECT j.type, p.job_id, p.worker_id, SUM(p.lumens) n FROM points p JOIN jobs j ON j.id=p.job_id "
                 f"WHERE p.worker_id IN {_in(wids)} GROUP BY p.worker_id, p.job_id"):
        if r["n"] > 1e-9:
            t = by_type.setdefault(r["type"], {"lumens": 0.0, "jobs": 0}); t["lumens"] = round(t["lumens"] + r["n"], 4); t["jobs"] += 1
    led = [{"t": r["t"], "job": r["job_id"], "type": r["type"], "label": label(r["type"], json.loads(r["params"]), r["level"]),
            "worker": r["worker_id"], "lumens": r["lumens"], "kind": r["kind"], "note": scrub(r["note"]) if r["note"] else None}
           for r in S.q(f"SELECT p.*, j.type, j.params, j.level FROM points p JOIN jobs j ON j.id=p.job_id "
                        f"WHERE p.worker_id IN {_in(wids)} ORDER BY p.id DESC LIMIT ?", (max(1, min(int(limit), 1000)),))]
    return {"lumens": m["lumens"], "verified_jobs": m["verified_jobs"], "pending": m["pending"],
            "rank": m["donor"]["rank"], "on_board": m["donor"]["on_board"], "leaderboard": m["donor"]["leaderboard"],
            "by_worker": [{"id": x["id"], "name": x["name"], "lumens": x["lumens"], "verified_jobs": x["verified_jobs"]}
                          for x in m["workers"]], "by_type": by_type, "ledger": led}


EVENT_KEYS = ("error", "lumens", "reason", "why", "pass", "files", "cards", "free_cards", "invalidated", "orig_state")


def my_events(S, w, after=0, limit=100):
    wids = [x["id"] for x in donor_workers(S, w)]
    names = {x["id"]: x["name"] for x in donor_workers(S, w)}
    # their workers' events, plus checker verdicts / invalidations of jobs one of their leases completed
    rows = S.q(f"SELECT * FROM events WHERE id>? AND (worker_id IN {_in(wids)} OR (worker_id IS NULL AND kind IN "
               f"('check','invalidate') AND job_id IN (SELECT job_id FROM leases WHERE state='done' AND worker_id IN {_in(wids)}))) "
               "ORDER BY id LIMIT ?", (int(after), max(1, min(int(limit), 1000))))
    out = []
    for r in rows:
        d = json.loads(r["detail"]) if r["detail"] else {}
        d = {k: (scrub(v) if isinstance(v, str) else v if isinstance(v, (int, float, bool)) or v is None else scrub(json.dumps(v)))
             for k, v in (d.items() if isinstance(d, dict) else ()) if k in EVENT_KEYS}
        out.append({"id": r["id"], "t": r["t"], "kind": r["kind"], "job": r["job_id"], "worker": r["worker_id"],
                    "worker_name": names.get(r["worker_id"]), "lease": (r["lease_id"] or "")[:8] or None, "detail": d})
    return {"events": out, "last": out[-1]["id"] if out else int(after)}


def set_profile(S, w, b):
    sets, args = [], []
    if "display_name" in b:
        sets.append("display_name=?"); args.append(clean_display_name(b["display_name"]))
    if "leaderboard" in b:
        if not isinstance(b["leaderboard"], bool): raise ValueError("leaderboard: true or false")
        sets.append("leaderboard=?"); args.append(int(b["leaderboard"]))
    if not sets: raise ValueError("nothing to set (display_name, leaderboard)")
    with S.tx() as db:
        did = w["donor_id"]
        d = db.execute("SELECT hidden, display_name FROM donors WHERE id=?", (did,)).fetchone() if did is not None else None
        if d is not None and d["hidden"] and "display_name" in b and args[0] != d["display_name"]:
            # hiding a donor hides its name for everyone showing it (board): a hidden donor may not move that to another name
            raise ValueError("display name: this donor's name was hidden by the operator and cannot be changed (ask the operator)")
        if did is None:
            did = S.new_donor(db, w["name"]); db.execute("UPDATE workers SET donor_id=? WHERE id=?", (did, w["id"]))
        db.execute(f"UPDATE donors SET {', '.join(sets)} WHERE id=?", (*args, did))
        S.event(db, "profile", worker_id=w["id"], detail={k: b[k] for k in ("display_name", "leaderboard") if k in b})
    return {"ok": True, "donor": did}


LINK_GONE = ("rqah link was removed: the leaderboard credits by display name, so use the same name on every machine "
             "(rqah config set --display-name NAME --leaderboard on)")
