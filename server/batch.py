# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Batches of eval_mix candidates for the search (stdlib only; admin.py batch-* and server.py's /api/v0/search/...).

A batch spec (JSON):
  {"name": "round-12", "model": "qwen38-27b", "revision": "r1", "base": "k3-i8", "golden": "xcorp-s07",
   "kv": "bf16", "max_model_len": 16384, "priority": 0,
   "candidates": [{"id": "c001", "mix": {"blk.0.ffn_down.weight": "k4-i8", ...}, "act": {"blk.3.ffn_down.weight": "a8g128"}},
                  ...]}
mix / act may also be lists of [weight, value] pairs. A weight mapped to the base level is dropped.

submit() checks the golden is marked public (golden_common.marker_ok, also its documents' sources), resolves each
candidate against the base level's skeleton and the manifest (every piece must exist), computes the assembled size
(common/radfile.mix_size: exact) and with the model's parameter count the bpw that sets the job's card class, and
enqueues one eval_mix job per candidate. A candidate identical to an existing job (same base, mix, act, golden, kv,
length) reuses that job and its result. status() / collect() report per candidate: state, KL mean, spot-check state,
and collect() copies each finished candidate's rows (radiance's FILE.rows layout) to OUT/<id>.rows.

0.5 (#4): a candidate may carry "pred_kld" (the search's predicted mean KL for the mix) and the spec a "band"
({"worse": x, "better": y}, inside eval_mix.PLAUSIBLE_LIMITS; default x3 / x0.5). The prediction stays on the server
(jobs.pred, never sent to a worker): a result far worse is refused at completion, a far better one is accepted as
suspect and re-run in full on a verifier. status() adds per candidate trust (confirmed / rerun-pending / spot-pending /
spot-pass / unverified), suspect, adopted, pred_kld, and "invalidated" for a result that no longer stands; adopt()
marks candidates the search acts on and queues a full trusted re-run of each: act on an adopted candidate only once its
trust is "confirmed". collect() deletes the rows of candidates that are no longer accepted."""
import json, os, re, shutil, time
import jobs
from jobs import eval_mix as E, golden_common as G
from common import radfile as R
from common.util import read_json, stamp, write_json_atomic

CAND_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class BatchError(ValueError):
    pass


def golden_info(data, man, g):
    """(golden.json, kld.json) of a golden that donors may receive; raises BatchError otherwise"""
    mk = read_json(os.path.join(data, "goldens", g, "golden.json"))
    ok, why = G.marker_ok(mk)
    if not ok: raise BatchError(f"golden {g}: {why}: donors never get it")
    for f in G.GOLDEN_FILES + ("golden.json",):
        if f"goldens/{g}/{f}" not in man: raise BatchError(f"golden {g}: {f} not in the manifest (run manifest)")
    km = read_json(os.path.join(data, "goldens", g, "kld.json"))
    bad = G.private_sources(d.get("source", "") for d in km["docs"])
    if bad: raise BatchError(f"golden {g}: documents from private sources {bad[:3]}")
    return mk, km


def _pairs(x):
    if isinstance(x, dict): return sorted(x.items())
    if isinstance(x, list): return sorted((str(a), str(b)) for a, b in x)
    raise BatchError("mix / act: an object or a list of pairs")


def submit(S, data, man, spec, cache=None):
    for k in ("model", "revision", "base", "golden", "candidates"):
        if k not in spec: raise BatchError(f"spec: missing {k}")
    cands = spec["candidates"]
    if not isinstance(cands, list) or not 1 <= len(cands) <= 10000: raise BatchError("spec: 1..10000 candidates")
    ids = [c.get("id") for c in cands]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and CAND_RE.fullmatch(i) for i in ids):
        raise BatchError("candidate ids: unique, [A-Za-z0-9._-]")
    _, km = golden_info(data, man, spec["golden"])
    info = read_json(os.path.join(data, "models", f"{spec['model']}@{spec['revision']}", "model.json"))
    if not info or not info.get("n_params"): raise BatchError("model.json with n_params missing (admin.py datapack / model-info)")
    base = {"model": spec["model"], "revision": spec["revision"], "base": spec["base"], "golden": spec["golden"],
            "kv": spec.get("kv", "bf16"), "max_model_len": int(spec.get("max_model_len", 16384)),
            "n_params": int(info["n_params"]), "rows": int(km["rows"])}
    sk_rel = E.skeleton(base)
    if sk_rel not in man: raise BatchError(f"{sk_rel} not in the manifest (admin.py ladder-add, then manifest)")
    cache = {} if cache is None else cache
    if spec.get("band") is not None and not isinstance(spec["band"], dict): raise BatchError("spec: band must be an object")
    band = E.band_of(spec.get("band"))
    skp = os.path.join(data, sk_rel)
    skel = cache.get(skp) or R.Rad(skp, skeleton=True); cache[skp] = skel
    names = [e["name_s"] for e in skel.entries]
    todo = []
    for c in cands:
        mix = [(w, lv) for w, lv in _pairs(c.get("mix", {})) if lv != base["base"]]
        act = _pairs(c.get("act", {}))
        for w, _ in mix + act:
            if w not in skel.idx: raise BatchError(f"candidate {c['id']}: the base has no weight {w}")
        p = dict(base, mix=[list(x) for x in mix], bytes=1)
        if act: p["act"] = [list(x) for x in act]
        rels = {}
        for w in names:
            lv = dict(mix).get(w, base["base"]); rel = E.piece(p, lv, w)
            if rel not in man: raise BatchError(f"candidate {c['id']}: {rel} not in the manifest")
            rels[w] = rel
        p["bytes"] = R.mix_size(skp, {w: os.path.join(data, r) for w, r in rels.items()}, dict(act), cache)
        try: m = jobs.check_params("eval_mix", p)
        except jobs.Rejected as e: raise BatchError(f"candidate {c['id']}: {e}")
        req, _ = m.inputs(p)
        inputs = list(req) + sorted(set(rels.values()) - set(req))
        if f"goldens/{p['golden']}/gen.u8" in man: inputs.append(f"goldens/{p['golden']}/gen.u8")
        jobs.check_inputs("eval_mix", p, inputs)
        pred = c.get("pred_kld")
        if pred is not None and not (isinstance(pred, (int, float)) and not isinstance(pred, bool) and 0 < pred < 100):
            raise BatchError(f"candidate {c['id']}: pred_kld must be a number in (0, 100)")
        todo.append((c["id"], p, inputs, None if pred is None else {"kld": float(pred), **band}))
    t = time.time(); out = {"new": 0, "reused": 0}
    with S.tx() as db:
        bid = db.execute("INSERT INTO batches(t, name, spec, n) VALUES(?,?,?,?)",
                         (t, str(spec.get("name", ""))[:64], json.dumps({k: v for k, v in spec.items() if k != "candidates"}), len(todo))).lastrowid
        for cid, p, inputs, pred in todo:
            ukey = f"eval_mix/v{E.VERSION}:{stamp(p, 24)}"
            r = db.execute("SELECT id FROM jobs WHERE ukey=?", (ukey,)).fetchone()
            if r:
                jid = r[0]; out["reused"] += 1
                if pred:                                    # the first prediction stands; a job not checked yet takes one
                    db.execute("UPDATE jobs SET pred=? WHERE id=? AND pred IS NULL AND state IN ('queued','leased')", (json.dumps(pred), jid))
            else:
                jid = S.enqueue(db, "eval_mix", E.VERSION, p, inputs, jobs.requirements(E, p),
                                sum(man[n]["size"] for n in inputs), level=f"batch-{bid}", ukey=ukey,
                                priority=int(spec.get("priority", 0)))
                if pred: db.execute("UPDATE jobs SET pred=? WHERE id=?", (json.dumps(pred), jid))
                out["new"] += 1
            db.execute("INSERT INTO batch_items(batch_id, cand, job_id) VALUES(?,?,?)", (bid, cid, jid))
        S.event(db, "batch", detail={"batch": bid, **out})
    return {"batch": bid, "candidates": len(todo), **out}


def trust(r, verifier):
    """how far the search may rely on an accepted candidate's rows (#4): confirmed (a verifier ran it, or re-ran it in
    full and it matched), rerun-pending (that full re-run is queued), spot-pending / spot-pass (a few of its documents
    re-scored on a verifier), unverified (a donor's rows, plausible against the prediction or without one)"""
    if verifier or r["trust_state"] == "pass": return "confirmed"
    if r["trust_state"] == "pending": return "rerun-pending"
    if r["verify_state"] in ("pending", "pass"): return f"spot-{r['verify_state']}"
    return "unverified"


def status(S, results, bid):
    b = S.q("SELECT * FROM batches WHERE id=?", (bid,))
    if not b: raise BatchError(f"no batch {bid}")
    items = []
    for r in S.q("SELECT i.cand, i.adopted, j.*, w.verifier AS by_verifier FROM batch_items i JOIN jobs j ON j.id=i.job_id "
                 "LEFT JOIN workers w ON w.id=j.accepted_by WHERE i.batch_id=? ORDER BY i.rowid", (bid,)):
        it = {"id": r["cand"], "job": r["id"], "state": r["state"], "verify": r["verify_state"]}
        pred = json.loads(r["pred"]) if r["pred"] else None
        if pred: it["pred_kld"] = pred["kld"]
        if r["adopted"]: it["adopted"] = r["adopted"]
        if r["state"] == "accepted":
            p = json.loads(r["params"]); res = json.loads(r["result"] or "{}")
            it.update(kld_mean=res.get("kld_mean"), bytes=p["bytes"], bpw=round(E.bpw(p), 4), by=r["accepted_by"],
                      trust=trust(r, r["by_verifier"]), suspect=r["check_state"] == "suspect")
        elif r["requeued"] and r["requeued"] > b[0]["t"]:
            it["invalidated"] = r["requeued"]                # a result seen earlier no longer stands (collect drops its rows)
        items.append(it)
    n = {}
    for it in items: n[it["state"]] = n.get(it["state"], 0) + 1
    conf = sum(1 for it in items if it.get("trust") == "confirmed")
    return {"batch": bid, "name": b[0]["name"], "counts": n, "done": n.get("accepted", 0) == len(items), "confirmed": conf,
            "adopted_confirmed": all(it.get("trust") == "confirmed" for it in items if it.get("adopted")), "items": items}


def adopt(S, bid, cands):
    """the search adopts these candidates: each accepted one gets a full re-run on a verifier (#4); the result counts
    for the search once its trust is confirmed (status). Refuses ids that are not accepted candidates of the batch."""
    from server import verify as V
    if not isinstance(cands, list) or not 1 <= len(cands) <= 1000 or not all(isinstance(c, str) for c in cands):
        raise BatchError("adopt: candidates must be a list of 1..1000 ids")
    out = {}
    with S.tx() as db:
        if not db.execute("SELECT 1 FROM batches WHERE id=?", (bid,)).fetchone(): raise BatchError(f"no batch {bid}")
        rows = {r["cand"]: r for r in db.execute("SELECT i.cand, j.* FROM batch_items i JOIN jobs j ON j.id=i.job_id "
                                                 "WHERE i.batch_id=?", (bid,)).fetchall()}
        bad = [c for c in cands if c not in rows or rows[c]["state"] != "accepted"]
        if bad: raise BatchError(f"adopt: not accepted candidates of batch {bid}: {bad[:10]}")
        t = time.time()
        for c in cands:
            db.execute("UPDATE batch_items SET adopted=COALESCE(adopted, ?) WHERE batch_id=? AND cand=?", (t, bid, c))
            rid = V.enqueue_trusted(S, db, rows[c], "adopt")
            out[c] = {"job": rows[c]["id"], "rerun": rid}
        S.event(db, "adopt", detail={"batch": bid, "n": len(cands), "reruns": sum(1 for v in out.values() if v["rerun"])})
    return {"batch": bid, "adopted": out}


def rows_path(results, job_id): return os.path.join(results, str(job_id), "rows.f32")


def collect(S, results, bid, out):
    """OUT/results.jsonl (status items) + OUT/<id>.rows for every accepted candidate (hard link, else a copy)"""
    st = status(S, results, bid)
    os.makedirs(out, exist_ok=True)
    n = 0
    for it in st["items"]:
        src, dst = rows_path(results, it["job"]), os.path.join(out, f"{it['id']}.rows")
        if it["state"] != "accepted":
            if os.path.exists(dst): os.remove(dst)          # collected before, invalidated since: stale
            continue
        if os.path.exists(dst): os.remove(dst)
        try: os.link(src, dst)
        except OSError: shutil.copyfile(src, dst + ".part"); os.replace(dst + ".part", dst)
        n += 1
    with open(os.path.join(out, "results.jsonl.part"), "w") as f:
        for it in st["items"]: f.write(json.dumps(it) + "\n")
    os.replace(os.path.join(out, "results.jsonl.part"), os.path.join(out, "results.jsonl"))
    return {"batch": bid, "collected": n, "of": len(st["items"])}
