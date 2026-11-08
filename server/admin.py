#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""admin.py --db QUEUE.sqlite [--data DATAROOT] [--results DIR] COMMAND ...  (on the server host; no network API)
(--db / --data / --results default to $RQAH_DB / $RQAH_DATA / $RQAH_RESULTS, as in the server container;
the 0.2 names $RADQUANT_* are still read in 0.3)

  invite [--uses N]                      an invite code for worker registration (printed once)
  add-worker NAME                        a worker token directly (printed once)
  workers                                registered workers, last seen, declared caps
  disable-worker ID [--enable]           revoke (or restore) a worker's token
  datapack MODEL REV CKPT HESS [--hout SNAP DIR] [--no-head]
                                         cut CKPT into per-layer weight files under DATAROOT/models/MODEL@REV/ and link
                                         the Hessians (and an H_out snapshot); then run `manifest`
  manifest [--jobs N]                    sha256 + size of every file under DATAROOT (incremental) -> manifest.json
  enqueue-level --model M --rev R --K K --codebook CB --level NAME [settings] [--layers A-B] [--head]
                                         one quant_layer job per layer (+ quant_head): a ladder level
  enqueue-noop N [--seconds S] [--size B] [--inputs a,b] [--tag T]
  list [--state S] [--level L] [--type T]   jobs
  events [JOB]                           the audit log
  requeue IDS|--level L|--state S [--force]   back to the queue (accepted ones only with --force); a queued job: attempts and per-worker exclusions reset
  cancel IDS|--level L
  export --level L OUT [--partial]       accepted shards of a level in ourq.py's layout (q-LNN.safetensors + index)
  check --level L [--rows-frac F]        run checker/check.py on the level's accepted results, store the verdict
                                         (a FAIL quarantines the donor unless --no-quarantine; a PASS settles points)
0.2 (rad-quant-at-home):
  model-info MODEL REV [--n-params N]    (re)write models/MODEL@REV/model.json (n_params: sets requirements / bpw)
  trust ID [--off] | verifier ID [--off] the per-worker flags: trusted (big-box job types), verifier (spot-check re-runs)
  quarantine ID --reason R | unquarantine ID
  probation ID [--off]                   a worker registered without an invite starts on probation (score-mode quant
                                         only until it has passed results; docs/VERIFY.md): --off lifts it, also
                                         the lock a quarantine puts on it; without --off puts a worker on it
  spot JOB                               re-run an accepted job on a verifier now (a manual spot check)
  ladder-add MODEL REV LEVEL FILE.rad    cut a ladder level's container into models/MODEL@REV/rad/LEVEL/ (skeleton +
                                         one piece per weight) for eval_mix; then `manifest`
  golden-add NAME DIR --public --corpus C [--pool P] [--note T]
                                         link a recorded golden into goldens/NAME/ with its privacy marker (refuses
                                         goldens with documents from our private sources); then `manifest`
  corpus-add NAME FILE.jsonl --public [--chunk-docs N]   a public corpus cut into chunks for golden_record
  enqueue-golden --model M --rev R --corpus C [--chunks a,b] --method tp2|staged [--top 512] [--max-model-len L]
  batch-submit SPEC.json | batch-status ID | batch-collect ID OUT     eval_mix batches (server/batch.py)
  points [--json]                        lumens per worker (verified work only); the leaderboard
w25 (donors, status page):
  invite --personal [--uses N]           one person's invite: every worker registered with it is one donor
  donors [--json]                        donors: display name, opt-in, hidden, workers, lumens
  donor ID [--display-name N] [--leaderboard on|off] [--hide | --unhide]   edit a donor (--hide: its display name
                                         off the public page for every donor showing it, as the board credits by
                                         name; --leaderboard off drops just this donor, who may opt in again)
  status-page OUT [--campaign F]         build the public status page once (server.py rebuilds it by itself)
0.3 (verify by score, docs/VERIFY.md; small-card jobs):
  enqueue-level ... --verify score [--unit group] [--copies N]
                                         score-mode jobs (no per-row claim; the server scores the codes); --unit group:
                                         one quant_tensor job per Hessian group (needs datapack --groups); --copies N:
                                         N copies per job with seed_offset +0..N-1 (best of N: export keeps the best)
  datapack ... --groups                  also cut one weight file per Hessian group (weights/LNN-gG.safetensors)
  score [--level L] [--loop S]           score the accepted score-mode results not scored yet (pass / fail /
                                         provisional; a fail quarantines; --loop: every S seconds until killed)
  score-import --level L DIR [--published NOTE]   our own ourq output dir (q-LNN.safetensors) for that level's jobs as
                                         trusted reference rows (or, --published, a released model's codes: replay guard)
  scores [--level L] [--job J]           the scored tensors: rel, basis, band, state
0.4 (docs/SCHED.md):
  enqueue-level ... [--priority-boost B | --priority P]
                                         default priority from K's granularity: integer K 40, halves 30, quarters 20,
                                         eighths 10, sixteenths 5, finer 0 (+ B); the head job one above its level
  reprioritize PATTERN [--priority P | --by-granularity] [--priority-boost B] [--dry-run]
                                         set the priority of the queued / leased jobs of every level matching the shell
                                         pattern (e.g. 'q27-k2.*'): P, or K's granularity default (+ B); heads +1
  queue [--limit N]                      queued jobs in scheduling order: priority classes, levels, sizes
0.5 (updates, RUNBOOK.md "Upgrades"):
  versions [--days D] [--json]           per job type and version: queued / running / accepted jobs, whether this
                                         server's code still accepts it, the workers seen in D days (7) that declare
                                         it; worker releases seen, and what raising --worker-min would refuse
  retire TYPE VERSION [--migrate] [--dry-run] [--undo]
                                         take an old job version out of service (nothing of it is leased or accepted
                                         any more). Refused while jobs of it run; refused while some are queued unless
                                         --migrate, which re-stamps them at the type's current version after checking
                                         their parameters and inputs against it (running jobs are never touched)
  release-fetch [X.Y.Z | latest] [--file F] [--out PATH]
                                         the release manifest (common/release.py: the release's image digests) from its
                                         GitHub Release (or a local file), checked, written next to the database (or
                                         --out); the running server picks it up and advertises that release
  release-show [--out PATH]              the installed manifest"""
import argparse, concurrent.futures as cf, json, os, re, shutil, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import read_json, write_json_atomic, sha256_file, st_header, safe_relpath, fmt_bytes, env, vtuple
from common.st import copy_tensors
from common import radfile, release as RL
import jobs
from jobs import qwen, quant_common as QC, golden_common as G
from server.store import Store, SchemaTooNew
from server import verify as V, batch as B, donor as D, statuspage as SP, sched as SCH


def man_files(data):
    return (read_json(os.path.join(data, "manifest.json"), {}) or {}).get("files", {})


def private_dir(data, rel_dir):
    """goldens/<g> or corpora/<c> whose marker does not say public: never listed, so never served"""
    parts = rel_dir.split("/")
    if len(parts) == 2 and parts[0] in ("goldens", "corpora"):
        ok, why = G.marker_ok(read_json(os.path.join(data, rel_dir, "golden.json" if parts[0] == "goldens" else "corpus.json")))
        return None if ok else why
    return None


def cmd_manifest(a):
    old = man_files(a.data); todo, out = [], {}
    for dp, dn, fn in os.walk(a.data, followlinks=True):
        keep = []
        for d in sorted(dn):
            if d.startswith("."): continue
            why = private_dir(a.data, os.path.relpath(os.path.join(dp, d), a.data))
            if why: print(f"skip {os.path.relpath(os.path.join(dp, d), a.data)}/: {why}"); continue
            keep.append(d)
        dn[:] = keep
        for f in sorted(fn):
            if f.startswith(".") or f.endswith(".part"): continue
            p = os.path.join(dp, f); rel = os.path.relpath(p, a.data)
            if rel == "manifest.json": continue
            if not safe_relpath(rel): print(f"skip (name not allowed): {rel}"); continue
            st = os.stat(p); o = old.get(rel)
            if o and o["size"] == st.st_size and o.get("mtime") == st.st_mtime: out[rel] = o
            else: todo.append((rel, p, st))
    t0 = time.time(); nb = sum(s.st_size for _, _, s in todo)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for (rel, p, st), hx in zip(todo, ex.map(lambda t: sha256_file(t[1]), todo)):
            out[rel] = {"sha256": hx, "size": st.st_size, "mtime": st.st_mtime}
    write_json_atomic(os.path.join(a.data, "manifest.json"), {"files": dict(sorted(out.items())), "built": time.time()})
    print(f"manifest: {len(out)} files, hashed {len(todo)} ({fmt_bytes(nb)}) in {time.time() - t0:.1f}s")


def cmd_datapack(a):
    root = os.path.join(a.data, "models", f"{a.model}@{a.rev}")
    for d in ("weights", "hessians"): os.makedirs(os.path.join(root, d), exist_ok=True)
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    cfg["quantization_config"] = {"quant_method": "exl3"}                   # ourq's template (as proxy2b/q2b.sh)
    write_json_atomic(os.path.join(root, "config.json"), cfg)
    wm = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
    nl = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    t0 = time.time()
    lo, hi = map(int, (a.layers or f"0-{nl - 1}").split("-"))
    for l in range(lo, hi + 1):
        out = os.path.join(root, "weights", f"L{l:02d}.safetensors")
        if os.path.exists(out): continue
        copy_tensors(a.ckpt, wm, [f"{k}.weight" for k in qwen.keys(qwen.layer_plan(a.arch, l))], out)
    for l in range(lo, hi + 1) if a.groups else ():
        for g, grp in enumerate(qwen.layer_plan(a.arch, l)):
            out = os.path.join(root, "weights", f"L{l:02d}-g{g}.safetensors")
            if not os.path.exists(out): copy_tensors(a.ckpt, wm, [f"{k}.weight" for k in grp[1]], out)
    if not a.no_head:
        out = os.path.join(root, "weights", "head.safetensors")
        if "lm_head.weight" not in wm: print("no lm_head.weight in the checkpoint (tied embeddings?): no head file")
        elif not os.path.exists(out): copy_tensors(a.ckpt, wm, ["lm_head.weight"], out)
    for f in sorted(os.listdir(a.hess)):
        if f.endswith(".safetensors") and not os.path.lexists(os.path.join(root, "hessians", f)):
            os.symlink(os.path.abspath(os.path.join(a.hess, f)), os.path.join(root, "hessians", f))
    if a.hout:
        snap, d = a.hout; od = os.path.join(root, "hout", snap); os.makedirs(od, exist_ok=True)
        for f in sorted(os.listdir(d)):
            if f.endswith(".safetensors") and not os.path.lexists(os.path.join(od, f)):
                os.symlink(os.path.abspath(os.path.join(d, f)), os.path.join(od, f))
    if a.bf16_rad and not os.path.lexists(os.path.join(root, "bf16.rad")):
        os.symlink(os.path.abspath(a.bf16_rad), os.path.join(root, "bf16.rad"))
    if a.hf:
        os.makedirs(os.path.join(root, "hf"), exist_ok=True)
        for f in sorted(os.listdir(a.ckpt)):
            if (f.endswith((".safetensors", ".json", ".jinja", ".txt", ".model")) and not os.path.lexists(os.path.join(root, "hf", f))
                    and not f.startswith(".")):
                os.symlink(os.path.abspath(os.path.join(a.ckpt, f)), os.path.join(root, "hf", f))
    write_model_info(a.data, a.model, a.rev, a.ckpt, a.n_params)
    print(f"datapack {a.model}@{a.rev}: layers {lo}-{hi} in {time.time() - t0:.0f}s -> {root}; now run `manifest`")


def write_model_info(data, model, rev, ckpt=None, n_params=None):
    """models/M@R/model.json: n_params (what bpw divides by; job requirements by model size). Default: every
    language-model tensor of the checkpoint plus the LM head (not a vision tower / drafter)"""
    root = os.path.join(data, "models", f"{model}@{rev}")
    if n_params is None and ckpt:
        wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        hdrs, n = {}, 0
        for k, f in wm.items():
            if not (k.startswith("model.language_model.") or k.startswith("lm_head.") or k.startswith("model.embed_tokens")
                    or k.startswith("model.layers.") or k.startswith("model.norm")): continue
            if f not in hdrs: hdrs[f] = st_header(os.path.join(ckpt, f))
            sh = hdrs[f][k]["shape"]; c = 1
            for x in sh: c *= x
            n += c
        n_params = n
    if not n_params: sys.exit("model-info: no parameter count (give --n-params)")
    write_json_atomic(os.path.join(root, "model.json"), {"model": model, "revision": rev, "n_params": int(n_params)})
    return int(n_params)


def settings_from(a):
    if a.settings_json: return json.load(open(a.settings_json))
    return {"sigma_reg": a.sigma_reg, "sigma_reg_in": a.sigma_reg_in, "sigma_reg_out": a.sigma_reg_out,
            "scaled_hessian": a.scaled_hessian, "qtf2": a.qtf2, "hout": a.hout_snap, "seed_offset": a.seed_offset}


DATA = None


def add_job(S, db, man, jtype, p, level, ukey, priority=0, grp=None):
    m = jobs.check_params(jtype, p)
    req, opt = m.inputs(p)
    import re
    names = list(req) + sorted(n for n in man if n not in req and any(re.fullmatch(rx, n) for rx in opt))
    miss = [n for n in req if n not in man]
    if miss: sys.exit(f"{jtype} {ukey}: inputs not in the manifest: {miss[:4]}")
    jobs.check_inputs(jtype, p, names)
    info = read_json(os.path.join(DATA, "models", f"{p['model']}@{p['revision']}", "model.json")) if "model" in p else None
    return S.enqueue(db, jtype, m.VERSION, p, names, jobs.requirements(m, p, info), sum(man[n]["size"] for n in names),
                     level=level, ukey=ukey, priority=priority, grp=grp)


def cmd_enqueue_level(a, S):
    man = man_files(a.data)
    base = {"model": a.model, "revision": a.rev, "arch": a.arch, "K": int(a.K) if float(a.K).is_integer() else float(a.K),
            "codebook": a.codebook, "level": a.level, "settings": settings_from(a)}
    base["stamp"] = QC.settings_stamp(base)
    cfg = read_json(os.path.join(a.data, "models", f"{a.model}@{a.rev}", "config.json"))
    nl = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    lo, hi = map(int, (a.layers or f"0-{nl - 1}").split("-"))
    if a.verify == "score": base["verify"] = "score"
    prio = a.priority if a.priority is not None else SCH.level_priority(a.K, a.priority_boost)
    n, stamps = 0, set()
    with S.tx() as db:
        for c in range(a.copies):                      # best of N: copy c quantizes with seed_offset + c
            b = dict(base, settings=dict(base["settings"], seed_offset=base["settings"]["seed_offset"] + c))
            b["stamp"] = QC.settings_stamp(b); stamps.add(b["stamp"])
            sfx = f"#c{c}" if a.copies > 1 else ""
            # the copies of one unit share a group: a donor is leased at most one of them (#11)
            grp = (lambda u: f"{a.level}:{u}") if a.copies > 1 else (lambda u: None)
            for l in range(lo, hi + 1):
                if a.unit == "group":
                    for g in range(len(qwen.layer_plan(a.arch, l))):
                        n += add_job(S, db, man, "quant_tensor", {**b, "layer": l, "group": g}, a.level,
                                     f"quant_tensor/v{jobs.get('quant_tensor').VERSION}:{b['stamp']}:L{l:02d}:g{g}{sfx}",
                                     priority=prio, grp=grp(f"L{l:02d}:g{g}")) is not None
                else:
                    n += add_job(S, db, man, "quant_layer", {**b, "layer": l}, a.level,
                                 f"quant_layer/v{jobs.get('quant_layer').VERSION}:{b['stamp']}:L{l:02d}{sfx}", priority=prio,
                                 grp=grp(f"L{l:02d}")) is not None
            if a.head:                                 # the longest job of a level starts first
                n += add_job(S, db, man, "quant_head", b, a.level, f"quant_head/v{jobs.get('quant_head').VERSION}:{b['stamp']}:head{sfx}",
                             priority=prio + 1, grp=grp("head")) is not None
    print(f"level {a.level} (stamp {', '.join(sorted(stamps))}{', verify by score' if a.verify == 'score' else ''}): {n} new jobs, "
          f"priority {prio}")


def version_table(S, days=7, now=None):
    """-> (rows per job type and version, Counter of the worker releases seen in `days`, recent worker rows)"""
    now = now or time.time(); ret = S.retired()
    have = {}
    for r in S.q("SELECT type, version, state, COUNT(*) n FROM jobs GROUP BY type, version, state"):
        have.setdefault((r["type"], r["version"]), Counter())[r["state"]] += r["n"]
    recent = S.q("SELECT id, caps, version FROM workers WHERE disabled=0 AND last_seen>=?", (now - days * 86400,))
    decl, soft = Counter(), Counter()
    for w in recent:
        caps = json.loads(w["caps"] or "{}")
        for t in caps.get("job_types") or []:
            v = (caps.get("versions") or {}).get(t)
            if isinstance(v, int): decl[(t, v)] += 1
        soft[w["version"] or "unknown"] += 1
    rows = []
    for t in sorted(set(jobs.REGISTRY) | {t for t, _ in have}):
        m = jobs.REGISTRY.get(t)
        for v in sorted({v for tt, v in list(have) + list(decl) if tt == t} | ({m.VERSION} if m else set())):
            st = have.get((t, v), Counter())
            served = m is not None and v in jobs.served(m) and v not in ret.get(t, ())
            rows.append({"type": t, "version": v, "current": bool(m and v == m.VERSION), "served": served,
                         "retired": v in ret.get(t, ()), "queued": st["queued"], "leased": st["leased"],
                         "accepted": st["accepted"], "workers": decl[(t, v)],
                         "stuck": st["queued"] > 0 and (not served or decl[(t, v)] == 0)})
    return rows, soft, recent


def cmd_versions(a, S):
    rows, soft, _ = version_table(S, a.days)
    cur, mn = S.meta_get("worker_current"), S.meta_get("worker_min")
    if a.json:
        print(json.dumps({"worker_current": cur, "worker_min": mn, "jobs": rows, "workers": dict(soft)}, indent=1)); return
    print(f"workers: current {cur or '?'}, minimum {mn or '?'} (what the server last advertised)")
    print(f"{'type':14s} {'ver':>3s}  {'':9s} {'queued':>6s} {'running':>7s} {'accepted':>8s} {'workers':>7s}")
    for r in rows:
        tag = "current" if r["current"] else "retired" if r["retired"] else "accepted" if r["served"] else "NOT SERVED"
        print(f"{r['type']:14s} {r['version']:>3d}  {tag:9s} {r['queued']:6d} {r['leased']:7d} {r['accepted']:8d} {r['workers']:7d}"
              + ("   <- stuck" if r["stuck"] else ""))
    print(f"\nworker releases seen in the last {a.days:g} days: "
          + (", ".join(f"{v}: {n}" for v, n in sorted(soft.items(), key=lambda kv: vtuple(kv[0]) or (-1,))) or "none"))
    for r in rows:
        if not r["stuck"]: continue
        m = jobs.REGISTRY.get(r["type"])
        print(f"STUCK {r['type']} v{r['version']}: {r['queued']} queued, " +
              ("this server does not accept it" + (f": `retire {r['type']} {r['version']} --migrate` moves them to v{m.VERSION}" if m else "")
               if not r["served"] else f"no worker seen in {a.days:g} days declares it (wait for one, or `retire --migrate`)"))
    if mn and cur:
        old = sum(n for v, n in soft.items() if v != "unknown" and vtuple(v) < vtuple(cur))
        if old: print(f"raising --worker-min to {cur} would refuse {old} of the workers seen (they finish what they hold)")


def cmd_retire(a, S):
    t, v = a.type, a.version
    m = jobs.get(t)
    ret = S.retired()
    if a.undo:
        ret.get(t, set()).discard(v)
        S.meta_set("retired", json.dumps({k: sorted(x) for k, x in ret.items() if x}))
        with S.tx() as db: S.event(db, "unretire", detail={"type": t, "version": v})
        print(f"{t} v{v}: back in service" + ("" if v in jobs.served(m) else f" (but this server's code does not accept it)")); return
    if v == m.VERSION: sys.exit(f"{t} v{v} is the current version of {t}: bump the type first")
    with S.tx() as db:
        leased = [r[0] for r in db.execute("SELECT id FROM jobs WHERE type=? AND version=? AND state='leased'", (t, v))]
        queued = db.execute("SELECT * FROM jobs WHERE type=? AND version=? AND state='queued' ORDER BY id", (t, v)).fetchall()
        if leased:
            sys.exit(f"{len(leased)} {t} v{v} jobs are running ({', '.join(map(str, leased[:10]))}): retire never touches running "
                     f"jobs. Wait for them (admin.py list --type {t} --state leased), or cancel them, then retire")
        if queued and not a.migrate:
            sys.exit(f"{len(queued)} queued {t} v{v} jobs still need it ({', '.join(str(r['id']) for r in queued[:10])}): "
                     f"--migrate re-stamps them at v{m.VERSION} after checking them against it, or cancel them")
        if queued and not a.data: sys.exit("--migrate needs --data (the manifest) to re-check the jobs' inputs")
        bad, moved = [], []
        man = man_files(a.data) if queued else {}
        for j in queued:
            p = json.loads(j["params"])
            try:
                jobs.check_params(t, p); jobs.check_inputs(t, p, json.loads(j["inputs"]))
                miss = [n for n in json.loads(j["inputs"]) if n not in man]
                if miss: raise jobs.Rejected(f"inputs not in the manifest: {miss[:3]}")
            except jobs.Rejected as e:
                bad.append(f"job {j['id']}: {e}"); continue
            info = read_json(os.path.join(a.data, "models", f"{p['model']}@{p['revision']}", "model.json")) if "model" in p else None
            uk = j["ukey"]
            if uk and uk.startswith(f"{t}/v{v}:"):
                uk = f"{t}/v{m.VERSION}:" + uk[len(f"{t}/v{v}:"):]
                if db.execute("SELECT 1 FROM jobs WHERE ukey=? AND id<>?", (uk, j["id"])).fetchone():
                    bad.append(f"job {j['id']}: the same unit is queued at v{m.VERSION} already ({uk}): cancel this one"); continue
            moved.append((j["id"], json.dumps(jobs.requirements(m, p, info)), uk))
        if bad:
            sys.exit(f"not retired: {len(bad)} queued jobs cannot move to v{m.VERSION}:\n  " + "\n  ".join(bad[:20]))
        if a.dry_run:
            print(f"dry run: {t} v{v} can be retired; {len(moved)} queued jobs would move to v{m.VERSION}"); return
        now = time.time()
        for jid, req, uk in moved:
            db.execute("UPDATE jobs SET version=?, requires=?, ukey=?, updated=? WHERE id=?", (m.VERSION, req, uk, now, jid))
            S.event(db, "migrate", jid, detail={"type": t, "from": v, "to": m.VERSION})
        ret.setdefault(t, set()).add(v)
        db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('retired', ?)", (json.dumps({k: sorted(x) for k, x in ret.items() if x}),))
        S.event(db, "retire", detail={"type": t, "version": v, "migrated": len(moved)})
    print(f"{t} v{v} retired" + (f"; {len(moved)} queued jobs moved to v{m.VERSION}" if moved else ""))


def release_path(a):
    return a.out or os.path.join(os.path.dirname(os.path.abspath(a.db)), "release.json")


def cmd_release_fetch(a):
    """a release manifest from GitHub (or --file), checked, installed atomically"""
    if a.file:
        b = open(a.file, "rb").read(1 << 20); src = a.file
    else:
        import urllib.request
        v = None if a.version in (None, "latest") else a.version
        if v is not None and not re.fullmatch(RL.VERSION_PAT, v): sys.exit("release-fetch: X.Y.Z or latest")
        src = RL.asset_url(v)
        try:
            with urllib.request.urlopen(urllib.request.Request(src, headers={"User-Agent": "rqah-admin"}), timeout=60) as r: b = r.read(1 << 20)
        except OSError as e: sys.exit(f"release-fetch: {src}: {e}")
    try: m = RL.loads(b)
    except ValueError as e: sys.exit(f"release-fetch: {src}: refused: {e}")
    if a.version not in (None, "latest") and m["version"] != a.version: sys.exit(f"release-fetch: {src} is {m['version']}, not {a.version}")
    p = release_path(a)
    old = read_json(p) if os.path.exists(p) else None
    write_json_atomic(p, m)
    print(f"{p}: release {m['version']} (was {old.get('version') if isinstance(old, dict) else 'none'}), published {m['created']}")
    for s, im in sorted(m["images"].items()): print(f"  {s:18s} {im['repo']}:{im['tag']}@{im['digest']}")
    print("the server advertises it at its next request (current = this release unless RQAH_WORKER_CURRENT is set)")


def cmd_release_show(a):
    p = release_path(a)
    if not os.path.exists(p): sys.exit(f"no release manifest at {p} (admin.py release-fetch)")
    try: m = RL.loads(open(p, "rb").read())
    except ValueError as e: sys.exit(f"{p}: {e}")
    print(json.dumps(m, indent=1, sort_keys=True))


def cmd_reprioritize(a, S):
    """the queued / leased jobs of the levels matching a pattern get priority P (or K's granularity default) + boost;
    quant_head jobs one more, as at enqueue. Accepted / failed / cancelled jobs keep theirs (it no longer matters)"""
    levels = [r[0] for r in S.q("SELECT DISTINCT level FROM jobs WHERE state IN ('queued','leased') AND verify_of IS NULL")]
    hit = SCH.match_levels(levels, a.pattern)
    if not hit: sys.exit(f"no queued level matches {a.pattern!r}")
    if a.priority is None and not a.by_granularity: sys.exit("give --priority P or --by-granularity")
    changes = Counter()
    with S.tx() as db:
        for lv in sorted(hit):
            for j in db.execute("SELECT id, type, params, priority FROM jobs WHERE level=? AND state IN ('queued','leased') "
                                "AND verify_of IS NULL", (lv,)).fetchall():
                p = json.loads(j["params"])
                if a.priority is not None: new = a.priority + a.priority_boost
                elif "K" in p: new = SCH.level_priority(p["K"], a.priority_boost)
                else: continue                          # not a ladder job (no K): left alone
                new += j["type"] == "quant_head"
                if new == j["priority"]: continue
                changes[(lv, j["priority"], new)] += 1
                if not a.dry_run: db.execute("UPDATE jobs SET priority=? WHERE id=?", (new, j["id"]))
            if not a.dry_run and any(k[0] == lv for k in changes):
                S.event(db, "reprioritize", detail={"level": lv, "to": sorted({k[2] for k in changes if k[0] == lv})})
    for (lv, old, new), n in sorted(changes.items()): print(f"{lv}: {n} jobs {old} -> {new}")
    print(f"{'would change' if a.dry_run else 'changed'} {sum(changes.values())} jobs in {len(hit)} matching levels")


def cmd_queue(a, S):
    """the queue in the order an unrestricted worker with an empty cache would get it (docs/SCHED.md)"""
    rows = S.q("SELECT * FROM jobs WHERE state='queued' ORDER BY priority DESC, id")
    with S.lock: ranks = SCH.level_ranks(S.db, [r["level"] for r in rows])
    rows.sort(key=lambda r: (-r["priority"], ranks.get(r["level"], r["id"]) if r["level"] else r["id"], r["id"]))
    for r in rows[:a.limit]:
        p = json.loads(r["params"]); what = f"L{p['layer']:02d}" if "layer" in p else ("head" if r["type"] == "quant_head" else "")
        print(f"{r['id']:6d} prio {r['priority']:4d} {r['type']:12s} {what:5s} lvl {r['level'] or '-':16s} K {p.get('K', '-')!s:6s} "
              f"in {fmt_bytes(r['input_bytes'])}")
    print(f"{len(rows)} queued" + (f" (first {a.limit} shown)" if len(rows) > a.limit else "") + "; by class: " +
          ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["priority"] for r in rows).items(), reverse=True)))


def cmd_enqueue_noop(a, S):
    man = man_files(a.data); ids = []
    with S.tx() as db:
        for i in range(a.n):
            p = {"seconds": a.seconds, "size": a.size, "seed": a.seed + i}
            if a.inputs: p["inputs"] = [x for x in a.inputs.split(",") if x]
            if a.tag: p["tag"] = a.tag
            if a.fail: p["fail"] = True
            for k in ("cards", "ram_gb", "vram_gb"):
                if getattr(a, k): p[k] = getattr(a, k)
            if a.proc: p["proc"] = True
            ids.append(add_job(S, db, man, "noop", p, a.tag or None, None, priority=a.priority))
    print("enqueued", ids)


def cmd_list(a, S):
    w, args = [], []
    for k in ("state", "level", "type"):
        if getattr(a, k): w.append(f"{k}=?"); args.append(getattr(a, k))
    rows = S.q("SELECT * FROM jobs" + (" WHERE " + " AND ".join(w) if w else "") + " ORDER BY id", args)
    for r in rows:
        p = json.loads(r["params"]); what = f"L{p['layer']:02d}" if "layer" in p else ("head" if r["type"] == "quant_head" else "")
        print(f"{r['id']:5d} {r['type']:11s} {what:5s} {r['state']:9s} att {r['attempts']}/{r['max_attempts']} "
              f"lvl {r['level'] or '-':12s} by {r['accepted_by'] or '-'} chk {r['check_state'] or '-'} {(r['error'] or '')[-80:]}")
    print(f"{len(rows)} jobs")


def ids_or(a, S, where):
    if a.ids: return [int(x) for x in a.ids]
    if a.level: return [r[0] for r in S.q(f"SELECT id FROM jobs WHERE level=? AND {where}", (a.level,))]
    if getattr(a, "state", None): return [r[0] for r in S.q(f"SELECT id FROM jobs WHERE state=? AND {where}", (a.state,))]
    sys.exit("give job ids, --level or --state")


def cmd_requeue(a, S):
    ids = ids_or(a, S, "1"); n = 0
    with S.tx() as db:
        for i in ids:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (i,)).fetchone()
            if j is None: continue
            if j["state"] == "queued":                  # still queued: a fresh chance for the workers that failed it
                db.execute("UPDATE jobs SET attempts=0, requeued=?, updated=? WHERE id=?", (time.time(), time.time(), i))
                S.event(db, "requeue", i); n += 1; continue
            if j["state"] == "accepted" and not a.force: print(f"{i}: accepted (use --force)"); continue
            if j["state"] == "accepted" and a.results:
                d = os.path.join(a.results, str(i))
                if os.path.exists(d): os.replace(d, d + f".old-{int(time.time())}")
            if j["state"] == "accepted":                # its scores no longer stand for a kept result
                db.execute("UPDATE scores SET state='void' WHERE job_id=? AND state<>'fail'", (i,))
            if j["lease_id"]: db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (time.time(), j["lease_id"]))
            db.execute("UPDATE jobs SET state='queued', attempts=0, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, "
                       "accepted_lease=NULL, accepted_by=NULL, accepted_at=NULL, check_state=NULL, requeued=?, updated=? WHERE id=?",
                       (time.time(), time.time(), i))
            S.event(db, "requeue", i); n += 1
    print(f"requeued {n}")


def cmd_cancel(a, S):
    ids = ids_or(a, S, "state IN ('queued','leased')")
    with S.tx() as db:
        for i in ids:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (i,)).fetchone()
            if j is None or j["state"] not in ("queued", "leased"): continue
            if j["lease_id"]: db.execute("UPDATE leases SET state='cancelled', ended=? WHERE id=?", (time.time(), j["lease_id"]))
            db.execute("UPDATE jobs SET state='cancelled', lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=? WHERE id=?",
                       (time.time(), i))
            S.event(db, "cancel", i)
    print(f"cancelled {len(ids)}")


def level_jobs(S, level):
    return [r for r in S.q("SELECT * FROM jobs WHERE level=? AND type IN ('quant_layer','quant_head','quant_tensor') "
                           "AND verify_of IS NULL ORDER BY id", (level,))]


def unit_of(r):
    """the piece of the level a job makes (copies of best-of-N share it): layer, layer/group or head"""
    p = json.loads(r["params"])
    return ("head",) if r["type"] == "quant_head" else (p["layer"], p.get("group", -1))


def cmd_export(a, S):
    from server import score as SC
    from common.st import merge_shards
    rows = level_jobs(S, a.level)
    units = {}
    for r in rows: units.setdefault(unit_of(r), []).append(r)
    pick = {}                                           # one accepted result per unit: the best passed copy (best of N)
    for u, rs in units.items():
        acc_u = [r for r in rs if r["state"] == "accepted"]
        best = SC.best_of_group(S, acc_u) if any(SC.is_score(json.loads(r["params"])) for r in acc_u) else None
        if best is None and acc_u: best = acc_u[0]
        if best is not None: pick[u] = best
    acc = list(pick.values())
    if len(pick) != len(units) and not a.partial: sys.exit(f"{len(pick)}/{len(units)} accepted (--partial to export anyway)")
    if a.checked and any(r["check_state"] != "pass" for r in acc): sys.exit("not every accepted job passed its check")
    os.makedirs(a.out, exist_ok=True)
    stamps, proxy, wm, K = [], [], {}, None
    def order(r): p = json.loads(r["params"]); return (p.get("layer", 10 ** 6), p.get("group", -1))
    groups = {}
    for r in sorted(acc, key=order):
        p = json.loads(r["params"]); K = p["K"]; src = os.path.join(a.results, str(r["id"]))
        m = jobs.get(r["type"])
        if r["type"] == "quant_tensor":                 # merged per layer below
            groups.setdefault(p["layer"], []).append(os.path.join(src, m.shard(p)))
        else:
            shard = m.shard(p)
            dst = os.path.join(a.out, shard)
            if os.path.exists(dst): os.remove(dst)
            try: os.link(os.path.join(src, shard), dst)
            except OSError: shutil.copyfile(os.path.join(src, shard), dst + ".part"); os.replace(dst + ".part", dst)
            for k in st_header(dst):
                if k != "__metadata__": wm[k] = shard
        proxy += [l for l in open(os.path.join(src, "proxy.jsonl")) if l.strip()]
        for s in json.load(open(os.path.join(src, "stamp.json"))):
            if s["stamp"] not in [x["stamp"] for x in stamps]: stamps.append(s)
    for l, parts in sorted(groups.items()):             # a layer's groups -> one q-LNN.safetensors (ourq's layout)
        shard = f"q-L{l:02d}.safetensors"
        merge_shards(parts, os.path.join(a.out, shard))
        for k in st_header(os.path.join(a.out, shard)):
            if k != "__metadata__": wm[k] = shard
    with open(os.path.join(a.out, "proxy.jsonl"), "w") as f: f.writelines(proxy)
    json.dump(stamps, open(os.path.join(a.out, "stamp.json"), "w"), indent=1)
    # as ourq.py write_index(): index over every shard, config = the template with the level's bits
    json.dump({"metadata": {}, "weight_map": dict(sorted(wm.items()))}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    p0 = json.loads(acc[0]["params"])
    c = read_json(os.path.join(a.data, "models", f"{p0['model']}@{p0['revision']}", "config.json"))
    c["quantization_config"].update({"bits": float(K), "head_bits": K, "calibration": {"source": "challenge calib Hessians"}})
    json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"exported {len(acc)} shards of {a.level} to {a.out} (ourq stamps: {sorted({s['stamp'] for s in stamps})})")


def cmd_check(a, S):
    from checker import check
    from server.score import is_score
    rows = [r for r in level_jobs(S, a.level) if r["state"] == "accepted" and (a.recheck or not r["check_state"])
            and not is_score(json.loads(r["params"]))]                     # score-mode results: admin.py score
    for r in rows:
        p = json.loads(r["params"]); res = os.path.join(a.results, str(r["id"]))
        try: rep = check.check_job(r["type"], p, res, lambda rel: os.path.join(a.data, rel), rows_frac=a.rows_frac, tol=a.tol)
        except Exception as e:                                 # the checker's fault or the server's, not the donor's: no fail, no quarantine
            rep = {"pass": False, "summary": f"error: {e!r}"[:300]}
            with S.tx() as db:
                db.execute("UPDATE jobs SET check_state='error', check_detail=? WHERE id=?", (json.dumps(rep), r["id"]))
                S.event(db, "check_error", r["id"], detail={"error": f"{e!r}"[:300]})
            print(f"{r['id']:5d} ERROR {rep['summary']}"); continue
        with S.tx() as db:
            db.execute("UPDATE jobs SET check_state=?, check_detail=? WHERE id=?", ("pass" if rep["pass"] else "fail", json.dumps(rep), r["id"]))
            S.event(db, "check", r["id"], detail={"pass": rep["pass"]})
            if rep["pass"]: V.settle(S, db, r["id"])
            elif not a.no_quarantine:
                w = V.worker(db, r["accepted_by"])
                if w is not None and not w["verifier"]:
                    V.quarantine(S, db, a.results, r["accepted_by"], f"checker failed on job {r['id']}: {rep['summary'][:200]}")
        print(f"{r['id']:5d} {'PASS' if rep['pass'] else 'FAIL'} {rep['summary']}")


def cmd_score(a, S):
    from server import score as SC
    dp = lambda rel: os.path.join(a.data, rel)
    while True:
        n = SC.run_pending(S, a.results, dp, a.level, log=print)
        if n: print("scored:", ", ".join(f"{v} {k}" for k, v in sorted(n.items())), flush=True)
        if not a.loop: break
        time.sleep(a.loop)


def cmd_score_import(a, S):
    """our own ourq output dir as reference rows for the level's jobs (matched by layer / head / group)"""
    import tempfile
    from server import score as SC
    from common.st import subset_shard
    dp = lambda rel: os.path.join(a.data, rel)
    done = 0
    for u, r in {unit_of(r): r for r in level_jobs(S, a.level)}.items():
        p = json.loads(r["params"]); m = jobs.get(r["type"]); d = a.dir
        src = f"q-L{p['layer']:02d}.safetensors" if r["type"] == "quant_tensor" else m.shard(p)   # ourq's own names
        if not os.path.exists(os.path.join(d, src)): continue
        if r["type"] == "quant_tensor":                 # the group's tensors of ourq's whole-layer shard
            d = tempfile.mkdtemp(prefix="rqah-import-")
            subset_shard(os.path.join(a.dir, src), [f"{k}.{x}" for k in qwen.keys(m.plan(p)) for x in ("trellis", "suh", "svh", "mul1")],
                         os.path.join(d, m.shard(p)))
        rows = SC.import_reference(S, dp, r["type"], p, d, kind="published" if a.published else "trusted", note=a.published or a.note)
        if d != a.dir: shutil.rmtree(d, ignore_errors=True)
        done += 1
        print(f"{u}: " + ", ".join(f"{x['key'].rsplit('.', 1)[-1]} {x['rel']:.4e}" for x in rows), flush=True)
    print(f"imported {done} shards of {a.dir} as {'published' if a.published else 'trusted'} rows for level {a.level}")


def cmd_scores(a, S):
    q, args = "SELECT s.*, j.level FROM scores s LEFT JOIN jobs j ON j.id=s.job_id WHERE 1", []
    if a.level: q += " AND j.level=?"; args.append(a.level)
    if a.job: q += " AND s.job_id=?"; args.append(a.job)
    for r in S.q(q + " ORDER BY s.id", args):
        print(f"{r['id']:6d} job {r['job_id'] or '-':>5} {r['kind']:9s} K{r['K']:<6g} {r['key'].replace('model.language_model.', ''):42s} "
              f"rel {r['rel']:.4e} {r['state']:11s} {r['basis'] or '-':7s} bound {r['bound'] or 0:.4e} {r['codes_sha'][:10]}")


def cmd_ladder_add(a, S):
    d = os.path.join(a.data, "models", f"{a.model}@{a.rev}", "rad", a.level)
    if not QC.re.fullmatch(QC.ID, a.level): sys.exit(f"bad level name {a.level!r}")
    t0 = time.time(); r = radfile.cut(a.rad, d)
    print(f"ladder-add {a.level}: skeleton + {len(r['pieces'])} pieces -> {d} in {time.time() - t0:.1f}s; now run `manifest`")


def cmd_golden_add(a, S):
    if not a.public: sys.exit("golden-add: only public goldens are added (--public); donors never get anything else")
    km = json.load(open(os.path.join(a.dir, "kld.json")))
    srcs = sorted({d.get("source", "") for d in km["docs"]})
    bad = G.private_sources(srcs)
    if bad: sys.exit(f"golden-add: documents from private sources {bad[:5]}: refusing (AGENTS.md: never to donors)")
    if km.get("format") != "radiance-kld-top-1": print(f"note: format {km.get('format')!r} (sparse top-N expected)")
    od = os.path.join(a.data, "goldens", a.name); os.makedirs(od, exist_ok=True)
    for f in G.GOLDEN_FILES + ("gen.u8",):
        src = os.path.join(a.dir, f)
        if os.path.exists(src) and not os.path.lexists(os.path.join(od, f)): os.symlink(os.path.abspath(src), os.path.join(od, f))
    write_json_atomic(os.path.join(od, "golden.json"), {"name": a.name, "privacy": "public", "corpus": a.corpus, "pool": a.pool,
                      "sources": srcs, "rows": km["rows"], "docs": len(km["docs"]), "note": a.note, "added": time.time()})
    print(f"golden {a.name}: {km['rows']} rows, {len(km['docs'])} docs, sources {srcs[:6]}; now run `manifest`")


def cmd_corpus_add(a, S):
    if not a.public: sys.exit("corpus-add: only public corpora are added (--public)")
    lines = [l for l in open(a.file, encoding="utf-8") if l.strip()]
    srcs = sorted({json.loads(l).get("source", "") for l in lines})
    bad = G.private_sources(srcs)
    if bad: sys.exit(f"corpus-add: documents from private sources {bad[:5]}: refusing")
    od = os.path.join(a.data, "corpora", a.name); os.makedirs(od, exist_ok=True)
    chunks = []
    for i in range(0, len(lines), a.chunk_docs):
        c = f"c{i // a.chunk_docs:04d}"; chunks.append(c)
        with open(os.path.join(od, c + ".jsonl.part"), "w", encoding="utf-8") as f: f.writelines(lines[i:i + a.chunk_docs])
        os.replace(os.path.join(od, c + ".jsonl.part"), os.path.join(od, c + ".jsonl"))
    write_json_atomic(os.path.join(od, "corpus.json"), {"name": a.name, "privacy": "public", "sources": srcs, "docs": len(lines),
                      "chunks": chunks, "chunk_docs": a.chunk_docs, "note": a.note, "added": time.time()})
    print(f"corpus {a.name}: {len(lines)} docs in {len(chunks)} chunks; now run `manifest`")


def cmd_enqueue_golden(a, S):
    man = man_files(a.data)
    cj = read_json(os.path.join(a.data, "corpora", a.corpus, "corpus.json"))
    ok, why = G.marker_ok(cj)
    if not ok: sys.exit(f"corpus {a.corpus}: {why}")
    chunks = a.chunks.split(",") if a.chunks else cj["chunks"]
    n = 0
    with S.tx() as db:
        for c in chunks:
            lines = [l for l in open(os.path.join(a.data, "corpora", a.corpus, c + ".jsonl"), encoding="utf-8") if l.strip()]
            pos = sum(max(0, len(json.loads(l)["prompt"].encode()) - 1 - int(json.loads(l).get("score_from", 0))) for l in lines)
            p = {"model": a.model, "revision": a.rev, "corpus": a.corpus, "chunk": c, "method": a.method, "top": a.top,
                 "max_model_len": a.max_model_len, "max_positions": int(pos * a.positions_slack) + 1024}
            if a.method == "staged": p["stages"] = a.stages
            n += add_job(S, db, man, "golden_record", p, f"golden-{a.corpus}", f"golden_record/v1:{a.model}@{a.rev}:{a.corpus}:{c}:{a.method}:{a.top}",
                         priority=a.priority) is not None
    print(f"golden_record: {n} new jobs ({a.corpus}, {len(chunks)} chunks, {a.method})")


def cmd_spot(a, S):
    with S.tx() as db:
        for i in a.ids:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (int(i),)).fetchone()
            if j is None or j["state"] != "accepted" or j["verify_of"]: print(f"{i}: not an accepted original"); continue
            print(f"{i}: re-run {V.enqueue_rerun(S, db, j, why='admin')}")


def cmd_points(a, S):
    st = V.standings(S)
    if a.json: print(json.dumps(st, indent=1)); return
    for r in st:
        flags = ("Q" if r["quarantined"] else "") + ("V" if r["verifier"] else "")
        print(f"{r['id']:4d} {r['name'][:24]:24s} {r['lumens']:10.2f} lumens  {r['jobs']:5d} jobs {flags}")


def cmd_donors(a, S):
    rows, _ = D.board(S)
    lum = {did: r for r in rows for did in r["donors"]}
    st = {r["id"]: r for r in V.standings(S)}
    out = []
    for d in S.q("SELECT * FROM donors ORDER BY id"):
        ws = S.q("SELECT id, name FROM workers WHERE donor_id=? ORDER BY id", (d["id"],))
        out.append({"id": d["id"], "display_name": d["display_name"], "leaderboard": bool(d["leaderboard"]), "hidden": bool(d["hidden"]),
                    "public_name_ok": D.public_name(d["display_name"]) is not None, "workers": [dict(w) for w in ws],
                    "lumens": round(sum(st[w["id"]]["lumens"] for w in ws if w["id"] in st), 4),
                    "rank": lum.get(d["id"], {}).get("rank")})
    if a.json: print(json.dumps(out, indent=1)); return
    for d in out:
        fl = ("board" if d["leaderboard"] else "-") + (",hidden" if d["hidden"] else "") + ("" if d["public_name_ok"] else ",name-not-shown")
        print(f"{d['id']:4d} {str(d['display_name'])[:28]:28s} {fl:24s} {d['lumens']:10.2f} lumens  rank {d['rank'] or '-':>3}  "
              "workers " + ",".join(str(w["id"]) + ":" + w["name"] for w in d["workers"]))


def cmd_donor(a, S):
    sets, args = [], []
    if a.display_name is not None:
        try: sets.append("display_name=?"); args.append(D.clean_display_name(a.display_name))
        except ValueError as e: sys.exit(str(e))
    if a.leaderboard: sets.append("leaderboard=?"); args.append(int(a.leaderboard == "on"))
    if a.hide or a.unhide: sets.append("hidden=?"); args.append(int(a.hide))
    if not sets: sys.exit("nothing to change")
    with S.tx() as db:
        if db.execute(f"UPDATE donors SET {', '.join(sets)} WHERE id=?", (*args, a.id)).rowcount != 1: sys.exit(f"no donor {a.id}")
        S.event(db, "donor-edit", detail={"donor": a.id, "set": [x.split("=")[0] for x in sets]})
    print(f"donor {a.id}: " + ", ".join(f"{k.split('=')[0]} = {v}" for k, v in zip(sets, args)))
    if a.hide: print("(the name is now off the public page for every donor showing it)")


def cmd_exclude_donor(a, S):
    """#4: every worker of the donor quarantined (no more leases), and every accepted result of theirs that is not
    confirmed independently (score-verified codes, a passed re-run; eval_mix: a full trusted re-run) invalidated:
    lumens revoked, the job queued again for anyone else, the search's batches show it as invalidated"""
    ws = S.q("SELECT id FROM workers WHERE donor_id=?", (a.donor,))
    if not ws: sys.exit(f"no donor {a.donor} (or no workers)")
    n = 0
    with S.tx() as db:
        for w in ws: n += V.quarantine(S, db, a.results, w["id"], f"donor {a.donor} excluded: {a.reason}")
        S.event(db, "exclude_donor", detail={"donor": a.donor, "workers": [w["id"] for w in ws], "invalidated": n, "reason": a.reason[:300]})
    print(f"donor {a.donor}: {len(ws)} workers quarantined, {n} results invalidated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=env("DB")); ap.add_argument("--data", default=env("DATA")); ap.add_argument("--results", default=env("RESULTS"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("invite"); x.add_argument("--uses", type=int, default=1); x.add_argument("--note", default="")
    x.add_argument("--personal", action="store_true", help="every worker registered with it belongs to one donor")
    x = sp.add_parser("donors"); x.add_argument("--json", action="store_true")
    x = sp.add_parser("donor"); x.add_argument("id", type=int); x.add_argument("--display-name")
    x.add_argument("--leaderboard", choices=("on", "off")); x.add_argument("--hide", action="store_true"); x.add_argument("--unhide", action="store_true")
    x = sp.add_parser("status-page"); x.add_argument("out"); x.add_argument("--campaign")
    x = sp.add_parser("add-worker"); x.add_argument("name")
    sp.add_parser("workers")
    x = sp.add_parser("disable-worker"); x.add_argument("id", type=int); x.add_argument("--enable", action="store_true")
    x = sp.add_parser("manifest"); x.add_argument("--jobs", type=int, default=4)
    x = sp.add_parser("datapack"); x.add_argument("model"); x.add_argument("rev"); x.add_argument("ckpt"); x.add_argument("hess")
    x.add_argument("--arch", default="qwen3_5"); x.add_argument("--hout", nargs=2, metavar=("SNAP", "DIR")); x.add_argument("--no-head", action="store_true")
    x.add_argument("--layers", help="A-B: only these layers' weight files (tests)")
    x.add_argument("--n-params", type=int, help="model.json's parameter count (default: from the checkpoint)")
    x.add_argument("--bf16-rad", help="link this bf16 .rad as models/M@R/bf16.rad (golden_record tp2)")
    x.add_argument("--hf", action="store_true", help="link the checkpoint's files into models/M@R/hf/ (golden_record staged)")
    x.add_argument("--groups", action="store_true", help="also one weight file per Hessian group (quant_tensor jobs)")
    x = sp.add_parser("model-info"); x.add_argument("model"); x.add_argument("rev"); x.add_argument("--ckpt"); x.add_argument("--n-params", type=int)
    for c in ("trust", "verifier"): x = sp.add_parser(c); x.add_argument("id", type=int); x.add_argument("--off", action="store_true")
    x = sp.add_parser("quarantine"); x.add_argument("id", type=int); x.add_argument("--reason", required=True)
    x = sp.add_parser("unquarantine"); x.add_argument("id", type=int)
    x = sp.add_parser("probation"); x.add_argument("id", type=int); x.add_argument("--off", action="store_true")
    x = sp.add_parser("spot"); x.add_argument("ids", nargs="+")
    x = sp.add_parser("ladder-add"); x.add_argument("model"); x.add_argument("rev"); x.add_argument("level"); x.add_argument("rad")
    x = sp.add_parser("golden-add"); x.add_argument("name"); x.add_argument("dir"); x.add_argument("--public", action="store_true")
    x.add_argument("--corpus", required=True, help="where its documents come from (xcorp, calib, ...)"); x.add_argument("--pool"); x.add_argument("--note", default="")
    x = sp.add_parser("corpus-add"); x.add_argument("name"); x.add_argument("file"); x.add_argument("--public", action="store_true")
    x.add_argument("--chunk-docs", type=int, default=64); x.add_argument("--note", default="")
    x = sp.add_parser("enqueue-golden")
    for k in ("model", "rev", "corpus"): x.add_argument(f"--{k}", required=True)
    x.add_argument("--chunks"); x.add_argument("--method", choices=("tp2", "staged"), required=True); x.add_argument("--top", type=int, default=512)
    x.add_argument("--stages", type=int, default=3); x.add_argument("--max-model-len", type=int, default=32768)
    x.add_argument("--positions-slack", type=float, default=1.0, help="max_positions = byte count of the chunk x this (an upper bound on tokens)")
    x.add_argument("--priority", type=int, default=0)
    x = sp.add_parser("batch-submit"); x.add_argument("spec")
    x = sp.add_parser("batch-status"); x.add_argument("id", type=int)
    x = sp.add_parser("batch-collect"); x.add_argument("id", type=int); x.add_argument("out")
    x = sp.add_parser("batch-adopt"); x.add_argument("id", type=int); x.add_argument("cands", nargs="+")
    x = sp.add_parser("exclude-donor", help="a donor caught faking: quarantine all its workers, drop every result of theirs "
                      "nothing confirmed (#4; per worker this happens on its own, donor-wide only by this command)")
    x.add_argument("donor", type=int); x.add_argument("--reason", required=True)
    x = sp.add_parser("points"); x.add_argument("--json", action="store_true")
    x = sp.add_parser("enqueue-level")
    for k in ("model", "rev", "K", "codebook", "level"): x.add_argument(f"--{k}", required=True)
    x.add_argument("--arch", default="qwen3_5"); x.add_argument("--layers"); x.add_argument("--head", action="store_true")
    x.add_argument("--settings-json"); x.add_argument("--sigma-reg", type=float, default=0.025)
    x.add_argument("--sigma-reg-in", type=float); x.add_argument("--sigma-reg-out", type=float)
    x.add_argument("--scaled-hessian", action="store_true"); x.add_argument("--qtf2", action="store_true")
    x.add_argument("--hout-snap"); x.add_argument("--seed-offset", type=int, default=0)
    x.add_argument("--verify", choices=("replicate", "score"), default="replicate",
                   help="score: the server scores the codes (docs/VERIFY.md); replicate: the 0.2 checker + spot checks")
    x.add_argument("--unit", choices=("layer", "group"), default="layer", help="group: one quant_tensor job per Hessian group")
    x.add_argument("--copies", type=int, default=1, help="best of N: N copies of every job (seed_offset + 0..N-1)")
    x.add_argument("--priority-boost", type=int, default=0, help="added to the default priority from K's granularity")
    x.add_argument("--priority", type=int, help="explicit priority (instead of the granularity default)")
    x = sp.add_parser("reprioritize"); x.add_argument("pattern", help="shell pattern over level names, e.g. 'q27-k2.*'")
    x.add_argument("--priority", type=int); x.add_argument("--by-granularity", action="store_true")
    x.add_argument("--priority-boost", type=int, default=0); x.add_argument("--dry-run", action="store_true")
    x = sp.add_parser("queue"); x.add_argument("--limit", type=int, default=50)
    x = sp.add_parser("enqueue-noop"); x.add_argument("n", type=int); x.add_argument("--seconds", type=float, default=1)
    x.add_argument("--size", type=int, default=1024); x.add_argument("--seed", type=int, default=0)
    x.add_argument("--inputs"); x.add_argument("--tag"); x.add_argument("--fail", action="store_true")
    x.add_argument("--cards", type=int, default=0); x.add_argument("--ram-gb", type=float, default=0); x.add_argument("--vram-gb", type=float, default=0)
    x.add_argument("--proc", action="store_true"); x.add_argument("--priority", type=int, default=0)
    x = sp.add_parser("list"); x.add_argument("--state"); x.add_argument("--level"); x.add_argument("--type")
    x = sp.add_parser("events"); x.add_argument("job", nargs="?", type=int)
    for c in ("requeue", "cancel"):
        x = sp.add_parser(c); x.add_argument("ids", nargs="*"); x.add_argument("--level"); x.add_argument("--state")
        x.add_argument("--force", action="store_true")
    x = sp.add_parser("export"); x.add_argument("--level", required=True); x.add_argument("out")
    x.add_argument("--partial", action="store_true"); x.add_argument("--checked", action="store_true", help="refuse unless every job passed the checker")
    x = sp.add_parser("check"); x.add_argument("--level", required=True); x.add_argument("--rows-frac", type=float, default=1.0)
    x.add_argument("--tol", type=float, default=None); x.add_argument("--recheck", action="store_true")
    x.add_argument("--no-quarantine", action="store_true", help="a FAIL does not quarantine the donor")
    x = sp.add_parser("score"); x.add_argument("--level"); x.add_argument("--loop", type=float, default=0)
    x = sp.add_parser("score-import"); x.add_argument("--level", required=True); x.add_argument("dir")
    x.add_argument("--published", metavar="NOTE", help="a released model's codes: replay guard, not a reference")
    x.add_argument("--note")
    x = sp.add_parser("scores"); x.add_argument("--level"); x.add_argument("--job", type=int)
    x = sp.add_parser("versions"); x.add_argument("--days", type=float, default=7); x.add_argument("--json", action="store_true")
    x = sp.add_parser("retire"); x.add_argument("type"); x.add_argument("version", type=int)
    x.add_argument("--migrate", action="store_true"); x.add_argument("--dry-run", action="store_true"); x.add_argument("--undo", action="store_true")
    x = sp.add_parser("release-fetch"); x.add_argument("version", nargs="?", default="latest"); x.add_argument("--file"); x.add_argument("--out")
    x = sp.add_parser("release-show"); x.add_argument("--out")
    a = ap.parse_args()
    if a.cmd in ("release-fetch", "release-show") and (a.out or a.db):      # no database access
        return cmd_release_fetch(a) if a.cmd == "release-fetch" else cmd_release_show(a)
    if not a.db: ap.error("--db (or RQAH_DB) is required")
    global DATA
    DATA = a.data
    try: S = Store(a.db)
    except SchemaTooNew as e: sys.exit(str(e))
    if a.cmd == "invite": print(S.add_invite(a.uses, a.note, a.personal))
    elif a.cmd == "donors": cmd_donors(a, S)
    elif a.cmd == "donor": cmd_donor(a, S)
    elif a.cmd == "status-page":
        page = SP.build(S, SP.load_campaign(a.campaign or os.path.join(os.path.dirname(os.path.abspath(a.db)), "campaign.json")))
        SP.write(a.out, page); print(f"{a.out}: {len(page)} bytes")
    elif a.cmd == "add-worker": wid, tok = S.add_worker(a.name); print(f"worker {wid} token {tok}")
    elif a.cmd == "workers":
        for r in S.q("SELECT * FROM workers ORDER BY id"):
            fl = ",".join(k for k in ("disabled", "trusted", "verifier", "quarantined") if r[k]) + \
                 {0: "", 1: ",probation", 2: ",probation-locked"}.get(r["probation"], "")
            print(r["id"], r["name"], f"donor {r['donor_id']}", fl, time.strftime("%Y-%m-%d %H:%M", time.gmtime(r["last_seen"] or 0)), r["caps"],
                  r["quarantine_reason"] or "")
    elif a.cmd == "disable-worker":
        with S.tx() as db: db.execute("UPDATE workers SET disabled=? WHERE id=?", (0 if a.enable else 1, a.id))
        print(f"worker {a.id} {'enabled' if a.enable else 'disabled'}")
    elif a.cmd == "manifest": cmd_manifest(a)
    elif a.cmd == "datapack": cmd_datapack(a)
    elif a.cmd == "enqueue-level": cmd_enqueue_level(a, S)
    elif a.cmd == "reprioritize": cmd_reprioritize(a, S)
    elif a.cmd == "queue": cmd_queue(a, S)
    elif a.cmd == "score": cmd_score(a, S)
    elif a.cmd == "score-import": cmd_score_import(a, S)
    elif a.cmd == "scores": cmd_scores(a, S)
    elif a.cmd == "enqueue-noop": cmd_enqueue_noop(a, S)
    elif a.cmd == "list": cmd_list(a, S)
    elif a.cmd == "events":
        rows = S.q("SELECT * FROM events" + (" WHERE job_id=?" if a.job else "") + " ORDER BY id", (a.job,) if a.job else ())
        for r in rows: print(f"{r['t']:.1f} job {r['job_id']} w {r['worker_id']} {r['kind']:14s} {(r['lease_id'] or '')[:8]} {r['detail'] or ''}")
    elif a.cmd == "requeue": cmd_requeue(a, S)
    elif a.cmd == "cancel": cmd_cancel(a, S)
    elif a.cmd == "export": cmd_export(a, S)
    elif a.cmd == "check": cmd_check(a, S)
    elif a.cmd == "model-info": print("n_params", write_model_info(a.data, a.model, a.rev, a.ckpt, a.n_params))
    elif a.cmd in ("trust", "verifier"):
        with S.tx() as db:
            db.execute(f"UPDATE workers SET {'trusted' if a.cmd == 'trust' else 'verifier'}=? WHERE id=?", (0 if a.off else 1, a.id))
            S.event(db, f"{a.cmd}{'-off' if a.off else ''}", worker_id=a.id)
        print(f"worker {a.id}: {a.cmd} {'off' if a.off else 'on'}")
    elif a.cmd == "quarantine":
        with S.tx() as db: n = V.quarantine(S, db, a.results, a.id, a.reason)
        print(f"worker {a.id} quarantined; {n} results invalidated")
    elif a.cmd == "unquarantine":
        with S.tx() as db:
            db.execute("UPDATE workers SET quarantined=0, quarantine_reason=NULL WHERE id=?", (a.id,)); S.event(db, "unquarantine", worker_id=a.id)
        print(f"worker {a.id}: quarantine lifted (its invalidated results stay requeued)")
    elif a.cmd == "probation":
        with S.tx() as db:
            if db.execute("UPDATE workers SET probation=? WHERE id=?", (0 if a.off else 1, a.id)).rowcount != 1:
                sys.exit(f"no worker {a.id}")
            S.event(db, f"probation{'-off' if a.off else ''}", worker_id=a.id)
        print(f"worker {a.id}: probation {'off' if a.off else 'on'}")
    elif a.cmd == "spot": cmd_spot(a, S)
    elif a.cmd == "ladder-add": cmd_ladder_add(a, S)
    elif a.cmd == "golden-add": cmd_golden_add(a, S)
    elif a.cmd == "corpus-add": cmd_corpus_add(a, S)
    elif a.cmd == "enqueue-golden": cmd_enqueue_golden(a, S)
    elif a.cmd == "batch-submit":
        try: print(json.dumps(B.submit(S, a.data, man_files(a.data), json.load(open(a.spec)))))
        except B.BatchError as e: sys.exit(f"batch refused: {e}")
    elif a.cmd == "batch-status": print(json.dumps(B.status(S, a.results, a.id), indent=1))
    elif a.cmd == "batch-collect": print(json.dumps(B.collect(S, a.results, a.id, a.out)))
    elif a.cmd == "batch-adopt":
        try: print(json.dumps(B.adopt(S, a.id, a.cands)))
        except B.BatchError as e: sys.exit(str(e))
    elif a.cmd == "exclude-donor": cmd_exclude_donor(a, S)
    elif a.cmd == "points": cmd_points(a, S)
    elif a.cmd == "versions": cmd_versions(a, S)
    elif a.cmd == "retire": cmd_retire(a, S)


if __name__ == "__main__":
    main()
