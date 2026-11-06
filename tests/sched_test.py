#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""sched_test.py [WORKDIR] -- the 0.4 scheduling rules (server/sched.py, docs/SCHED.md) in process, no HTTP, no GPU
(a few seconds): the server's Queue.lease against a fixture database and manifest.

Checks: K-granularity priorities (enqueue-level defaults, --priority-boost, reprioritize by pattern, heads +1);
priority classes strictly in order; inside a class whole levels first even when their job ids interleave; cache
affinity inside a level but never across classes; a budgeted worker gets the least download first; the budget
filter (download = uncached inputs only, upload = the type's expected upload) and a randomized property check that no
granted job exceeds the declared budget; the over-budget answer; the cache report protocol (full list, rev only,
unknown rev after a server restart); the class-wait drain (a busy worker passing over an aged higher-class job is
drained for it, not for a same-class one, not before the window); the expect field of a granted job."""
import argparse, json, os, random, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server.store import Store
from server.server import Queue
from server import sched as SCH
from common.util import write_json_atomic
import jobs
from jobs import qwen

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="rqah-sched-"))
RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def sha(i): return f"{i:016x}" + "ab" * 24


def setup(tag):
    d = f"{W}/{tag}"
    shutil.rmtree(d, ignore_errors=True)
    for x in ("data", "spool", "results"): os.makedirs(f"{d}/{x}")
    a = argparse.Namespace(db=f"{d}/q.sqlite", data=f"{d}/data", spool=f"{d}/spool", results=f"{d}/results", lease_s=120,
                           heartbeat_s=30, chunk_max_mb=64, upload_max_mb=0, spool_max_gb=0, admin_token_file=None,
                           search_token_file=None, spot_rate=0.0, spot_first=0, reserve_max_s=6 * 3600, campaign=None,
                           status_dir=None, retry_s=30, max_leases=64, gang_wait_s=0, class_wait_s=900, class_scan=5000)
    return d, a


def manifest(d, files):
    """files: {name: (sha index, size)}"""
    write_json_atomic(f"{d}/data/manifest.json", {"files": {n: {"sha256": sha(i), "size": s, "mtime": 0} for n, (i, s) in files.items()}})


def noop(S, inputs=(), size=1000, prio=0, level=None, ram=0, cards=0, sizes=None):
    p = {"seconds": 0, "size": size, "seed": 0}
    if inputs: p["inputs"] = list(inputs)
    if ram: p["ram_gb"] = ram
    m = jobs.get("noop")
    with S.tx() as db:
        return S.enqueue(db, "noop", m.VERSION, p, list(inputs), jobs.requirements(m, p), sum((sizes or {}).get(n, 0) for n in inputs),
                         level=level, priority=prio)


def caps(**kw):
    c = {"job_types": ["noop"], "versions": {"noop": jobs.get("noop").VERSION}, "gpu": False, "cards": 0, "slots": 64,
         "ram_gb": 64, "disk_gb": 1000}
    c.update(kw); return c


def drain_order(Q, w, c, n):
    out = []
    for _ in range(n):
        r = Q.lease(w, c)
        if not r.get("job"): break
        out.append(r["job"]["id"])
    return out


def worker(S, name):
    wid, tok = S.add_worker(name)
    return S.auth(tok)


def main():
    os.makedirs(W, exist_ok=True)
    # ---------------------------------------------------------------- priorities from K
    lp = SCH.level_priority
    got = [lp(3), lp("2.5"), lp(2.25), lp(2.125), lp(2.0625), lp(2.03125), lp(4, 3), lp(2.75, -5)]
    ok("level_priority: integer 40, half 30, quarter 20, 1/8 10, 1/16 5, finer 0, + boost", got == [40, 30, 20, 10, 5, 0, 43, 15], got)

    # enqueue-level defaults and reprioritize (admin.py on a fake datapack: the manifest only lists the inputs)
    d, a = setup("admin")
    root = "models/m@r"
    os.makedirs(f"{d}/data/{root}")
    json.dump({"num_hidden_layers": 2, "quantization_config": {}}, open(f"{d}/data/{root}/config.json", "w"))
    files = {f"{root}/config.json": (1, 100), f"{root}/weights/head.safetensors": (2, 1000), f"{root}/hessians/lm_head.input.safetensors": (3, 100)}
    for l in range(2):
        files[f"{root}/weights/L{l:02d}.safetensors"] = (10 + l, 1000)
        for i, (h, _) in enumerate(qwen.layer_plan("qwen3_5", l)): files[f"{root}/hessians/{h}.safetensors"] = (100 + 10 * l + i, 100)
    manifest(d, files)
    adm = lambda *x: subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", a.db, "--data", a.data, *x],
                                    capture_output=True, text=True)
    for K, lv, extra in (("3", "q-k3", []), ("2.5", "q-k2.5", []), ("2.25", "q-k2.25", ["--head"]), ("2.125", "q-k2.125", []),
                         ("2.75", "q-k2.75", ["--priority-boost", "5"])):
        r = adm("enqueue-level", "--model", "m", "--rev", "r", "--K", K, "--codebook", "i8", "--level", lv, *extra)
        if r.returncode: print(r.stdout, r.stderr)
    S = Store(a.db)
    pr = {(r["level"], r["type"]): r["priority"] for r in S.q("SELECT level, type, priority FROM jobs")}
    want = {("q-k3", "quant_layer"): 40, ("q-k2.5", "quant_layer"): 30, ("q-k2.25", "quant_layer"): 20, ("q-k2.25", "quant_head"): 21,
            ("q-k2.125", "quant_layer"): 10, ("q-k2.75", "quant_layer"): 25}
    ok("enqueue-level: default priority from K's granularity (+ boost; head one above its level)", pr == want, pr)
    r = adm("reprioritize", "q-k2.*", "--by-granularity", "--priority-boost", "1", "--dry-run")
    pr2 = {(x["level"], x["type"]): x["priority"] for x in S.q("SELECT level, type, priority FROM jobs")}
    ok("reprioritize --dry-run changes nothing", pr2 == want and "would change 9 jobs" in r.stdout, r.stdout.strip().splitlines()[-1:])
    r = adm("reprioritize", "q-k2.*", "--by-granularity", "--priority-boost", "1")
    pr3 = {(x["level"], x["type"]): x["priority"] for x in S.q("SELECT level, type, priority FROM jobs")}
    ok("reprioritize 'q-k2.*' --by-granularity --priority-boost 1: matching levels only, heads +1",
       pr3 == {**want, ("q-k2.5", "quant_layer"): 31, ("q-k2.25", "quant_layer"): 21, ("q-k2.25", "quant_head"): 22,
               ("q-k2.125", "quant_layer"): 11, ("q-k2.75", "quant_layer"): 21}, pr3)
    with S.tx() as db: db.execute("UPDATE jobs SET state='accepted' WHERE level='q-k3' AND id=(SELECT MIN(id) FROM jobs WHERE level='q-k3')")
    r = adm("reprioritize", "q-k3", "--priority", "7")
    st = {(x["state"], x["priority"]) for x in S.q("SELECT state, priority FROM jobs WHERE level='q-k3'")}
    ok("reprioritize --priority P: queued jobs only (an accepted one keeps its priority)", st == {("accepted", 40), ("queued", 7)}, st)
    r = adm("queue", "--limit", "3")
    ok("admin queue: scheduling order listing", r.returncode == 0 and "by class" in r.stdout, r.stdout.strip().splitlines()[-1:])
    r = adm("reprioritize", "nomatch*", "--by-granularity")
    ok("reprioritize: a pattern that matches nothing is an error", r.returncode != 0)

    # ---------------------------------------------------------------- ordering
    d, a = setup("order")
    files = {f"test/f{i}": (i, 1000 * (i + 1)) for i in range(40)}
    manifest(d, files); sizes = {n: s for n, (_, s) in files.items()}
    Q = Queue(a); S = Q.store
    low = noop(S, ["test/f0"], prio=10, level="lowA", sizes=sizes)
    A1 = noop(S, ["test/f1"], prio=40, level="A", sizes=sizes)
    B1 = noop(S, ["test/f2"], prio=40, level="B", sizes=sizes)
    A2 = noop(S, ["test/f3"], prio=40, level="A", sizes=sizes)
    B2 = noop(S, ["test/f4"], prio=40, level="B", sizes=sizes)
    A3 = noop(S, ["test/f5"], prio=40, level="A", sizes=sizes)
    H = noop(S, ["test/f6"], prio=41, level="A", sizes=sizes)
    w = worker(S, "w1")
    order = drain_order(Q, w, caps(), 10)
    ok("classes strictly in order; inside a class whole levels first (A before B though their ids interleave)",
       order == [H, A1, A2, A3, B1, B2, low], order)

    # cache affinity: inside the level, never across classes
    d, a = setup("affinity")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    hi1 = noop(S, ["test/f1"], prio=40, level="A", sizes=sizes)
    hi2 = noop(S, ["test/f2"], prio=40, level="A", sizes=sizes)
    hi3 = noop(S, ["test/f3"], prio=40, level="A", sizes=sizes)
    hb1 = noop(S, ["test/f7"], prio=40, level="B", sizes=sizes)
    lo1 = noop(S, ["test/f4"], prio=30, level="C", sizes=sizes)
    w = worker(S, "w2")
    cached = [sha(3)[:16], sha(4)[:16], sha(7)[:16]]
    c = caps(cache={"rev": "0" * 15 + "1", "shas": cached})
    order = drain_order(Q, w, c, 10)
    ok("cache affinity inside the level (A3 cached goes first), level order kept (A before B), never across classes (C last)",
       order == [hi3, hi1, hi2, hb1, lo1], order)

    # budgeted worker: least download first
    d, a = setup("budgeted")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    j_big = noop(S, ["test/f30"], prio=40, level="A", sizes=sizes)
    j_mid = noop(S, ["test/f10"], prio=40, level="A", sizes=sizes)
    j_cached = noop(S, ["test/f35"], prio=40, level="B", sizes=sizes)
    j_small = noop(S, ["test/f1"], prio=40, level="B", sizes=sizes)
    w = worker(S, "w3")
    c = caps(budget={"down": 10 ** 9, "up": None}, cache={"rev": "0" * 15 + "2", "shas": [sha(35)[:16]]})
    order = drain_order(Q, w, c, 10)
    ok("a budgeted worker takes the least download first (cached, then small)", order == [j_cached, j_small, j_mid, j_big], order)

    # ---------------------------------------------------------------- budget filter
    d, a = setup("budget")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    big_in = noop(S, ["test/f20"], size=100, prio=40, sizes=sizes)            # 21000 bytes down
    big_out = noop(S, ["test/f0"], size=50000, prio=40, sizes=sizes)          # 50000 bytes up
    cached_big = noop(S, ["test/f25"], size=100, prio=40, sizes=sizes)        # 26000 bytes, cached
    w = worker(S, "w4")
    c = caps(budget={"down": 5000, "up": 10000}, cache={"rev": "0" * 15 + "3", "shas": [sha(25)[:16]]})
    r = Q.lease(w, c)
    ok("budget: only the job whose uncached download and upload fit (its input is cached)",
       r.get("job") and r["job"]["id"] == cached_big and r["job"]["expect"] == {"down": 0, "up": 100}, r.get("job", {}).get("expect"))
    r = Q.lease(w, dict(c, cache={"rev": "0" * 15 + "3"}))
    ok("budget: nothing else fits -> over_budget answer, longer retry", r.get("job") is None and r.get("over_budget") == 2 and r["retry_s"] >= 60,
       {k: r.get(k) for k in ("over_budget", "retry_s", "why")})
    r = Q.lease(w, dict(c, budget={"down": None, "up": 60000}, cache={"rev": "0" * 15 + "3"}))
    ok("unlimited download, upload 60000: the big-input job comes next (expect.down = its size)",
       r.get("job") and r["job"]["id"] == big_in and r["job"]["expect"]["down"] == 21000, r.get("job", {}).get("expect"))
    r = Q.lease(w, dict(c, budget={"down": None, "up": 49999}, cache={"rev": "0" * 15 + "3"}))
    ok("upload budget 1 byte short of the job's expected upload: not offered", r.get("job") is None and r.get("over_budget") == 1)

    # randomized: no granted job ever exceeds the declared budget
    d, a = setup("fuzz")
    rnd = random.Random(7)
    fz = {f"test/z{i}": (1000 + i, rnd.randint(1, 10 ** 6)) for i in range(60)}
    manifest(d, fz); zs = {n: s for n, (_, s) in fz.items()}
    Q = Queue(a); S = Q.store
    for i in range(300):
        ins = rnd.sample(sorted(fz), rnd.randint(0, 3))
        noop(S, ins, size=rnd.randint(0, 10 ** 6), prio=rnd.choice([0, 10, 20, 30, 40]), level=rnd.choice(["a", "b", "c", None]), sizes=zs)
    w = worker(S, "w5")
    bad, granted, refused = [], 0, 0
    for t in range(250):
        cs = rnd.sample(sorted(fz), rnd.randint(0, 30))
        shas = sorted({sha(fz[n][0])[:16] for n in cs})
        bd = rnd.choice([None, rnd.randint(0, 3 * 10 ** 6), rnd.randint(0, 10 ** 4)])
        bu = rnd.choice([None, rnd.randint(0, 10 ** 6), rnd.randint(0, 10 ** 3)])
        c = caps(slots=1, budget={"down": bd, "up": bu}, cache={"rev": f"{t:016x}", "shas": shas})
        r = Q.lease(w, c)
        if not r.get("job"): refused += 1; continue
        j = r["job"]; granted += 1
        down = sum(i["size"] for i in j["inputs"] if i["sha256"][:16] not in shas)
        up = j["params"]["size"]
        if (bd is not None and down > bd) or (bu is not None and up > bu) or j["expect"] != {"down": down, "up": up}: bad.append(j["id"])
        with S.tx() as db:                                  # give it back so the next round sees a full queue
            db.execute("UPDATE leases SET state='abandoned' WHERE job_id=?", (j["id"],))
            db.execute("UPDATE jobs SET state='queued', lease_id=NULL WHERE id=?", (j["id"],))
    ok("randomized (250 lease requests, 300 jobs): no granted job exceeds the budget, expect = true down / up",
       not bad and granted > 50 and refused > 5, f"granted {granted}, refused {refused}, bad {bad[:5]}")

    # ---------------------------------------------------------------- cache report protocol
    d, a = setup("cacheproto")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    j1 = noop(S, ["test/f1"], prio=0, sizes=sizes)
    w = worker(S, "w6")
    r = Q.lease(w, caps(cache={"rev": "a" * 16}))
    ok("cache report: an unknown rev without the list -> cache_unknown (the worker resends it)", r.get("cache_unknown") is True)
    r = Q.lease(w, caps(cache={"rev": "a" * 16, "shas": [sha(1)[:16]]}))
    ok("cache report: the full list is stored; the job's expect.down counts the cache", r.get("job") and r["job"]["expect"]["down"] == 0)
    stored = json.loads(S.q("SELECT caps FROM workers WHERE id=?", (w["id"],))[0][0])
    ok("the stored caps keep the cache rev and count, not the list", stored.get("cache") == {"rev": "a" * 16, "n": 1}, stored.get("cache"))
    Q2 = Queue(a)                                           # a restarted server has no reports
    r = Q2.lease(w, caps(cache={"rev": "a" * 16}))
    ok("after a server restart the known rev is unknown again", r.get("cache_unknown") is True)

    # ---------------------------------------------------------------- class-wait drain
    d, a = setup("starve")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    busy = noop(S, prio=50, ram=8, sizes=sizes)
    w = worker(S, "w7")
    c = caps(ram_gb=16, slots=4)
    r = Q.lease(w, c)
    ok("starve setup: a running 8 GB job", r.get("job") and r["job"]["id"] == busy)
    hi = noop(S, prio=40, ram=12, sizes=sizes)              # does not fit while the 8 GB job runs
    lo = noop(S, prio=10, ram=2, sizes=sizes)
    r = Q.lease(w, c)
    ok("a fresh higher-class job that does not fit now: the lower-class job is leased", r.get("job") and r["job"]["id"] == lo)
    with S.tx() as db:
        db.execute("UPDATE leases SET state='abandoned' WHERE job_id=?", (lo,))
        db.execute("UPDATE jobs SET state='queued', lease_id=NULL WHERE id=?", (lo,))
        db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 1000, hi))
    r = Q.lease(w, c)
    res = S.q("SELECT * FROM reservations")
    ok("after --class-wait-s the worker is drained for it (reservation, nothing lower leased)",
       r.get("job") is None and r.get("draining") == hi and [x["job_id"] for x in res] == [hi], r.get("why"))
    r = Q.lease(w, c)
    ok("while draining: still nothing", r.get("job") is None and r.get("draining") == hi)
    with S.tx() as db:                                      # the 8 GB job ends
        db.execute("UPDATE leases SET state='done' WHERE job_id=?", (busy,)); db.execute("UPDATE jobs SET state='accepted' WHERE id=?", (busy,))
    r = Q.lease(w, c)
    ok("once room frees up the drained-for job is leased", r.get("job") and r["job"]["id"] == hi and not S.q("SELECT * FROM reservations"))
    d, a = setup("starve-same")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    busy = noop(S, prio=50, ram=8, sizes=sizes); w = worker(S, "w8"); Q.lease(w, c)
    hi = noop(S, prio=40, ram=12, sizes=sizes); same = noop(S, prio=40, ram=2, sizes=sizes)
    with S.tx() as db: db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 1000, hi))
    r = Q.lease(w, c)
    ok("an aged job does not drain a worker for work of its own class", r.get("job") and r["job"]["id"] == same)
    d, a = setup("starve-budget")
    manifest(d, files)
    Q = Queue(a); S = Q.store
    busy = noop(S, prio=50, ram=8, sizes=sizes); w = worker(S, "w9"); Q.lease(w, c)
    hi = noop(S, ["test/f30"], prio=40, ram=12, sizes=sizes); lo = noop(S, prio=10, ram=2, sizes=sizes)
    with S.tx() as db: db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 1000, hi))
    r = Q.lease(w, dict(c, budget={"down": 100, "up": None}))
    ok("a job over the worker's budget never drains it (it could not run it anyway)", r.get("job") and r["job"]["id"] == lo)

    n = sum(1 for _, p in RESULTS if p)
    print(f"\n{n}/{len(RESULTS)} checks passed")
    return 0 if n == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
