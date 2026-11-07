#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""probation_test.py [WORKDIR] -- open registration and probation (server/server.py register, server/store.py
register_open, server/sched.py tier) in process, a few seconds: the server's Queue and request handler on a loopback
port, a fixture database and manifest, no GPU.

Checks: /register without an invite -> a worker on probation (with an invite: none, a bad invite still 403); the
per-address limit (IPv4, and IPv6 by /64) and the global hourly limit (429 with retry_s); the client address from
X-Forwarded-For only behind a trusted proxy, rightmost entry; addresses never stored; closed registration refuses a
registration without an invite and still takes invites; a probation worker leases score-mode quant jobs only (not
replicate-mode quant, eval_mix or noop), one at a time; after --probation-pass passed results more at once, still no
eval_mix; after --eval-unlock eval_mix too; an invited worker gets eval_mix at once; a quarantined probation worker
stays locked even when unquarantined (admin.py probation --off lifts it); /me shows the tier, the status page does
not."""
import argparse, json, os, shutil, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server.store import Store
from server.server import Queue, Handler
from server import sched as SCH, verify as V, donor as D, statuspage as SP
from common.util import write_json_atomic
import jobs
from jobs import qwen

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="rqah-probation-"))
RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def sha(i): return f"{i:016x}" + "ab" * 24


def setup(tag, **kw):
    d = f"{W}/{tag}"
    shutil.rmtree(d, ignore_errors=True)
    for x in ("data", "spool", "results"): os.makedirs(f"{d}/{x}")
    a = argparse.Namespace(db=f"{d}/q.sqlite", data=f"{d}/data", spool=f"{d}/spool", results=f"{d}/results", lease_s=120,
                           heartbeat_s=30, chunk_max_mb=64, upload_max_mb=0, spool_max_gb=0, admin_token_file=None,
                           search_token_file=None, spot_rate=0.0, spot_first=0, reserve_max_s=6 * 3600, campaign=None,
                           status_dir=None, retry_s=30, max_leases=64, gang_wait_s=0, class_wait_s=900, class_scan=5000,
                           open_registration=True, reg_per_ip_day=5, reg_global_hour=60, trusted_proxy="",
                           probation_pass=3, eval_unlock=5)
    for k, v in kw.items(): setattr(a, k, v)
    return d, a


class Srv:
    """Queue + the real request handler on 127.0.0.1:<free port>"""
    def __init__(self, a):
        self.Q = Queue(a)
        H = type("H", (Handler,), {"Q": self.Q, "log_message": lambda *x: None})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H); self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def req(self, method, path, body=None, token=None, headers=None):
        h = dict(headers or {})
        if token: h["Authorization"] = f"Bearer {token}"
        data = json.dumps(body).encode() if body is not None else None
        if data is not None: h["Content-Type"] = "application/json"
        try:
            r = urllib.request.urlopen(urllib.request.Request(self.url + path, data=data, method=method, headers=h), timeout=30)
            return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def register(self, name, invite=None, xff=None):
        b = {"name": name}
        if invite: b["invite"] = invite
        return self.req("POST", "/api/v0/register", b, headers={"X-Forwarded-For": xff} if xff else None)

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


def datapack(d, a):
    """a fake 2-layer model: enough for admin.py enqueue-level"""
    root = "models/m@r"
    os.makedirs(f"{d}/data/{root}")
    json.dump({"num_hidden_layers": 2, "quantization_config": {}}, open(f"{d}/data/{root}/config.json", "w"))
    files = {f"{root}/config.json": (1, 100), f"{root}/weights/head.safetensors": (2, 1000), f"{root}/hessians/lm_head.input.safetensors": (3, 100)}
    for l in range(2):
        files[f"{root}/weights/L{l:02d}.safetensors"] = (10 + l, 1000)
        for i, (h, _) in enumerate(qwen.layer_plan("qwen3_5", l)): files[f"{root}/hessians/{h}.safetensors"] = (100 + 10 * l + i, 100)
    write_json_atomic(f"{d}/data/manifest.json", {"files": {n: {"sha256": sha(i), "size": s, "mtime": 0} for n, (i, s) in files.items()}})
    for K, lv, extra in [(K, f"q-k{K}", ["--verify", "score"]) for K in ("2", "2.5", "3", "3.5", "4", "5", "6", "7", "8")] + [("2.25", "q-rep", [])]:
        r = subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", a.db, "--data", a.data, "enqueue-level", "--model", "m",
                            "--rev", "r", "--K", K, "--codebook", "i8", "--level", lv, *extra], capture_output=True, text=True)
        if r.returncode: print(r.stdout, r.stderr)


def eval_job(S, prio=100):
    m = jobs.get("eval_mix")
    p = {"model": "m", "revision": "r", "base": "q-k3", "mix": [], "golden": "g", "kv": "bf16", "max_model_len": 4096,
         "bytes": 10 ** 10, "n_params": 27 * 10 ** 9, "rows": 1000}
    with S.tx() as db: return S.enqueue(db, "eval_mix", m.VERSION, p, [], jobs.requirements(m, p), 0, priority=prio)


def noop_job(S, prio=100):
    m = jobs.get("noop"); p = {"seconds": 0, "size": 10, "seed": 0}
    with S.tx() as db: return S.enqueue(db, "noop", m.VERSION, p, [], jobs.requirements(m, p), 0, priority=prio)


TYPES = ("quant_layer", "quant_tensor", "quant_head", "eval_mix", "noop")
CAPS = {"job_types": list(TYPES), "versions": {t: jobs.get(t).VERSION for t in TYPES}, "gpu": True, "cards": 4, "slots": 4,
        "vram_gb": 32, "ram_gb": 256, "disk_gb": 1000}


def give_passes(S, wid, n, base_lumens=1.0):
    """n quant jobs accepted from wid and passed by the scorer, lumens granted (what score.py would leave behind)"""
    with S.tx() as db:
        js = db.execute("SELECT id FROM jobs WHERE state='queued' AND type='quant_layer' AND params LIKE '%\"verify\": \"score\"%' "
                        "ORDER BY id LIMIT ?", (n,)).fetchall()
        for (j,) in js:
            db.execute("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=?, check_state='pass' WHERE id=?", (wid, time.time(), j))
            db.execute("INSERT INTO points(t, worker_id, job_id, lumens, kind) VALUES(?,?,?,?, 'grant')", (time.time(), wid, j, base_lumens))
    return len(js)


def release(S, wid):
    """hand back every active lease of wid (the worker abandons them), so the next lease starts clean"""
    with S.tx() as db:
        for l in db.execute("SELECT * FROM leases WHERE worker_id=? AND state='active'", (wid,)).fetchall():
            S.end_lease(db, l, "abandoned", count_attempt=False)


def lease_all(Q, w, caps, n=8):
    got = []
    for _ in range(n):
        r = Q.lease(w, caps)
        if not r.get("job"): return got, r
        got.append(r["job"])
    return got, {}


def main():
    os.makedirs(W, exist_ok=True)
    # ---------------------------------------------------------------- registration
    d, a = setup("reg", reg_per_ip_day=2, reg_global_hour=6)
    srv = Srv(a); S = srv.Q.store
    inv = S.add_invite(uses=5)
    c1, r1 = srv.register("open1")
    c2, r2 = srv.register("inv1", invite=inv)
    c3, r3 = srv.register("bad1", invite="ri-not-an-invite")
    row = lambda wid: S.q("SELECT * FROM workers WHERE id=?", (wid,))[0]
    ok("register without an invite -> a worker on probation with a token that works; with an invite -> no probation; "
       "a wrong invite -> 403 (not quietly probation)",
       c1 == 200 and r1["probation"] is True and row(r1["worker_id"])["probation"] == 1 and
       srv.req("GET", "/api/v0/status", token=r1["token"])[1]["me"]["tier"] == "probation" and
       c2 == 200 and r2["probation"] is False and row(r2["worker_id"])["probation"] == 0 and c3 == 403, f"{c1} {c2} {c3}")
    c4, _ = srv.register("open2")
    c5, r5 = srv.register("open3")
    c6, _ = srv.register("inv2", invite=inv)
    ok("per-address limit (2/day): the third registration without an invite from one address -> 429 with retry_s ~ 1 day; "
       "invites are not limited", c4 == 200 and c5 == 429 and 86000 < r5.get("retry_s", 0) <= 86401 and c6 == 200, f"{c5} {r5}")
    cx, _ = srv.register("spoof", xff="203.0.113.9")
    ok("X-Forwarded-For from a peer that is not a trusted proxy is ignored (still the loopback's limit)", cx == 429)
    srv.close()

    d, a = setup("proxy", reg_per_ip_day=2, reg_global_hour=8, trusted_proxy="127.0.0.1")
    srv = Srv(a); S = srv.Q.store
    res = [srv.register(f"p{i}", xff=x)[0] for i, x in enumerate(["198.51.100.7", "198.51.100.7", "198.51.100.7",
                                                                  "203.0.113.5, 198.51.100.8", "203.0.113.6, 198.51.100.8",
                                                                  "203.0.113.7, 198.51.100.8"])]
    ok("behind a trusted proxy: the client address is the rightmost X-Forwarded-For entry; entries the client sent "
       "(further left) do not reset the limit", res == [200, 200, 429, 200, 200, 429], str(res))
    v6 = [srv.register(f"v6{i}", xff=x)[0] for i, x in enumerate(["2001:db8:1:2::10", "2001:db8:1:2:aaaa::1", "2001:db8:1:2::ffff",
                                                                  "2001:db8:1:3::1"])]
    ok("IPv6 counts per /64 (three hosts of one /64: the third refused; another /64 is fine)", v6 == [200, 200, 429, 200], str(v6))
    g0, g = srv.register("global1", xff="192.0.2.77"), srv.register("global2", xff="192.0.2.78")
    ok("global limit (8/h): the 8th registration passes, the next one from a fresh address -> 429",
       g0[0] == 200 and g[0] == 429 and 0 < g[1].get("retry_s", 0) <= 3601, str(g))
    raw = b"".join(open(f, "rb").read() for f in (a.db, a.db + "-wal") if os.path.exists(f))
    ok("client addresses are never stored (only keyed hashes in reg_log)",
       not any(x.encode() in raw for x in ("198.51.100.7", "198.51.100.8", "203.0.113.5", "2001:db8:1:2", "127.0.0.1")) and
       S.q("SELECT COUNT(*) FROM reg_log")[0][0] == 8)
    srv.close()

    d, a = setup("closed", open_registration=False)
    srv = Srv(a); S = srv.Q.store
    c1, r1 = srv.register("x1"); c2, r2 = srv.register("x2", invite=S.add_invite())
    ok("--no-open-registration: no invite -> 403 with the reason; an invite still registers",
       c1 == 403 and "invite" in r1.get("error", "") and c2 == 200 and r2["probation"] is False, f"{c1} {r1}")
    srv.close()

    # ---------------------------------------------------------------- probation tiers
    d, a = setup("tiers")
    datapack(d, a)
    Q = Queue(a); S = Q.store
    ev = eval_job(S); nj = noop_job(S)
    with S.tx() as db:                                      # the replicate-mode level first: it must still be skipped
        db.execute("UPDATE jobs SET priority=200 WHERE level='q-rep'")
    wp, tp = S.register_open("prob", "192.0.2.10")
    w = S.auth(tp)
    got, why = lease_all(Q, w, CAPS)
    ok("probation: one job at a time, a score-mode quant job (not the replicate-mode level, eval_mix or noop, all "
       "queued at higher priority); the refusal says why",
       len(got) == 1 and got[0]["type"] in SCH.QUANT and got[0]["params"].get("verify") == "score" and "probation" in why.get("why", ""),
       f"{[(j['type'], j['params'].get('level')) for j in got]} {why}")
    release(S, wp)
    me0 = D.me(S, w)["workers"][0]
    queued = lambda: S.q("SELECT type, params FROM jobs WHERE state='queued'")
    n0 = sum(1 for j in queued() if SCH.score_quant(j))
    give_passes(S, wp, 3)
    got, _ = lease_all(Q, w, CAPS)
    ok("after 3 passed results: several at once (4 cards), still score-mode quant only (no eval_mix, no replicate-mode, no noop)",
       len(got) == 4 and all(j["type"] in SCH.QUANT and j["params"].get("verify") == "score" for j in got),
       str([(j["type"], j["params"].get("level")) for j in got]))
    release(S, wp)
    me1 = D.me(S, w)["workers"][0]
    give_passes(S, wp, 2)
    me2 = D.me(S, w)["workers"][0]; n2 = len(queued())
    got, _ = lease_all(Q, w, CAPS)
    unlocked = lambda got: [j for j in got if j["id"] in (ev, nj) or j["params"].get("level") == "q-rep"]
    ok("after 5 passed results: every type unlocked: the replicate-mode level (priority 200) and eval_mix / noop (100) "
       "come first", len(got) == 4 and len(unlocked(got)) == 4 and ev in [j["id"] for j in got], str([(j["id"], j["type"]) for j in got]))
    ok("/me shows the tier (probation 0 -> quant 3 -> full 5, with the thresholds); could_run follows it",
       me0["tier"] == {"tier": "probation", "passed": 0, "probation_pass": 3, "eval_unlock": 5} and me1["tier"]["tier"] == "quant"
       and me1["tier"]["passed"] == 3 and me2["tier"]["tier"] == "full" and me2["tier"]["passed"] == 5
       and me0["could_run"] == n0 and me2["could_run"] == n2,
       f"{me0['tier']} {me1['tier']} {me2['tier']} could_run {me0['could_run']}/{n0} {me2['could_run']}/{n2}")
    release(S, wp)
    # a duplicate (passed, no lumens) does not count
    wd, td = S.register_open("dup", "192.0.2.11")
    give_passes(S, wd, 3, base_lumens=0.0)
    ok("passed results that earned nothing (duplicates of another donor's codes) do not count",
       SCH.passed_quant(S.q, wd) == 0 and Q.tier(S.q, S.auth(td))[0] == "probation")
    # an invited worker: eval_mix at once
    wi, ti = S.add_worker("invited")
    got, _ = lease_all(Q, S.auth(ti), CAPS)
    ok("a worker registered with an invite (or by the operator) skips probation: replicate-mode quant and eval_mix at once",
       len(got) == 4 and len(unlocked(got)) == 4 and ev in [j["id"] for j in got], str([(j["id"], j["type"]) for j in got]))
    release(S, wi)
    # quarantine locks it for good
    wq, tq = S.register_open("caught", "192.0.2.12")
    with S.tx() as db: V.quarantine(S, db, a.results, wq, "spot check failed")
    with S.tx() as db: db.execute("UPDATE workers SET quarantined=0 WHERE id=?", (wq,))        # admin.py unquarantine
    give_passes(S, wq, 6)
    got, why = lease_all(Q, S.auth(tq), CAPS)
    adm = subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", a.db, "--data", a.data, "probation", str(wq), "--off"],
                         capture_output=True, text=True)
    release(S, wq)
    got2, _ = lease_all(Q, S.auth(tq), CAPS)
    ok("a quarantined probation worker stays locked after unquarantine, whatever it passed (one score-mode quant job at a "
       "time); admin.py probation --off lifts it",
       S.q("SELECT probation FROM workers WHERE id=?", (wq,))[0][0] == 0 and len(got) == 1 and
       got[0]["params"].get("verify") == "score" and adm.returncode == 0 and len(got2) == 4 and len(unlocked(got2)) == 4,
       f"{[(j['type'], j['params'].get('level')) for j in got]} {why} / {[(j['type'], j['params'].get('level')) for j in got2]}")
    page = SP.build(S).decode()
    ok("the status page says nothing about probation or tiers", "probation" not in page.lower() and "tier" not in page.lower())
    n_ok = sum(r[1] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
