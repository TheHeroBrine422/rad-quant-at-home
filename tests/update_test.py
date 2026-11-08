#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""update_test.py [WORKDIR] -- updates in process, no GPU, no network beyond loopback, ~30 s (README "Updating",
RUNBOOK.md "Upgrades"; tests/upgrade_e2e.py runs the whole path with real processes and the 0.4.0 worker).

Checks:
- versions: X-RQAH-Version, the user agents of 0.3 / 0.4 workers, anything else unknown; GET /api/v0/version without
  a token; every lease answer, /status and /me carry {current, min}; the server records each worker's release
- below the minimum: 426 on /lease with the way to update (and retry_s), while the jobs it holds still heartbeat,
  upload and complete; a worker without a version (the CLI, curl) is not refused
- job versions: a job is leased only at the version the worker declares and only if this server's code accepts it
  (VERSION + ACCEPTS); a running job whose version the server stopped accepting gets a clear 409, not an odd failure;
  the startup report names queued jobs nobody will be leased; admin.py versions / retire (refused while running or
  queued, --migrate re-stamps queued jobs and their dedupe keys, --dry-run, --undo, the lease filter honours it)
- schema: a new database starts at SCHEMA_VERSION with no backup; a pre-0.5 one is backed up once and migrated
  (columns, donors); reopening is a no-op; a database from a newer server is refused by Store, server.py and admin.py
- restart: leases that were active when the server went down survive a downtime longer than the lease (expiry gets
  a fresh lease_s, the deadline moves by the downtime); a server_start event
- the worker (worker.py against this server): state.json written; an available update logged exactly once and kept in
  update.json; a 426 logged once as UPDATE NEEDED, nothing leased, the worker keeps running; a state from a newer
  release has its job records set aside; a cache index entry of the wrong size and a foreign index are rebuilt
  safely; a resumed lease at another job version is handed back (abandoned, not counted)
- rqah status: the VERSION column, "update available: X -> Y" with `rqah update` and the exact image to pull, the
  below-minimum line; rqah update refuses without a config
- the status page: the share of recent workers on the current release and on older ones
- the setup wizard's --update against a simulated Docker: the order pull -> drain -> rewrite -> rm -> up, the
  restart policy off during the drain, never a volume removal, the pin and the invite kept; --release, the drain
  timeout (second SIGTERM), --dry-run, a failed pull changes nothing, an unreachable queue, a malformed or older
  advertised release, an image override, up to date; --start drains a running worker first"""
import argparse, contextlib, hashlib, io, json, os, re, shutil, sqlite3, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server import store as ST, admin as AD, statuspage as SP
from server.store import Store, SchemaTooNew
from server.server import Queue, Handler
from common.util import VERSION, VERSION_HEADER, vtuple, vstr, write_json_atomic
from client import rqah as CLI, setup as WZ
from common import release as RL
import jobs
from jobs import noop

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="rqah-update-"))
RESULTS = []
NEXT = vstr(vtuple(VERSION)[:1] + (vtuple(VERSION)[1] + 1, 0))            # the release after this one
PREV = "0.4.0" if vtuple(VERSION) > (0, 4, 0) else "0.3.0"


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def args(d, **kw):
    a = argparse.Namespace(db=f"{d}/q.sqlite", data=f"{d}/data", spool=f"{d}/spool", results=f"{d}/results", lease_s=120,
                           heartbeat_s=30, chunk_max_mb=64, upload_max_mb=0, spool_max_gb=0, admin_token_file=None,
                           search_token_file=None, spot_rate=0.0, spot_first=0, reserve_max_s=6 * 3600, campaign=None,
                           status_dir=None, retry_s=1, max_leases=64, gang_wait_s=0, class_wait_s=900, class_scan=5000,
                           open_registration=True, reg_per_ip_day=50, reg_global_hour=600, trusted_proxy="",
                           probation_pass=3, eval_unlock=5, worker_current=None, worker_min=None)
    for k, v in kw.items(): setattr(a, k, v)
    return a


def fresh(tag):
    d = f"{W}/{tag}"
    shutil.rmtree(d, ignore_errors=True)
    for x in ("data", "spool", "results"): os.makedirs(f"{d}/{x}")
    write_json_atomic(f"{d}/data/manifest.json", {"files": {}})
    return d


class Srv:
    def __init__(self, a):
        self.Q = Queue(a)
        H = type("H", (Handler,), {"Q": self.Q, "log_message": lambda *x: None})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H); self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def req(self, method, path, body=None, token=None, headers=None, raw=None):
        h = dict(headers or {})
        if token: h["Authorization"] = f"Bearer {token}"
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        try:
            r = urllib.request.urlopen(urllib.request.Request(self.url + path, data=data, method=method, headers=h), timeout=30)
            return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def close(self): self.httpd.shutdown(); self.httpd.server_close()


def caps(v=None, types=("noop",)):
    return {"job_types": list(types), "versions": {t: (v if v is not None else jobs.get(t).VERSION) for t in types}, "gpu": False,
            "cards": 0, "slots": 8, "ram_gb": 8, "disk_gb": 10}


def noop_job(S, version=None, seed=0, seconds=0, ukey=None):
    p = {"seconds": seconds, "size": 10, "seed": seed}
    with S.tx() as db:
        return S.enqueue(db, "noop", version or noop.VERSION, p, [], jobs.requirements(noop, p), 0,
                         ukey=ukey or f"noop/v{version or noop.VERSION}:s{seed}")


def wv(v): return {VERSION_HEADER: v}


@contextlib.contextmanager
def noop_as(version, accepts=()):
    """this process's noop module as if it were at another version (the next release's server code)"""
    old = (noop.VERSION, getattr(noop, "ACCEPTS", None))
    noop.VERSION = version; noop.ACCEPTS = tuple(accepts)
    try: yield
    finally:
        noop.VERSION = old[0]
        if old[1] is None: del noop.ACCEPTS
        else: noop.ACCEPTS = old[1]


def capture(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try: rc = fn(*a, **kw)
        except SystemExit as e: rc = e.code
    return rc, out.getvalue()


# ================================================================ versions on the wire
def t_versions():
    d = fresh("ver"); s = Srv(args(d, worker_current=NEXT, worker_min=PREV)); S = s.Q.store
    cv = Queue.client_version
    ok("client version: header wins, else the 0.4 / 0.3 user agent as X.Y.0, else unknown",
       cv({VERSION_HEADER: "0.5.0", "User-Agent": "rad-quant-at-home-worker/0.4"}) == "0.5.0"
       and cv({"User-Agent": "rad-quant-at-home-worker/0.4"}) == "0.4.0" and cv({"User-Agent": "rad-quant-at-home-worker/0.2"}) == "0.2.0"
       and cv({"User-Agent": "rqah-cli/0.5.0"}) is None and cv({VERSION_HEADER: "0.5.0; x", "User-Agent": "curl/8"}) is None
       and cv({}) is None)
    st, v = s.req("GET", "/api/v0/version")
    ok("GET /api/v0/version needs no token", st == 200 and v["worker"] == {"current": NEXT, "min": PREV} and v["server"] == VERSION, str(v))
    ok("the advertised versions are in meta (status page)", S.meta_get("worker_current") == NEXT and S.meta_get("worker_min") == PREV)

    wid, tok = S.add_worker("w-new"); wid2, tok2 = S.add_worker("w-old"); wid3, tok3 = S.add_worker("w-ancient")
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok, wv(VERSION))
    ok("a lease answer carries {current, min} and what it makes of this worker", st == 200 and r["update"]["current"] == NEXT
       and r["update"]["min"] == PREV and r["update"]["yours"] == VERSION and r["update"]["available"] is True
       and r["update"]["supported"] is True, str(r))
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok2, {"User-Agent": "rad-quant-at-home-worker/0.4"})
    vers = {x["id"]: x["version"] for x in S.q("SELECT id, version FROM workers")}
    ok("the server records each worker's release (0.4: from its user agent)", vers[wid] == VERSION and vers[wid2] == "0.4.0", str(vers))
    j = noop_job(S, seed=1)
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok3, wv("0.1.0"))
    ok("below the minimum: 426 with the update command and retry_s, no job", st == 426 and "rqah update" in r["error"]
       and r["update"]["min"] == PREV and r["update"]["supported"] is False and r["retry_s"] >= 600
       and S.q("SELECT state FROM jobs WHERE id=?", (j,))[0][0] == "queued", f"{st} {r}")
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok3)
    ok("a request without a version (CLI, curl) is not refused", st == 200 and r.get("job", {}).get("id") == j, str(st))
    # the minimum goes up while an old worker holds a job: it finishes it
    j2 = noop_job(S, seed=2)
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok2, wv(PREV)); job = r["job"]
    s.Q.worker_min = VERSION if vtuple(VERSION) > vtuple(PREV) else NEXT
    st1, hb = s.req("POST", "/api/v0/heartbeat", {"job_id": job["id"], "lease_id": job["lease_id"]}, tok2, wv(PREV))
    st2, _ = s.req("PUT", f"/api/v0/upload/{job['id']}/{job['lease_id']}/out.bin?offset=0", None, tok2, wv(PREV), raw=noop.content(2, 10))
    import hashlib
    st3, c = s.req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "result": {},
                                                 "files": {"out.bin": {"sha256": hashlib.sha256(noop.content(2, 10)).hexdigest(), "size": 10}}},
                   tok2, wv(PREV))
    st4, r4 = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok2, wv(PREV))
    ok("min raised while it holds a job: heartbeat, upload, complete still work; the next lease is 426",
       (st1, hb.get("ok"), st2, st3, c.get("ok"), st4) == (200, True, 200, 200, True, 426), f"{st1} {st2} {st3} {st4}")
    s.Q.worker_min = PREV
    st, me = s.req("GET", "/api/v0/me", None, tok2)
    st5, stat = s.req("GET", "/api/v0/status", None, tok2)
    ok("/me and /status carry the update block, /me each worker's release", me["update"] == {"current": NEXT, "min": PREV}
       and [w["version"] for w in me["workers"]] == [PREV] and stat["update"]["yours"] == PREV and stat["update"]["available"], str(me["update"]))
    # rqah status, as the donor sees it in the worker container
    os.environ["RQAH_IMAGE_ID"] = f"ghcr.io/theherobrine422/rad-quant-at-home-worker:{PREV}-cu128+abc1234"
    rc, out = capture(CLI.main, ["--server", s.url, "--token", tok2, "status"])
    ok("rqah status: VERSION column, update available X -> Y, rqah update and the exact image",
       rc == 0 and re.search(r"\bVERSION\b", out) and f"update available: {PREV} -> {NEXT}" in out and "rqah update" in out
       and f"docker pull ghcr.io/theherobrine422/rad-quant-at-home-worker:{NEXT}-cu128," in out, out[-900:])
    rc, out = capture(CLI.main, ["--server", s.url, "--token", tok2, "--json", "status"])
    js = json.loads(out)
    ok("rqah status --json has the notice", rc == 0 and js["update"]["current"] == NEXT and js["update_notice"], str(js.get("update_notice")))
    s.Q.worker_min = NEXT
    rc, out = capture(CLI.main, ["--server", s.url, "--token", tok2, "status"])
    ok("rqah status below the minimum says it gets no new jobs", "below the minimum" in out and "NO new jobs" in out, out[-400:])
    s.Q.worker_min = PREV
    rc, out = capture(CLI.main, ["update", "--config-dir", f"{d}/no-such-config", "--yes"])
    ok("rqah update without a setup config refuses and points at the docs", rc == 2 and "no configuration" in out and "Updating" in out, out)
    del os.environ["RQAH_IMAGE_ID"]
    # status page: shares by release (verifiers and stale workers do not count)
    page = SP.render(SP.collect(S))
    ok("status page: the share of recent workers on the current release and on older ones",
       f"worker release {NEXT} (current): 0% of the 3 workers" in page and f"{VERSION}: 33%" in page and "rqah status" in page,
       re.search(r"worker release[^<]*", page).group(0) if "worker release" in page else "missing")
    s.close()


# ================================================================ job versions, served / retired
def t_job_versions():
    d = fresh("jobver"); s = Srv(args(d, worker_current=VERSION, worker_min=PREV)); S, Q = s.Q.store, s.Q
    old, new, oldest = noop.VERSION, noop.VERSION + 1, noop.VERSION - 1
    jo = [noop_job(S, old, seed=i) for i in range(3)]
    jx = noop_job(S, oldest, seed=9)
    wa, ta = S.add_worker("a-old"); wb, tb = S.add_worker("b-new")
    wr = lambda i: S.q("SELECT * FROM workers WHERE id=?", (i,))[0]
    with noop_as(new, accepts=(old,)):                      # the next release's server: noop new, still accepts old
        jn = [noop_job(S, new, seed=10 + i) for i in range(2)]
        rep = Q.version_report()
        ok("startup report: queued jobs of a version this server does not accept", len(rep) == 1 and f"noop v{oldest}" in rep[0], str(rep))
        got_a = [Q.lease(wr(wa), caps(old))["job"] for _ in range(2)]
        got_b = [Q.lease(wr(wb), caps(new))["job"] for _ in range(3)]
        ok("an old worker gets only its version, a new one only the new", [g["version"] for g in got_a] == [old, old]
           and [g["version"] for g in got_b[:2]] == [new, new] and got_b[2] is None, f"{[g and g['version'] for g in got_b]}")
        ok("a version this server no longer accepts is leased to nobody", Q.lease(wr(wa), caps(oldest))["job"] is None
           and S.q("SELECT state FROM jobs WHERE id=?", (jx,))[0][0] == "queued")
        rc, out = capture(AD.cmd_versions, argparse.Namespace(days=7, json=False), S)
        ok("admin versions: per version counts, NOT SERVED and stuck", re.search(rf"noop\s+{oldest}\s+NOT SERVED\s+1\b.*stuck", out)
           and re.search(rf"noop\s+{old}\s+accepted\s+1\s+2\b", out) and "STUCK noop" in out, out)
        ra = lambda **kw: argparse.Namespace(**{"type": "noop", "version": old, "migrate": False, "dry_run": False, "undo": False,
                                                "data": f"{d}/data", **kw})
        rc, out = capture(AD.cmd_retire, ra(), S)
        ok("retire refused while jobs of that version run (never touches them)", rc and "running" in str(rc), str(rc)[:200])
        for g in got_a:                                     # the old worker finishes its jobs
            with S.tx() as db:
                l = db.execute("SELECT * FROM leases WHERE id=?", (g["lease_id"],)).fetchone()
                S.end_lease(db, l, "done", job_state="accepted")
        rc, out = capture(AD.cmd_retire, ra(), S)
        ok("retire refused while queued jobs still need it, unless --migrate", rc and "still need it" in str(rc), str(rc)[:200])
        rc, out = capture(AD.cmd_retire, ra(migrate=True, dry_run=True), S)
        ok("retire --migrate --dry-run changes nothing", "dry run" in out and
           S.q("SELECT version FROM jobs WHERE id=?", (jo[2],))[0][0] == old and not S.retired(), out)
        rc, out = capture(AD.cmd_retire, ra(migrate=True), S)
        j2 = S.q("SELECT version, ukey FROM jobs WHERE id=?", (jo[2],))[0]
        ok("retire --migrate: queued jobs re-stamped at the new version (dedupe key too), the version retired",
           tuple(j2) == (new, f"noop/v{new}:s2") and S.retired() == {"noop": {old}}
           and S.q("SELECT COUNT(*) FROM events WHERE kind='migrate'")[0][0] == 1, f"{tuple(j2)} {out}")
        r = Q.lease(wr(wb), caps(new))["job"]
        ok("the migrated job goes to a new worker; nothing at the retired version to an old one", r and r["id"] == jo[2]
           and Q.lease(wr(wa), caps(old))["job"] is None)
        rc, out = capture(AD.cmd_retire, ra(undo=True), S)
        ok("retire --undo", S.retired() == {} and "back in service" in out, out)
    # a running job whose version the server stops accepting: a clear 409
    j = noop_job(S, old, seed=50)
    r = s.req("POST", "/api/v0/lease", {"caps": caps(old)}, ta)[1]["job"]
    with noop_as(new):                                      # a server that accepts only the new version
        st, c = s.req("POST", "/api/v0/complete", {"job_id": r["id"], "lease_id": r["lease_id"], "result": {},
                                                    "files": {"out.bin": {"sha256": "0" * 64, "size": 10}}}, ta)
    ok("complete of a version the server no longer accepts: 409 that says so", st == 409 and "no longer accepts" in c["error"]
       and c.get("retired"), f"{st} {c}")
    s.close()


# ================================================================ schema and restart
def t_schema():
    d = fresh("schema")
    S = Store(f"{d}/new.sqlite")
    ok("a new database starts at the current schema, no backup", S.db.execute("PRAGMA user_version").fetchone()[0] == ST.SCHEMA_VERSION
       and not [f for f in os.listdir(d) if ".bak" in f] and S.migrated == list(range(1, ST.SCHEMA_VERSION + 1)))
    # a database in the pre-0.5 shape (v0 tables, no migrations): one worker from before donors, one job, one lease
    p = f"{d}/old.sqlite"
    c = sqlite3.connect(p); c.executescript(ST.SCHEMA)
    c.execute("INSERT INTO workers(name, token_hash, created) VALUES('old', 'h', 1)")
    c.execute("INSERT INTO jobs(type, version, params, inputs, requires, input_bytes, state, created) VALUES('noop', 2, '{}', '[]', '{}', 0, 'queued', 1)")
    c.commit(); c.close()
    S = Store(p)
    cols = {r[1] for r in S.db.execute("PRAGMA table_info(workers)")}
    ok("a pre-0.5 database: backed up once, then migrated (columns, a donor per worker)", S.migrated == [1, 2]
       and os.path.exists(f"{p}.schema0.bak") and {"version", "probation", "donor_id"} <= cols
       and S.q("SELECT donor_id FROM workers")[0][0] is not None and S.q("SELECT COUNT(*) FROM jobs")[0][0] == 1)
    b = sqlite3.connect(f"{p}.schema0.bak")
    ok("the backup is the old database", b.execute("PRAGMA user_version").fetchone()[0] == 0
       and "version" not in {r[1] for r in b.execute("PRAGMA table_info(workers)")} and b.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1)
    b.close(); S.db.close()
    S = Store(p)
    ok("reopening is a no-op", S.migrated == [] and len([f for f in os.listdir(d) if f.startswith("old.sqlite.schema")]) == 1)
    S.db.execute(f"PRAGMA user_version={ST.SCHEMA_VERSION + 1}"); S.db.close()
    try: Store(p); refused = None
    except SchemaTooNew as e: refused = str(e)
    ok("a database from a newer server is refused, with the way out", refused and "newer" in refused and ".bak" in refused, str(refused))
    r = subprocess.run([sys.executable, "-I", f"{HERE}/server/server.py", "--db", p, "--data", f"{d}/data", "--spool", f"{d}/spool",
                        "--results", f"{d}/results", "--port", "0", "--status-interval", "0"], capture_output=True, text=True, timeout=60)
    r2 = subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", p, "list"], capture_output=True, text=True, timeout=60)
    ok("server.py and admin.py refuse it (exit with the message, no traceback)", r.returncode == 1 and "newer" in r.stderr
       and "Traceback" not in r.stderr and r2.returncode == 1 and "newer" in r2.stderr, (r.stderr + r2.stderr)[-300:])


def t_restart():
    d = fresh("restart"); S = Store(f"{d}/q.sqlite")
    wid, _ = S.add_worker("w"); j = noop_job(S)
    now = time.time()
    with S.tx() as db:
        db.execute("INSERT INTO leases(id, job_id, worker_id, issued, expires, state, deadline) VALUES('a'||hex(randomblob(15)), ?, ?, ?, ?, 'active', ?)",
                   (j, wid, now - 3700, now - 3500, now + 100))
        db.execute("UPDATE jobs SET state='leased', lease_id=(SELECT id FROM leases), lease_owner=?, lease_expires=? WHERE id=?", (wid, now - 3500, j))
    S.meta_set("alive_t", now - 3600)                       # the server went down an hour ago, its lease 100 s later
    down, n = S.after_restart(120, now)
    l = S.q("SELECT * FROM leases")[0]
    ok("restart after an hour: the lease gets a fresh lease_s, its deadline moves by the downtime",
       n == 1 and abs(down - 3600) < 1 and l["expires"] >= now + 119 and abs(l["deadline"] - (now + 100 + 3600)) < 2
       and S.q("SELECT lease_expires FROM jobs")[0][0] == l["expires"], f"{down} {dict(l)}")
    ok("... so the reaper does not expire it, and a server_start event says why", S.expire(now + 5) == 0
       and S.q("SELECT state FROM jobs")[0][0] == "leased" and S.q("SELECT COUNT(*) FROM events WHERE kind='server_start'")[0][0] == 1)
    ok("a worker that never comes back: the lease expires one lease_s later", S.expire(now + 125) == 1
       and S.q("SELECT state FROM jobs")[0][0] == "queued")


# ================================================================ the worker
def worker(s, d, name, *extra, env=None):
    lf = open(f"{d}/{name}.log", "a")
    e = dict(os.environ, **(env or {}))
    return subprocess.Popen([sys.executable, "-I", f"{HERE}/worker/worker.py", "--server", s.url, "--token-file", f"{d}/{name}/token",
                             "--state", f"{d}/{name}", "--poll", "0.3", "--job-types", "noop", *extra],
                            stdout=lf, stderr=subprocess.STDOUT, env=e)


def wait(cond, timeout):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(0.2)
    return cond()


def t_worker():
    d = fresh("worker"); s = Srv(args(d, worker_current=NEXT, worker_min=PREV)); S = s.Q.store
    wid, tok = S.add_worker("wk"); os.makedirs(f"{d}/wk"); open(f"{d}/wk/token", "w").write(tok + "\n")
    js = [noop_job(S, seed=i) for i in range(3)]
    p = worker(s, d, "wk", "--idle-exit", "4"); p.wait(60)
    log = open(f"{d}/wk.log").read()
    st = json.load(open(f"{d}/wk/state.json")); up = json.load(open(f"{d}/wk/update.json"))
    ok("worker: jobs done, state.json format 2 with its release", p.returncode == 0 and all(
        S.q("SELECT state FROM jobs WHERE id=?", (j,))[0][0] == "accepted" for j in js) and st["format"] == 2 and st["version"] == VERSION, str(st))
    ok("an available update is logged exactly once and kept in update.json", log.count(f"update available: {VERSION} -> {NEXT}") == 1
       and up["current"] == NEXT and up["available"] and not up["refused"], f"{log.count('update available')} {up}")
    ok("the server recorded the worker's release from its header", S.q("SELECT version FROM workers WHERE id=?", (wid,))[0][0] == VERSION)
    # below the minimum: 426, logged once, nothing leased, the worker keeps going until --idle-exit
    s.Q.worker_min = NEXT
    j = noop_job(S, seed=20)
    p = worker(s, d, "wk", "--idle-exit", "4"); p.wait(60)
    log = open(f"{d}/wk.log").read()
    ok("below the minimum: UPDATE NEEDED logged once, no job leased, no crash", p.returncode == 0 and log.count("UPDATE NEEDED") == 1
       and "rqah update" in log and S.q("SELECT state FROM jobs WHERE id=?", (j,))[0][0] == "queued" and "Traceback" not in log
       and json.load(open(f"{d}/wk/update.json"))["refused"], log[-500:])
    s.Q.worker_min = PREV
    with S.tx() as db: db.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (j,))
    # a resumed lease leased at another job version (the release before an update) is handed back
    j = noop_job(S, seed=30, seconds=30)
    r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok)[1]["job"]
    r["version"] = noop.VERSION - 1                         # as the previous release had it
    write_json_atomic(f"{d}/wk/current/{r['id']}-{r['lease_id'][:8]}.json", {"job": r, "phase": "run", "t_lease": time.time(), "devices": []})
    # and a cache: one good blob, one of the wrong size, one file nobody knows
    blobs = f"{d}/wk/cache/blobs"; good, bad, stray = "a" * 64, "b" * 64, "c" * 64
    for n, size in ((good, 100), (bad, 50), (stray, 10)): open(f"{blobs}/{n}", "wb").write(b"x" * size)
    write_json_atomic(f"{d}/wk/cache/index.json", {good: {"size": 100, "last": 1, "name": "test/g"}, bad: {"size": 100, "last": 1, "name": "test/b"}})
    p = worker(s, d, "wk", "--idle-exit", "3"); p.wait(60)
    row = S.q("SELECT state, attempts FROM jobs WHERE id=?", (j,))[0]; log = open(f"{d}/wk.log").read()
    kinds = [r[0] for r in S.q("SELECT kind FROM events WHERE job_id=? ORDER BY id", (j,))]
    # (the queue still has the job at this worker's version, so it leases it again, fresh, and runs it)
    ok("a resumed lease at another job version is handed back (abandoned, attempt not counted)",
       kinds[:5] == ["enqueue", "lease", "abandon", "lease", "accept"] and tuple(row) == ("accepted", 1) and "before an update" in log,
       f"{kinds} {tuple(row)}")
    ok("cache: a blob of the wrong size and an unknown file dropped, the good one kept", os.path.exists(f"{blobs}/{good}")
       and not os.path.exists(f"{blobs}/{bad}") and not os.path.exists(f"{blobs}/{stray}") and "wrong size" in log)
    with S.tx() as db: db.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (j,))
    # a foreign (newer or torn) index: the cache is rebuilt (blobs dropped), the token stays
    open(f"{d}/wk/cache/index.json", "w").write('{"format": 9, "blobs": [')
    # a state from a newer release: its job records are set aside
    write_json_atomic(f"{d}/wk/state.json", {"format": 99, "version": "9.9.9"})
    write_json_atomic(f"{d}/wk/current/1-abcdef01.json", {"job": {"id": 1, "lease_id": "abcdef01" * 4}, "phase": "run"})
    n_workers = S.q("SELECT COUNT(*) FROM workers")[0][0]
    p = worker(s, d, "wk", "--idle-exit", "2"); p.wait(60)
    log = open(f"{d}/wk.log").read()
    ok("a state from a newer release: job records set aside, the token kept (no new registration), state.json rewritten",
       p.returncode == 0 and glob_one(f"{d}/wk/current.format99-*/1-abcdef01.json") and not os.listdir(f"{d}/wk/current")
       and S.q("SELECT COUNT(*) FROM workers")[0][0] == n_workers and json.load(open(f"{d}/wk/state.json"))["format"] == 2
       and "set aside" in log, log[-300:])
    ok("a torn cache index: rebuilt empty (the blobs are fetched again when needed)", not os.path.exists(f"{blobs}/{good}")
       and p.returncode == 0)
    s.close()


def glob_one(pat):
    import glob
    return len(glob.glob(pat)) == 1


# ================================================================ the wizard's --update (a simulated Docker)
class FakeDocker:
    """docker for WZ.Host.run: one container, a registry {tag: digest}, a clock that only moves when the wizard sleeps.
    finish_s: how long the worker takes to finish its jobs after the first SIGTERM (None: never); a second SIGTERM
    stops it within 3 s"""
    def __init__(self, cfgdir, image, running=True, finish_s=20, pull_ok=True, registry=None):
        self.cfgdir, self.t, self.cmds, self.finish_s, self.pull_ok = cfgdir, 0.0, [], finish_s, pull_ok
        self.c = {"image": image, "running": running, "restart": "unless-stopped", "terms": 0, "stop_at": None} if image else None
        self.registry, self.local = dict(registry or {}), {}

    def clock(self): return self.t
    def sleep(self, s): self.t += s

    def _tick(self):
        if self.c and self.c["running"] and self.c["stop_at"] is not None and self.t >= self.c["stop_at"]: self.c["running"] = False

    def run(self, argv, timeout=15):
        self.cmds.append(list(argv)); self._tick(); c = self.c
        if argv[0] == "systemctl": return 0, "", ""
        a = argv[1:] if argv[0] == "docker" else argv
        if argv[0] == "sh": return self._create()
        if a[:3] == ["image", "inspect", "--format"]:
            d = self.local.get(a[4])
            return (0, json.dumps([f"{a[4].rsplit(':', 1)[0]}@{d}"]), "") if d else (1, "", "No such image")
        if a[:1] == ["inspect"]:
            return (0, f"{'true' if c['running'] else 'false'} {c['image']}\n", "") if c else (1, "", "No such object")
        if a[:1] == ["pull"]:
            if not self.pull_ok or a[1] not in self.registry: return 1, "", "manifest unknown"
            self.local[a[1]] = self.registry[a[1]]; return 0, "pulled", ""
        if a[:2] == ["update", "--restart"]: c["restart"] = a[2]; return 0, "", ""
        if a[:3] == ["kill", "--signal", "TERM"]:
            c["terms"] += 1
            if c["terms"] == 1 and self.finish_s is not None: c["stop_at"] = self.t + self.finish_s
            if c["terms"] >= 2: c["stop_at"] = min(c["stop_at"] or 1e18, self.t + 3)
            return 0, "", ""
        if a[:1] == ["stop"]: c["running"] = False; return 0, "", ""
        if a[:1] == ["rm"]:
            if c["running"]: return 1, "", "container is running"
            self.c = None; return 0, "", ""
        if a[:1] == ["compose"] and "up" in a: return self._create()
        return 1, "", "fake docker: unknown " + " ".join(a)

    def _create(self):
        if self.c: return 1, "", "name in use"
        img = re.search(r'image: "([^"]+)"', open(os.path.join(self.cfgdir, "compose.yaml")).read()).group(1)
        self.c = {"image": img, "running": True, "restart": "unless-stopped", "terms": 0, "stop_at": None}
        return 0, "created", ""


def hw_fake(compose=True):
    return {"gpus": [{"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 4070", "arch": "8.9", "vram": 12 << 30}],
            "ram": 64 << 30, "video_gid": 39, "render_gid": 105, "localtime": True, "selinux_enforcing": False,
            "docker": {"present": True, "access": True, "compose": "5.6.0" if compose else None, "error": None}}


def wz_args(**kw):
    return argparse.Namespace(**{"to": None, "yes": True, "release": False, "drain_timeout": None, "dry_run": False, "digest": None,
                                 "no_digest_check": False, "auto": False, **kw})


def wz_setup(tag, version="", **cfgkw):
    cfgdir = f"{W}/wz/{tag}"; shutil.rmtree(cfgdir, ignore_errors=True)
    cfg = dict(WZ.DEFAULTS, server="https://queue.example.org", vendor="nvidia", gpus=[0], gpu_name="rtx4070", vram_gb=11,
               ram_gb=24, mem_limit_gb=31, cache_gb=50, version=version, **cfgkw)
    WZ.write_all(cfgdir, cfg, hw_fake(), "riINVITECODE123")
    return cfgdir, cfg


DG = {v: "sha256:" + hashlib.sha256(v.encode()).hexdigest() for v in (PREV, VERSION, NEXT)}
TAGREF = lambda v: f"{WZ.IMAGE_REPO}:{v}-cu128"


def adv(v, published="2026-01-01T00:00:00Z", digest=None, image=None, **extra):
    return {"api": "v0", "server": v, "worker": {"current": v, "min": PREV, "notes": f"{WZ.RELEASES}/tag/v{v}", "published": published,
                                                 "images": {"cu128": {"image": image or TAGREF(v), "digest": digest or DG[v]}}, **extra}}


def run_update(tag, advert=None, running=True, finish_s=20, pull_ok=True, version="", fetch=None, fetch_rel=None, registry=None,
               cfgkw=None, now=None, **kw):
    cfgdir, cfg = wz_setup(tag, version, **(cfgkw or {}))
    fd = FakeDocker(cfgdir, WZ.image_for(cfg), running=running, finish_s=finish_s, pull_ok=pull_ok,
                    registry=registry if registry is not None else {TAGREF(v): d for v, d in DG.items()})
    if now is not None: fd.t = now
    said = []
    advert = advert or adv(NEXT)
    host = WZ.Host("/", {"docker": fd.run, "sh": fd.run, "systemctl": fd.run})
    rc = WZ.update(host, hw_fake(), cfg, cfgdir, "riINVITECODE123", wz_args(**kw), say=said.append, fetch=fetch or (lambda url: advert),
                   fetch_rel=fetch_rel or (lambda v: (_ for _ in ()).throw(urllib.error.URLError("no network in tests"))),
                   sleep=fd.sleep, clock=fd.clock)
    return rc, fd, said, cfgdir


def verbs(fd):
    out = []
    for c in fd.cmds:
        a = c[1:] if c[0] == "docker" else c
        if a[0] == "inspect": continue
        out.append(" ".join(a[:3]) if a[0] in ("kill", "update") else "compose up" if a[0] == "compose" else
                   "digest" if a[:2] == ["image", "inspect"] else a[0])
    return out


def t_wizard():
    new_img = f"{TAGREF(NEXT)}@{DG[NEXT]}"
    rc, fd, said, cfgdir = run_update("plain")
    ok("update: pull the tag, check its digest, drain (restart policy off, SIGTERM, wait for the jobs), rm, compose up",
       rc == 0 and verbs(fd) == ["pull", "digest", "update --restart no", "kill --signal TERM", "rm", "compose up"]
       and fd.c["image"] == new_img and fd.c["running"] and fd.t >= 20 and any("digest checked" in x for x in said),
       f"{rc} {verbs(fd)} {said[-2:]}")
    allc = [" ".join(c) for c in fd.cmds]
    ok("never a volume removal or compose down", not any(x in c for c in allc for x in ("volume", " -v", "down", "--volumes")))
    comp, env = open(f"{cfgdir}/compose.yaml").read(), open(f"{cfgdir}/worker.env").read()
    sj = json.load(open(f"{cfgdir}/setup.json"))
    ok("the files pin TAG@DIGEST (setup.json: version and digest), the invite and the state volume stay, *.prev kept",
       f'image: "{new_img}"' in comp and "rqah-state:/var/lib/rqah" in comp and sj["settings"]["digest"] == DG[NEXT]
       and sj["settings"]["version"] in (NEXT, "") and "RQAH_INVITE=riINVITECODE123" in env
       and os.path.exists(f"{cfgdir}/compose.yaml.prev") and new_img in open(f"{cfgdir}/docker-run.sh").read())
    ok("the release notes link is shown", any(f"releases/tag/v{NEXT}" in x for x in said))
    moved = {TAGREF(NEXT): "sha256:" + "e" * 64}
    rc, fd, said, cfgdir = run_update("moved", registry=moved)
    ok("a moved / tampered tag (other digest on the registry): REFUSED after the pull, before any drain; files untouched",
       rc == 1 and verbs(fd) == ["pull", "digest"] and fd.c["running"] and any("REFUSED" in x for x in said)
       and NEXT not in open(f"{cfgdir}/compose.yaml").read(), said[-1])
    rc, fd, said, _ = run_update("nodigest", advert={"worker": {"current": NEXT, "min": PREV}})
    ok("no digest advertised (and no release.json): refused, nothing done", rc == 1 and verbs(fd) == [] and "digest" in said[-1], said[-1])
    rc, fd, said, _ = run_update("otherrepo", advert=adv(NEXT, image=f"ghcr.io/evil/worker:{NEXT}-cu128"))
    ok("an advertised image outside our repository: refused", rc == 1 and verbs(fd) == [] and "not" in said[-1], said[-1])
    rc, fd, said, _ = run_update("nocheck", advert={"worker": {"current": NEXT, "min": PREV}}, no_digest_check=True)
    ok("--no-digest-check: pulls and runs the bare tag", rc == 0 and fd.c["image"] == TAGREF(NEXT) and "digest" not in verbs(fd), str(verbs(fd)))
    rel = {"version": PREV, "notes_url": f"{WZ.RELEASES}/tag/v{PREV}", "created": "2026-01-01T00:00:00Z",
           "images": {"cu128": {"repo": WZ.IMAGE_REPO, "tag": f"{PREV}-cu128", "digest": DG[PREV]}}}
    rc, fd, said, _ = run_update("downto", to=PREV, version=VERSION, fetch_rel=lambda v: rel if v == PREV else {})
    ok("--to another release: its digest from that release's release.json", rc == 0 and fd.c["image"] == f"{TAGREF(PREV)}@{DG[PREV]}", str(rc))
    rc, fd, said, _ = run_update("release", release=True)
    ok("--release: two SIGTERMs at once (the jobs go back to the queue), no waiting for them",
       rc == 0 and verbs(fd)[2:6] == ["update --restart no", "kill --signal TERM", "kill --signal TERM", "rm"] and fd.t < 20, f"{verbs(fd)} {fd.t}")
    rc, fd, said, _ = run_update("timeout", finish_s=None, drain_timeout=1.0)
    ok("drain timeout: after it, a second SIGTERM hands the jobs back; then the update goes on",
       rc == 0 and verbs(fd)[2:5] == ["update --restart no", "kill --signal TERM", "kill --signal TERM"]
       and 60 <= fd.t < 200 and any("handing its jobs back" in x for x in said), f"{verbs(fd)} {fd.t}")
    rc, fd, said, cfgdir = run_update("dry", dry_run=True)
    ok("--dry-run: says what it would do, runs nothing", rc == 0 and verbs(fd) == [] and any("dry run" in x for x in said)
       and NEXT not in open(f"{cfgdir}/compose.yaml").read())
    rc, fd, said, cfgdir = run_update("pullfail", pull_ok=False)
    ok("a failed pull changes nothing (no drain, files untouched)", rc == 1 and verbs(fd) == ["pull"] and fd.c["running"]
       and NEXT not in open(f"{cfgdir}/compose.yaml").read(), str(said[-1:]))

    def down(url): raise urllib.error.URLError("connection refused")
    rc, fd, said, _ = run_update("unreach", fetch=down)
    ok("queue unreachable: a clear message, nothing done", rc == 1 and verbs(fd) == [] and "--to" in said[-1], said[-1])
    rc, fd, said, _ = run_update("badver", advert={"worker": {"current": "latest; rm -rf /", "min": PREV}})
    ok("a malformed advertised release is not used", rc == 1 and verbs(fd) == [], said[-1])
    rc, fd, said, _ = run_update("older", advert=adv(PREV), version=VERSION)
    ok("an older advertised release is not a downgrade unless --to", rc == 2 and verbs(fd) == [] and "--to" in said[-1], said[-1])
    rc, fd, said, _ = run_update("setupmin", advert=adv(NEXT, setup_min="9.0.0"))
    ok("a release that needs a newer setup.py (setup_min): refused with the way out", rc == 2 and verbs(fd) == [] and "git pull" in said[-1], said[-1])
    cfgdir, cfg = wz_setup("override"); cfg["image"] = "registry.example/mine:1"
    r = WZ.update(WZ.Host("/", {"docker": lambda a: (1, "", "")}), hw_fake(), cfg, cfgdir, None, wz_args(), say=lambda x: None,
                  fetch=lambda u: adv(NEXT))
    ok("an image override is left alone (refused with the way to change it)", r == 2)
    rc, fd, said, _ = run_update("stopped", running=False)
    ok("a stopped worker: no drain, removed and recreated on the new release", rc == 0 and verbs(fd) == ["pull", "digest", "rm", "compose up"]
       and fd.c["image"] == new_img)
    # up to date: pinned to NEXT@digest and running it
    cfgdir, cfg = wz_setup("current", version=NEXT, digest=DG[NEXT])
    fd = FakeDocker(cfgdir, WZ.image_for(cfg)); said = []
    r = WZ.update(WZ.Host("/", {"docker": fd.run}), hw_fake(), cfg, cfgdir, None, wz_args(), say=said.append, fetch=lambda u: adv(NEXT))
    ok("up to date: nothing done", r == 0 and verbs(fd) == [] and "up to date" in said[-1], said[-1])

    # --auto (the timer)
    pub = WZ.datetime.datetime(2026, 11, 1, tzinfo=WZ.datetime.timezone.utc).timestamp()
    stamp = "2026-11-01T00:00:00Z"
    rc, fd, said, _ = run_update("auto-off", advert=adv(NEXT, stamp), auto=True, now=pub + 86400 * 3)
    ok("--auto with auto_update off in setup.json: does nothing", rc == 0 and verbs(fd) == [] and "off" in said[-1], said[-1])
    rc, fd, said, _ = run_update("auto-young", advert=adv(NEXT, stamp), auto=True, cfgkw={"auto_update": True}, now=pub + 3600)
    ok("--auto: a release younger than a day waits", rc == 0 and verbs(fd) == [] and "24 h old" in said[-1], said[-1])
    rc, fd, said, _ = run_update("auto-go", advert=adv(NEXT, stamp), auto=True, cfgkw={"auto_update": True}, now=pub + 86400 * 2,
                                 finish_s=5000)
    ok("--auto: a day-old release, digest-checked, drained at a job boundary (the jobs finish, up to 6 h), recreated",
       rc == 0 and verbs(fd) == ["pull", "digest", "update --restart no", "kill --signal TERM", "rm", "compose up"]
       and fd.c["image"] == new_img and fd.t - (pub + 86400 * 2) >= 5000, f"{verbs(fd)} {said[-1]}")
    rc, fd, said, _ = run_update("auto-old", advert=adv(PREV, stamp), auto=True, version=VERSION, cfgkw={"auto_update": True}, now=pub + 86400 * 2)
    ok("--auto never downgrades", rc == 2 and verbs(fd) == [], said[-1])
    rc, fd, said, _ = run_update("auto-nodig", advert={"worker": {"current": NEXT, "min": PREV, "published": stamp}}, auto=True,
                                 cfgkw={"auto_update": True}, now=pub + 86400 * 2, no_digest_check=True)
    ok("--auto never runs without a digest (even with --no-digest-check)", rc == 2 and verbs(fd) == [], said[-1])
    # the timer
    u = WZ.render_units("/home/ada/.config/rad-quant-at-home", "/usr/bin/python3", "/home/ada/rqah/client/setup.py")
    svc, tmr = u["rqah-update.service"], u["rqah-update.timer"]
    ok("timer units: a oneshot running setup.py --update --auto on this config, hourly with jitter, persistent",
       "ExecStart=/usr/bin/python3 -I /home/ada/rqah/client/setup.py --update --auto --config-dir /home/ada/.config/rad-quant-at-home" in svc
       and "Type=oneshot" in svc and "OnUnitInactiveSec=1h" in tmr and "Persistent=true" in tmr and "RandomizedDelaySec" in tmr)
    calls = []
    host = WZ.Host("/", {"systemctl": lambda a: (calls.append(a) or (0, "", ""))})
    host.which = lambda n: "/usr/bin/" + n
    on = WZ.install_timer(host, "/cfg", True, say=lambda x: None); off = WZ.install_timer(host, "/cfg", False, say=lambda x: None)
    ok("install_timer: links the service, enables the timer; off disables it", on and off and [c[1:4] for c in calls] == [
        ["--user", "link", "/cfg/rqah-update.service"], ["--user", "enable", "--now"], ["--user", "disable", "--now"]], str(calls))
    host2 = WZ.Host("/", {}); host2.which = lambda n: None; said = []
    ok("no systemctl: says to run it from cron", not WZ.install_timer(host2, "/cfg", True, say=said.append) and "cron" in said[-1])
    # --start (setup --pull --start) on a running worker drains it first
    cfgdir, cfg = wz_setup("start")
    fd = FakeDocker(cfgdir, WZ.image_for(cfg)); said = []
    okk = WZ.start(WZ.Host("/", {"docker": fd.run, "sh": fd.run}), hw_fake(compose=False), cfg, cfgdir, said.append,
                   sleep=fd.sleep, clock=fd.clock)
    ok("--start drains a running worker, then recreates it (docker-run.sh path)", okk and verbs(fd)[:3] == ["update --restart no",
       "kill --signal TERM", "rm"] and fd.cmds[-1][0] == "sh", str(verbs(fd)))


# ================================================================ releases: the manifest, tools/release.py, the server, admin, the CLI
def t_release():
    d = fresh("release")
    m = RL.make(NEXT, "a" * 40, {"cu128": DG[NEXT], "rocm7.14-gfx1201": DG[VERSION]}, {"rad_exl3_mix": "x@y"}, "2026-11-08T12:00:00Z")
    bad = []
    for mut in (lambda x: x.update(tag="v9.9.9"), lambda x: x["images"]["cu128"].update(repo="ghcr.io/evil/w"),
                lambda x: x["images"]["cu128"].update(tag=f"{NEXT}-rocm"), lambda x: x["images"]["cu128"].update(digest="sha256:12"),
                lambda x: x.update(notes_url="https://evil.example/x"), lambda x: x.update(extra=1), lambda x: x.update(images={})):
        y = json.loads(json.dumps(m)); mut(y)
        try: RL.check(y); bad.append(False)
        except ValueError: bad.append(True)
    ok("release manifest: a good one passes; wrong tag, repo, image tag, digest, notes page, unknown key, no images refused",
       RL.check(m) is m and all(bad), str(bad))
    tr = lambda *a: subprocess.run([sys.executable, "-I", f"{HERE}/tools/release.py", *a], capture_output=True, text=True, cwd=HERE)
    r1, r2 = tr("check-tag", f"v{VERSION}"), tr("check-tag", "v0.0.1")
    ok("release.py check-tag: v<VERSION> only", r1.returncode == 0 and r2.returncode == 1 and "not v" in r2.stderr, r2.stderr)
    r3, r4 = tr("pins"), tr("pins", "--need-radiance")
    pinned = dict(l.split("=", 1) for l in r3.stdout.splitlines())
    ok("release.py pins: the pin file is well-formed; the ROCm build stops while RADIANCE_IMAGE is empty",
       r3.returncode == 0 and len(pinned["RXM_COMMIT"]) == 40 and (r4.returncode == 0) == bool(pinned["RADIANCE_IMAGE"]), r4.stderr[-120:])
    open(f"{d}/pins.env", "w").write(open(f"{HERE}/docker/pins.env").read().replace("RXM_COMMIT=", "RXM_COMMIT=zz"))
    ok("release.py pins: a malformed pin is refused", tr("pins", "--file", f"{d}/pins.env").returncode == 1)
    r5 = tr("manifest", "--digest", f"cu128={DG[VERSION]}", "--digest", f"rocm7.14-gfx1201={DG[PREV]}", "--commit", "b" * 40)
    mf = f"{d}/built.json"; open(mf, "w").write(r5.stdout)
    r6 = tr("notes", "--manifest", mf)
    ok("release.py manifest + notes: a checkable release.json; notes with the tags, digests, rqah update, the changes",
       r5.returncode == 0 and RL.loads(r5.stdout)["images"]["cu128"]["digest"] == DG[VERSION] and r6.returncode == 0
       and f"`{RL.IMAGE_REPO}:{VERSION}-cu128` | `{DG[VERSION]}`" in r6.stdout and "rqah update" in r6.stdout and "Changes" in r6.stdout,
       (r5.stderr + r6.stderr)[-300:])
    # the server: current and digests from the manifest
    s = Srv(args(d, worker_min=PREV)); S = s.Q.store
    ok("no manifest: current = this server's release, no images", s.Q.update_info() == {"current": VERSION, "min": PREV})
    rcf = subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", f"{d}/q.sqlite", "release-fetch", "--file", mf],
                         capture_output=True, text=True)
    st, v = s.req("GET", "/api/v0/version")
    ok("admin release-fetch --file installs it; the server advertises that release with its digests and notes",
       rcf.returncode == 0 and st == 200 and v["worker"]["current"] == VERSION and v["worker"]["images"]["cu128"] ==
       {"image": f"{RL.IMAGE_REPO}:{VERSION}-cu128", "digest": DG[VERSION]} and v["worker"]["notes"].endswith(f"/tag/v{VERSION}"),
       rcf.stdout + rcf.stderr)
    m2 = RL.make(NEXT, "c" * 40, {"cu128": DG[NEXT]}, created="2026-11-08T12:00:00Z")
    open(f"{d}/next.json", "w").write(json.dumps(m2))
    subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", f"{d}/q.sqlite", "release-fetch", "--file", f"{d}/next.json"],
                   capture_output=True)
    wid, tok = S.add_worker("w")
    st, r = s.req("POST", "/api/v0/lease", {"caps": caps()}, tok, wv(VERSION))
    s.Q.sync_meta()
    ok("a new manifest is picked up without a restart: lease answers advertise it, meta follows (status page)",
       r["update"]["current"] == NEXT and r["update"]["available"] and r["update"]["images"]["cu128"]["digest"] == DG[NEXT]
       and S.meta_get("worker_current") == NEXT, str(r["update"]))
    open(f"{d}/evil.json", "w").write(json.dumps(dict(m2, images={"cu128": {"repo": "ghcr.io/evil/w", "tag": f"{NEXT}-cu128", "digest": DG[NEXT]}})))
    rce = subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", f"{d}/q.sqlite", "release-fetch", "--file", f"{d}/evil.json"],
                         capture_output=True, text=True)
    open(f"{d}/release.json", "w").write('{"format": 1, "torn')            # a broken file in place: the last good one stays
    time.sleep(0.05)
    ok("release-fetch refuses a manifest of another repository; a broken file in place: the server keeps the last good one",
       rce.returncode == 1 and "refused" in rce.stderr and s.Q.update_info()["current"] == NEXT, rce.stderr[-200:])
    os.environ["RQAH_IMAGE_ID"] = f"{RL.IMAGE_REPO}:{VERSION}-cu128+abc1234"
    rc, out = capture(CLI.main, ["--server", s.url, "--token", tok, "status"])
    del os.environ["RQAH_IMAGE_ID"]
    ok("rqah status: the exact image by digest and the release notes link", f"docker pull {RL.IMAGE_REPO}:{NEXT}-cu128@{DG[NEXT]}" in out
       and f"release notes: {RL.notes_url(NEXT)}" in out, out[-600:])
    s.close()


def main():
    os.makedirs(W, exist_ok=True)
    for t in (t_versions, t_job_versions, t_schema, t_restart, t_worker, t_wizard, t_release):
        print(f"--- {t.__name__}", flush=True)
        try: t()
        except Exception:
            import traceback; traceback.print_exc(); ok(f"{t.__name__} ran", False)
    n = sum(1 for _, c in RESULTS if c)
    print(f"\n{n}/{len(RESULTS)} checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
