#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""e2e_noop.py [WORKDIR] -- end-to-end tests of server + worker with noop jobs (localhost, ~2 min, a few MB).

E2E_SERVER=local (default) runs server.py as a subprocess; E2E_SERVER=docker runs the server container from
docker/server/compose.yaml (image $E2E_IMAGE, default rad-quant-at-home-server:<VERSION>; project rqah-e2e) with WORKDIR's db/, data/,
out/ and secrets/ bind-mounted, while the workers, admin.py and the test itself run on the host against the same files.
Both modes also restart the server mid-run three ways (graceful restart, SIGKILL, remove + recreate) and check that
every job is accepted exactly once with the right bytes.

Scenarios (TESTS.md): tokens / invites, 3 workers with one killed mid-job (lease expires, job reassigned, accepted
once), a killed worker restarted before / after its lease expired, duplicate completion, corrupted upload, upload
caps, disk-cap eviction and the disk-based lease filter, Range resume of a partial download, data access rules,
malformed jobs from the real server (known type, bad params) and from a fake malicious server (unknown type, extra
keys, path traversal, inputs the type does not allow, a malformed eval_mix, a bad stamp, a job id that is a path, an
empty lease id), graceful stop (finish / abandon), time windows, env-only configuration (RQAH_*, then the 0.2
RADQUANT_* names).

0.2 (run_v02, TESTS.md section 8; the GPU side faked by tests/fakes.py, fake_radiance.py, fake_mkgold27.py):
requirements (VRAM / RAM classes), a 4-card worker (one process per card, device variables, the RAM budget), gang
jobs and the drain, eval_mix batches over the search API (assembly = the server's pieces, rows exact, size classes,
TP2, cache reuse and LRU), privacy refusals (golden-add, manifest, batch, serve time), spot checks with two cheating
donors, quarantine and requeue, lumens, golden_record (trusted only, TP2 and staged, accept checks), the spot-check
draw rate and eval_mix's accept() in process.

w25 (run_w25, TESTS.md section 9): the donor CLI client/rqah.py against the running server (status, jobs, points,
logs, config, link; --json everywhere; exit codes), donor isolation, personal invites and leaderboard opt-in at
registration, the display-name rules, and the public status page served at / (headers, read-only, no tokens, worker
names, addresses or paths; the opt-in name appears; the --status-dir copy). tests/statuspage_test.py checks the page
itself against a fixture database."""
import collections, glob, hashlib, json, os, shutil, signal, sqlite3, subprocess, sys, threading, time
import urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
from worker.worker import in_windows
from common.util import sha256_file
from common import radfile as RF
import fakes

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w13-e2e")
PORT = int(os.environ.get("E2E_PORT", "18751"))
URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable
RESULTS = []
MODE = os.environ.get("E2E_SERVER", "local")
DBP, DATA, RES, SPOOL = f"{W}/db/queue.sqlite", f"{W}/data", f"{W}/out/results", f"{W}/out/spool"
ADMIN_TOKEN = "e2e-admin-" + hashlib.sha256(W.encode()).hexdigest()[:40]
SEARCH_TOKEN = "e2e-search-" + hashlib.sha256(W.encode()).hexdigest()[24:64]
STATUS_DIR = f"{W}/out/status" if MODE == "docker" else f"{W}/status"
TOKENS = {}


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def admin(*args):
    r = subprocess.run([PY, "-I", f"{HERE}/server/admin.py", "--db", DBP, "--data", DATA, "--results", RES, *args],
                       capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr)
    return r.stdout


def admin_rc(*args):
    r = subprocess.run([PY, "-I", f"{HERE}/server/admin.py", "--db", DBP, "--data", DATA, "--results", RES, *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


_DB = []


def db():
    """one connection for the whole test (a fresh one per call is freed only by the cyclic GC under Python 3.14: they
    piled up to the fd limit, 1024 on the queue host)"""
    if not _DB:
        c = sqlite3.connect(DBP, timeout=30); c.row_factory = sqlite3.Row; _DB.append(c)
    return _DB[0]


def req(method, path, body=None, token=None, headers=None):
    h = dict(headers or {})
    if token: h["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    r = urllib.request.Request(URL + path, data=data, method=method, headers=h)
    try:
        resp = urllib.request.urlopen(r, timeout=30); b = resp.read()
        return resp.status, (json.loads(b) if b and resp.headers.get("Content-Type") == "application/json" else b)
    except urllib.error.HTTPError as e:
        b = e.read(); return e.code, (json.loads(b) if b else {})
    except (urllib.error.URLError, OSError) as e:          # incl. resets while a container's port proxy starts
        return None, {"error": repr(e)}


def worker(name, token, *extra, log=None):
    lf = open(f"{W}/{name}.log", "a")
    return subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", URL, f"--token={token}", "--state", f"{W}/{name}",
                             "--poll", "0.5", *extra], stdout=lf, stderr=subprocess.STDOUT)


def wait(cond, timeout, step=0.2):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(step)
    return cond()


def states():
    return {r["id"]: r["state"] for r in db().execute("SELECT id, state FROM jobs")}


def events(job=None, kind=None):
    q, a = "SELECT * FROM events WHERE 1", []
    if job is not None: q += " AND job_id=?"; a.append(job)
    if kind: q += " AND kind=?"; a.append(kind)
    return db().execute(q, a).fetchall()


def jobs_of(tag):
    return [r["id"] for r in db().execute("SELECT id FROM jobs WHERE level=? ORDER BY id", (tag,))]


def stats(name):
    p = f"{W}/{name}/worker.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---------------------------------------------------------------- fake malicious server
class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    queue, fails, data_hits = [], [], []
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.endswith("/lease"): self._send({"job": Fake.queue.pop(0) if Fake.queue else None, "retry_s": 0.2})
        elif self.path.endswith("/fail"): Fake.fails.append(b); self._send({"ok": True})
        else: self._send({"ok": True, "cancel": False})
    def do_GET(self):
        Fake.data_hits.append(self.path); self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()


# ---------------------------------------------------------------- the server under test: a subprocess or the container
class LocalServer:
    def start(self):
        # RADQUANT_UPLOAD_MAX_MB: a 0.2 name on purpose (still read in 0.3: the 413 check depends on it)
        e = {k: v for k, v in os.environ.items() if not k.startswith(("RADQUANT_", "RQAH_"))}
        e.update(RADQUANT_UPLOAD_MAX_MB="48", RQAH_SPOT_RATE="0", RQAH_SPOT_FIRST="1", RQAH_SEARCH_TOKEN_FILE=f"{W}/secrets/search-token")
        self.p = subprocess.Popen([PY, "-I", f"{HERE}/server/server.py", "--db", DBP, "--data", DATA, "--spool", SPOOL,
                                   "--results", RES, "--port", str(PORT), "--lease-s", "6", "--heartbeat-s", "1.5", "--retry-s", "0.5",
                                   "--admin-token-file", f"{W}/secrets/admin-token", "--status-interval", "1", "--status-dir", STATUS_DIR],
                                  stdout=open(f"{W}/server.log", "a"), stderr=subprocess.STDOUT, env=e)
    def stop(self):
        if self.p.poll() is None: self.p.send_signal(signal.SIGTERM); self.p.wait(30)
        return self.p.returncode
    def restart(self): rc = self.stop(); self.start(); return f"SIGTERM exit {rc}"
    def arm(self): pass
    def crash(self, after_kill=lambda: None):
        self.p.kill(); self.p.wait(); after_kill(); self.start(); return "SIGKILL, respawned"
    def recreate(self): return self.restart()
    def down(self):
        if self.p.poll() is None: self.p.kill()


class DockerServer:
    """docker/server/compose.yaml with bind mounts into W (project rqah-e2e); the port is published on loopback"""
    def __init__(self):
        self.envf = f"{W}/e2e.env"
        self.cmd = [os.environ.get("E2E_DOCKER", "docker"), "compose", "-f", f"{HERE}/docker/server/compose.yaml",
                    "-p", "rqah-e2e", "--env-file", self.envf]
    def dc(self, *a, check=True):
        r = subprocess.run(self.cmd + list(a), capture_output=True, text=True)
        if check and r.returncode: print(r.stdout, r.stderr)
        return r
    def write_env(self):
        image = os.environ.get("E2E_IMAGE") or "rad-quant-at-home-server:" + open(f"{HERE}/VERSION").read().strip()
        # RADQUANT_UPLOAD_MAX_MB: a 0.2 name on purpose (still read in 0.3: the 413 check below depends on it)
        env = {"RQAH_IMAGE": image, "RQAH_DB_DIR": f"{W}/db", "RQAH_DATA_DIR": DATA, "RQAH_OUT_DIR": f"{W}/out",
               "RQAH_SOURCES_DIR": f"{W}/sources", "RQAH_SECRETS_DIR": f"{W}/secrets", "RQAH_PUBLISH": f"127.0.0.1:{PORT}",
               "RQAH_LEASE_S": "6", "RQAH_HEARTBEAT_S": "1.5", "RQAH_RETRY_S": "0.5", "RADQUANT_UPLOAD_MAX_MB": "48",
               "RQAH_ADMIN_TOKEN_FILE": "/run/secrets/rqah/admin-token", "RQAH_SPOT_RATE": "0", "RQAH_SPOT_FIRST": "1",
               "RQAH_SEARCH_TOKEN_FILE": "/run/secrets/rqah/search-token", "RQAH_STATUS_INTERVAL": "1",
               "RQAH_STATUS_DIR": "/srv/rqah/out/status"}
        if os.path.exists("/sys/fs/selinux/enforce") and open("/sys/fs/selinux/enforce").read().strip() == "1":
            env["RQAH_SELINUX_RELABEL"] = "1"
        with open(self.envf, "w") as f: f.writelines(f"{k}={v}\n" for k, v in env.items())
    def cid(self): return self.dc("ps", "-q", "server").stdout.strip()
    def inspect(self, fmt):
        return subprocess.run(["docker", "inspect", "-f", fmt, self.cid()], capture_output=True, text=True).stdout.strip()
    def start(self): self.dc("up", "-d", "server")
    def restart(self): t = time.time(); self.dc("restart", "server"); return f"docker compose restart ({time.time() - t:.1f}s)"
    def arm(self):                             # look up the PID now: the docker CLI takes ~1 s, the kill must not
        self.armed = (int(self.inspect("{{.RestartCount}}") or 0), int(self.inspect("{{.State.Pid}}")))
    def crash(self, after_kill=lambda: None):
        n0, pid = getattr(self, "armed", None) or (int(self.inspect("{{.RestartCount}}") or 0), int(self.inspect("{{.State.Pid}}")))
        self.armed = None
        os.kill(pid, signal.SIGKILL)                                        # the server process (PID 1 in the container)
        after_kill()
        back = wait(lambda: int(self.inspect("{{.RestartCount}}") or 0) > n0 and self.inspect("{{.State.Running}}") == "true", 30)
        return f"SIGKILL of PID 1; restart policy brought it back: {back} (RestartCount {self.inspect('{{.RestartCount}}')})"
    def recreate(self):
        old = self.cid(); self.dc("down"); self.dc("up", "-d", "server")
        return f"docker compose down + up (new container: {self.cid() != old})"
    def down(self): self.dc("down", "-v", check=False)


def main():
    srv = DockerServer() if MODE == "docker" else LocalServer()
    if MODE == "docker": srv.down()                                          # a leftover run holds W's mounts
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(f"{DATA}/test"); os.makedirs(f"{DATA}/models/x@1")
    for d in ("db", "out/results", "out/spool", "sources", "secrets"): os.makedirs(f"{W}/{d}")
    for n in "abcd":
        with open(f"{DATA}/test/{n}.bin", "wb") as f: f.write(hashlib.sha256(n.encode()).digest() * 9375)   # 300 kB
    open(f"{DATA}/models/x@1/config.json", "w").write("{}")
    for n, t in (("admin-token", ADMIN_TOKEN), ("search-token", SEARCH_TOKEN)):
        fd = os.open(f"{W}/secrets/{n}", os.O_WRONLY | os.O_CREAT, 0o400)
        with os.fdopen(fd, "w") as f: f.write(t + "\n")
    if MODE == "docker":                       # the container runs as 10001: it owns what it writes (db, out, its secret)
        srv.write_env()
        for d in ("db", "out", "secrets"):
            for dp, dn, fn in os.walk(f"{W}/{d}"):
                for x in [dp] + [os.path.join(dp, f) for f in fn]: os.chown(x, 10001, 10001)
    procs = []
    try:
        srv.start()                            # first, so the server (not the host admin) creates the database
        assert wait(lambda: req("GET", "/healthz")[0] == 200, 60), "server did not start"
        admin("manifest")
        run(procs, srv)
    finally:
        for p in procs:
            if p.poll() is None: p.kill()
        if MODE == "docker":
            open(f"{W}/server.log", "w").write(srv.dc("logs", "--no-color", "server", check=False).stdout)
        srv.down()
    n_ok = sum(r[1] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    json.dump(RESULTS, open(f"{W}/results.json", "w"), indent=1)
    sys.exit(0 if n_ok == len(RESULTS) else 1)


def run(procs, srv):
    ok("/healthz without a token -> 200", req("GET", "/healthz")[0] == 200)
    if MODE == "docker": docker_checks(srv)
    # ---- tokens and invites
    ok("no token -> 401", req("POST", "/api/v0/lease", {"caps": {"job_types": ["noop"]}})[0] == 401)
    ok("bad token -> 401", req("POST", "/api/v0/lease", {"caps": {"job_types": ["noop"]}}, token="x" * 43)[0] == 401)
    ok("bad invite -> 403", req("POST", "/api/v0/register", {"invite": "nope", "name": "w"})[0] == 403)
    inv = admin("invite", "--uses", "3").strip()
    toks = {}
    for n in ("wa", "wb", "wc"):
        c, b = req("POST", "/api/v0/register", {"invite": inv, "name": n}); toks[n] = b.get("token")
    ok("invite registers 3 workers", all(toks.values()))
    ok("used-up invite -> 403", req("POST", "/api/v0/register", {"invite": inv, "name": "wd"})[0] == 403)
    t_client = admin("add-worker", "client").split()[3]

    # ---- data access rules
    ok("data without a lease -> 403", req("GET", "/data/test/a.bin", token=t_client)[0] == 403)
    ok("data not in manifest -> 404", req("GET", "/data/test/zzz.bin", token=t_client)[0] == 404)
    ok("data path traversal -> 404", req("GET", "/data/../q.sqlite", token=t_client)[0] in (400, 404))

    # ---- 3 workers, one killed mid-job
    admin("enqueue-noop", "9", "--seconds", "3", "--size", "200000", "--inputs", "test/a.bin", "--tag", "kill")
    ids = jobs_of("kill")
    wa = worker("wa", toks["wa"], "--idle-exit", "30"); wb = worker("wb", toks["wb"], "--idle-exit", "30")
    wc = worker("wc", toks["wc"], "--idle-exit", "30"); procs += [wa, wb, wc]
    wid = {n: db().execute("SELECT id FROM workers WHERE name=?", (n,)).fetchone()[0] for n in ("wa", "wb", "wc")}
    victim = None
    if wait(lambda: db().execute("SELECT job_id FROM leases WHERE worker_id=? AND state='active'", (wid["wa"],)).fetchone(), 15):
        time.sleep(1.0)
        victim = db().execute("SELECT job_id FROM leases WHERE worker_id=? AND state='active'", (wid["wa"],)).fetchone()[0]
        wa.send_signal(signal.SIGKILL); wa.wait()
    ok("worker wa killed mid-job", victim is not None, f"job {victim}")
    ok("all 9 jobs accepted", wait(lambda: all(states()[i] == "accepted" for i in ids), 60))
    ev = [e["kind"] for e in events(victim)]
    ok("killed job: lease expired, reassigned, accepted once",
       ev.count("expire") >= 1 and ev.count("lease") >= 2 and ev.count("accept") + ev.count("accept_late") == 1, str(ev))
    acc = db().execute("SELECT accepted_by FROM jobs WHERE id=?", (victim,)).fetchone()[0]
    ok("killed job accepted from another worker", acc != wid["wa"], f"by {acc}")
    ok("results match the expected bytes", all(
        sha256_file(f"{RES}/{i}/out.bin") == hashlib.sha256(__import__("jobs.noop", fromlist=["x"]).content(
            json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (i,)).fetchone()[0])["seed"], 200000)).hexdigest() for i in ids))
    # restart the killed worker after its lease expired and the job was done elsewhere: it must drop the job cleanly
    wa2 = worker("wa", toks["wa"], "--idle-exit", "2"); procs.append(wa2); wa2.wait(30)
    ok("restarted worker drops its dead lease", "dropping it" in open(f"{W}/wa.log").read() and not os.listdir(f"{W}/wa/current"))
    for p in (wb, wc): p.wait(30)

    # ---- killed and restarted before the lease expires: resumes the same lease
    admin("enqueue-noop", "1", "--seconds", "4", "--size", "1000", "--seed", "100", "--tag", "resume")
    j = jobs_of("resume")[0]
    w = worker("wb", toks["wb"], "--idle-exit", "2"); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(1); w.send_signal(signal.SIGKILL); w.wait()
    w = worker("wb", toks["wb"], "--idle-exit", "2"); procs.append(w); w.wait(30)
    ev = [e["kind"] for e in events(j)]
    ok("restart within the lease resumes it", states()[j] == "accepted" and ev.count("lease") == 1 and "expire" not in ev, str(ev))

    # ---- duplicate completion, corrupted upload, caps (a hand-written client)
    admin("enqueue-noop", "2", "--seconds", "0", "--size", "5000", "--seed", "200", "--tag", "client")
    caps = {"caps": {"job_types": ["noop"], "disk_gb": 1}}
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job = r["job"]
    body = __import__("jobs.noop", fromlist=["x"]).content(job["params"]["seed"], 5000)
    sha = hashlib.sha256(body).hexdigest()
    up = f"/api/v0/upload/{job['id']}/{job['lease_id']}"
    ok("upload of a non-output name -> 403", req("PUT", f"{up}/evil.sh?offset=0", b"x", token=t_client)[0] == 403)
    ok("upload over the cap -> 413", req("PUT", f"{up}/out.bin?offset=0", b"x" * 5001, token=t_client)[0] == 413)
    bad = bytearray(body); bad[100] ^= 1
    req("PUT", f"{up}/out.bin?offset=0", bytes(bad), token=t_client)
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": sha, "size": 5000}}}, token=t_client)
    ok("corrupted upload rejected (sha mismatch -> 422)", c == 422 and r.get("bad") == ["out.bin"], f"{c} {r}")
    ok("corrupted upload deleted on the server", req("GET", f"{up}/out.bin", token=t_client)[1].get("size") == 0)
    ok("wrong offset -> 409 with the resume point", req("PUT", f"{up}/out.bin?offset=7", body, token=t_client)[0] == 409)
    req("PUT", f"{up}/out.bin?offset=0", body[:3000], token=t_client)
    c, r = req("PUT", f"{up}/out.bin?offset=3000", body[3000:], token=t_client)
    ok("chunked upload resumes at the offset", c == 200 and r["size"] == 5000)
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": sha, "size": 5000}}, "result": {"x": 1}}, token=t_client)
    ok("re-uploaded result accepted", c == 200 and r.get("duplicate") is False, f"{c} {r}")
    c, r = req("POST", "/api/v0/complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": {"out.bin": {"sha256": "0" * 64, "size": 5000}}}, token=t_client)
    ok("duplicate completion ignored", c == 200 and r.get("duplicate") is True, f"{c} {r}")
    ok("upload to an accepted job -> 409", req("PUT", f"{up}/out.bin?offset=0", body, token=t_client)[0] == 409)
    ok("accepted result unchanged", sha256_file(f"{RES}/{job['id']}/out.bin") == sha and
       [e["kind"] for e in events(job["id"])].count("accept") == 1)
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job2 = r["job"]
    ok("another worker's lease -> 403", req("POST", "/api/v0/heartbeat", {"job_id": job2["id"], "lease_id": job2["lease_id"]}, token=toks["wc"])[0] == 403)
    req("POST", "/api/v0/fail", {"job_id": job2["id"], "lease_id": job2["lease_id"], "error": "test", "abandon": True}, token=t_client)
    r2 = db().execute("SELECT state, attempts FROM jobs WHERE id=?", (job2["id"],)).fetchone()
    ok("abandon returns the job without counting an attempt", tuple(r2) == ("queued", 0), str(tuple(r2)))
    admin("cancel", str(job2["id"]))
    # the server-wide upload ceiling (--upload-max-mb 48 = 50331648 B) under a job type's own cap (64 MiB)
    admin("enqueue-noop", "1", "--seconds", "0", "--size", str(64 << 20), "--seed", "250", "--tag", "bigout")
    c, r = req("POST", "/api/v0/lease", caps, token=t_client); job3 = r["job"]
    u3 = f"/api/v0/upload/{job3['id']}/{job3['lease_id']}/out.bin"
    c1, _ = req("PUT", f"{u3}?offset=0", b"x" * ((48 << 20) - 10), token=t_client)
    c, r = req("PUT", f"{u3}?offset={(48 << 20) - 10}", b"x" * 11, token=t_client)
    ok("upload over the server's ceiling -> 413", c1 == 200 and c == 413 and "50331648" in r.get("error", ""), f"{c1} {c} {r}")
    admin("cancel", str(job3["id"]))

    # ---- read-only admin summary
    S = "/api/v0/admin/summary"
    ok("admin summary: no / worker / wrong token -> 401", req("GET", S)[0] == 401 and req("GET", S, token=toks["wb"])[0] == 401
       and req("GET", S, token=ADMIN_TOKEN[:-1] + "x")[0] == 401)
    c, r = req("GET", S, token=ADMIN_TOKEN)
    ok("admin summary with the admin token", c == 200 and any(l["level"] == "kill" and l["state"] == "accepted" and l["n"] == 9
       for l in r.get("levels", [])) and "token" not in json.dumps(r.get("workers")), f"{c} {len(r.get('events', []))} events")

    # ---- disk cap: eviction, and jobs larger than the cache are not leased to it
    t_small = admin("add-worker", "small").split()[3]
    for i, n in enumerate("abca"):                 # one level each: inside a level 0.4 would take the cached a again first
        admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", str(300 + i), "--inputs", f"test/{n}.bin", "--tag", f"disk{i}")
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "310", "--inputs", "test/a.bin,test/b.bin,test/c.bin", "--tag", "big")
    w = worker("small", t_small, "--cache-gb", "0.0007", "--idle-exit", "4"); procs.append(w); w.wait(40)
    st = stats("small"); used = sum(os.path.getsize(os.path.join(f"{W}/small/cache/blobs", f)) for f in os.listdir(f"{W}/small/cache/blobs"))
    ok("disk cap: 4 jobs done with eviction", all(states()[i] == "accepted" for t in range(4) for i in jobs_of(f"disk{t}")) and st and st[-1]["evicted"] >= 2,
       f"evicted {st[-1]['evicted'] if st else None}, cache {used} B <= 700000")
    ok("cache stays under its cap", used <= 700_000)
    ok("job bigger than the cache not leased to it", states()[jobs_of("big")[0]] == "queued")
    # Range resume: half of d.bin already in the cache as .part
    dsha = sha256_file(f"{W}/data/test/d.bin")
    os.makedirs(f"{W}/rng/cache/blobs", exist_ok=True)
    open(f"{W}/rng/cache/blobs/{dsha}.part", "wb").write(open(f"{W}/data/test/d.bin", "rb").read()[:120000])
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "320", "--inputs", "test/d.bin", "--tag", "range")
    t_rng = admin("add-worker", "rng").split()[3]
    admin("cancel", str(jobs_of("big")[0]))
    w = worker("rng", t_rng, "--idle-exit", "2"); procs.append(w); w.wait(30)
    st = stats("rng")
    ok("partial download resumed with a Range request", st and st[-1]["down_bytes"] == 300000 - 120000 and states()[jobs_of("range")[0]] == "accepted",
       f"down {st[-1]['down_bytes'] if st else None} of 300000")

    # ---- malformed jobs through the real server (the queue holds a known type with bad parameters)
    c = db()
    for p in ({"seconds": "3", "size": 1, "seed": 1}, {"seconds": 1, "size": 1, "seed": 1, "cmd": "rm -rf /"}):
        c.execute("INSERT INTO jobs(type, version, params, inputs, requires, input_bytes, state, max_attempts, level, created, updated) "
                  "VALUES('noop', 2, ?, '[]', '{}', 0, 'queued', 2, 'malformed', 0, 0)", (json.dumps(p),))
    c.commit()
    wb2 = worker("wb", toks["wb"], "--idle-exit", "4"); wc2 = worker("wc", toks["wc"], "--idle-exit", "4"); procs += [wb2, wc2]
    wb2.wait(40); wc2.wait(40)
    mids = jobs_of("malformed")
    ok("malformed jobs rejected by workers, then failed", all(states()[i] == "failed" for i in mids) and
       all(len(events(i, "reject")) == 2 for i in mids), str([states()[i] for i in mids]))
    ok("workers survive malformed jobs", wb2.returncode == 0 and wc2.returncode == 0)

    # ---- a fake malicious server
    good = {"type": "noop", "version": 1, "params": {"seconds": 0, "size": 1, "seed": 1}, "inputs": [], "lease_id": "f" * 32, "heartbeat_s": 5}
    Fake.queue = [dict(good, id=1, type="shell", params={"cmd": "id"}),
                  dict(good, id=2, params={"seconds": 0, "size": 1, "seed": 1, "argv": ["sh"]}),
                  dict(good, id=3, params={"seconds": 0, "size": 1, "seed": 1, "inputs": ["test/../../etc/passwd"]},
                       inputs=[{"name": "test/../../etc/passwd", "sha256": "0" * 64, "size": 1}]),
                  dict(good, id=4, params={"seconds": 0, "size": 1, "seed": 1, "inputs": ["test/a.bin"]},
                       inputs=[{"name": "test/a.bin", "sha256": "0" * 64, "size": 1}, {"name": "models/x@1/config.json", "sha256": "0" * 64, "size": 1}]),
                  dict(good, id=5, type="eval_mix", version=0, params={"model": "m", "revision": "r", "recipe": "a", "golden": "g", "kv": "bf16"}),
                  dict(good, id=6, type="quant_layer", params={"model": "m", "revision": "r", "arch": "qwen3_5", "K": 3, "codebook": "i8",
                       "level": "l", "layer": 1, "stamp": "0" * 16, "settings": {"sigma_reg": 0.025, "sigma_reg_in": None, "sigma_reg_out": None,
                       "scaled_hessian": True, "qtf2": True, "hout": None, "seed_offset": 0}}),
                  dict(good, id=f"{W}/escape/x", lease_id="f" * 32),
                  dict(good, id=7, lease_id="")]
    fs = ThreadingHTTPServer(("127.0.0.1", PORT + 1), Fake); threading.Thread(target=fs.serve_forever, daemon=True).start()
    lf = open(f"{W}/fake.log", "w")
    p = subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", f"http://127.0.0.1:{PORT + 1}", "--token", "t", "--state", f"{W}/fake",
                          "--poll", "0.2", "--idle-exit", "3"], stdout=lf, stderr=subprocess.STDOUT); procs.append(p); p.wait(30); fs.shutdown()
    rej = {f["job_id"]: f for f in Fake.fails}
    ok("fake server: all 8 bad jobs rejected", sorted(rej, key=str) == sorted([1, 2, 3, 4, 5, 6, 7, f"{W}/escape/x"], key=str)
       and all(f.get("reject") for f in rej.values()), "; ".join(f"{k}: {v['error'][19:80]}" for k, v in sorted(rej.items(), key=lambda kv: str(kv[0]))))
    ok("fake server: a job id that is a path creates nothing", not os.path.exists(f"{W}/escape") and
       not glob.glob(f"{W}/fake/current/*escape*") and not glob.glob(f"{W}/fake/jobs/*escape*"))
    ok("fake server: no download attempted", not Fake.data_hits, str(Fake.data_hits))

    # ---- graceful stop: finish, then abandon
    admin("enqueue-noop", "1", "--seconds", "3", "--size", "100", "--seed", "400", "--tag", "stopfin")
    j = jobs_of("stopfin")[0]; w = worker("wc", toks["wc"]); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(0.5); w.send_signal(signal.SIGTERM); w.wait(30)
    ok("SIGTERM: finishes the current job, then exits", states()[j] == "accepted" and w.returncode == 0)
    admin("enqueue-noop", "1", "--seconds", "30", "--size", "100", "--seed", "401", "--tag", "stopab")
    j = jobs_of("stopab")[0]; w = worker("wc", toks["wc"], "--on-stop", "abandon"); procs.append(w)
    wait(lambda: states()[j] == "leased", 10); time.sleep(0.5); t0 = time.time(); w.send_signal(signal.SIGTERM); w.wait(30)
    r = db().execute("SELECT state, attempts FROM jobs WHERE id=?", (j,)).fetchone()
    ok("--on-stop abandon: job handed back at once, attempt not counted", tuple(r) == ("queued", 0) and time.time() - t0 < 5, str(tuple(r)))
    admin("cancel", str(j))

    # ---- time windows
    lt = time.localtime(); m = lt.tm_hour * 60 + lt.tm_min
    hm = lambda x: f"{(x // 60) % 24:02d}:{x % 60:02d}"
    ok("window logic", in_windows([f"{hm(m - 5)}-{hm(m + 5)}"]) and not in_windows([f"{hm(m + 10)}-{hm(m + 20)}"]) and
       in_windows([f"{hm(m + 10)}-{hm(m + 5)}"]) and in_windows([]))
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--seed", "500", "--tag", "win")
    j = jobs_of("win")[0]
    w = worker("wb", toks["wb"], "--window", f"{hm(m + 30)}-{hm(m + 40)}"); procs.append(w); time.sleep(4)
    ok("outside its window a worker leases nothing", states()[j] == "queued")
    w.send_signal(signal.SIGTERM); w.wait(10)
    admin("cancel", str(j))

    # ---- a worker configured by environment only (the image's way): invite -> token file (0600) -> token file
    inv = admin("invite").strip(); tf = f"{W}/envw/token"
    admin("enqueue-noop", "2", "--seconds", "0", "--size", "100", "--seed", "550", "--tag", "envw")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RADQUANT_", "RQAH_"))}
    e1 = dict(env, RQAH_SERVER=URL, RQAH_STATE=f"{W}/envw", RQAH_TOKEN_FILE=tf, RQAH_INVITE=inv,
              RQAH_NAME="envw", RQAH_IMAGE_ID="e2e-env")
    a = [PY, "-I", f"{HERE}/worker/worker.py", "--poll", "0.5", "--idle-exit", "2", "--max-jobs", "1"]
    r1 = subprocess.run(a, env=e1, capture_output=True, text=True, timeout=60)
    # the second start with the 0.2 names only (still read in 0.3)
    e2 = dict(env, RADQUANT_SERVER=URL, RADQUANT_STATE=f"{W}/envw", RADQUANT_TOKEN_FILE=tf, RADQUANT_NAME="envw", RADQUANT_IMAGE_ID="e2e-env")
    r2 = subprocess.run(a, env=e2, capture_output=True, text=True, timeout=60)
    wrow = db().execute("SELECT caps FROM workers WHERE name='envw'").fetchall()
    ok("worker from env only (RQAH_*, then the 0.2 RADQUANT_* names): registers once, token file 0600, reuses it", r1.returncode == 0 and r2.returncode == 0 and
       len(wrow) == 1 and os.stat(tf).st_mode & 0o777 == 0o600 and json.loads(wrow[0][0])["image"] == "e2e-env" and
       all(states()[i] == "accepted" for i in jobs_of("envw")) and "RADQUANT_SERVER is deprecated" in r2.stdout + r2.stderr,
       f"rc {r1.returncode}/{r2.returncode}, workers {len(wrow)}, "
       f"mode {oct(os.stat(tf).st_mode & 0o777) if os.path.exists(tf) else None}")

    # ---- the server goes away mid-run: graceful restart, crash (SIGKILL), remove + recreate
    for k, how in enumerate(("restart", "crash", "recreate")):
        tag = f"rs-{how}"
        admin("enqueue-noop", "6", "--seconds", "3", "--size", "200000", "--seed", str(600 + 10 * k), "--tag", tag)
        ids = jobs_of(tag)
        ws = [worker(n, toks[n], "--idle-exit", "20") for n in ("wb", "wc")]; procs += ws
        mid_ok = wait(lambda: sum(states()[i] == "accepted" for i in ids) >= 1 and sum(states()[i] == "leased" for i in ids) >= 2, 40, 0.1)
        mid = collections.Counter(states()[i] for i in ids)
        t0 = time.time(); how_s = getattr(srv, how)()
        back = wait(lambda: req("GET", "/healthz")[0] == 200, 60)
        ok(f"{how}: server back mid-run", mid_ok and back, f"{how_s}; healthz after {time.time() - t0:.1f}s; jobs at the time {dict(mid)}")
        done = wait(lambda: all(states()[i] == "accepted" for i in ids), 120)
        for w in ws: w.wait(60)
        acc = {i: [e["kind"] for e in events(i)] for i in ids}
        once = all(v.count("accept") + v.count("accept_late") == 1 for v in acc.values())
        seed = lambda i: json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (i,)).fetchone()[0])["seed"]
        exact = all(os.path.exists(f"{RES}/{i}/out.bin") and sha256_file(f"{RES}/{i}/out.bin") == hashlib.sha256(
            __import__("jobs.noop", fromlist=["x"]).content(seed(i), 200000)).hexdigest() for i in ids)
        n_exp = sum(v.count("expire") for v in acc.values()); n_late = sum(v.count("accept_late") for v in acc.values())
        integ = db().execute("PRAGMA integrity_check").fetchone()[0]
        ok(f"{how}: all 6 jobs accepted once each, byte-exact, db intact", done and once and exact and integ == "ok",
           f"expired leases {n_exp}, late accepts {n_late}, integrity {integ}")
        ok(f"{how}: workers rode it out", all(w.returncode == 0 for w in ws), str([w.returncode for w in ws]))
    # ---- killed mid-upload: the partial upload survives in the spool and the worker resumes it
    admin("enqueue-noop", "1", "--seconds", "0", "--size", str(40 << 20), "--seed", "700", "--tag", "rs-upload")
    j = jobs_of("rs-upload")[0]; n0 = len(stats("wb")); srv.arm()
    w = worker("wb", toks["wb"], "--idle-exit", "20", "--chunk-mb", "1"); procs.append(w)
    part = lambda: max([os.path.getsize(f) for f in glob.glob(f"{SPOOL}/{j}/*/out.bin")] or [0])
    hit = wait(lambda: 8 << 20 <= part() <= 32 << 20, 60, 0.002); s0 = part()
    s1 = []; how_s = srv.crash(lambda: s1.append(part())); s1 = s1[0]   # spool right after the kill
    done = wait(lambda: states()[j] == "accepted", 90); w.wait(60)
    st = stats("wb")[n0:]; up = st[-1].get("up_bytes") if st else None
    seed = json.loads(db().execute("SELECT params FROM jobs WHERE id=?", (j,)).fetchone()[0])["seed"]
    exact = done and sha256_file(f"{RES}/{j}/out.bin") == hashlib.sha256(__import__("jobs.noop", fromlist=["x"]).content(seed, 40 << 20)).hexdigest()
    ok("crash mid-upload: partial upload kept, resumed, accepted once, byte-exact",
       hit and s1 >= s0 - (1 << 20) and s1 > 0 and exact and up is not None and up <= (41 << 20) and
       [e["kind"] for e in events(j)].count("accept") + [e["kind"] for e in events(j)].count("accept_late") == 1,
       f"{how_s}; spool {s0 / 2**20:.1f} MiB before the kill, {s1 / 2**20:.1f} MiB just after; worker sent {None if up is None else up / 2**20:.1f} MiB of 40")
    run_v02(procs, srv)
    run_w25(procs, srv)
    if MODE == "docker":
        ok("container healthy at the end", wait(lambda: srv.inspect("{{.State.Health.Status}}") == "healthy", 40),
           srv.inspect("{{.State.Health.Status}}"))


def docker_checks(srv):
    ok("container: healthcheck turns healthy", wait(lambda: srv.inspect("{{.State.Health.Status}}") == "healthy", 40),
       srv.inspect("{{.State.Health.Status}}"))
    uid = srv.dc("exec", "-T", "server", "id", "-u").stdout.strip()
    ps = srv.dc("exec", "-T", "server", "cat", "/proc/1/status").stdout
    puid = [l.split()[1] for l in ps.splitlines() if l.startswith("Uid:")]
    ok("container: server runs as uid 10001 (non-root)", uid == "10001" and puid == ["10001"], f"exec {uid}, PID 1 {puid}")
    ok("container: read-only rootfs, no capabilities, restart unless-stopped",
       srv.inspect("{{.HostConfig.ReadonlyRootfs}}") == "true" and srv.inspect("{{.HostConfig.RestartPolicy.Name}}") == "unless-stopped"
       and "ALL" in srv.inspect("{{.HostConfig.CapDrop}}"),
       f"{srv.inspect('{{.HostConfig.ReadonlyRootfs}}')} {srv.inspect('{{.HostConfig.CapDrop}}')} {srv.inspect('{{.HostConfig.RestartPolicy.Name}}')}")
    r = srv.dc("exec", "-T", "server", "rad-quant-at-home", "admin", "invite", "--note", "from-inside")
    r0 = srv.dc("exec", "-T", "server", "rqah", "version")
    ok("container: `rad-quant-at-home admin` works inside the running server (and the alias `rqah`)",
       r.returncode == 0 and len(r.stdout.strip()) >= 20 and r0.returncode == 0 and "rad-quant-at-home-server" in r0.stdout, r0.stdout.strip())
    r = srv.dc("run", "--rm", "-T", "admin", "list")
    ok("container: `compose run --rm admin` works beside it", r.returncode == 0, (r.stderr or "").strip()[-200:])
    ok("container: port published on loopback only", srv.dc("port", "server", "8740").stdout.strip() == f"127.0.0.1:{PORT}",
       srv.dc("port", "server", "8740").stdout.strip())


# ======================================================================== 0.2 (rad-quant-at-home): w24
FAKE_RAD, FAKE_GOLD = f"{HERE}/tests/fake_radiance.py", f"{HERE}/tests/fake_mkgold27.py"
EM_TOL = {"row_atol": 1e-3, "row_rtol": 0.05, "rows_frac": 0.995, "mean_rel": 2e-3, "top1_flip": 0.01}
GPU = ("--gpu-env", "CUDA_VISIBLE_DEVICES")


def tok(name):
    TOKENS[name] = admin("add-worker", name).split()[3]
    return TOKENS[name]
def wid(name): return db().execute("SELECT id FROM workers WHERE name=?", (name,)).fetchone()[0]
def jrow(j): return db().execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
def result(j): return json.loads(jrow(j)["result"] or "{}")
def manifest(): return json.load(open(f"{DATA}/manifest.json"))["files"]


def interval(j):
    """(lease time, accept time) of the accepted attempt"""
    ev = events(j); acc = [e for e in ev if e["kind"] in ("accept", "accept_late")]
    if not acc: return None
    lid = acc[-1]["lease_id"]
    return [e["t"] for e in ev if e["kind"] == "lease" and e["lease_id"] == lid][0], acc[-1]["t"]


def overlap(ivs):
    pts = sorted([(a, 1) for a, b in ivs] + [(b, -1) for a, b in ivs], key=lambda x: (x[0], x[1]))
    m = c = 0
    for _, d in pts: c += d; m = max(m, c)
    return m


def run_v02(procs, srv):
    # the v0 scenarios queued spot-check re-runs (first job of each donor) with no verifier running: drop them
    c = db(); c.execute("UPDATE jobs SET state='cancelled' WHERE verify_of IS NOT NULL AND state='queued'")
    c.execute("UPDATE jobs SET verify_state=NULL WHERE verify_state='pending'"); c.commit()
    v02_requirements(procs)
    v02_multicard(procs)
    E = v02_eval_setup()
    v02_eval(procs, E)
    v02_spot(procs, E)
    v02_golden(procs)
    v02_units(E)


def v02_requirements(procs):
    spec = {"req-c16": ("1", "16", "8"), "req-c32": ("1", "32", "8"), "req-ram40": ("0", "0", "40"),
            "req-head": ("1", "8", "48"), "req-none": ("1", "64", "8")}
    for i, (tag, (cards, vram, ram)) in enumerate(spec.items()):
        admin("enqueue-noop", "1", "--seconds", "0.3", "--size", "100", "--seed", str(1000 + i), "--tag", tag,
              "--cards", cards, "--vram-gb", vram, "--ram-gb", ram)
    J = {t: jobs_of(t)[0] for t in spec}
    w16 = worker("s16", tok("s16"), "--gpu-name", "fake16", "--cards", "1", "--vram-gb", "16", "--ram-gb", "32", *GPU,
                 "--idle-exit", "6"); procs.append(w16); w16.wait(40)
    st = {t: states()[j] for t, j in J.items()}
    ok("requirements: a 16 GB / 32 GB RAM worker gets only the job that fits it",
       st["req-c16"] == "accepted" and all(st[t] == "queued" for t in spec if t != "req-c16"), str(st))
    w32 = worker("s32", tok("s32"), "--gpu-name", "fake32", "--cards", "1", "--vram-gb", "32", "--ram-gb", "64", *GPU,
                 "--idle-exit", "6"); procs.append(w32); w32.wait(40)
    by = {t: jrow(j)["accepted_by"] for t, j in J.items()}
    ok("requirements: a 32 GB / 64 GB RAM worker gets the 32 GB, 40 GB-RAM and head-sized (48 GB RAM) jobs; "
       "a job nobody fits stays queued", all(by[t] == wid("s32") for t in ("req-c32", "req-ram40", "req-head")) and
       states()[J["req-none"]] == "queued" and not [e for e in events(J["req-none"]) if e["kind"] == "lease"], str(by))
    admin("cancel", str(J["req-none"]))


def v02_multicard(procs):
    t = tok("m4")
    admin("enqueue-noop", "6", "--seconds", "2", "--size", "1000", "--seed", "1100", "--tag", "mc-a", "--cards", "1", "--proc")
    admin("enqueue-noop", "1", "--seconds", "1", "--size", "1000", "--seed", "1120", "--tag", "mc-g", "--cards", "4", "--proc")
    admin("enqueue-noop", "2", "--seconds", "1", "--size", "1000", "--seed", "1130", "--tag", "mc-b", "--cards", "1", "--proc")
    A, G, B2 = jobs_of("mc-a"), jobs_of("mc-g")[0], jobs_of("mc-b")
    w = worker("m4", t, "--gpu-name", "fake32", "--gpus", "0,1,2,3", "--vram-gb", "32", "--ram-gb", "64", *GPU,
               "--idle-exit", "6", "--job-types", "noop"); procs.append(w)
    done = wait(lambda: all(states()[j] == "accepted" for j in A + [G] + B2), 90); w.wait(40)
    iv = {j: interval(j) for j in A + [G] + B2}
    singles = A + B2
    ok("multi-card: 4-card worker ran single-card jobs side by side (at most 4 at once)",
       done and 3 <= overlap([iv[j] for j in singles]) <= 4, f"max concurrent {overlap([iv[j] for j in singles]) if done else None}")
    devs_ok = all(len(result(j)["devices"]) == 1 and result(j)["child_env"] == {"CUDA_VISIBLE_DEVICES": str(result(j)["devices"][0])}
                  for j in singles)
    clash = [(a, b) for a in singles for b in singles if a < b and iv[a][0] < iv[b][1] and iv[b][0] < iv[a][1]
             and result(a)["devices"] == result(b)["devices"]]
    ok("multi-card: each job process sees only its own card; concurrent jobs never share one", devs_ok and not clash,
       f"clashes {clash[:3]}")
    g = result(G)
    ok("gang job: all 4 cards, one process seeing CUDA_VISIBLE_DEVICES=0,1,2,3",
       g.get("devices") == [0, 1, 2, 3] and g.get("child_env") == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, str(g.get("child_env")))
    res = [e for e in events(G) if e["kind"] == "reserve"]
    lease_g = iv[G][0]
    m4 = wid("m4")
    between = [e for e in db().execute("SELECT * FROM events WHERE kind='lease' AND worker_id=? AND t>? AND t<?",
                                       (m4, res[0]["t"] if res else 0, lease_g)).fetchall()]
    ok("gang job: the worker drained for it (reservation, no new single job until it ran), not starved by later jobs",
       len(res) == 1 and not between and all(iv[j][0] > lease_g for j in B2) and
       all(not (iv[j][0] < iv[G][1] and iv[G][0] < iv[j][1]) for j in singles),
       f"reserve at +{(res[0]['t'] - min(v[0] for v in iv.values())) if res else -1:.1f}s, gang lease at "
       f"+{lease_g - min(v[0] for v in iv.values()):.1f}s; leases during the drain {len(between)}")
    admin("enqueue-noop", "3", "--seconds", "1", "--size", "100", "--seed", "1140", "--tag", "mc-ram", "--cards", "1", "--ram-gb", "40")
    R_ = jobs_of("mc-ram")
    w = worker("m4", t, "--gpu-name", "fake32", "--gpus", "0,1,2,3", "--vram-gb", "32", "--ram-gb", "64", *GPU,
               "--idle-exit", "6", "--job-types", "noop"); procs.append(w)
    done = wait(lambda: all(states()[j] == "accepted" for j in R_), 60); w.wait(40)
    ok("multi-card: one host-RAM budget (3 jobs x 40 GB on a 64 GB worker with 4 free cards run one at a time)",
       done and overlap([interval(j) for j in R_]) == 1)
    # --stop-file with two jobs running (both heartbeat threads see it): finish both, abandon none, exit 0
    admin("enqueue-noop", "3", "--seconds", "3", "--size", "100", "--seed", "1150", "--tag", "mc-stop", "--cards", "1")
    S_ = jobs_of("mc-stop"); sf = f"{W}/m4.stop"
    w = worker("m4", t, "--gpu-name", "fake32", "--gpus", "0,1", "--vram-gb", "32", "--ram-gb", "64", *GPU, "--stop-file", sf,
               "--job-types", "noop"); procs.append(w)
    wait(lambda: sum(states()[j] == "leased" for j in S_) == 2, 20); time.sleep(1.6); open(sf, "w").close(); w.wait(30)
    st_ = [states()[j] for j in S_]
    ok("multi-card --stop-file: both running jobs finish (none abandoned), nothing new leased, exit 0",
       w.returncode == 0 and sorted(st_) == ["accepted", "accepted", "queued"] and
       not any(e["kind"] == "abandon" for j in S_ for e in events(j)), f"{st_}, rc {w.returncode}")
    admin("cancel", *[str(j) for j in S_ if states()[j] == "queued"])


def v02_eval_setup():
    src = f"{W}/v02src"; os.makedirs(src, exist_ok=True)
    lv = fakes.ladder(f"{src}/levels", Ks=(2, 3, 4))
    for rev in ("1", "2"):
        for k, p in lv.items(): admin("ladder-add", "fm", rev, k, p)
    docs = [{"prompt": (f"doc {i}: " + " ".join(f"word{(i * 7 + j) % 23}" for j in range(30 + 9 * i))), "source": s,
             "score_from": i % 3} for i, s in enumerate(["xcorp/web", "xcorp/code", "xcorp/web", "xcorp/chat", "xcorp/code", "xcorp/web"])]
    with open(f"{src}/xc.jsonl", "w") as f: f.writelines(json.dumps(d) + "\n" for d in docs)
    gd = f"{DATA}/goldens/xg"; fakes.record_golden(f"{src}/xc.jsonl", gd, top=16)
    rc1, o1 = admin_rc("golden-add", "xg", gd, "--public", "--corpus", "xcorp", "--pool", "search")
    with open(f"{src}/tr.jsonl", "w") as f: f.writelines(json.dumps(dict(d, source="agent-trace/run7")) + "\n" for d in docs[:2])
    fakes.record_golden(f"{src}/tr.jsonl", f"{DATA}/goldens/pg", top=16)
    rc2, o2 = admin_rc("golden-add", "pg", f"{DATA}/goldens/pg", "--public", "--corpus", "traces")
    rc3, o3 = admin_rc("golden-add", "xg2", gd, "--corpus", "xcorp")
    json.dump({"privacy": "public", "sources": ["xcorp/web"]}, open(f"{DATA}/goldens/pg/golden.json", "w"))   # a marker that lies
    fakes.record_golden(f"{src}/xc.jsonl", f"{DATA}/goldens/pv", top=16)
    json.dump({"privacy": "private", "sources": ["xcorp/web"]}, open(f"{DATA}/goldens/pv/golden.json", "w"))
    sk = f"{DATA}/models/fm@1/rad/k2/skeleton.radh"
    base = RF.mix_size(sk, {e["name_s"]: f"{DATA}/models/fm@1/rad/k2/t/{e['name_s']}.radw" for e in RF.Rad(sk, skeleton=True).entries})
    admin("model-info", "fm", "1", "--n-params", str(int(base * 8 / 4.5) + 1))     # the base mix just under 4.5 bpw
    admin("model-info", "fm", "2", "--n-params", str(int(base * 8 / 8.0)))         # ~8 bpw: TP2
    admin("manifest")
    man = manifest()
    ok("golden-add: public golden linked with its marker; a golden with documents from agent traces refused; "
       "nothing added without --public", rc1 == 0 and "goldens/xg/golden.json" in man and rc2 != 0 and "private sources" in o2
       and rc3 != 0, f"{o2.strip()[-80:]} | {o3.strip()[-60:]}")
    ok("manifest: a golden marked private is never listed (so never served)",
       not any(k.startswith("goldens/pv/") for k in man) and "goldens/pg/kld.json" in man)
    return {"src": src, "levels": lv, "docs": docs, "gold": gd, "names": fakes.weight_names()}


def expected(E, rev, base, mix, act=None):
    """assemble the mix here from the server's pieces and score it with the fake radiance: (sha256, rows bytes)"""
    root = f"{DATA}/models/fm@{rev}/rad"
    pp = {n: f"{root}/{mix.get(n, base)}/t/{n}.radw" for n in E["names"]}
    out = f"{E['src']}/exp.rad"
    info = RF.assemble(out, f"{root}/{base}/skeleton.radh", pp, act or {})
    fakes.score(out, E["gold"], f"{E['src']}/exp.json")
    return info["sha256"], open(f"{E['src']}/exp.json.rows", "rb").read()


LIN = [n for n in fakes.weight_names() if any(f".{x}." in n for x in fakes.LINEARS)]


def v02_eval(procs, E):
    S = "/api/v0/search/batch"
    cands = [{"id": "c0", "mix": {}},
             {"id": "c1", "mix": {"blk.0.attn_q.weight": "k3", "blk.1.ffn_up.weight": "k3"}},
             {"id": "c2", "mix": {n: "k4" for n in LIN}},
             {"id": "c3", "mix": {"blk.2.ffn_down.weight": "k4", "output.weight": "k3"},
              "act": {"blk.2.ffn_down.weight": "a8g128", "blk.0.attn_k.weight": "a8t"}},
             {"id": "c4", "mix": [["blk.1.ffn_up.weight", "k3"], ["blk.0.attn_q.weight", "k3"]]},
             {"id": "c5", "mix": {"blk.1.attn_k.weight": "k2"}}]
    spec = {"name": "r1", "model": "fm", "revision": "1", "base": "k2", "golden": "xg", "kv": "bf16", "max_model_len": 4096,
            "candidates": cands}
    ok("search API: wrong / worker token -> 401", req("POST", S, spec, token="x" * 40)[0] == 401 and
       req("POST", S, spec, token=admin("add-worker", "nosearch").split()[3])[0] == 401)
    c1, r1 = req("POST", S, dict(spec, golden="pg"), token=SEARCH_TOKEN)
    c2, r2 = req("POST", S, dict(spec, golden="pv"), token=SEARCH_TOKEN)
    c3, r3 = req("POST", S, dict(spec, candidates=[{"id": "x", "mix": {"blk.0.attn_q.weight": "k9"}}]), token=SEARCH_TOKEN)
    ok("search API: refuses a golden with private documents (even under a lying marker), a golden marked private, "
       "a piece that does not exist", c1 == 400 and "private sources" in r1.get("error", "") and c2 == 400 and
       c3 == 400 and "not in the manifest" in r3.get("error", ""), f"{r1.get('error', '')[:70]} | {r2.get('error', '')[:60]}")
    c, b1 = req("POST", S, spec, token=SEARCH_TOKEN)
    ok("search API: a batch of 6 candidates -> 4 new eval_mix jobs, 2 reused (a duplicate; a mix naming only the base)",
       c == 200 and b1.get("new") == 4 and b1.get("reused") == 2, str(b1))
    c, b2 = req("POST", S, dict(spec, name="tp2", revision="2", candidates=cands[:2]), token=SEARCH_TOKEN)
    st1 = req("GET", f"{S}/{b1['batch']}", token=SEARCH_TOKEN)[1]
    J = {it["id"]: it["job"] for it in st1["items"]}
    J2 = {it["id"]: it["job"] for it in req("GET", f"{S}/{b2['batch']}", token=SEARCH_TOKEN)[1]["items"]}
    rq = {k: json.loads(jrow(j)["requires"]) for k, j in J.items()}
    ok("eval_mix requirements by size: base-like mixes 16 GB cards, the all-k4 mix 32 GB, the ~8 bpw model 2 cards",
       rq["c0"]["vram_gb"] == 16 and rq["c2"]["vram_gb"] == 32 and all(json.loads(jrow(j)["requires"])["cards"] == 2 for j in J2.values()),
       f"bpw c0 {json.loads(jrow(J['c0'])['params'])['bytes'] * 8 / json.loads(jrow(J['c0'])['params'])['n_params']:.3f}, "
       f"c2 {json.loads(jrow(J['c2'])['params'])['bytes'] * 8 / json.loads(jrow(J['c2'])['params'])['n_params']:.3f}")
    # serve-time privacy: a lease's golden file is refused while its marker says private
    tcl = admin("add-worker", "evclient").split()[3]
    c, r = req("POST", "/api/v0/lease", {"caps": {"job_types": ["eval_mix"], "gpu": True, "cards": 1, "vram_gb": 32, "ram_gb": 64,
                                                  "disk_gb": 1}}, token=tcl)
    jb = r.get("job") or {}
    gj = f"{DATA}/goldens/xg/golden.json"; orig = open(gj, "rb").read()
    a1 = req("GET", "/data/goldens/xg/kld.json", token=tcl)[0]
    json.dump({"privacy": "private"}, open(gj, "w")); a2, b_ = req("GET", "/data/goldens/xg/kld.json", token=tcl)
    open(gj, "wb").write(orig)
    req("POST", "/api/v0/fail", {"job_id": jb.get("id"), "lease_id": jb.get("lease_id"), "error": "test", "abandon": True}, token=tcl)
    ok("data: a golden file is served only while its marker says public (checked per request)", a1 == 200 and a2 == 403, f"{a1} {a2} {b_}")
    # 1. a 16 GB card with a small cache: the 16 GB-class candidates, one at a time, evicting as it goes
    ib = max(jrow(j)["input_bytes"] for k, j in J.items() if json.loads(jrow(j)["requires"])["vram_gb"] <= 16)
    cap = ib + 4096                    # one candidate's inputs fit; the next one's two new pieces need an eviction
    e16 = worker("e16", tok("e16"), "--gpu-name", "fake16", "--cards", "1", "--vram-gb", "16", "--ram-gb", "32", *GPU,
                 "--radiance", FAKE_RAD, "--job-types", "eval_mix", "--cache-gb", f"{cap / 1e9:.9f}", "--idle-exit", "6")
    procs.append(e16); e16.wait(90)
    # 2. a 32 GB card and a 2-card box
    e32 = worker("e32", tok("e32"), "--gpu-name", "fake32", "--cards", "1", "--vram-gb", "32", "--ram-gb", "64", *GPU,
                 "--radiance", FAKE_RAD, "--job-types", "eval_mix", "--idle-exit", "6")
    procs.append(e32); wait(lambda: states()[J["c2"]] == "accepted", 60)
    m2 = worker("m2", tok("m2"), "--gpu-name", "fake32", "--gpus", "0,1", "--vram-gb", "32", "--ram-gb", "64", *GPU,
                "--radiance", FAKE_RAD, "--job-types", "eval_mix,golden_record", "--mkgold27", FAKE_GOLD, "--idle-exit", "6")
    procs.append(m2)
    done = wait(lambda: req("GET", f"{S}/{b1['batch']}", token=SEARCH_TOKEN)[1].get("done") and
                req("GET", f"{S}/{b2['batch']}", token=SEARCH_TOKEN)[1].get("done"), 120)
    e32.wait(60); m2.wait(60)
    st1 = req("GET", f"{S}/{b1['batch']}", token=SEARCH_TOKEN)[1]; st2 = req("GET", f"{S}/{b2['batch']}", token=SEARCH_TOKEN)[1]
    ok("eval_mix: both batches finished (status over the search API)", done, f"{st1.get('counts')} {st2.get('counts')}")
    by = {k: jrow(j)["accepted_by"] for k, j in J.items()}
    big = {k for k, j in J.items() if json.loads(jrow(j)["requires"])["vram_gb"] > 16}
    ok("eval_mix: the 32 GB-class mixes (> 4.5 bpw) never went to the 16 GB card; it did all the others",
       big == {"c2", "c3"} and all(by[k] == wid("e32") for k in big) and all(by[k] == wid("e16") for k in J if k not in big),
       f"{by}, 32 GB class {sorted(big)}")
    good = True; det = []
    for k, j in list(J.items()) + [("tp2-" + k, j) for k, j in J2.items()]:
        cand = next(c for c in cands if c["id"] == k.replace("tp2-", ""))
        mix = dict(cand["mix"]) if isinstance(cand["mix"], dict) else dict(map(tuple, cand["mix"]))
        mix = {a: b for a, b in mix.items() if b != "k2"}
        sha, rows = expected(E, "2" if k.startswith("tp2-") else "1", "k2", mix, cand.get("act"))
        meta = json.load(open(f"{RES}/{j}/job.json"))
        same = meta["container_sha256"] == sha and open(f"{RES}/{j}/rows.f32", "rb").read() == rows
        good &= same
        if not same: det.append(k)
    ok("eval_mix: every container assembled on the donor = assembled here (sha256); rows = the fake radiance's on it",
       good, f"mismatch {det}")
    c, rb = req("GET", f"{S}/{b1['batch']}/rows/c3", token=SEARCH_TOKEN)
    out = f"{W}/collect1"; admin("batch-collect", str(b1["batch"]), out)
    ok("eval_mix: rows over the search API and batch-collect = the stored result",
       c == 200 and rb == open(f"{RES}/{J['c3']}/rows.f32", "rb").read() == open(f"{out}/c3.rows", "rb").read()
       and len(open(f"{out}/results.jsonl").readlines()) == 6)
    open(f"{W}/search-token", "w").write(SEARCH_TOKEN)
    sq = subprocess.run([PY, "-I", f"{HERE}/client/searchq.py", "--server", URL, "--token-file", f"{W}/search-token", "collect",
                         str(b1["batch"]), f"{W}/collect2"], capture_output=True, text=True)
    ok("client/searchq.py collect: the same rows and results lines", sq.returncode == 0 and all(
       open(f"{W}/collect2/{k}.rows", "rb").read() == open(f"{RES}/{j}/rows.f32", "rb").read() for k, j in J.items()), sq.stdout.strip())
    g = [json.load(open(f"{RES}/{j}/kld.json")) for j in J2.values()]
    ok("eval_mix TP2 (gang): on the 2-card worker, radiance --tp 2 seeing cards 0,1",
       all(jrow(j)["accepted_by"] == wid("m2") for j in J2.values()) and all(x["tp"] == 2 and x["devices"] == {"CUDA_VISIBLE_DEVICES": "0,1"} for x in g))
    man = manifest()
    st = {s_["job"]: s_ for s_ in stats("m2")}
    g0, g1 = J2["c0"], J2["c1"]
    k3 = man["models/fm@2/rad/k3/t/blk.0.attn_q.weight.radw"]["size"] + man["models/fm@2/rad/k3/t/blk.1.ffn_up.weight.radw"]["size"]
    ok("cache: the 2nd candidate downloaded only the 2 pieces it did not hold (content-addressed, sha256)",
       st[g0]["down_bytes"] > k3 and st[g1]["down_bytes"] == k3,
       f"first {st[g0]['down_bytes']} of {jrow(g0)['input_bytes']} B, second {st[g1]['down_bytes']} B (k3 pieces {k3} B)")
    s16 = stats("e16"); used = sum(os.path.getsize(os.path.join(f"{W}/e16/cache/blobs", f)) for f in os.listdir(f"{W}/e16/cache/blobs"))
    ok("cache: LRU eviction under a cap of one candidate's inputs + 4 KiB, results still exact",
       s16 and s16[-1]["evicted"] >= 1 and used <= cap, f"evicted {s16[-1]['evicted'] if s16 else None}, used {used} <= {cap:.0f}")
    E.update(J=J, J2=J2, batch1=b1["batch"])


def v02_spot(procs, E):
    S = "/api/v0/search/batch"
    vt = tok("verif"); admin("verifier", str(wid("verif")))
    tcn, tce, th = tok("cheatn"), tok("cheate"), tok("honest")
    admin("enqueue-noop", "3", "--seconds", "0.2", "--size", "5000", "--seed", "1200", "--tag", "cheat-n")
    CN = jobs_of("cheat-n")
    w = worker("cheatn", tcn, "--env", "RQAH_TEST_NOOP_SALT=evil", "--job-types", "noop", "--idle-exit", "4"); procs.append(w); w.wait(40)
    cands = [{"id": f"s{i}", "mix": {LIN[i]: "k3", LIN[i + 3]: "k4"}} for i in range(3)]
    c, b3 = req("POST", S, {"name": "r3", "model": "fm", "revision": "1", "base": "k2", "golden": "xg", "kv": "bf16",
                            "max_model_len": 4096, "candidates": cands}, token=SEARCH_TOKEN)
    CE = [it["job"] for it in req("GET", f"{S}/{b3['batch']}", token=SEARCH_TOKEN)[1]["items"]]
    w = worker("cheate", tce, "--gpu-name", "fake32", "--cards", "1", "--vram-gb", "32", "--ram-gb", "64", *GPU, "--radiance", FAKE_RAD,
               "--env", "FAKE_RAD_BIAS=0.01", "--job-types", "eval_mix", "--idle-exit", "4"); procs.append(w); w.wait(60)
    acc_before = {j: jrow(j)["accepted_by"] for j in CN + CE}
    pts_before = {n: db().execute("SELECT COALESCE(SUM(lumens),0) FROM points WHERE worker_id=?", (wid(n),)).fetchone()[0] for n in ("cheatn", "cheate")}
    # the verifier (our own box) re-runs everything drawn; jitter of 1e-4 per row stands in for another GPU
    v = worker("verif", vt, "--gpu-name", "fake32", "--gpus", "0,1", "--vram-gb", "32", "--ram-gb", "96", *GPU, "--radiance", FAKE_RAD,
               "--mkgold27", FAKE_GOLD, "--env", "FAKE_RAD_NOISE=0.0001", "--env", "FAKE_GOLD_TOP=32", "--idle-exit", "16")
    h = worker("honest", th, "--gpu-name", "fake32", "--cards", "1", "--vram-gb", "32", "--ram-gb", "64", *GPU, "--radiance", FAKE_RAD,
               "--job-types", "noop,eval_mix", "--idle-exit", "16")
    procs += [v, h]
    fin = wait(lambda: all(states()[j] == "accepted" and jrow(j)["accepted_by"] not in (wid("cheatn"), wid("cheate")) for j in CN + CE) and
               not db().execute("SELECT 1 FROM jobs WHERE verify_of IS NOT NULL AND state IN ('queued','leased')").fetchone(), 150)
    h.wait(60); v.wait(60)                      # (a verifier may also run trusted-only types: let it go before the golden tests)
    q = {n: db().execute("SELECT quarantined, quarantine_reason FROM workers WHERE id=?", (wid(n),)).fetchone() for n in ("cheatn", "cheate")}
    ok("spot checks: a donor's first job is re-run on a verifier; a salted noop and biased eval rows fail -> both donors quarantined",
       all(acc_before[j] == wid("cheatn") for j in CN) and all(acc_before[j] == wid("cheate") for j in CE) and
       q["cheatn"][0] == 1 and q["cheate"][0] == 1, f"{q['cheatn'][1]} | {q['cheate'][1]}")
    exact_n = all(sha256_file(f"{RES}/{j}/out.bin") == hashlib.sha256(__import__("jobs.noop", fromlist=["x"]).content(1200 + i, 5000)).hexdigest()
                  for i, j in enumerate(CN))
    from jobs import golden_common as G_
    exact_e = True
    for i, j in enumerate(CE):                 # a verifier may redo one too: its rows carry the 1e-4 jitter
        open(f"{W}/exp.rows", "wb").write(expected(E, "1", "k2", cands[i]["mix"])[1])
        exact_e &= G_.compare_rows(f"{W}/exp.rows", f"{RES}/{j}/rows.f32", dict(EM_TOL, row_atol=1e-6, row_rtol=1e-3))[0]
    ok("quarantine: every result of theirs no re-run confirmed was invalidated and redone by someone else (right rows)",
       fin and exact_n and exact_e and any(f.startswith(f"{CE[1]}.invalid-") for f in os.listdir(RES)),
       f"noop exact {exact_n}, eval rows right {exact_e}; redone by {sorted({jrow(j)['accepted_by'] for j in CN + CE})}")
    c, r = req("POST", "/api/v0/lease", {"caps": {"job_types": ["noop"]}}, token=tcn)
    ok("quarantine: the donor's lease requests get nothing", c == 200 and r.get("job") is None and r.get("quarantined") is True)
    vs = {r_["verify_state"]: r_["n"] for r_ in db().execute("SELECT verify_state, COUNT(*) n FROM jobs WHERE verify_state IS NOT NULL GROUP BY verify_state")}
    honest_pass = [r_["id"] for r_ in db().execute("SELECT j.id FROM jobs j JOIN workers w ON w.id=j.accepted_by WHERE j.verify_state='pass' "
                                                    "AND j.type='eval_mix' AND w.name IN ('e16','e32','m2')")]
    nfail = db().execute("SELECT COUNT(*) FROM events WHERE kind='spot_fail'").fetchone()[0]
    ok("spot checks: honest donors' drawn results pass (eval rows within tolerance of the verifier's jittered re-run)",
       nfail == 2 and len(honest_pass) == 3 and vs.get("pass", 0) >= 8, f"{vs}, spot_fail events {nfail}; honest eval passes {honest_pass}")
    pts = {r_["name"]: r_ for r_ in json.loads(admin("points", "--json"))}
    rev = db().execute("SELECT COUNT(*) FROM points WHERE kind='revoke' AND worker_id IN (?,?)", (wid("cheatn"), wid("cheate"))).fetchone()[0]
    bonus = db().execute("SELECT COUNT(*) FROM points WHERE note='quick-return bonus'").fetchone()[0]
    ok("lumens: verified work only (cheaters' grants revoked to 0, the verifier earns none, honest donors earn; quick-return bonus)",
       pts["cheatn"]["lumens"] == 0 and pts["cheate"]["lumens"] == 0 and rev >= 1 and pts_before["cheatn"] > 0 and
       pts["verif"]["lumens"] == 0 and pts["honest"]["lumens"] > 0 and pts["e16"]["lumens"] > 0 and bonus > 0,
       f"before {pts_before}; honest {pts['honest']['lumens']:.2f}, e16 {pts['e16']['lumens']:.2f}; revokes {rev}")
    c, r = req("GET", "/api/v0/admin/summary", token=ADMIN_TOKEN)
    ok("admin summary: lumens, spot-check counts, quarantine flags", c == 200 and r.get("lumens") and r.get("spot") and
       any(w_["quarantined"] for w_ in r.get("workers", [])))


def v02_golden(procs):
    src = f"{W}/v02src"
    docs = [{"prompt": f"public text {i}: " + "lorem ipsum dolor " * (5 + i), "source": f"xcorp/{s}", "score_from": i}
            for i, s in enumerate(["web", "code", "web", "chat", "code"])]
    with open(f"{src}/pub.jsonl", "w") as f: f.writelines(json.dumps(d) + "\n" for d in docs)
    with open(f"{src}/priv.jsonl", "w") as f: f.writelines(json.dumps(dict(d, source="pi-session/2026-11-02")) + "\n" for d in docs[:2])
    rc1, o1 = admin_rc("corpus-add", "pubc", f"{src}/pub.jsonl", "--public", "--chunk-docs", "3")
    rc2, o2 = admin_rc("corpus-add", "privc", f"{src}/priv.jsonl", "--public")
    root = f"{DATA}/models/fm@1"
    open(f"{root}/bf16.rad", "wb").write(b"fake bf16 container" * 64)
    os.makedirs(f"{root}/hf", exist_ok=True)
    json.dump({"model_type": "qwen3_5"}, open(f"{root}/hf/config.json", "w"))
    json.dump({"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}, open(f"{root}/hf/model.safetensors.index.json", "w"))
    open(f"{root}/hf/model-00001-of-00001.safetensors", "wb").write(b"\0" * 4096)
    admin("manifest")
    a1 = admin("enqueue-golden", "--model", "fm", "--rev", "1", "--corpus", "pubc", "--method", "tp2", "--top", "32", "--max-model-len", "4096")
    a2 = admin("enqueue-golden", "--model", "fm", "--rev", "1", "--corpus", "pubc", "--method", "staged", "--chunks", "c0000", "--top", "32",
               "--max-model-len", "4096")
    ok("corpus-add: public corpus cut into chunks; our session logs refused", rc1 == 0 and rc2 != 0 and "private sources" in o2, o2.strip()[-70:])
    GJ = jobs_of("golden-pubc")
    tb = tok("bigbox")
    bb = ("--gpu-name", "fake32", "--gpus", "0,1", "--vram-gb", "32", "--ram-gb", "96", *GPU, "--radiance", FAKE_RAD,
          "--mkgold27", FAKE_GOLD, "--env", "FAKE_GOLD_TOP=32", "--job-types", "golden_record", "--idle-exit", "6")
    w = worker("bigbox", tb, *bb); procs.append(w); w.wait(40)
    ok("golden_record: an untrusted 2-card worker is never leased one", len(GJ) == 3 and all(states()[j] == "queued" for j in GJ)
       and not any(e["kind"] == "lease" for j in GJ for e in events(j)), f"{a1.strip()} / {a2.strip()}")
    admin("trust", str(wid("bigbox")))
    w = worker("bigbox", tb, *bb); procs.append(w)
    wait(lambda: all(states()[j] == "accepted" for j in GJ), 60); w.wait(40)        # then a verifier for the re-run
    v = worker("verif", admin("add-worker", "verif2").split()[3], "--gpu-name", "fake32", "--gpus", "0,1", "--vram-gb", "32",
               "--ram-gb", "96", *GPU, "--radiance", FAKE_RAD, "--mkgold27", FAKE_GOLD, "--env", "FAKE_GOLD_TOP=32", "--idle-exit", "12")
    admin("verifier", str(wid("verif2"))); procs.append(v)
    done = wait(lambda: all(states()[j] == "accepted" for j in GJ) and
                not db().execute("SELECT 1 FROM jobs WHERE verify_of IS NOT NULL AND state IN ('queued','leased')").fetchone(), 90)
    v.wait(60)
    P = {j: json.loads(jrow(j)["params"]) for j in GJ}
    tp2 = [j for j in GJ if P[j]["method"] == "tp2"]; stg = [j for j in GJ if P[j]["method"] == "staged"][0]
    ok("golden_record: trusted -> TP2 (gang, cards 0,1) and staged (1 card) recorded and accepted (layout, corpus hash, row checks)",
       done and all(jrow(j)["accepted_by"] == wid("bigbox") for j in GJ) and all(result(j)["rows"] > 0 for j in GJ) and
       all(json.load(open(f"{RES}/{j}/job.json"))["devices"] == [0, 1] for j in tp2) and len(json.load(open(f"{RES}/{stg}/job.json"))["devices"]) == 1)
    from jobs import golden_common as G_
    c0 = [j for j in tp2 if P[j]["chunk"] == "c0000"][0]
    okc, det = G_.compare_goldens(f"{RES}/{c0}", f"{RES}/{stg}", {"nll_mae": 0.02, "top1_agree": 0.98})
    vs = [jrow(j)["verify_state"] for j in GJ]
    ok("golden_record: the two recorders agree on one chunk; the trusted donor's first golden passed its spot check",
       okc and "pass" in vs, f"{det}; verify states {vs}")
    # tampering a recorded golden: what accept() catches
    import shutil as sh_
    t = f"{W}/gtamper"; bad = []
    for case in ("nll", "order", "trunc", "corpus"):
        if os.path.exists(t): sh_.rmtree(t)
        sh_.copytree(f"{RES}/{c0}", t)
        if case == "nll":
            b = bytearray(open(f"{t}/rows.f32", "rb").read())
            for i in range(0, len(b), 12): b[i:i + 4] = __import__("struct").pack("<f", 0.123)     # every row's nll
            open(f"{t}/rows.f32", "wb").write(b)
        elif case == "order":
            b = bytearray(open(f"{t}/top.bin", "rb").read()); T = 32
            for r in range(len(b) // (8 * T)):
                o = r * 8 * T; b[o:o + 8], b[o + 8:o + 16] = b[o + 8:o + 16], b[o:o + 8]
            open(f"{t}/top.bin", "wb").write(b)
        elif case == "trunc":
            b = open(f"{t}/top.bin", "rb").read(); open(f"{t}/top.bin", "wb").write(b[:-8])
        files = {f: f"{t}/{f}" for f in ("kld.json", "tokens.i32", "top.bin", "rows.f32", "gen.u8") if os.path.exists(f"{t}/{f}")}
        corpus = f"{DATA}/corpora/pubc/{'c0001' if case == 'corpus' else 'c0000'}.jsonl"
        try: G_.check_golden(files, top=32, corpus=corpus); bad.append(case)
        except ValueError: pass
    ok("golden_record accept: a wrong nll column, an unsorted head, a truncated top.bin, another chunk -> all refused", not bad, str(bad))


def v02_units(E):
    """in-process: the spot-check draw rate, and what eval_mix's accept() refuses"""
    sys.path.insert(0, HERE)
    from server.store import Store
    from server import verify as Vf
    from jobs import eval_mix as EM
    S = Store(":memory:")
    with S.tx() as d:
        d.execute("INSERT INTO workers(id, name, token_hash) VALUES(1, 'donor', 'a'), (2, 'ours', 'b')")
        d.execute("UPDATE workers SET verifier=1 WHERE id=2")
        for i in range(6000):
            d.execute("INSERT INTO jobs(id, type, version, params, inputs, requires, input_bytes, state, accepted_by, accepted_at) "
                      "VALUES(?, 'noop', 2, ?, '[]', '{}', 0, 'accepted', ?, 0)", (i + 1, json.dumps({"seconds": 0, "size": 1, "seed": i}), 1 if i < 5000 else 2))
    pol = Vf.Policy(0.04, 1)
    for i in range(6000): Vf.after_accept(S, "/nonexistent", i + 1, pol)
    n1 = S.q("SELECT COUNT(*) FROM jobs WHERE verify_state='pending' AND accepted_by=1")[0][0]
    n2 = S.q("SELECT COUNT(*) FROM jobs WHERE verify_state='pending' AND accepted_by=2")[0][0]
    first = S.q("SELECT verify_state FROM jobs WHERE id=1")[0][0]
    ok("spot-check draws: 3-5% of a donor's jobs (4% + its first), none of a verifier's", 0.03 <= n1 / 5000 <= 0.05 and n2 == 0 and first == "pending",
       f"{n1} of 5000 ({n1 / 50:.2f}%), verifier {n2}")
    j = E["J"]["c1"]; p = json.loads(jrow(j)["params"]); t = f"{W}/etamper"
    dp = lambda rel: os.path.join(DATA, rel)
    refused = []
    for case in ("ok", "nll_p", "neg", "mean", "short"):
        if os.path.exists(t): __import__("shutil").rmtree(t)
        __import__("shutil").copytree(f"{RES}/{j}", t)
        import array as ar
        a = ar.array("f"); a.frombytes(open(f"{t}/rows.f32", "rb").read())
        if case == "nll_p": a[2 + 4 * 7] += 0.5
        if case == "neg": a[4 * 3] = -0.5
        if case == "mean": a[0] += 0.3
        if case == "short": a = a[:-4]
        open(f"{t}/rows.f32", "wb").write(a.tobytes())
        try: EM.accept(p, {f: f"{t}/{f}" for f in ("rows.f32", "kld.json", "job.json")}, {}, dp)
        except ValueError: refused.append(case)
    ok("eval_mix accept: refuses a changed reference NLL, a negative KL, rows that disagree with the report, a short file; takes the real one",
       refused == ["nll_p", "neg", "mean", "short"], str(refused))


# ======================================================================== w25: donor CLI + public status page
import re as _re
LEAK = {"token": _re.compile(r"\b(?:rq|ri)[A-Za-z0-9_-]{20,}"), "IPv4": _re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "abs path": _re.compile(r"(?<![\w.<])/(?:root|data|srv|home|workspace|tmp|var|etc|mnt|opt)\b"),
        "url": _re.compile(r"(?i)\bhttps?://")}


def tokfile(token):
    p = f"{W}/cli-{hashlib.sha256(token.encode()).hexdigest()[:8]}.tok"
    if not os.path.exists(p): open(p, "w").write(token + "\n")
    return p


def cli(token, *args, env=None, server=URL, state=None):
    """the donor CLI as a donor runs it: server and token file from the environment -> (exit code, JSON or None, output)"""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("RQAH_", "RADQUANT_"))}
    e["RQAH_SERVER"] = server
    if token is not None: e["RQAH_TOKEN_FILE"] = tokfile(token)
    if state: e["RQAH_STATE"] = state
    e.update(env or {})
    r = subprocess.run([PY, "-I", f"{HERE}/client/rqah.py", *args], capture_output=True, text=True, env=e, timeout=60)
    try: j = json.loads(r.stdout) if "--json" in args else None
    except ValueError: j = None
    return r.returncode, j, r.stdout + r.stderr


def leaks(text):
    return {k: rx.findall(text)[:3] for k, rx in LEAK.items() if rx.search(text)}


def page():
    c, b = req("GET", "/")
    return c, (b.decode() if isinstance(b, bytes) else "")


def run_w25(procs, srv):
    th, tcn, t16, t32 = TOKENS["honest"], TOKENS["cheatn"], TOKENS["e16"], TOKENS["e32"]
    pts = {r["name"]: r for r in json.loads(admin("points", "--json"))}
    hid = wid("honest")
    # status
    rc, st, out = cli(th, "status", "--json", state=f"{W}/honest")
    rc2, st2, _ = cli(th, "--json", "status")
    rch, _, human = cli(th, "status")
    ok("cli status --json: own worker only, lumens = the server's ledger, queue depth, could-run count, local jobs",
       rc == 0 and st["ok"] and [w["name"] for w in st["workers"]] == ["honest"] and abs(st["lumens"] - pts["honest"]["lumens"]) < 1e-6
       and st["verified_jobs"] == pts["honest"]["jobs"] > 0 and isinstance(st["queue"]["by_type"], dict) and
       isinstance(st["workers"][0]["could_run"], int) and st["local"] == [] and st["donor"]["leaderboard"] is False and
       rc2 == 0 and st2["worker"] == st["worker"] and rch == 0 and "lumens" in human,
       out[-300:] if rc or not st else f"{st['lumens']:.2f} lumens, {st['verified_jobs']} jobs")
    # jobs
    from server.donor import VERIFICATION
    rc, jb, out = cli(th, "jobs", "--json", "--limit", "500")
    vs = collections.Counter(j["verification"] for j in (jb or {}).get("jobs", []))
    ok("cli jobs --json: the worker's leases with verification states; lumens of verified rows = its total",
       rc == 0 and jb["jobs"] and all(j["worker"] == hid for j in jb["jobs"]) and set(vs) <= set(VERIFICATION) and vs["verified"] > 0
       and abs(sum(j["lumens"] for j in jb["jobs"] if j["verification"] == "verified") - pts["honest"]["lumens"]) < 1e-6, dict(vs))
    rc, iso, _ = cli(th, "jobs", "--json", "--worker", str(wid("e16")))
    rch, _, human = cli(th, "jobs")
    ok("cli jobs --worker <someone else's>: nothing (donor isolation); human table", rc == 0 and iso["jobs"] == [] and iso.get("error")
       and rch == 0 and "verified" in human)
    # the quarantined donor's view
    rc, cs, _ = cli(tcn, "status", "--json"); rc2, cj, _ = cli(tcn, "jobs", "--json"); rc3, cp, _ = cli(tcn, "points", "--json")
    ok("cli for a quarantined donor: flagged with a scrubbed reason, results 'voided', ledger shows the revoke, 0 lumens",
       rc == rc2 == rc3 == 0 and cs["workers"][0]["quarantined"] and "spot check" in cs["workers"][0]["quarantine_reason"] and
       "voided" in {j["verification"] for j in cj["jobs"]} and any(l["kind"] == "revoke" for l in cp["ledger"]) and cp["lumens"] == 0
       and not leaks(json.dumps(cs)), (cs or {}).get("workers", [{}])[0].get("quarantine_reason"))
    # points
    rc, pp, _ = cli(th, "points", "--json")
    rch, _, human = cli(th, "points")
    ok("cli points --json: total, per worker and type, ledger of grants; human form too",
       rc == 0 and abs(pp["lumens"] - pts["honest"]["lumens"]) < 1e-6 and pp["by_worker"][0]["id"] == hid and pp["by_type"] and
       any(l["kind"] == "grant" for l in pp["ledger"]) and rch == 0 and "verified jobs" in human)
    # logs
    rc, lg, out = cli(th, "logs", "--json", "--limit", "500")
    kinds = {e["kind"] for e in (lg or {}).get("events", [])}
    ok("cli logs --json: the server's events of its own workers only (lease, accept, points), scrubbed",
       rc == 0 and {"lease", "accept", "points"} <= kinds and all(e["worker"] in (hid, None) for e in lg["events"]) and
       not leaks(out), f"{sorted(kinds)} {leaks(out)}")
    rc, ll, _ = cli(th, "logs", "--local", "--json", state=f"{W}/honest")
    ok("cli logs --local: this machine's finished-job records", rc == 0 and ll["local"] and "run_s" in ll["local"][-1])
    # config
    rc, cf, out = cli(th, "config", "--json", env={"RQAH_RAM_GB": "64", "RQAH_WINDOWS": "22:00-08:00"})
    ok("cli config --json: server, token fingerprint (never the token), env caps, profile, this worker's declared caps",
       rc == 0 and cf["local"]["token"]["fingerprint"] == hashlib.sha256(th.encode()).hexdigest()[:12] and th not in out and
       cf["local"]["env"].get("RQAH_RAM_GB") == "64" and cf["profile"]["leaderboard"] is False and cf["worker"]["id"] == hid
       and cf["worker"]["gpu_name"] == "fake32", out[-300:])
    bad = [cli(th, "config", "set", "--display-name", n, "--json") for n in ("10.1.2.3", "<b>x</b>", "me@example.org", "")]
    ok("cli config set: an address, markup, an e-mail, an empty name -> refused (HTTP 400, exit 1)",
       all(r[0] == 1 and r[1] and r[1]["http"] == 400 for r in bad), str([r[1] for r in bad])[:300])
    c0, p0 = page()
    rc, cs, _ = cli(th, "config", "set", "--display-name", "Honest Donor", "--leaderboard", "on", "--json")
    shown = wait(lambda: "Honest Donor" in page()[1], 15)
    rc2, st, _ = cli(th, "status", "--json")
    ok("leaderboard opt-in: not on the page before; config set -> listed within the rebuild interval, rank #1 of 1",
       c0 == 200 and "Honest Donor" not in p0 and rc == 0 and shown and st["donor"]["rank"] == 1 and st["donor"]["on_board"] == 1,
       str((st or {}).get("donor")))
    # the page
    r = urllib.request.urlopen(urllib.request.Request(URL + "/"), timeout=30); hdr = r.headers; body = r.read().decode()
    rh = urllib.request.urlopen(urllib.request.Request(URL + "/", method="HEAD"), timeout=30)
    cpost = req("POST", "/", {"x": 1})[0]; cput = req("PUT", "/index.html", b"x")[0]
    ok("status page at /: text/html, CSP default-src 'none', nosniff, no-referrer; HEAD works; POST / PUT do nothing",
       hdr["Content-Type"].startswith("text/html") and "default-src 'none'" in hdr["Content-Security-Policy"] and
       hdr["X-Content-Type-Options"] == "nosniff" and hdr["Referrer-Policy"] == "no-referrer" and rh.status == 200 and
       rh.read() == b"" and cpost in (401, 404) and cput in (401, 404), f"POST {cpost} PUT {cput}")
    names = [n for n in list(TOKENS) + ["client", "small", "rng", "verif2", "pa", "pb"] if _re.search(rf"\b{_re.escape(n)}\b", body)]
    secrets_ = [t for t in list(TOKENS.values()) + [ADMIN_TOKEN, SEARCH_TOKEN, "127.0.0.1", W, str(PORT)] if t in body]
    ok("status page: no token / address / path / URL patterns, none of this run's tokens, worker names, port or work dir",
       not leaks(body) and not names and not secrets_ and "<script" not in body.lower() and "<form" not in body.lower(),
       f"{leaks(body)} names {names} secrets {len(secrets_)}")
    import xml.etree.ElementTree as ET_
    try: ET_.fromstring(body.split("\n", 1)[1]); wf = True
    except ET_.ParseError: wf = False
    nev = db().execute("SELECT COUNT(*) FROM jobs WHERE type='eval_mix' AND state='accepted' AND verify_of IS NULL").fetchone()[0]
    ok("status page: well-formed, search evals = accepted eval_mix jobs, the noop row, recent work, last update",
       wf and f"search {nev:,} evals" in body and "test job" in body and "Recent work" in body and "Last update" in body, f"{nev} evals")
    same = wait(lambda: os.path.exists(f"{STATUS_DIR}/index.html") and open(f"{STATUS_DIR}/index.html").read() == page()[1], 10)
    ok("--status-dir: the same page written for a static host (atomic, 0644, nothing else in the directory)",
       same and oct(os.stat(f"{STATUS_DIR}/index.html").st_mode & 0o777) == "0o644" and os.listdir(STATUS_DIR) == ["index.html"],
       str(os.listdir(STATUS_DIR)) if os.path.exists(STATUS_DIR) else "no dir")
    rc_ = admin_rc("status-page", f"{W}/admin-page.html")
    ok("admin.py status-page OUT and donors", rc_[0] == 0 and "Honest Donor" in open(f"{W}/admin-page.html").read() and
       "Honest Donor" in admin("donors"), rc_[1][-200:])
    # linking two of one donor's workers, personal invites, opt-in at registration
    rc, lk, _ = cli(t16, "link", "--other-token-file", tokfile(t32), "--json")
    rc2, s16, _ = cli(t16, "status", "--json"); rc3, s32, _ = cli(t32, "status", "--json")
    rcb, bl, _ = cli(t16, "link", "--other-token-file", tokfile(t16), "--json")
    rcj, jl, _ = cli(t16, "link", "--other-token-file", tokfile("rq" + "x" * 43), "--json")
    ok("cli link: two workers of one donor -> status from either shows both, lumens summed; a token not of this server -> 403",
       rc == 0 and lk["moved"] == 1 and rc2 == rc3 == 0 and sorted(w["name"] for w in s16["workers"]) == ["e16", "e32"] and
       s16["donor"]["id"] == s32["donor"]["id"] and abs(s16["lumens"] - pts["e16"]["lumens"] - pts["e32"]["lumens"]) < 1e-6 and
       rcb == 0 and bl["moved"] == 0 and rcj == 1 and jl["http"] == 403, f"{lk} / {jl}")
    inv = admin("invite", "--uses", "2", "--personal").strip()
    reg = [req("POST", "/api/v0/register", {"invite": inv, "name": n})[1] for n in ("pa", "pb")]
    rc, pa, _ = cli(reg[0]["token"], "status", "--json")
    r3 = req("POST", "/api/v0/register", {"invite": admin("invite").strip(), "name": "boardie", "leaderboard": True})[1]
    rc2, bd, _ = cli(r3["token"], "status", "--json")
    ok("personal invite: both registrations are one donor; register with leaderboard=true -> opted in under its name",
       rc == 0 and sorted(w["name"] for w in pa["workers"]) == ["pa", "pb"] and rc2 == 0 and bd["donor"]["leaderboard"] is True
       and bd["donor"]["display_name"] == "boardie")
    # errors and exit codes
    rc1, j1, _ = cli("rq" + "y" * 43, "status", "--json")
    rc2, j2, _ = cli(th, "status", "--json", server="http://127.0.0.1:1")
    rc3, j3, _ = cli(None, "status", "--json")
    c401 = req("GET", "/api/v0/me")[0]
    ok("cli exit codes: bad token 3, unreachable 4, no token 2 (JSON errors); /api/v0/me without a token -> 401",
       (rc1, rc2, rc3) == (3, 4, 2) and j1["ok"] is False and j2["ok"] is False and j3["ok"] is False and c401 == 401,
       f"{(rc1, rc2, rc3)}")


if __name__ == "__main__":
    main()
