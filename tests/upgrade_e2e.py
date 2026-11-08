#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""upgrade_e2e.py [WORKDIR] -- the whole update path with real processes and three releases, no GPU, ~4 min.

Trees: OLD = this repository at UPGRADE_OLD_REF (default f179831, the 0.4.0 release; `git archive`, so CI checks out
with fetch-depth 0), NEW = this working tree, NEXT = a copy of it posing as the release after: VERSION one minor up,
noop bumped to version 3 (ACCEPTS 2) and a new job type noop_next. Servers, workers, admin.py and the CLI run as
subprocesses of their own tree against one queue database on loopback (lease 8 s, heartbeat 1 s).

  A  the 0.4.0 server and two 0.4.0 workers fill the database (accepted, failed, running jobs; a full input cache)
  B  server upgrade 0.4.0 -> NEW, down 2.5 leases while an old worker runs a job: the database is migrated (checked on
     a copy: every row kept, a backup, the 0.4.0 admin and server still open it), the running job finishes on its
     lease (no expiry, one lease); old workers keep working with the new server; the 0.4.0 CLI still works; the new
     CLI and the status page show the update
  C  worker updates 0.4.0 -> NEW on the same state: one drained (SIGTERM, then the new code), one killed mid-job and
     restarted with the new code; same tokens (no new registration), the full cache kept (the resumed job downloads
     nothing), eviction still works, the release recorded
  D  --worker-min raised while a 0.4.0 worker runs a job: it finishes it, then gets 426 (logged with the way to
     update), stays up, leases nothing; a NEW worker takes the queue
  E  server NEW -> NEXT while NEW workers run noop v2 jobs: the one left running is accepted (nothing requeued); one
     killed and restarted as a NEXT worker hands its v2 job back (abandoned, not counted); old and new workers never
     get noop v3 or noop_next, the NEXT worker gets both and never v2; retire noop 2 is refused while v2 runs or is
     queued, then --migrate re-stamps leftovers at v3 for the NEXT worker; admin.py versions"""
import glob, hashlib, json, os, re, shutil, signal, sqlite3, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rqah-upgrade-e2e")
PORT = int(os.environ.get("UPGRADE_PORT", "18781"))
URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable
OLD_REF = os.environ.get("UPGRADE_OLD_REF", "f179831")
LEASE = 8
RESULTS, PROCS = [], []
DB = f"{W}/q/queue.sqlite"
DATA = f"{W}/q/data"


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def vt(v): return tuple(int(x) for x in v.split("."))


def wait(cond, timeout, step=0.25):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(step)
    return cond()


# ---------------------------------------------------------------- trees
def trees():
    old, nxt = f"{W}/old", f"{W}/next"
    for d in (old, nxt): shutil.rmtree(d, ignore_errors=True)
    os.makedirs(old)
    a = subprocess.run(["git", "-C", HERE, "archive", "--format=tar", OLD_REF], capture_output=True)
    if a.returncode:
        sys.exit(f"upgrade_e2e: no {OLD_REF} in this checkout ({a.stderr.decode().strip()[:200]}): it needs the git history "
                 "(CI: actions/checkout with fetch-depth: 0), or UPGRADE_OLD_REF=<a 0.4.x commit>")
    subprocess.run(["tar", "-x", "-C", old], input=a.stdout, check=True)
    shutil.copytree(HERE, nxt, ignore=shutil.ignore_patterns(".git", "__pycache__", "tests", "research-kit", "docker"))
    new_v = open(f"{HERE}/VERSION").read().strip()
    next_v = f"{vt(new_v)[0]}.{vt(new_v)[1] + 1}.0"
    open(f"{nxt}/VERSION", "w").write(next_v + "\n")
    s = open(f"{nxt}/jobs/noop.py").read()
    s2 = s.replace('NAME, VERSION, IMPLEMENTED = "noop", 2, True', 'NAME, VERSION, IMPLEMENTED = "noop", 3, True\nACCEPTS = (2,)')
    assert s2 != s, "jobs/noop.py: the VERSION line moved; update upgrade_e2e.trees()"
    open(f"{nxt}/jobs/noop.py", "w").write(s2)
    open(f"{nxt}/jobs/noop_next.py", "w").write(s.replace('NAME, VERSION, IMPLEMENTED = "noop", 2, True',
                                                          'NAME, VERSION, IMPLEMENTED = "noop_next", 1, True'))
    s = open(f"{nxt}/jobs/__init__.py").read()
    s2 = s.replace("from . import noop, quant_layer", "from . import noop, noop_next, quant_layer").replace(
        "REGISTRY = {m.NAME: m for m in (noop, quant_layer", "REGISTRY = {m.NAME: m for m in (noop, noop_next, quant_layer")
    assert s2.count("noop_next") == 2, "jobs/__init__.py changed; update upgrade_e2e.trees()"
    open(f"{nxt}/jobs/__init__.py", "w").write(s2)
    old_v = open(f"{old}/VERSION").read().strip()
    return {"old": old, "new": HERE, "next": nxt}, old_v, new_v, next_v


# ---------------------------------------------------------------- processes
class Server:
    def __init__(self, tree, *extra, db=DB, port=PORT, tag="server"):
        self.tree = tree
        lf = open(f"{W}/{tag}.log", "a"); lf.write(f"\n==== {tree} {' '.join(extra)} {time.ctime()}\n"); lf.flush()
        self.p = subprocess.Popen([PY, "-I", f"{tree}/server/server.py", "--db", db, "--data", DATA, "--spool", f"{W}/q/spool",
                                   "--results", f"{W}/q/results", "--port", str(port), "--lease-s", str(LEASE), "--heartbeat-s", "1",
                                   "--retry-s", "0.5", "--spot-rate", "0", "--spot-first", "0", "--status-interval", "1", *extra],
                                  stdout=lf, stderr=subprocess.STDOUT)
        PROCS.append(self.p)
        url = f"http://127.0.0.1:{port}"
        assert wait(lambda: self.p.poll() is None and get(url + "/healthz")[0] == 200, 30), f"server {tree} did not start"

    def stop(self, sig=signal.SIGTERM):
        self.p.send_signal(sig); return self.p.wait(30)


def get(url, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=10)
        return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()
    except (urllib.error.URLError, OSError) as e: return None, repr(e).encode()


def admin(tree, *args, check=True):
    r = subprocess.run([PY, "-I", f"{tree}/server/admin.py", "--db", DB, "--data", DATA, "--results", f"{W}/q/results", *args],
                       capture_output=True, text=True, timeout=120)
    if check and r.returncode: print(r.stdout, r.stderr)
    return r


def worker(tree, name, *extra, types="noop"):
    lf = open(f"{W}/{name}.log", "a"); lf.write(f"\n==== {tree} {time.ctime()}\n"); lf.flush()
    p = subprocess.Popen([PY, "-I", f"{tree}/worker/worker.py", "--server", URL, "--token-file", f"{W}/{name}/token",
                          "--invite", INVITE[0], "--state", f"{W}/{name}", "--name", name, "--poll", "0.3", "--job-types", types,
                          *extra], stdout=lf, stderr=subprocess.STDOUT)
    PROCS.append(p); return p


INVITE = [None]


def con():
    c = sqlite3.connect(DB, timeout=30); c.row_factory = sqlite3.Row; return c


def q(sql, a=()):
    c = con()
    try: return c.execute(sql, a).fetchall()
    finally: c.close()


def tagged(tag): return [r[0] for r in q("SELECT id FROM jobs WHERE level=? ORDER BY id", (tag,))]
def state(j): return q("SELECT state FROM jobs WHERE id=?", (j,))[0][0]
def kinds(j): return [r[0] for r in q("SELECT kind FROM events WHERE job_id=? ORDER BY id", (j,))]
def wid(name): return q("SELECT id FROM workers WHERE name=?", (name,))[0][0]
def all_in(ids, st): return all(state(j) == st for j in ids)
def holder(j): return (q("SELECT lease_owner FROM jobs WHERE id=?", (j,)) or [[None]])[0][0]


def enqueue(tree, n, tag, seconds=0.0, seed=0, inputs=None):
    a = ["enqueue-noop", str(n), "--seconds", str(seconds), "--size", "1000", "--seed", str(seed), "--tag", tag]
    if inputs: a += ["--inputs", inputs]
    admin(tree, *a); return tagged(tag)


def counts(path):
    c = sqlite3.connect(path)
    try:
        return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for (t,) in
                c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()}
    finally: c.close()


def rqah(tree, token, *args):
    return subprocess.run([PY, "-I", f"{tree}/client/rqah.py", "--server", URL, "--token", token, *args], capture_output=True,
                          text=True, timeout=60, env=dict(os.environ, RQAH_IMAGE_ID=f"ghcr.io/x/rad-quant-at-home-worker:{OLD_V}-cu128+abc"))


def stop_all():
    for p in PROCS:
        if p.poll() is None:
            p.kill()
            try: p.wait(10)
            except subprocess.TimeoutExpired: pass


# ---------------------------------------------------------------- the path
def main():
    global OLD_V
    shutil.rmtree(W, ignore_errors=True); os.makedirs(f"{W}/q/data/test")
    T, OLD_V, NEW_V, NEXT_V = trees()
    print(f"trees: old {OLD_V} ({OLD_REF}), new {NEW_V}, next {NEXT_V}", flush=True)
    assert vt(OLD_V) < vt(NEW_V) < vt(NEXT_V), "VERSION must be above the old release"
    for i in range(5): open(f"{DATA}/test/in{i}.bin", "wb").write(hashlib.sha256(str(i).encode()).digest() * 9375)   # 300 kB
    admin(T["old"], "manifest")

    # ---- A: the 0.4.0 fleet
    s = Server(T["old"])
    INVITE[0] = admin(T["old"], "invite", "--uses", "50").stdout.strip().split()[-1]
    done = enqueue(T["old"], 4, "a-done", 0.2, 100)
    w1 = worker(T["old"], "w1")
    assert wait(lambda: all_in(done, "accepted"), 60), "0.4.0 fleet: jobs not accepted"
    fail = enqueue(T["old"], 1, "a-fail", 0, 150); admin(T["old"], "cancel", *map(str, fail))
    longj = enqueue(T["old"], 1, "a-long", 40, 200)[0]
    assert wait(lambda: state(longj) == "leased", 20)
    ins = [enqueue(T["old"], 1, f"a-in{i}", 0, 300 + i, f"test/in{i}.bin")[0] for i in range(3)]   # 900 kB through a 700 kB cache
    w2 = worker(T["old"], "w2", "--cache-gb", "0.0007")
    assert wait(lambda: all_in(ins, "accepted"), 60), "inputs jobs"
    blobs = lambda: sorted(os.listdir(f"{W}/w2/cache/blobs"))
    full = blobs()
    ok("A: the 0.4.0 server and workers ran jobs; w2's cache is at its cap", len(full) == 2 and
       sum(os.path.getsize(f"{W}/w2/cache/blobs/{b}") for b in full) == 600000 and state(longj) == "leased", str(full))

    # ---- B: server 0.4.0 -> NEW, down 2.5 leases while w1 runs longj
    time.sleep(2)
    t_down = time.time(); s.stop()
    before = counts(DB)
    shutil.copy(DB, f"{W}/copy.sqlite")
    for x in ("-wal", "-shm"):
        if os.path.exists(DB + x): shutil.copy(DB + x, f"{W}/copy.sqlite{x}")
    m = subprocess.run([PY, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); from server.store import Store, SCHEMA_VERSION; "
                        "S = Store(sys.argv[2]); print(S.migrated, SCHEMA_VERSION)", HERE, f"{W}/copy.sqlite"],
                       capture_output=True, text=True)
    after = counts(f"{W}/copy.sqlite")
    cols = {r[1] for r in sqlite3.connect(f"{W}/copy.sqlite").execute("PRAGMA table_info(workers)")}
    ok("B: migration on a copy of the populated 0.4.0 database: every row kept, version column, a backup",
       m.returncode == 0 and all(after.get(t) == n for t, n in before.items()) and "version" in cols
       and os.path.exists(f"{W}/copy.sqlite.schema0.bak") and counts(f"{W}/copy.sqlite.schema0.bak") == before,
       f"{m.stdout.strip()} {m.stderr.strip()[-200:]} {[(t, before[t], after.get(t)) for t in before if after.get(t) != before[t]]}")
    r = subprocess.run([PY, "-I", f"{T['old']}/server/admin.py", "--db", f"{W}/copy.sqlite", "list"], capture_output=True, text=True)
    s_old = Server(T["old"], db=f"{W}/copy.sqlite", port=PORT + 1, tag="server-rollback")
    st, _ = get(f"http://127.0.0.1:{PORT + 1}/api/v0/status")
    s_old.stop()
    ok("B: the 0.4.0 admin and server still open the migrated database (one-release rollback)",
       r.returncode == 0 and f"{before['jobs']} jobs" in r.stdout and st == 401, f"{r.stdout[-60:]!r} {st}")
    time.sleep(max(0.0, 2.5 * LEASE - (time.time() - t_down)))
    s = Server(T["new"])
    down = time.time() - t_down
    assert wait(lambda: state(longj) == "accepted", 60), f"the long job: {state(longj)}"
    k = kinds(longj)
    ok(f"B: down {down:.0f} s (lease {LEASE} s) while w1 ran a job: it finished on its lease (no expiry, one lease)",
       k.count("lease") == 1 and "expire" not in k and k[-1] in ("accept", "points") and q(
           "SELECT detail FROM events WHERE kind='server_start'"), str(k))
    more = enqueue(T["new"], 4, "b-more", 0.2, 400)
    ok("B: the 0.4.0 workers keep working with the new server", wait(lambda: all_in(more, "accepted"), 60)
       and {q("SELECT accepted_by FROM jobs WHERE id=?", (j,))[0][0] for j in more} <= {wid("w1"), wid("w2")})
    ok("B: the server recorded them as 0.4.0 (from their user agent)", {r[0] for r in q("SELECT version FROM workers")} == {f"{vt(OLD_V)[0]}.{vt(OLD_V)[1]}.0"})
    tok1 = open(f"{W}/w1/token").read().strip()
    r_old = rqah(T["old"], tok1, "status"); r_new = rqah(T["new"], tok1, "status")
    ok("B: the 0.4.0 CLI still works against the new server", r_old.returncode == 0 and "lumens" in r_old.stdout, r_old.stderr[-200:])
    ok("B: the new CLI shows the update and the image", r_new.returncode == 0 and f"update available: {OLD_V} -> {NEW_V}" in r_new.stdout
       and f"docker pull ghcr.io/x/rad-quant-at-home-worker:{NEW_V}-cu128" in r_new.stdout, r_new.stdout[-500:])
    time.sleep(1.5)
    page = get(URL + "/")[1].decode()
    ok("B: the status page shows the share of workers on old releases", f"worker release {NEW_V} (current): 0% of the 2 workers" in page
       and f"{OLD_V}: 100%" in page, re.search(r"worker release[^<]*", page).group(0) if "worker release" in page else "none")

    # ---- C: worker updates on the same state: w1 drained, w2 killed mid-job; the cache is full
    nw = q("SELECT COUNT(*) FROM workers")[0][0]
    w1.send_signal(signal.SIGTERM); rc1 = w1.wait(30)
    cj = enqueue(T["new"], 1, "c-cached", 15, 500, "test/in2.bin")[0]
    assert wait(lambda: state(cj) == "leased", 20)
    time.sleep(3); w2.kill(); w2.wait()
    w1 = worker(T["new"], "w1"); w2 = worker(T["new"], "w2", "--cache-gb", "0.0007")
    assert wait(lambda: state(cj) == "accepted", 60), f"c-cached {state(cj)}"
    rec = [json.loads(l) for l in open(f"{W}/w2/worker.jsonl") if l.strip()]
    rj = [x for x in rec if x.get("job") == cj]
    ok("C: drained w1 exited 0; killed w2 restarted with the new code resumed its lease (one lease, nothing downloaded)",
       rc1 == 0 and kinds(cj).count("lease") == 1 and "expire" not in kinds(cj) and rj and rj[0]["resumed"] and rj[0]["down_bytes"] == 0,
       f"{rc1} {kinds(cj)} {rj}")
    ok("C: same tokens (no new registration), the full cache kept, state.json written",
       q("SELECT COUNT(*) FROM workers")[0][0] == nw and blobs() == full and json.load(open(f"{W}/w1/state.json"))["version"] == NEW_V
       and "state from worker before 0.5" in open(f"{W}/w1.log").read())
    w1.send_signal(signal.SIGTERM); w1.wait(30)                    # so that w2 (the full cache) takes the next one
    ej = enqueue(T["new"], 1, "c-evict", 0, 510, "test/in3.bin")
    assert wait(lambda: all_in(ej, "accepted"), 60)
    ok("C: a new input with the cache full evicts and runs", len(blobs()) == 2 and blobs() != full
       and sum(os.path.getsize(f"{W}/w2/cache/blobs/{b}") for b in blobs()) <= 700000)
    ok("C: the server records the new release", {r[0] for r in q("SELECT version FROM workers WHERE name IN ('w1','w2')")} == {NEW_V})

    # ---- D: the minimum goes up while a 0.4.0 worker runs a job
    w2.send_signal(signal.SIGTERM); w2.wait(30)
    w3 = worker(T["old"], "w3")
    dj = enqueue(T["new"], 1, "d-long", 12, 600)[0]
    assert wait(lambda: state(dj) == "leased" and holder(dj) == wid("w3"), 20)
    s.stop(); s = Server(T["new"], "--worker-min", NEW_V)
    assert wait(lambda: state(dj) == "accepted", 60), f"d-long {state(dj)}"
    dq = enqueue(T["new"], 2, "d-queued", 0, 610)
    time.sleep(6)
    log3 = open(f"{W}/w3.log").read()
    ok("D: min raised mid-job: the 0.4.0 worker finished it, then got 426 with the way to update, and stays up idle",
       all_in(dq, "queued") and w3.poll() is None and "426" in log3 and "rqah update" in log3, log3[-300:])
    w1 = worker(T["new"], "w1")
    ok("D: a new worker takes the queue", wait(lambda: all_in(dq, "accepted"), 30))
    w3.send_signal(signal.SIGTERM); w3.wait(30)

    # ---- E: server NEW -> NEXT (noop 3, noop_next) while NEW workers run noop v2 jobs
    w2 = worker(T["new"], "w2")
    l1 = enqueue(T["new"], 1, "e-l1", 20, 700)[0]
    assert wait(lambda: state(l1) == "leased", 20)
    l2 = enqueue(T["new"], 1, "e-l2", 12, 710)[0]
    assert wait(lambda: state(l2) == "leased", 20)
    s.stop(); s = Server(T["next"], "--worker-min", OLD_V)
    v3 = enqueue(T["next"], 3, "e-v3", 0.2, 720)
    nn = subprocess.run([PY, "-I", "-c", "import sys, json; sys.path.insert(0, sys.argv[1]); from server.store import Store; import jobs; "
                         "m = jobs.get('noop_next'); S = Store(sys.argv[2]); ids = []\n"
                         "with S.tx() as db:\n"
                         " for i in range(3):\n"
                         "  p = {'seconds': 0.2, 'size': 1000, 'seed': 730 + i, 'tag': 'e-next'}\n"
                         "  ids.append(S.enqueue(db, 'noop_next', m.VERSION, p, [], jobs.requirements(m, p), 0, level='e-next'))\n"
                         "print(json.dumps(ids))", T["next"], DB], capture_output=True, text=True)
    nxt = json.loads(nn.stdout or "[]")
    v2q = enqueue(T["new"], 2, "e-v2", 0.2, 740)                    # v2 jobs still queued (enqueued by the NEW admin)
    r = admin(T["next"], "retire", "noop", "2", check=False)
    ok("E: retire noop 2 refused while v2 jobs run", r.returncode != 0 and "running" in r.stderr, r.stderr[-160:])
    holders = {holder(l1), holder(l2)}
    victim = "w1" if holder(l1) == wid("w1") else "w2"
    vp = w1 if victim == "w1" else w2
    vp.kill(); vp.wait()
    wn = worker(T["next"], victim, types="noop,noop_next")          # the victim's state, updated to NEXT mid-job
    assert wait(lambda: state(l1) != "leased" or holder(l1) != wid(victim), 30)
    ok("E: a NEXT worker resuming a v2 lease hands it back (abandoned, not counted)", "abandon" in kinds(l1)
       and q("SELECT attempts FROM jobs WHERE id=?", (l1,))[0][0] == 0 and "before an update" in open(f"{W}/{victim}.log").read(),
       f"{kinds(l1)} {holders}")
    ok("E: the v2 job left running across the server bump is accepted, nothing requeued",
       wait(lambda: state(l2) == "accepted", 40) and kinds(l2).count("lease") == 1 and not {"expire", "requeue", "abandon"} & set(kinds(l2)),
       str(kinds(l2)))
    other = w1 if victim == "w2" else w2                          # the other NEW worker leaves (finishing what it holds)
    other.send_signal(signal.SIGTERM); other.wait(60)
    r = admin(T["next"], "retire", "noop", "2", check=False)
    ok("E: retire noop 2 refused while v2 jobs are queued (without --migrate)", r.returncode != 0 and "still need it" in r.stderr, r.stderr[-160:])
    w3 = worker(T["old"], "w3")
    assert wait(lambda: all_in([l1] + v2q, "accepted") and all_in(v3 + nxt, "accepted"), 90), \
        f"E: {[(j, state(j)) for j in [l1] + v2q + v3 + nxt]}"
    by = {r["id"]: (r["type"], r["version"], r["accepted_by"]) for r in q("SELECT id, type, version, accepted_by FROM jobs")}
    leases = q("SELECT w.name, j.type, j.version FROM leases l JOIN workers w ON w.id=l.worker_id JOIN jobs j ON j.id=l.job_id "
               "WHERE j.level LIKE 'e-%'")
    old_new = {"w3", "w1", "w2"} - {victim}
    ok("E: old and new workers never leased noop v3 or noop_next", all((t, v) == ("noop", 2) for n, t, v in leases if n in old_new),
       str(sorted({(n, t, v) for n, t, v in leases})))
    ok("E: the NEXT worker got both noop v3 and noop_next, never v2 (after handing back the one it held)",
       {(t, v) for n, t, v in leases if n == victim and (t, v) != ("noop", 2)} == {("noop", 3), ("noop_next", 1)}
       and sum(1 for n, t, v in leases if n == victim and (t, v) == ("noop", 2)) == 1
       and all(by[j][2] == wid(victim) for j in v3 + nxt) and len(nxt) == 3)
    w3.send_signal(signal.SIGTERM); w3.wait(30)                    # the last v2 worker leaves
    left = enqueue(T["new"], 2, "e-left", 0.2, 760)                 # leftovers at v2 nobody runs any more
    r = admin(T["next"], "retire", "noop", "2", "--migrate", check=False)
    got = wait(lambda: all_in(left, "accepted"), 30)
    vb = [tuple(q("SELECT version, accepted_by FROM jobs WHERE id=?", (j,))[0]) for j in left]
    ok("E: retire noop 2 --migrate re-stamps the leftovers at v3; the NEXT worker runs them",
       r.returncode == 0 and "2 queued jobs moved to v3" in r.stdout and got and vb == [(3, wid(victim))] * 2,
       f"{vb} {r.stdout} {r.stderr[-200:]}")
    r = admin(T["next"], "versions")
    ok("E: admin.py versions shows v2 retired, v3 current, noop_next", r.returncode == 0 and re.search(r"noop\s+2\s+retired", r.stdout)
       and re.search(r"noop\s+3\s+current", r.stdout) and re.search(r"noop_next\s+1\s+current", r.stdout), r.stdout)
    for p in (wn, w1, w2):
        if p.poll() is None: p.send_signal(signal.SIGTERM)
    s.stop()


if __name__ == "__main__":
    t0 = time.time()
    try: main()
    except (AssertionError, SystemExit) as e:
        ok(f"ran to the end ({e})", False)
    finally:
        stop_all()
    n = sum(1 for _, c in RESULTS if c)
    print(f"\n{n}/{len(RESULTS)} checks passed ({time.time() - t0:.0f} s)")
    sys.exit(0 if n == len(RESULTS) and n else 1)
