#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bandwidth_e2e.py [WORKDIR] -- the worker's bandwidth limits (worker/bandwidth.py) end to end with noop jobs:
server.py and worker.py as processes on localhost (~3 min, ~25 MB of disk).

Checks: the budget counters in process (calendar day and budget month in local time with a reset day, persistence
across instances, next reset, reservations); rate caps: a 7.5 MB fetch and a 7.5 MB upload at 2 Mbps take 30 s
each (+-10%), a Range-resumed fetch pays only its remaining half; download budget: a worker leases only while its
budget covers the job, keeps the count across a SIGKILL and a restart, stops leasing when spent (over_budget from the
server), shows it in `rqah status` (server view and this machine's counters) and on the public status page (a
count); upload budget the same with output-only jobs; a fully cached job still runs on a spent download budget; cache
affinity with a real worker (a level's job whose input it holds comes first)."""
import json, os, re, shutil, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from worker.bandwidth import Budget, Meter, Bucket, MBPS
from common.util import sha256_file

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rqah-bw-e2e")
PORT = int(os.environ.get("E2E_PORT", "18761"))
URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable
DB, DATA, RES, SPOOL = f"{W}/db/q.sqlite", f"{W}/data", f"{W}/results", f"{W}/spool"
RESULTS = []
N = 7_500_000                                               # 30 s at 2 Mbps


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def admin(*a):
    r = subprocess.run([PY, "-I", f"{HERE}/server/admin.py", "--db", DB, "--data", DATA, "--results", RES, *a], capture_output=True, text=True)
    if r.returncode: print(r.stdout, r.stderr)
    return r.stdout


def add_worker(name):
    out = admin("add-worker", name)
    wid, tok = re.search(r"worker (\d+) token (\S+)", out).groups()
    p = f"{W}/{name}.token"; open(p, "w").write(tok + "\n"); os.chmod(p, 0o600)
    return int(wid), p


def worker(name, tokfile, *extra):
    lf = open(f"{W}/{name}.log", "a")
    return subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", URL, "--token-file", tokfile, "--state", f"{W}/{name}",
                             "--poll", "0.5", *extra], stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "TZ": "America/Chicago"})


def stats(name):
    p = f"{W}/{name}/worker.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def wait(cond, timeout, step=0.25):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(step)
    return cond()


def sql(q, a=()):
    import sqlite3
    c = sqlite3.connect(DB, timeout=30); c.row_factory = sqlite3.Row
    try: return c.execute(q, a).fetchall()
    finally: c.close()


def blob(name, n, seed):
    import hashlib
    with open(f"{DATA}/test/{name}", "wb") as f:
        i = 0
        while n > 0:
            b = hashlib.sha256(f"{seed}:{i}".encode()).digest() * 2048; f.write(b[:n]); n -= len(b[:n]); i += 1


def unit():
    """the counters with a fake clock (TZ America/Chicago, reset day 15)"""
    os.environ["TZ"] = "America/Chicago"; time.tzset()
    d = f"{W}/unit"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    t = [time.mktime((2026, 11, 14, 23, 59, 0, 0, 0, -1))]
    clock = lambda: t[0]
    b = Budget(d, {"down_day": 10 ** 6, "down_month": 3 * 10 ** 6, "up_day": None, "up_month": None}, 15, clock)
    b.add("down", 400_000); b.flush()
    ok("budget: remaining = the tighter of day and month", b.remaining("down") == 600_000 and b.remaining("up") is None)
    b2 = Budget(d, b.limits, 15, clock)
    ok("budget: counters persist across instances (a restart)", b2.remaining("down") == 600_000, b2.snapshot()["used"])
    b2.add("down", 600_000)
    nr = b2.next_reset("down")
    ok("budget: day spent -> next reset is local midnight", b2.remaining("down") == 0 and
       time.localtime(nr)[:5] == (2026, 11, 15, 0, 0), time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(nr)))
    t[0] += 120                                             # 2026-11-15 00:01: new day and (reset day 15) new month
    s = b2.snapshot()
    ok("budget: a new day and a new budget month (reset day 15) zero their counters",
       b2.remaining("down") == 10 ** 6 and s["month"] == "2026-11-15" and s["used"]["down_month"] == 0, s)
    t[0] = time.mktime((2026, 12, 3, 12, 0, 0, 0, 0, -1))
    b2.add("down", 999_999); t[0] += 86400; b2.add("down", 999_999); t[0] += 86400; b2.add("down", 999_999)
    nr = b2.next_reset("down")
    ok("budget: month spent (and the day) -> next reset is the 15th", b2.remaining("down") == 1 and
       time.localtime(nr)[:3] == (2026, 12, 15), time.strftime("%Y-%m-%d", time.localtime(nr)))
    t[0] = time.mktime((2027, 1, 14, 12, 0, 0, 0, 0, -1))
    ok("budget: January 14 still counts in the month that started December 15 (year wrap)", b2.snapshot()["month"] == "2026-12-15")
    m = Meter(f"{W}/unit2", limits={"down_day": 10 ** 7, "down_month": None, "up_day": None, "up_month": 5 * 10 ** 6})
    m.reserve(1, 3 * 10 ** 6, 10 ** 6); m.reserve(2, 2 * 10 ** 6, 5 * 10 ** 5)
    m.transfer("down", 10 ** 6, 1)
    dec = m.declare()
    ok("meter: declared budget = left minus running jobs' reservations (paid bytes release them)",
       dec["down"] == 10 ** 7 - 10 ** 6 - 2 * 10 ** 6 - 2 * 10 ** 6 and dec["up"] == 5 * 10 ** 6 - 15 * 10 ** 5 and dec["spent"] == [], dec)
    bk = Bucket(2 * MBPS); t0 = time.monotonic()
    for _ in range(20): bk.take(25_000)
    el = time.monotonic() - t0
    ok("bucket: 500 kB at 2 Mbps in ~2 s (the first quarter second is burst)", 1.6 <= el <= 2.2, f"{el:.2f} s")
    del os.environ["TZ"]; time.tzset()


def main():
    shutil.rmtree(W, ignore_errors=True)
    for d in (f"{W}/db", f"{DATA}/test", RES, SPOOL): os.makedirs(d)
    unit()
    blob("big.bin", N, 1)
    for i in range(4): blob(f"m{i}.bin", 1_000_000, 10 + i)
    blob("c0.bin", 1_000_000, 20)
    admin("manifest")
    srv = subprocess.Popen([PY, "-I", f"{HERE}/server/server.py", "--db", DB, "--data", DATA, "--spool", SPOOL, "--results", RES,
                            "--port", str(PORT), "--lease-s", "30", "--heartbeat-s", "2", "--status-interval", "1", "--retry-s", "5"],
                           stdout=open(f"{W}/server.log", "a"), stderr=subprocess.STDOUT)
    procs = [srv]
    try:
        assert wait(healthy, 20), "server up"
        run(procs)
    finally:
        for p in procs:
            if p.poll() is None: p.send_signal(signal.SIGKILL); p.wait()
    n = sum(1 for _, p in RESULTS if p)
    print(f"\n{n}/{len(RESULTS)} checks passed")
    return 0 if n == len(RESULTS) else 1


def run(procs):
    # ---------------------------------------------------------------- rate caps
    _, tA = add_worker("rate")
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "7500000", "--inputs", "test/big.bin", "--tag", "rate")
    p = worker("rate", tA, "--max-down-mbps", "2", "--max-up-mbps", "2", "--max-jobs", "1"); procs.append(p)
    p.wait(timeout=150)
    s = stats("rate")
    f_s, u_s = (s[0].get("fetch_s"), s[0].get("upload_s")) if s else (None, None)
    ok("rate cap: 7.5 MB fetch at 2 Mbps takes 30 s +-10%", s and 27 <= f_s <= 33, f"fetch {f_s} s ({s[0].get('down_bytes') if s else '-'} B)")
    ok("rate cap: 7.5 MB upload at 2 Mbps takes 30 s +-10%", s and 27 <= u_s <= 33, f"upload {u_s} s ({s[0].get('up_bytes') if s else '-'} B)")
    # Range resume: half of the blob already in .part
    sha = sha256_file(f"{DATA}/test/big.bin")
    shutil.rmtree(f"{W}/rate/cache", ignore_errors=True); os.makedirs(f"{W}/rate/cache/blobs")
    with open(f"{DATA}/test/big.bin", "rb") as f, open(f"{W}/rate/cache/blobs/{sha}.part", "wb") as g: g.write(f.read(N // 2))
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "1000", "--inputs", "test/big.bin", "--tag", "resume", "--seed", "5")
    p = worker("rate", tA, "--max-down-mbps", "2", "--max-jobs", "1"); procs.append(p)
    p.wait(timeout=120)
    s = stats("rate")
    r = s[-1] if len(s) == 2 else {}
    ok("rate cap on a Range-resumed fetch: only the missing half, 15 s +-10%", r.get("down_bytes") == N - N // 2 and 13.5 <= r.get("fetch_s", 0) <= 16.5,
       f"{r.get('down_bytes')} B in {r.get('fetch_s')} s")

    # ---------------------------------------------------------------- download budget across a SIGKILL and a restart
    wid, tB = add_worker("cap")
    for i in range(4):                                     # one distinct 1 MB input each
        admin("enqueue-noop", "1", "--seconds", "0", "--size", "1000", "--inputs", f"test/m{i}.bin", "--tag", "dcap", "--seed", str(i))
    ids = [r["id"] for r in sql("SELECT id FROM jobs WHERE level='dcap' ORDER BY id")]
    bud = ["--max-down-gb-day", "0.0025"]                   # 2.5 MB: two 1 MB jobs, then 0.5 MB < 1 MB left
    p = worker("cap", tB, *bud); procs.append(p)
    wait(lambda: len(stats("cap")) >= 1, 60)
    p.send_signal(signal.SIGKILL); p.wait()
    bw = json.load(open(f"{W}/cap/bandwidth.json"))
    ok("download budget: counted bytes are on disk after a SIGKILL", bw.get("down_day", 0) >= 1_000_000, f"down_day {bw.get('down_day')}")
    done1 = len(stats("cap"))
    p = worker("cap", tB, *bud, "--idle-exit", "6"); procs.append(p)
    p.wait(timeout=120)
    bw = json.load(open(f"{W}/cap/bandwidth.json"))
    acc = sql("SELECT COUNT(*) n FROM jobs WHERE level='dcap' AND state='accepted'")[0]["n"]
    ok("download budget across the restart: exactly 2 of 4 jobs (2 MB of 2.5), the rest stay queued",
       acc == 2 and 2_000_000 <= bw["down_day"] <= 2_100_000, f"accepted {acc}, down_day {bw['down_day']}, first run {done1}")
    log = open(f"{W}/cap.log").read()
    ok("the worker logs the spent budget and the server's over-budget answer",
       "data budget spent (download)" in log and "need more data than its budget has left" in log)
    caps = json.loads(sql("SELECT caps FROM workers WHERE id=?", (wid,))[0]["caps"])
    ok("the lease request declared it (budget.spent = [down], down < 1 MB, reset at local midnight)",
       caps.get("budget", {}).get("spent") == ["down"] and caps["budget"]["down"] < 1 << 20 and caps["budget"].get("reset"), caps.get("budget"))
    r = subprocess.run([PY, "-I", f"{HERE}/client/rqah.py", "--server", URL, "--token-file", tB, "--state", f"{W}/cap", "status"],
                       capture_output=True, text=True, env={**os.environ, "TZ": "America/Chicago"})
    ok("rqah status: the worker's DATA LEFT column, the spent note and this machine's counters",
       "DATA LEFT" in r.stdout and "data budget spent (download)" in r.stdout and "data budget (this machine): download" in r.stdout
       and "SPENT: down" in r.stdout, r.stdout[-400:].replace("\n", " | "))
    rj = json.loads(subprocess.run([PY, "-I", f"{HERE}/client/rqah.py", "--json", "--server", URL, "--token-file", tB, "--state",
                                    f"{W}/cap", "status"], capture_output=True, text=True).stdout)
    ok("rqah --json status: budget per worker and local_budget", rj["workers"][0]["budget"]["spent"] == ["down"] and
       rj["local_budget"]["spent"] == ["down"], rj["workers"][0]["budget"])
    time.sleep(2)
    page = urllib.request.urlopen(URL + "/").read().decode()
    ok("status page: a count of workers at their data cap (no names, no amounts)", "1 workers at their data cap until it resets" in page
       and "cap.token" not in page)
    # a spent download budget still runs a job whose inputs are all cached
    admin("enqueue-noop", "1", "--seconds", "0", "--size", "1000", "--inputs", "test/m0.bin", "--tag", "cached", "--seed", "99")
    p = worker("cap", tB, *bud, "--idle-exit", "6"); procs.append(p)
    p.wait(timeout=60)
    acc = sql("SELECT COUNT(*) n FROM jobs WHERE level='cached' AND state='accepted'")[0]["n"]
    st = stats("cap")[-1]
    ok("spent download budget: a job whose inputs are cached still runs (0 bytes down)", acc == 1 and st.get("down_bytes") == 0, st)

    # ---------------------------------------------------------------- upload budget
    admin("cancel", "--level", "dcap")                     # the two jobs over the first worker's budget
    _, tC = add_worker("upcap")
    admin("enqueue-noop", "3", "--seconds", "0", "--size", "1000000", "--tag", "ucap")
    p = worker("upcap", tC, "--max-up-gb-month", "0.0025", "--budget-reset-day", "5", "--idle-exit", "6"); procs.append(p)
    p.wait(timeout=90)
    bw = json.load(open(f"{W}/upcap/bandwidth.json"))
    acc = sql("SELECT COUNT(*) n FROM jobs WHERE level='ucap' AND state='accepted'")[0]["n"]
    ok("upload budget (month from the 5th): 2 of 3 1 MB results uploaded, then nothing new",
       acc == 2 and 2_000_000 <= bw["up_month"] <= 2_100_000 and bw["month"].endswith("-05"), f"accepted {acc}, {bw}")
    page = urllib.request.urlopen(URL + "/").read().decode()
    time.sleep(1.5); page = urllib.request.urlopen(URL + "/").read().decode()
    ok("status page counts both capped workers", "2 workers at their data cap" in page)

    # ---------------------------------------------------------------- cache affinity with a real worker
    admin("cancel", "--level", "ucap")                     # the one over the upload budget
    _, tD = add_worker("aff")
    for i, f in enumerate(("m1", "m2", "m1")):
        admin("enqueue-noop", "1", "--seconds", "0", "--size", "100", "--inputs", f"test/{f}.bin", "--tag", "aff", "--seed", str(200 + i))
    ids = [r["id"] for r in sql("SELECT id FROM jobs WHERE level='aff' ORDER BY id")]
    p = worker("aff", tD, "--idle-exit", "4"); procs.append(p)
    p.wait(timeout=60)
    order = [r["job"] for r in stats("aff")]
    ok("cache affinity: after the m1 job the worker gets the other m1 job of the level before the m2 one (1 MB saved)",
       order == [ids[0], ids[2], ids[1]] and [r["down_bytes"] for r in stats("aff")] == [1_000_000, 0, 1_000_000], order)


def healthy():
    try: return urllib.request.urlopen(URL + "/healthz", timeout=2).status == 200
    except OSError: return False


if __name__ == "__main__":
    sys.exit(main())
