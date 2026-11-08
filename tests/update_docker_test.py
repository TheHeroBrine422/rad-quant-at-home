#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""update_docker_test.py [WORKDIR] -- the drain of `rqah update` (client/setup.py drain()) against real Docker, ~2 min.
Manual (needs a Docker daemon and an image with Python 3.10+: $UPDATE_DOCKER_IMAGE, default the local server image
rad-quant-at-home-server:<VERSION>, else python:3.12-slim). No GPU: the worker runs noop jobs from this tree
(copied into WORKDIR and bind-mounted read-only, `:z` for SELinux) in a container with `--restart unless-stopped` and the host's network namespace, against a
server.py on 127.0.0.1 (nothing listens beyond loopback).

Checks: drain() lets the running job finish (accepted, one lease), the container stops and Docker does not restart it
(restart policy "no" during the drain, RestartCount 0); drain(release=True) hands the job back at once (queued,
attempt not counted); start()'s `docker rm` keeps the state directory (the token is reused). Also reports, without
judging, whether Docker restarts an unless-stopped container whose process exits after `docker kill --signal TERM`
(the reason drain() turns the restart policy off first)."""
import json, os, shutil, signal, sqlite3, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from client import setup as WZ

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rqah-update-docker")
PORT = int(os.environ.get("UPDATE_DOCKER_PORT", "18791"))
PY = sys.executable
VER = open(f"{HERE}/VERSION").read().strip()
RESULTS = []
NAME = "rqah-update-test"


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def sh(*a, check=False):
    r = subprocess.run(list(a), capture_output=True, text=True, timeout=300)
    if check and r.returncode: raise RuntimeError(f"{a}: {r.stderr}")
    return r


def image():
    if os.environ.get("UPDATE_DOCKER_IMAGE"): return os.environ["UPDATE_DOCKER_IMAGE"]
    for i in (f"rad-quant-at-home-server:{VER}", "rad-quant-at-home-server:0.4.0", "rad-quant-at-home-server:latest"):
        if sh("docker", "image", "inspect", i).returncode == 0: return i
    return "python:3.12-slim"


def q(sql, a=()):
    c = sqlite3.connect(f"{W}/q.sqlite", timeout=30)
    try: return c.execute(sql, a).fetchall()
    finally: c.close()


def wait(cond, timeout):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(0.5)
    return cond()


def admin(*a): return sh(PY, "-I", f"{HERE}/server/admin.py", "--db", f"{W}/q.sqlite", "--data", f"{W}/data", *a, check=True).stdout


def run_worker(img):
    sh("docker", "rm", "-f", NAME)
    uid = f"{os.getuid()}:{os.getgid()}"
    return sh("docker", "run", "-d", "--name", NAME, "--restart", "unless-stopped", "--network", "host", "--user", uid,
              "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{W}/app:/app:ro,z", "-v", f"{W}/state:/state:z",
              "--entrypoint", "python", img, "-I", "/app/worker/worker.py", "--server", f"http://127.0.0.1:{PORT}",
              "--token-file", "/state/token", "--state", "/state", "--poll", "0.3", "--job-types", "noop", check=True)


def inspect():
    r = sh("docker", "inspect", "--format", "{{json .State}}|{{json .HostConfig.RestartPolicy}}|{{.RestartCount}}", NAME)
    if r.returncode: return None
    st, pol, n = r.stdout.strip().split("|")
    return json.loads(st), json.loads(pol), int(n)


def main():
    if shutil.which("docker") is None or sh("docker", "info").returncode: sys.exit("update_docker_test: needs a Docker daemon")
    shutil.rmtree(W, ignore_errors=True)
    for d in ("data", "spool", "results", "state"): os.makedirs(f"{W}/{d}")
    open(f"{W}/data/manifest.json", "w").write('{"files": {}}')
    os.makedirs(f"{W}/app")
    for x in ("VERSION", "common", "jobs", "worker"):
        (shutil.copytree if os.path.isdir(f"{HERE}/{x}") else shutil.copy)(f"{HERE}/{x}", f"{W}/app/{x}")
    img = image(); print(f"image {img}", flush=True)
    srv = subprocess.Popen([PY, "-I", f"{HERE}/server/server.py", "--db", f"{W}/q.sqlite", "--data", f"{W}/data", "--spool", f"{W}/spool",
                            "--results", f"{W}/results", "--port", str(PORT), "--lease-s", "8", "--heartbeat-s", "1", "--retry-s", "0.5",
                            "--spot-rate", "0", "--spot-first", "0", "--status-interval", "0"],
                           stdout=open(f"{W}/server.log", "w"), stderr=subprocess.STDOUT)
    try:
        assert wait(lambda: sh("curl", "-sf", f"http://127.0.0.1:{PORT}/healthz").returncode == 0, 30), "server"
        tok = admin("add-worker", "docker-w").split()[-1]
        open(f"{W}/state/token", "w").write(tok + "\n")
        host = WZ.Host("/")
        # 1. drain: the job finishes, the container stays stopped
        run_worker(img)
        j = int(admin("enqueue-noop", "1", "--seconds", "15", "--size", "100", "--seed", "1").split("[")[1].split("]")[0])
        assert wait(lambda: q("SELECT state FROM jobs WHERE id=?", (j,))[0][0] == "leased", 60), "job 1 not leased"
        t0 = time.time()
        stopped = WZ.drain(host, NAME, timeout_s=120, say=print)
        dt = time.time() - t0
        time.sleep(5)
        st, pol, n = inspect()
        ok("drain: the running job finished (accepted, one lease)", q("SELECT state FROM jobs WHERE id=?", (j,))[0][0] == "accepted"
           and q("SELECT COUNT(*) FROM leases WHERE job_id=?", (j,))[0][0] == 1, f"{dt:.0f} s")
        ok("drain: the container stopped by itself (exit 0) and Docker did not restart it", stopped and not st["Running"]
           and st["ExitCode"] == 0 and n == 0 and pol["Name"] == "no", f"{st['Status']} exit {st['ExitCode']} policy {pol['Name']} restarts {n}")
        # 2. start()'s removal keeps the state: a new container uses the same token
        rc = sh("docker", "rm", NAME).returncode
        run_worker(img)
        j2 = int(admin("enqueue-noop", "1", "--seconds", "60", "--size", "100", "--seed", "2").split("[")[1].split("]")[0])
        assert wait(lambda: q("SELECT state FROM jobs WHERE id=?", (j2,))[0][0] == "leased", 60), "job 2 not leased"
        ok("docker rm + a new container: same state, same token (no new worker)", rc == 0 and q("SELECT COUNT(*) FROM workers")[0][0] == 1)
        # 3. drain --release: handed back at once
        t0 = time.time()
        stopped = WZ.drain(host, NAME, timeout_s=120, release=True, say=print)
        dt = time.time() - t0
        r = q("SELECT state, attempts FROM jobs WHERE id=?", (j2,))[0]
        ok("drain --release: the job is handed back at once (queued, attempt not counted), the container stopped",
           stopped and tuple(r) == ("queued", 0) and dt < 30 and not inspect()[0]["Running"], f"{tuple(r)} {dt:.0f} s")
        admin("cancel", str(j2))
        # 4. for the record: an unless-stopped container whose process exits after `docker kill --signal TERM`
        sh("docker", "rm", NAME); run_worker(img); time.sleep(3)
        sh("docker", "kill", "--signal", "TERM", NAME); time.sleep(8)
        st, pol, n = inspect()
        print(f"note: after `docker kill --signal TERM` (policy {pol['Name']}) Docker {'RESTARTED' if n or st['Running'] else 'did not restart'} "
              f"the container (running {st['Running']}, restarts {n}); drain() sets the policy to no first either way", flush=True)
    finally:
        sh("docker", "rm", "-f", NAME)
        srv.send_signal(signal.SIGTERM)
        try: srv.wait(30)
        except subprocess.TimeoutExpired: srv.kill()
    n = sum(1 for _, c in RESULTS if c)
    print(f"\n{n}/{len(RESULTS)} checks passed")
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
