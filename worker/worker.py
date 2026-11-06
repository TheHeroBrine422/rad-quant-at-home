#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""worker.py --server URL --state DIR [--cache DIR --cache-gb G] [caps] [runtime paths] [--window HH:MM-HH:MM ...]

The rad-quant-at-home donor side: lease -> fetch inputs (content-addressed cache with a disk cap) -> run the built-in job
-> upload (resumable) -> complete, with heartbeats throughout. Everything it does is in jobs/: the server names a job
type and sends parameters, which must match that type's schema; inputs must be the files the type allows for those
parameters and match their sha256; only the type's output files are uploaded, and only to the configured server.

Several cards (0.2): --gpus 0,1,2,3 (or --cards 4) runs up to one job per card at once, each in its own thread whose
job processes see only their cards (--gpu-env: ROCR_VISIBLE_DEVICES on ROCm, CUDA_VISIBLE_DEVICES on CUDA), with one
input cache and one host-RAM budget (--ram-gb) for all of them. A gang job (several cards) gets that many free cards;
the server drains the worker for it first (no new jobs until the cards are free).

Restart-safe: each running job and its phase live in STATE/current/<job>-<lease>.json (written atomically); downloads
go to <blob>.part (resumed with HTTP ranges) and are renamed only after their sha256 matches; uploads resume from the
server's offset. A killed worker restarted with the same --state picks its jobs up again if their leases are alive.
Stop: SIGINT / SIGTERM or the --stop-file once = finish the current jobs and exit (--on-stop abandon: give them back
now); twice = abandon now. Time windows: no new leases outside them. Every flag has a RQAH_* environment variable
(the 0.2 names RADQUANT_* are still read in 0.3).

Bandwidth (0.4, worker/bandwidth.py): --max-down-mbps / --max-up-mbps cap the transfer rates (all jobs together);
--max-{down,up}-gb-{day,month} are data budgets counted in STATE/bandwidth.json. The worker declares what is left
(minus what its running jobs still have to move) and which inputs it caches when it leases; the server offers only
jobs that fit. A spent budget: the running jobs finish, nothing new is leased until the day / month resets."""
import argparse, fcntl, hashlib, json, os, re, shutil, signal, subprocess, sys, threading, time, traceback
import urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import API, read_json, write_json_atomic, sha256_file, safe_relpath, env, fmt_bytes
from worker.bandwidth import Meter, GB
import jobs

UA = "rad-quant-at-home-worker/0.4"
CACHE_REPORT_MAX = 32768                                 # cached blobs reported at lease time (most recently used)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class APIError(Exception):
    def __init__(self, code, body): super().__init__(f"HTTP {code}: {body}"); self.code, self.body = code, body


class Client:
    def __init__(self, url, token=None, timeout=60):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _req(self, method, path, body=None, headers=None, raw=False):
        h = {"User-Agent": UA, **(headers or {})}
        if self.token: h["Authorization"] = f"Bearer {self.token}"
        data = None
        if isinstance(body, MeteredBody):                                    # a throttled upload chunk
            body.reset(); data = body; h["Content-Length"] = str(len(body))
        elif body is not None and not isinstance(body, (bytes, bytearray)):
            data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
        elif body is not None: data = bytes(body)
        r = urllib.request.Request(self.url + path, data=data, method=method, headers=h)
        try:
            resp = urllib.request.urlopen(r, timeout=self.timeout)
            return resp if raw else json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try: b = json.loads(e.read() or b"{}")
            except ValueError: b = {}
            raise APIError(e.code, b)

    def call(self, method, path, body=None, tries=None, stop=None, **kw):
        """retries transient failures (connection, 5xx) with backoff; 4xx are final"""
        d, n = 2.0, 0
        while True:
            try: return self._req(method, path, body, **kw)
            except APIError as e:
                if e.code < 500: raise
                err = e
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                err = e
            n += 1
            if (tries is not None and n >= tries) or (stop is not None and stop()): raise err
            log(f"server unreachable ({err!r}); retry in {d:.0f}s")
            time.sleep(d); d = min(d * 2, 120)

    def api(self, method, ep, body=None, **kw): return self.call(method, f"/api/{API}/{ep}", body, **kw)


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


class MeteredBody:
    """an upload chunk that pays the up bucket and counts against the budget as http.client reads it (8 KB at a
    time); reset() before every attempt (a retried request sends it again, and that counts again)"""
    def __init__(self, b, meter, job):
        self.b, self.meter, self.job, self.pos = memoryview(b), meter, job, 0
    def __len__(self): return len(self.b)
    def reset(self): self.pos = 0
    def read(self, n=-1):
        if n is None or n < 0: n = len(self.b) - self.pos
        x = bytes(self.b[self.pos:self.pos + n]); self.pos += len(x)
        if x: self.meter.transfer("up", len(x), self.job)
        return x


# ---------------------------------------------------------------- the input cache
class CacheFull(Exception): pass


class Cache:
    """blobs/<sha256> with a byte cap, shared by every running job: blobs a running job needs are pinned; least
    recently used unpinned blobs are evicted; a blob is downloaded once even if two jobs need it at the same time; a
    job that cannot get room while other jobs hold pins waits for them"""
    def __init__(self, d, cap):
        self.d, self.cap = d, cap
        self.blobs = os.path.join(d, "blobs"); os.makedirs(self.blobs, exist_ok=True)
        self.ix_path = os.path.join(d, "index.json")
        self.ix = read_json(self.ix_path, {}) or {}
        on_disk = {f for f in os.listdir(self.blobs)}
        self.ix = {k: v for k, v in self.ix.items() if k in on_disk}       # forget what is gone
        for f in on_disk:                                                     # files nobody knows: drop
            if f not in self.ix and not f.endswith(".part"): os.remove(os.path.join(self.blobs, f))
        self.evicted = 0
        self.cv = threading.Condition()
        self.pins, self.fetching = {}, {}                                    # sha -> count / sha -> size

    def used(self):
        return sum(os.path.getsize(os.path.join(self.blobs, f)) for f in os.listdir(self.blobs) if not f.endswith(".part"))

    def path(self, sha): return os.path.join(self.blobs, sha)
    def save(self): write_json_atomic(self.ix_path, self.ix)
    def have(self, i): return os.path.exists(self.path(i["sha256"])) and os.path.getsize(self.path(i["sha256"])) == i["size"]

    def missing_bytes(self, inputs):
        """what fetching these inputs would still download (partial downloads resume)"""
        n = 0
        for i in {i["sha256"]: i for i in inputs}.values():
            if self.have(i): continue
            part = self.path(i["sha256"]) + ".part"
            n += i["size"] - min(i["size"], os.path.getsize(part) if os.path.exists(part) else 0)
        return n

    def report(self):
        """(rev, [sha256[:16] of complete blobs, most recently used first]) for the lease request: the server prefers
        jobs whose inputs are here and charges only the rest against a download budget"""
        with self.cv:
            ks = [k for _, k in sorted(((v.get("last", 0), k) for k, v in self.ix.items()), reverse=True)
                  if os.path.exists(self.path(k))][:CACHE_REPORT_MAX]
        shas = sorted(k[:16] for k in ks)
        return hashlib.sha256(",".join(shas).encode()).hexdigest()[:16], shas

    def _room(self, need, mine):
        """under cv: evict until need fits beside complete blobs and in-flight downloads. True / False (others' pins
        or downloads hold the space: wait) / raises CacheFull (it can never fit)"""
        if need > self.cap: raise CacheFull(f"job needs {need} bytes, cache cap {self.cap}")
        for f in os.listdir(self.blobs):                                      # stale partial downloads first
            if f.endswith(".part") and f[:-5] not in self.pins and f[:-5] not in self.fetching: os.remove(os.path.join(self.blobs, f))
        while self.used() + sum(self.fetching.values()) + need > self.cap:
            cand = sorted((v["last"], k) for k, v in self.ix.items() if k not in self.pins)
            if not cand:
                if any(s not in mine for s in self.pins) or self.fetching: return False
                raise CacheFull(f"cannot free {need} bytes: everything cached is needed now")
            _, k = cand[0]; os.remove(self.path(k)); del self.ix[k]; self.evicted += 1
            log(f"cache: evicted {k[:12]} ({self.evicted} so far)")
        self.save()
        return True

    def pin(self, inputs):
        with self.cv:
            for s in {i["sha256"] for i in inputs}: self.pins[s] = self.pins.get(s, 0) + 1

    def unpin(self, inputs):
        with self.cv:
            for s in {i["sha256"] for i in inputs}:
                self.pins[s] -= 1
                if not self.pins[s]: del self.pins[s]
            self.cv.notify_all()

    def ensure(self, inputs, fetch, cancel):
        """inputs [{name, sha256, size}] (pinned by the caller) -> ({name: local path}, bytes fetched)"""
        mine = {i["sha256"] for i in inputs}
        got = 0
        for i in {i["sha256"]: i for i in inputs}.values():
            s = i["sha256"]; part = self.path(s) + ".part"
            with self.cv:
                while True:
                    if cancel.is_set(): raise InterruptedError("cancelled")
                    if self.have(i): break
                    if s in self.fetching: self.cv.wait(1); continue          # another job is downloading it
                    have = os.path.getsize(part) if os.path.exists(part) else 0
                    if self._room(max(0, i["size"] - have), mine): self.fetching[s] = i["size"]; break
                    log(f"cache: waiting for other jobs to free room for {i['name']}"); self.cv.wait(5)
            if s not in self.fetching: continue
            try:
                got += fetch(i, part)
                if sha256_file(part) != s:
                    os.remove(part); raise IOError(f"{i['name']}: sha256 mismatch after download")
                os.replace(part, self.path(s))
            finally:
                with self.cv: self.fetching.pop(s, None); self.cv.notify_all()
        with self.cv:
            t = time.time()
            for i in inputs: self.ix[i["sha256"]] = {"size": i["size"], "last": t, "name": i["name"]}
            self.save()
        return {i["name"]: self.path(i["sha256"]) for i in inputs}, got


# ---------------------------------------------------------------- one job's runtime
class JobContext:
    """what a job module sees: its validated inputs, its own out / stage dirs, its cards (job processes see only them,
    as device 0..n-1, unless --gpu-env none), cancellation"""
    def __init__(self, cfg, job, paths, out_dir, stage_dir, cancel, devices):
        self.cfg, self.jtype, self.jversion, self.params = cfg, job["type"], job["version"], job["params"]
        self._paths, self.input_names = paths, list(paths)
        self.out_dir, self.stage_dir, self._cancel = out_dir, stage_dir, cancel
        self.devices = list(devices)
        ge = cfg.gpu_env
        self.dev_env = {ge: ",".join(map(str, devices))} if ge != "none" and devices else {}
        self.device = 0 if self.dev_env or not devices else devices[0]

    def input(self, name): return self._paths[name]
    def cancelled(self): return self._cancel.is_set()

    def run_cmd(self, argv, env=None, cwd=None, log=None):
        e = dict(os.environ); e.update(self.cfg.env); e.update(self.dev_env); e.update(env or {})
        with open(log or os.devnull, "ab") as lf:
            p = subprocess.Popen(argv, env=e, cwd=cwd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            while p.poll() is None:
                if self._cancel.wait(0.5):
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(15)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    raise InterruptedError("cancelled")
            return p.returncode


class GpuLock:
    """flock on shared lock files (a box's one-GPU-job-at-a-time locks), interruptible. A path with {card} is one
    lock per card (e.g. /var/lock/gpu-card{card}.lock), taken for each of the job's cards in order"""
    def __init__(self, path, cancel, devices=()):
        self.cancel, self.fds = cancel, []
        self.paths = [] if not path else sorted({path.format(card=d) for d in devices}) if "{card}" in path else [path]
    def __enter__(self):
        for p in self.paths:
            fd = open(p, "a"); t0 = time.time(); said = False
            while True:
                try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if not said: log(f"waiting for {p}"); said = True
                    if self.cancel.wait(2): fd.close(); self.__exit__(); raise InterruptedError("cancelled while waiting for the GPU lock")
            if said: log(f"got {p} after {time.time() - t0:.0f}s")
            self.fds.append(fd)
        return self
    def __exit__(self, *x):
        for fd in self.fds: fcntl.flock(fd, fcntl.LOCK_UN); fd.close()
        self.fds = []


def in_windows(windows, t=None):
    if not windows: return True
    lt = time.localtime(t); m = lt.tm_hour * 60 + lt.tm_min
    for w in windows:
        a, b = [int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in w.split("-")]
        if (a <= m < b) if a <= b else (m >= a or m < b): return True
    return False


# ---------------------------------------------------------------- the worker
class Worker:
    def __init__(self, a):
        self.a = a
        a.state = os.path.abspath(a.state)
        os.makedirs(os.path.join(a.state, "current"), exist_ok=True)
        # what job modules read through ctx.cfg: this install's paths (never from the server)
        self.state, self.python, self.ourq, self.exl3_src = a.state, a.python, a.ourq, a.exl3_src
        self.radiance, self.mkgold27, self.headroom_mib = a.radiance, a.mkgold27, a.headroom_mib
        self.image, self.gpu_name = a.image, a.gpu_name
        self.vram_cap_gb, self.capwrap = a.vram_cap_gb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "capwrap.py")
        self.gpus = [int(x) for x in a.gpus.split(",") if x != ""] if a.gpus else (list(range(a.cards)) if a.cards else
                                                                                 ([a.device] if a.gpu_name else []))
        self.gpu_env = a.gpu_env if a.gpu_env != "auto" else ("ROCR_VISIBLE_DEVICES" if os.path.exists("/dev/kfd") else "CUDA_VISIBLE_DEVICES")
        self.stats_path = os.path.join(a.state, "worker.jsonl")
        legacy = os.path.join(a.state, "current.json")                       # a v0 state dir: one job
        if os.path.exists(legacy):
            c = read_json(legacy)
            if not (c and self.ids_ok(c.get("job") or {})): log(f"{legacy}: bad job or lease id; leaving it in place")
            else:
                c.setdefault("devices", self.gpus[:1])
                write_json_atomic(self.cur_path(c["job"]), c); os.remove(legacy)
        tok = a.token or (open(a.token_file).read().strip() if a.token_file and os.path.exists(a.token_file) else None)
        self.c = Client(a.server, tok)
        if not tok:
            if not a.invite: sys.exit("no token: give --token / --token-file, or --invite CODE --name NAME to register")
            r = self.c.api("POST", "register", {"invite": a.invite, "name": a.name, "leaderboard": bool(a.leaderboard)})
            tok = r["token"]; self.c.token = tok
            if a.token_file:
                fd = os.open(a.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f: f.write(tok + "\n")
            log(f"registered as worker {r['worker_id']}")
        self.cache = Cache(a.cache or os.path.join(a.state, "cache"), int(a.cache_gb * 1e9))
        lim = {f"{d}_{p}": int(getattr(a, f"max_{d}_gb_{p}") * GB) if getattr(a, f"max_{d}_gb_{p}") else None
               for d in ("down", "up") for p in ("day", "month")}
        self.meter = Meter(a.state, a.max_down_mbps, a.max_up_mbps, lim, a.budget_reset_day)
        self.cache_sent = None                                                # cache report rev the server holds
        self.legacy_caps = False                                              # a 0.3 server: no budget / cache keys
        self.budget_note = None
        self.lease_pause = 0.0                                                # after handing back an over-budget job
        self.stop, self.abandon = threading.Event(), threading.Event()
        self.stop_lock, self.stop_file_seen = threading.Lock(), False
        impl = {n for n, m in jobs.REGISTRY.items() if m.IMPLEMENTED}
        self.types = sorted(impl & set(a.job_types.split(","))) if a.job_types else sorted(impl)
        self.slots = a.slots or max(len(self.gpus), 1)
        self.caps = {"job_types": self.types, "versions": {n: jobs.REGISTRY[n].VERSION for n in self.types},
                     "gpu": bool(self.gpus), "gpu_name": a.gpu_name or "", "cards": len(self.gpus), "slots": self.slots,
                     "vram_gb": a.vram_gb, "ram_gb": a.ram_gb, "disk_gb": a.cache_gb, "image": a.image}
        self.env = dict(kv.split("=", 1) for kv in a.env)
        self.lock = threading.Lock()
        self.running = {}                                                     # job id -> {thread, cancel, devices, ram}
        self.jobs_done = 0

    # signals / stop file
    def on_signal(self, *_):
        if self.stop.is_set() or self.a.on_stop == "abandon":
            log("stop: abandoning the current jobs"); self.abandon.set()
            with self.lock:
                for r in self.running.values(): r["cancel"].set()
        else: log("stop: finishing the current jobs, then exiting (again to abandon them)")
        self.stop.set()

    def check_stop_file(self):
        """main loop and every job's heartbeat thread look; only the first sighting counts as a stop request"""
        if self.a.stop_file and not self.stop.is_set() and os.path.exists(self.a.stop_file):
            with self.stop_lock:
                if self.stop_file_seen: return
                self.stop_file_seen = True
            log("stop file seen"); self.on_signal()

    def record(self, **kw):
        with self.lock:
            with open(self.stats_path, "a") as f: f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    # ------------------------------------------------------------ resources
    def free(self):
        with self.lock:
            busy = {d for r in self.running.values() for d in r["devices"]}
            return [d for d in self.gpus if d not in busy], self.a.ram_gb - sum(r["ram"] for r in self.running.values())

    def reap(self):
        with self.lock:
            for k in [k for k, r in self.running.items() if not r["thread"].is_alive()]: del self.running[k]
            return len(self.running)

    # ------------------------------------------------------------ main loop
    def loop(self):
        for f in sorted(os.listdir(os.path.join(self.a.state, "current"))):
            cur = read_json(os.path.join(self.a.state, "current", f))
            if cur: self.resume(cur)
        idle = 0
        while True:
            self.check_stop_file()
            n_run = self.reap()
            if self.stop.is_set():
                if not n_run: break
                time.sleep(0.3); continue
            if self.a.max_jobs and self.jobs_done + n_run >= self.a.max_jobs:
                if not n_run: break
                time.sleep(0.3); continue
            if n_run >= self.slots or not in_windows(self.a.window) or time.time() < self.lease_pause:
                self.stop.wait(0.3 if n_run or time.time() < self.lease_pause else 30); continue
            caps = dict(self.caps, running=sorted(self.running))
            if not self.legacy_caps:
                bud = self.meter.declare()
                if bud is not None: caps["budget"] = bud
                rev, shas = self.cache.report()
                caps["cache"] = {"rev": rev} if rev == self.cache_sent else {"rev": rev, "shas": shas}
                self.note_budget(bud)
            try: r = self.c.api("POST", "lease", {"caps": caps}, stop=self.stop.is_set)
            except APIError as e:
                if e.code == 400 and not self.legacy_caps and "unknown key" in str(e.body.get("error", "")):
                    self.legacy_caps = True
                    log("server predates 0.4 (no data budgets, no cache report): leasing with the 0.3 caps; jobs over "
                        "the budget are handed back"); continue
                log(f"lease refused: {e}")
                if e.code == 401: self.stop.set(); log("token rejected by the server"); continue
                self.stop.wait(30); continue
            except Exception:
                if self.stop.is_set(): continue
                raise
            if r.get("cache_unknown"): self.cache_sent = None; continue          # the server lost our report: resend
            if "cache" in caps: self.cache_sent = caps["cache"]["rev"]
            if not r.get("job"):
                if r.get("quarantined") and not n_run: log(f"server: {r.get('why')}")
                elif (r.get("draining") or r.get("over_budget")) and not idle: log(f"server: {r.get('why')}")
                if not n_run: idle += 1
                if self.a.idle_exit and idle >= self.a.idle_exit and not n_run: log("idle: exiting"); break
                self.stop.wait(min(r.get("retry_s", 30), self.a.poll)); continue
            idle = 0
            self.start(r["job"])
        self.meter.budget.flush()
        log("worker stopped")

    def note_budget(self, bud):
        """log when the data budget becomes spent / available again"""
        if not bud: return
        spent = bud.get("spent") or []
        note = None
        if spent:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(bud["reset"])) if bud.get("reset") else "?"
            note = (f"data budget spent ({', '.join('download' if d == 'down' else 'upload' for d in spent)}): no new jobs "
                    + ("except ones whose inputs are all cached " if spent == ["down"] else "") + f"until {when} local time")
        if note != self.budget_note:
            log(note or "data budget: available again")
            self.budget_note = note

    def budget_check(self, job):
        """the worker's own check of a fresh lease against its budget (the server should never offer more; a 0.3
        server does not know the budget): None if it fits, else why not. Reserves the job's transfers if it fits"""
        down = self.cache.missing_bytes(job["inputs"])
        up = int((job.get("expect") or {}).get("up") or sum(job.get("outputs", {}).values()))
        for d, n in (("down", down), ("up", up)):
            left = self.meter.available(d, but=job["id"])
            if left is not None and n > left:
                return f"over this worker's data budget: {d}load {fmt_bytes(n)}, {fmt_bytes(left)} left"
        self.meter.reserve(job["id"], down, up)
        return None
    @staticmethod
    def ids_ok(job):
        """the job id and lease id the server (or a state file) gave us are what the server itself produces: a
        non-negative integer and 32 hex characters; nothing builds a path from them before this is true"""
        jid, lid = job.get("id"), job.get("lease_id")
        return isinstance(jid, int) and not isinstance(jid, bool) and jid >= 0 and isinstance(lid, str) and bool(re.fullmatch(r"[0-9a-f]{32}", lid))

    def validate(self, job):
        """is this a job we run, with parameters and inputs it allows? (no code from the server, ever) -> (module,
        its requirements); raises jobs.Rejected"""
        if not self.ids_ok(job): raise jobs.Rejected("bad job id or lease id")
        m = jobs.check_params(job["type"], job["params"])
        if not m.IMPLEMENTED: raise jobs.Rejected(f"{job['type']} is not implemented in this worker")
        if job["type"] not in self.types: raise jobs.Rejected(f"{job['type']} not enabled here")
        for i in job["inputs"]:
            if not (safe_relpath(i.get("name")) and isinstance(i.get("size"), int) and i["size"] >= 0 and
                    isinstance(i.get("sha256"), str) and len(i["sha256"]) == 64 and all(c in "0123456789abcdef" for c in i["sha256"])):
                raise jobs.Rejected(f"bad input entry {str(i)[:100]}")
        jobs.check_inputs(job["type"], job["params"], [i["name"] for i in job["inputs"]])
        r = jobs.requirements(m, job["params"])
        if r["gpu"] and not self.gpus: raise jobs.Rejected("a GPU job, and this worker declared no GPU")
        return m, r

    def start(self, job, cur=None):
        """check a leased (or resumed) job, give it its cards and RAM and a thread; refuse what does not fit"""
        try: m, own = self.validate(job)
        except jobs.Rejected as e:
            log(f"job {str(job.get('id'))[:80]!r}: REJECTED: {e}")
            self.fail(job, f"rejected by worker: {e}", reject=True)
            if cur and self.ids_ok(job): self.cleanup(cur)
            return
        sreq = job.get("requires") or {}
        need = max(int(sreq.get("cards", 0)), own["cards"]); ram = max(float(sreq.get("ram_gb", 0)), own["ram_gb"])
        free_devs, free_ram = self.free()
        t_end = time.time() + 10                         # threads of jobs the server has just accepted may still be ending
        while not (cur and cur.get("devices") is not None) and (need > len(free_devs) or ram > free_ram + 1e-9) and time.time() < t_end:
            time.sleep(0.1); self.reap(); free_devs, free_ram = self.free()
        if cur and cur.get("devices") is not None: devs = cur["devices"]
        elif need > len(free_devs) or ram > free_ram + 1e-9:
            log(f"job {job['id']}: needs {need} cards / {ram} GB RAM, {len(free_devs)} / {free_ram} free: handing it back")
            self.fail(job, f"does not fit this worker now ({need} cards, {ram} GB RAM)", abandon=True); return
        else: devs = free_devs[:need]
        cur = cur or {"job": job, "phase": "fetch", "t_lease": time.time()}
        cur["devices"] = devs
        cancel = threading.Event()
        if self.abandon.is_set(): cancel.set()
        t = threading.Thread(target=self.do_job, args=(cur, cancel, m, own, bool(cur.get("resumed"))), daemon=True)
        with self.lock: self.running[job["id"]] = {"thread": t, "cancel": cancel, "devices": devs, "ram": ram}
        t.start()

    def resume(self, cur):
        j = cur["job"]
        if not self.ids_ok(j):
            log(f"state file for job {str(j.get('id'))[:80]!r}: bad id or lease id; ignoring it"); return
        try: hb = self.c.api("POST", "heartbeat", {"job_id": j["id"], "lease_id": j["lease_id"]}, tries=5)
        except Exception as e: hb = {"ok": False, "cancel": True, "error": repr(e)}
        if hb.get("cancel"):
            log(f"job {j['id']}: lease gone after restart ({hb}); dropping it")
            self.cleanup(cur); return
        log(f"job {j['id']}: resuming at phase {cur['phase']}")
        cur["resumed"] = True
        self.start(j, cur)

    def heartbeat_loop(self, job, done, cancel):
        while not done.wait(job.get("heartbeat_s", 30)):
            self.check_stop_file()
            try:
                hb = self.c.api("POST", "heartbeat", {"job_id": job["id"], "lease_id": job["lease_id"]}, tries=3)
                if hb.get("cancel"):
                    log(f"job {job['id']}: server says stop ({hb.get('state')}, job {hb.get('job_state')})"); cancel.set()
            except APIError as e:
                if e.code in (401, 403): cancel.set()
            except Exception as e:
                log(f"heartbeat failed: {e!r}")

    def cur_path(self, job): return os.path.join(self.a.state, "current", f"{job['id']}-{job['lease_id'][:8]}.json")
    def save(self, cur): write_json_atomic(self.cur_path(cur["job"]), cur)

    def dirs(self, job):
        d = os.path.join(self.a.state, "jobs", f"{job['id']}-{job['lease_id'][:8]}")
        return d, os.path.join(d, "out"), os.path.join(d, "stage")

    def cleanup(self, cur):
        d, _, _ = self.dirs(cur["job"])
        shutil.rmtree(d, ignore_errors=True)
        if os.path.exists(self.cur_path(cur["job"])): os.remove(self.cur_path(cur["job"]))

    def fail(self, job, err, **kw):
        try: self.c.api("POST", "fail", {"job_id": job["id"], "lease_id": job["lease_id"], "error": err[-1500:], **kw}, tries=5)
        except Exception as e: log(f"could not report the failure: {e!r}")

    def do_job(self, cur, cancel, m, r, resumed=False):
        """one job in its own thread (validated by start()): fetch, run, upload, complete"""
        job = cur["job"]
        done = threading.Event()
        hb = threading.Thread(target=self.heartbeat_loop, args=(job, done, cancel), daemon=True); hb.start()
        st = {"job": job["id"], "type": job["type"], "resumed": resumed, "devices": cur["devices"]}
        pinned = False
        try:
            self.cache.pin(job["inputs"]); pinned = True
            if resumed: self.meter.reserve(job["id"], self.cache.missing_bytes(job["inputs"]) if cur["phase"] == "fetch" else 0,
                                           (job.get("expect") or {}).get("up", 0))   # a running job always finishes
            else:
                why = self.budget_check(job)
                if why:
                    log(f"job {job['id']}: {why}: handing it back"); self.lease_pause = time.time() + 60
                    self.fail(job, why, abandon=True); self.cleanup(cur); return
            self.save(cur)
            jd, out, stage = self.dirs(job); os.makedirs(out, exist_ok=True)
            # 2. fetch
            t0 = time.time()
            paths, got = self.cache.ensure(job["inputs"], lambda i, part: self.fetch(i, part, cancel, job["id"]), cancel)
            st.update(fetch_s=round(time.time() - t0, 2), down_bytes=got, input_bytes=sum(i["size"] for i in job["inputs"]),
                      evicted=self.cache.evicted)
            if cancel.is_set(): raise InterruptedError("cancelled")
            # 3. run
            if cur["phase"] in ("fetch", "run"):
                cur["phase"] = "run"; self.save(cur)
                ctx = JobContext(self, job, paths, out, stage, cancel, cur["devices"])
                t0 = time.time()
                with GpuLock(self.a.gpu_lock if r["gpu"] else None, cancel, cur["devices"]):
                    t1 = time.time()
                    result = m.run(job["params"], ctx)
                st.update(lock_wait_s=round(t1 - t0, 2), run_s=round(time.time() - t1, 2))
                cur.update(phase="upload", result=result); self.save(cur)
            # 4. upload + complete
            t0 = time.time()
            up = self.upload_and_complete(job, out, cur.get("result", {}), m, cancel)
            with self.lock: self.running.pop(job["id"], None); self.jobs_done += 1   # its cards and RAM are free from here
            st.update(upload_s=round(time.time() - t0, 2), **up)
            st["total_s"] = round(time.time() - cur.get("t_lease", time.time()), 2)
            with self.meter.lock: st["link_bytes"] = dict(self.meter.moved.get(job["id"], {}))   # incl. retries
            self.record(**st)
            log(f"job {job['id']} done: {json.dumps(st)}")
            self.cleanup(cur)
        except InterruptedError:
            if self.abandon.is_set():
                log(f"job {job['id']}: abandoned"); self.fail(job, "abandoned by the worker", abandon=True)
            else: log(f"job {job['id']}: cancelled by the server")
            self.cleanup(cur)
        except CacheFull as e:
            log(f"job {job['id']}: {e}"); self.fail(job, f"cache: {e}", abandon=True); self.cleanup(cur)
        except Exception as e:
            tb = traceback.format_exc(); log(f"job {job['id']} FAILED: {e!r}")
            self.fail(job, tb); self.cleanup(cur)
        finally:
            done.set()
            if pinned: self.cache.unpin(job["inputs"])
            self.meter.release(job["id"])
            with self.lock: self.running.pop(job["id"], None)

    def fetch(self, inp, part, cancel, job_id=None):
        """one input into part, resuming from its size with a Range request; returns bytes transferred"""
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > inp["size"]: os.remove(part); have = 0
        if have == inp["size"]: return 0
        r = self.c.call("GET", "/data/" + inp["name"], headers={"Range": f"bytes={have}-"}, raw=True, tries=5)
        if r.status == 200 and have: have = 0; mode = "wb"                    # server ignored the range
        else: mode = "ab"
        n, chunk = 0, self.meter.chunk("down")
        try:
            with open(part, mode) as f:
                while True:
                    b = r.read(chunk)
                    if not b: break
                    f.write(b); n += len(b)
                    self.meter.transfer("down", len(b), job_id, cancel)          # rate cap + budget
                    if cancel.is_set(): raise InterruptedError("cancelled")
        finally: self.meter.budget.flush()
        return n

    def upload_and_complete(self, job, out, result, m, cancel):
        outs = m.outputs(job["params"]); req = m.required_outputs(job["params"])
        names = sorted(n for n in outs if os.path.exists(os.path.join(out, n)))
        if req - set(names): raise RuntimeError(f"job produced no {sorted(req - set(names))}")
        files = {}
        for n in names:
            p = os.path.join(out, n); sz = os.path.getsize(p)
            if sz > outs[n]: raise RuntimeError(f"{n}: {sz} bytes over the cap {outs[n]}")
            files[n] = {"sha256": sha256_file(p), "size": sz}
        sent = 0
        for attempt in range(3):
            for n in names:
                sent += self.upload(job, n, os.path.join(out, n), files[n]["size"], cancel)
            try:
                r = self.c.api("POST", "complete", {"job_id": job["id"], "lease_id": job["lease_id"], "files": files, "result": result})
                return {"up_bytes": sent, "duplicate": r.get("duplicate", False)}
            except APIError as e:
                if e.code == 422 and e.body.get("bad"): log(f"server rejected {e.body['bad']} (sha / size): re-uploading"); continue
                raise
        raise RuntimeError("upload kept failing the sha256 check")

    def upload(self, job, name, path, size, cancel):
        ep = f"upload/{job['id']}/{job['lease_id']}/{name}"
        try: off = self.c.api("GET", ep, tries=5)["size"]
        except APIError as e:
            if e.code == 409 and e.body.get("accepted"): return 0
            raise
        if off > size: off = None                                              # server holds junk: it drops it at complete
        if off is None or off == size: return 0
        sent, chunk = 0, self.a.chunk_mb << 20
        with open(path, "rb") as f:
            f.seek(off)
            while off < size:
                b = f.read(min(chunk, size - off))
                try: r = self.c.api("PUT", f"{ep}?offset={off}", MeteredBody(b, self.meter, job["id"]), tries=5)
                except APIError as e:
                    if e.code == 409 and "size" in e.body: off = e.body["size"]; f.seek(off); continue
                    if e.code == 409 and e.body.get("accepted"): return sent
                    raise
                off = r["size"]; sent += len(b)
                if cancel.is_set() and self.abandon.is_set(): raise InterruptedError("cancelled")
        self.meter.budget.flush()
        return sent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    e = lambda k, d=None: env(k, d, side="worker")                  # $RQAH_<K> (0.2: $RADQUANT_<K>); a flag wins
    ap.add_argument("--server", default=e("SERVER"), help="queue URL ($RQAH_SERVER)")
    ap.add_argument("--token", default=e("TOKEN"), help="worker token ($RQAH_TOKEN; prefer --token-file)")
    ap.add_argument("--token-file", default=e("TOKEN_FILE"), help="token file ($RQAH_TOKEN_FILE); written at registration")
    ap.add_argument("--invite", default=e("INVITE"), help="one-time invite code to register ($RQAH_INVITE)")
    ap.add_argument("--name", default=e("NAME", "worker"), help="worker name at registration ($RQAH_NAME)")
    ap.add_argument("--leaderboard", action="store_true", default=e("LEADERBOARD", "0").lower() in ("1", "true", "yes", "on"),
                    help="at registration: list the name on the public status page ($RQAH_LEADERBOARD=1; later: "
                         "`rqah config set --leaderboard on|off`)")
    ap.add_argument("--state", default=e("STATE"), help="state dir (current jobs, jobs/, cache/, worker.jsonl)")
    ap.add_argument("--cache", help="input cache dir (default STATE/cache)"); ap.add_argument("--cache-gb", type=float, default=float(e("CACHE_GB", 200)))
    ap.add_argument("--job-types", default=e("JOB_TYPES"), help="comma list (default: every implemented type)")
    ap.add_argument("--gpu-name", default=e("GPU_NAME", ""), help="the GPU model (empty, without --gpus / --cards: a CPU-only worker)")
    ap.add_argument("--gpus", default=e("GPUS"), help="device indices this worker uses, e.g. 0,1,2,3 ($RQAH_GPUS)")
    ap.add_argument("--cards", type=int, default=int(e("CARDS", 0)), help="shorthand for --gpus 0..N-1")
    ap.add_argument("--slots", type=int, default=int(e("SLOTS", 0)), help="jobs at once (default: one per card, at least 1)")
    ap.add_argument("--vram-gb", type=float, default=float(e("VRAM_GB", 0)), help="VRAM per card")
    ap.add_argument("--ram-gb", type=float, default=float(e("RAM_GB", 0)), help="host RAM all jobs together may use")
    ap.add_argument("--device", type=int, default=0, help="v0: the one device of a single-card worker")
    ap.add_argument("--gpu-env", default=e("GPU_ENV", "auto"),
                    help="variable that gives each job process its cards: ROCR_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, none "
                         "(pass the device index instead), auto (ROCR if /dev/kfd exists)")
    ap.add_argument("--vram-cap-gb", type=float, default=float(e("VRAM_CAP_GB", 0)), help="cap torch's allocator in job processes (shared card)")
    ap.add_argument("--gpu-lock", default=e("GPU_LOCK"), help="flock this file around every GPU job; {card} = one lock per card")
    ap.add_argument("--python", default=sys.executable); ap.add_argument("--ourq", help="quantize/ourq.py")
    ap.add_argument("--exl3-src", help="the fork's exllamav3_ext sources (ourq --frac-ext)")
    ap.add_argument("--radiance", default=e("RADIANCE", "radiance"), help="the radiance binary (eval_mix, golden_record tp2)")
    ap.add_argument("--mkgold27", default=e("MKGOLD27", os.path.join(REPO, "eval", "gold27", "mkgold27.py")),
                    help="the staged golden recorder (golden_record staged)")
    ap.add_argument("--headroom-mib", type=int, default=int(e("HEADROOM_MIB", 0)), help="radiance --host-headroom-mib (0: the job's default)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment for job processes")
    ap.add_argument("--image", default=e("IMAGE_ID", "local"), help="image / install id reported with results ($RQAH_IMAGE_ID)")
    ap.add_argument("--window", action="append", default=[], metavar="HH:MM-HH:MM",
                    help="local-time windows for new jobs (or $RQAH_WINDOWS, comma-separated; TZ sets the zone)")
    ap.add_argument("--on-stop", choices=("finish", "abandon"), default=e("ON_STOP", "finish")); ap.add_argument("--stop-file", default=e("STOP_FILE"))
    ap.add_argument("--poll", type=float, default=30); ap.add_argument("--idle-exit", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=0); ap.add_argument("--chunk-mb", type=int, default=16)
    bw = ap.add_argument_group("bandwidth (0 = unlimited; all jobs of this worker together)")
    bw.add_argument("--max-down-mbps", type=float, default=float(e("MAX_DOWN_MBPS", 0)), help="download rate cap, Mbit/s ($RQAH_MAX_DOWN_MBPS)")
    bw.add_argument("--max-up-mbps", type=float, default=float(e("MAX_UP_MBPS", 0)), help="upload rate cap, Mbit/s ($RQAH_MAX_UP_MBPS)")
    for d in ("down", "up"):
        for p in ("day", "month"):
            bw.add_argument(f"--max-{d}-gb-{p}", type=float, default=float(e(f"MAX_{d.upper()}_GB_{p.upper()}", 0)),
                            help=f"{'download' if d == 'down' else 'upload'} budget per {'calendar day' if p == 'day' else 'budget month'}, "
                                 f"GB = 1e9 bytes ($RQAH_MAX_{d.upper()}_GB_{p.upper()})")
    bw.add_argument("--budget-reset-day", type=int, default=int(e("BUDGET_RESET_DAY", 1)),
                    help="day of the month (1-28) the monthly budgets start again, local time ($RQAH_BUDGET_RESET_DAY)")
    a = ap.parse_args()
    if not 1 <= a.budget_reset_day <= 28: ap.error("--budget-reset-day: 1..28")
    for k in ("max_down_mbps", "max_up_mbps", "max_down_gb_day", "max_down_gb_month", "max_up_gb_day", "max_up_gb_month"):
        if getattr(a, k) < 0: ap.error(f"--{k.replace('_', '-')}: not negative")
    if not a.server or not a.state: ap.error("--server and --state are required (or $RQAH_SERVER / $RQAH_STATE)")
    if not a.window: a.window = [x for x in (e("WINDOWS", "")).replace(" ", ",").split(",") if x]
    for w_ in a.window:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d", w_): ap.error(f"bad window {w_!r}: HH:MM-HH:MM")
    if a.gpus and not re.fullmatch(r"\d+(,\d+)*", a.gpus): ap.error("--gpus: a comma list of device indices")
    w = Worker(a)
    signal.signal(signal.SIGINT, w.on_signal); signal.signal(signal.SIGTERM, w.on_signal)
    log(f"rad-quant-at-home worker up: types {w.types}, cards {w.gpus or 'none'} ({w.gpu_env}), slots {w.slots}, "
        f"RAM {a.ram_gb} GB, cache {a.cache_gb} GB, windows {a.window or 'always'}"
        + (f", rate caps down {a.max_down_mbps or '-'} / up {a.max_up_mbps or '-'} Mbps" if a.max_down_mbps or a.max_up_mbps else "")
        + (", data budget " + ", ".join(f"{d} {getattr(a, f'max_{d}_gb_{p}'):g} GB/{p}" for d in ("down", "up") for p in ("day", "month")
                                         if getattr(a, f"max_{d}_gb_{p}")) + f" (month from day {a.budget_reset_day})"
           if w.meter.budget.limited() else ""))
    w.loop()


if __name__ == "__main__":
    main()
