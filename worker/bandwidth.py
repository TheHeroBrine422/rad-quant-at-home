# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The donor's bandwidth limits (0.4): rate caps and data budgets for one worker, stdlib only.

Rate caps (--max-down-mbps / --max-up-mbps, megabits per second, 1 Mbps = 125,000 bytes/s): one token bucket per
direction, shared by every job thread of the worker, so the caps hold for the whole box. Fetches (fresh and resumed
Range transfers) pay for the bytes they have read from the socket, uploads for the bytes they hand to it; a bucket
holds at most a quarter second of tokens, so the rate over any transfer longer than a few seconds is the cap.

Data budgets (--max-{down,up}-gb-{day,month}, 1 GB = 1e9 bytes): byte counters per direction for the current
calendar day and month in the donor's local time (TZ), the month starting on --budget-reset-day (1-28, e.g. the
day the ISP's billing month starts). They live in STATE/bandwidth.json (the state volume in the container), written
atomically at least every 4 MB / 2 s and after every transfer, so a restart or a recreated container keeps them.
What a running job will still move (its inputs not fetched yet, its expected upload) is reserved against the budget
at once, so the remaining budget the worker declares when it leases is what is really left for new jobs."""
import os, threading, time
from common.util import read_json, write_json_atomic

MBPS = 125_000                                          # bytes/s per Mbps
GB = 1_000_000_000
DIRS = ("down", "up")
PERIODS = ("day", "month")
SPENT_BYTES = 1 << 20                                   # less than this left: the direction counts as spent


class Bucket:
    """token bucket in bytes; take(n) pays n (may go into debt, then sleeps it off): exact over long transfers"""
    def __init__(self, rate, burst_s=0.25):
        self.rate = float(rate); self.cap = max(self.rate * burst_s, 16384.0)
        self.level, self.t = self.cap, time.monotonic()
        self.lock = threading.Lock()

    def take(self, n, cancel=None):
        with self.lock:
            now = time.monotonic()
            self.level = min(self.cap, self.level + (now - self.t) * self.rate); self.t = now
            self.level -= n
            wait = -self.level / self.rate if self.level < 0 else 0.0
        t_end = time.monotonic() + wait
        while True:
            left = t_end - time.monotonic()
            if left <= 0: return
            if cancel is not None:
                if cancel.wait(min(left, 0.5)): raise InterruptedError("cancelled")
            else: time.sleep(min(left, 0.5))

    def chunk(self):
        """a read size that keeps the pacing smooth: ~0.1 s of the rate, 16 KB .. 1 MB"""
        return int(min(1 << 20, max(16384, self.rate / 10)))


def month_start(lt, reset_day):
    """(year, month) of the budget month that local date lt (struct_time) falls in"""
    y, m = lt.tm_year, lt.tm_mon
    if lt.tm_mday < reset_day: y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return y, m


def local_midnight(y, m, d):
    return time.mktime((y, m, d, 0, 0, 0, 0, 0, -1))


class Budget:
    """persistent per-direction counters for the current day and budget month (see the module docstring)"""
    FILE = "bandwidth.json"

    def __init__(self, state_dir, limits=None, reset_day=1, clock=time.time):
        """limits: {"down_day": bytes or None, "down_month", "up_day", "up_month"}"""
        self.path = os.path.join(state_dir, self.FILE)
        self.limits = {f"{d}_{p}": (limits or {}).get(f"{d}_{p}") for d in DIRS for p in PERIODS}
        if not 1 <= reset_day <= 28: raise ValueError("budget reset day: 1..28")
        self.reset_day, self.clock = reset_day, clock
        self.lock = threading.RLock()
        s = read_json(self.path, {}) or {}
        self.s = {"day": s.get("day"), "month": s.get("month"), **{k: int(s.get(k, 0)) for k in self.limits}}
        self.dirty, self.t_flush, self.unflushed = False, 0.0, 0
        self._roll()
        self.flush(force=True)

    def limited(self, d=None):
        return any(v is not None for k, v in self.limits.items() if d is None or k.startswith(d + "_"))

    def _keys(self, t=None):
        lt = time.localtime(self.clock() if t is None else t)
        y, m = month_start(lt, self.reset_day)
        return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}", f"{y:04d}-{m:02d}-{self.reset_day:02d}"

    def _roll(self):
        day, month = self._keys()
        if self.s["day"] != day:
            self.s["day"] = day; self.dirty = True
            for d in DIRS: self.s[f"{d}_day"] = 0
        if self.s["month"] != month:
            self.s["month"] = month; self.dirty = True
            for d in DIRS: self.s[f"{d}_month"] = 0

    def add(self, d, n):
        if n <= 0: return
        with self.lock:
            self._roll()
            for p in PERIODS: self.s[f"{d}_{p}"] += n
            self.dirty = True; self.unflushed += n
            if self.unflushed >= (4 << 20) or time.time() - self.t_flush >= 2: self.flush()

    def flush(self, force=False):
        with self.lock:
            if not (self.dirty or force): return
            write_json_atomic(self.path, {**self.s, "limits": self.limits, "reset_day": self.reset_day, "updated": self.clock()})
            self.dirty, self.t_flush, self.unflushed = False, time.time(), 0

    def remaining(self, d):
        """bytes left in direction d (the tighter of day and month), None if unlimited"""
        with self.lock:
            self._roll()
            left = [self.limits[f"{d}_{p}"] - self.s[f"{d}_{p}"] for p in PERIODS if self.limits[f"{d}_{p}"] is not None]
            return max(0, min(left)) if left else None

    def next_reset(self, d=None, t=None):
        """when the binding limit of d (or of any direction) next resets: local midnight of the next day, or the next
        budget month's first day"""
        t = self.clock() if t is None else t
        lt = time.localtime(t)
        nd = time.localtime(local_midnight(lt.tm_year, lt.tm_mon, lt.tm_mday) + 36 * 3600)   # tomorrow (DST-safe)
        day_end = local_midnight(nd.tm_year, nd.tm_mon, nd.tm_mday)
        y, m = month_start(lt, self.reset_day)
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
        month_end = local_midnight(y, m, self.reset_day)
        with self.lock:
            out = []
            for dd in ([d] if d else DIRS):
                for p, end in (("day", day_end), ("month", month_end)):
                    lim = self.limits[f"{dd}_{p}"]
                    if lim is not None and lim - self.s[f"{dd}_{p}"] < SPENT_BYTES: out.append(end)
            return max(out) if out else None

    def snapshot(self):
        with self.lock:
            self._roll()
            return {"used": {k: self.s[k] for k in self.limits}, "limits": dict(self.limits), "reset_day": self.reset_day,
                    "day": self.s["day"], "month": self.s["month"]}


class Meter:
    """rate caps + budget + what running jobs still reserve; one per worker"""
    def __init__(self, state_dir, down_mbps=0, up_mbps=0, limits=None, reset_day=1, clock=time.time):
        self.buckets = {"down": Bucket(down_mbps * MBPS) if down_mbps else None, "up": Bucket(up_mbps * MBPS) if up_mbps else None}
        self.budget = Budget(state_dir, limits, reset_day, clock)
        self.resv, self.moved = {}, {}                     # job id -> {dir: bytes still reserved} / {dir: bytes moved}
        self.lock = threading.Lock()

    def chunk(self, d): return self.buckets[d].chunk() if self.buckets[d] else 1 << 20

    def transfer(self, d, n, job=None, cancel=None):
        """n bytes went (or are about to go) over the link in direction d: pay the bucket, count, release the job's
        reservation"""
        if self.buckets[d]: self.buckets[d].take(n, cancel)
        self.budget.add(d, n)
        if job is not None:
            with self.lock:
                m = self.moved.setdefault(job, {"down": 0, "up": 0}); m[d] += n
                r = self.resv.get(job)
                if r: r[d] = max(0, r[d] - n)

    def reserve(self, job, down, up):
        with self.lock: self.resv[job] = {"down": max(0, int(down)), "up": max(0, int(up))}

    def release(self, job):
        with self.lock: self.resv.pop(job, None); self.moved.pop(job, None)
        self.budget.flush()

    def reserved(self, d, but=None):
        with self.lock: return sum(r[d] for j, r in self.resv.items() if j != but)

    def available(self, d, but=None):
        """budget left for new work in direction d after every running job's reservation (but: not this job's); None
        = unlimited"""
        r = self.budget.remaining(d)
        return None if r is None else max(0, r - self.reserved(d, but))

    def declare(self):
        """what the worker tells the server when it leases (None: no budget configured)"""
        if not self.budget.limited(): return None
        out = {d: self.available(d) for d in DIRS}
        spent = [d for d in DIRS if out[d] is not None and self.budget.remaining(d) < SPENT_BYTES]
        out["spent"] = spent
        nr = self.budget.next_reset()
        if nr: out["reset"] = int(nr)
        return out
