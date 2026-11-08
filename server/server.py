#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""server.py --db QUEUE.sqlite --data DATAROOT --spool DIR --results DIR [--bind 127.0.0.1] [--port 8740]

The rad-quant-at-home job queue (stdlib only). Workers authenticate with a bearer token (from /register, with an
invite code or, with open registration, without one: then on probation; or admin.py add-worker). Plain HTTP: put it
behind a TLS terminator or a tunnel before anything leaves localhost / LAN. Every flag can also come from the environment as RQAH_<FLAG> (e.g. RQAH_DB, RQAH_PORT,
RQAH_ADMIN_TOKEN_FILE; the 0.2 names RADQUANT_<FLAG> are still read in 0.3); a flag on the command line wins. The
container image (docker/server/) configures it that way.

Scheduling (0.2): a worker declares its cards, VRAM per card, host RAM and disk; every job type declares what one job
needs (jobs/*.requires). The server leases a job only if the worker's declaration covers it and enough cards / RAM are
free now (it counts the worker's active leases). A gang job (cards > 1) that a busy multi-card worker could run gets a
reservation on that worker: it is leased nothing else until enough of its cards drain, then the gang job; another
worker that can run it at once still may. Trusted-only types go to workers flagged trusted; spot-check re-runs to
verifiers only (server/verify.py); quarantined workers get nothing.

Scheduling (0.4, server/sched.py, docs/SCHED.md): priority classes strictly in order; inside a class whole levels
first, then the jobs whose inputs the worker caches (its lease request reports its cache: sha256 prefixes, resent
when they change). A worker may declare a data budget (bytes left to download / upload); it is offered only jobs whose
uncached inputs and expected upload (jobs/*.upload_bytes) fit. A higher-class job a busy worker could run but keeps
passing over drains that worker once it has waited --class-wait-s.

  POST /api/v0/register   {invite?, name, leaderboard?}               -> {worker_id, token, probation}

Open registration (0.5, docs/VERIFY.md "Identity"): /register without an invite makes a worker on probation (score-mode
quant jobs only, one lease at a time, until --probation-pass of its results passed; eval_mix and every type whose check
takes our hardware after --eval-unlock), rate-limited per client address (--reg-per-ip-day, IPv6 per /64) and in total
(--reg-global-hour): 429 with retry_s. --no-open-registration: invites only (403). The client address is the peer's,
or, when the peer is one of --trusted-proxy, the rightmost X-Forwarded-For entry that is not a trusted proxy.
  POST /api/v0/lease      {caps}                                      -> {job: {...}} | {job: null, retry_s}
  POST /api/v0/heartbeat  {job_id, lease_id}                          -> {ok, cancel, lease_expires}
  GET  /api/v0/upload/J/LEASE/NAME                                    -> {size} (resume point)
  PUT  /api/v0/upload/J/LEASE/NAME?offset=N   body: bytes             -> {size}
  POST /api/v0/complete   {job_id, lease_id, files: {name: {sha256, size}}, result}
  POST /api/v0/fail       {job_id, lease_id, error, abandon, reject}
  GET  /data/RELPATH      (Range: bytes=a-[b])  only files in the manifest that a job leased by this worker lists
  GET  /api/v0/status
  GET  /api/v0/me[/jobs|/points|/events]  POST /api/v0/me/profile   the donor's own view (server/donor.py;
                          the donor CLI client/rqah.py); /api/v0/me/link was removed: 410
  GET  /  (/index.html)   no auth: the public status page (server/statuspage.py), rebuilt every --status-interval s;
                          with --status-dir also written there for a static host
  GET  /healthz           no auth: 200 if the database answers (container healthcheck)
  GET  /api/v0/version    no auth (0.5): {api, server, worker: {current, min, images, notes, published}} -- what
                          `rqah update` moves to (images: {stack: {image, digest}} from the release manifest)
  GET  /api/v0/admin/summary  read-only queue overview, only with --admin-token-file, bearer = that token
  POST /api/v0/search/batch           an eval_mix batch (server/batch.py)      } only with --search-token-file,
  GET  /api/v0/search/batch/ID        its status and per-candidate results    } bearer = that token; enqueues
  GET  /api/v0/search/batch/ID/rows/C candidate C's rows ([rows, 4] f32)       } eval_mix jobs on public goldens only
  POST /api/v0/search/batch/ID/adopt  {candidates: [C, ...]}: the search adopts them; each gets a full trusted re-run

Nothing else that changes the queue goes over HTTP: admin.py works on the database and the data root directly, on
this host (or in the server container: `rqah admin ...`).

Versions (0.5, README "Updating", RUNBOOK.md "Upgrades"): a worker sends its release in X-RQAH-Version (before 0.5:
only in its user agent, rad-quant-at-home-worker/0.4; 0.3 said 0.2). The server records it (workers.version), and
every lease answer, /status and /me carry {"update": {current, min, images, notes, published}}: current is the
release in the release manifest (--release-manifest, default release.json next to the database: common/release.py,
written by the release workflow, installed with `admin.py release-fetch`; re-read when it changes), else
--worker-current, else this server's release; images are that release's image digests per stack (only when the
manifest is the current release). --worker-min (default WORKER_MIN) is the oldest release still served. A worker
below it gets 426 on /lease with the way to update; the jobs it holds still heartbeat, upload and complete. API v0 changes are additive: the previous release's workers and CLI keep
working against this server (tests/upgrade_e2e.py runs the 0.4.0 worker against it). A restart extends the leases
that were active by the downtime (Store.after_restart), so a server outage longer than --lease-s loses no work."""
import argparse, errno, hashlib, hmac, json, os, re, secrets, shutil, signal, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs, unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import API, NAME_RE, VERSION, VERSION_HEADER, VERSION_RE, safe_relpath, sha256_file, read_json, env, vtuple
from common.schema import validate, SchemaError
import jobs
from jobs import golden_common as G
from server.store import Store, SchemaTooNew
from common import release as RL
from server import verify as V, batch as B, donor as D, statuspage as SP
from server import sched as SCH
from server.sched import norm_req, job_filter

CHUNK_MAX = 64 << 20
WORKER_MIN = "0.4.0"                   # the oldest worker release this server leases to by default (one release back)
UPDATE_HOW = ("on the host: `rqah update` (python3 client/setup.py --update), or pull the new image tag and recreate the "
              "container with the same state volume (README \"Updating\")")
IMPLAUSIBLE_TO_VERIFIER = 2            # #4: after this many donors' implausible results a job goes to our verifiers
JSON_MAX = 1 << 20
BATCH_MAX = 32 << 20
CAPS = {"type": "object", "required": ["job_types"], "properties": {
    "job_types": {"type": "array", "maxItems": 32, "items": {"type": "string", "pattern": "[a-z0-9_]{1,32}"}},
    "gpu": {"type": "boolean"}, "gpu_name": {"type": "string", "maxLength": 128},
    "cards": {"type": "integer", "minimum": 0, "maximum": 64, "description": "GPUs this worker runs jobs on"},
    "slots": {"type": "integer", "minimum": 1, "maximum": 64, "description": "jobs at once (default: cards, at least 1)"},
    "vram_gb": {"type": "number", "minimum": 0, "maximum": 4096, "description": "per card"},
    "ram_gb": {"type": "number", "minimum": 0, "maximum": 65536, "description": "host RAM for all its jobs together"},
    "disk_gb": {"type": "number", "minimum": 0, "maximum": 1e6}, "image": {"type": "string", "maxLength": 128},
    "running": {"type": "array", "maxItems": 64, "items": {"type": "integer"}},
    "versions": {"type": "object", "additionalProperties": {"type": "integer"}},
    # 0.4: the data budget left (bytes, null = unlimited) and the input cache report (docs/SCHED.md)
    "budget": {"type": "object", "properties": {
        "down": {"type": ["integer", "null"], "minimum": 0}, "up": {"type": ["integer", "null"], "minimum": 0},
        "spent": {"type": "array", "maxItems": 2, "items": {"enum": ["down", "up"]}},
        "reset": {"type": "integer", "minimum": 0, "description": "when the binding limit resets (unix time)"}}},
    "cache": {"type": "object", "required": ["rev"], "properties": {
        "rev": {"type": "string", "pattern": "[0-9a-f]{16}"},
        "shas": {"type": "array", "maxItems": 32768, "items": {"type": "string", "pattern": "[0-9a-f]{16}"},
                 "description": "sha256[:16] of every complete blob in the worker's input cache; sent when rev changes"}}}}}


class HTTPError(Exception):
    def __init__(self, code, msg, **extra): super().__init__(msg); self.code, self.msg, self.extra = code, msg, extra


def _dir_bytes(path):
    n = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            try: n += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return n


class Queue:
    def __init__(self, a):
        self.a = a
        self.store = Store(a.db)
        self.data = os.path.abspath(a.data); self.spool = os.path.abspath(a.spool); self.results = os.path.abspath(a.results)
        for d in (self.spool, self.results): os.makedirs(d, exist_ok=True)
        self._man, self._man_mtime = {}, None
        self.lease_s, self.hb_s = a.lease_s, a.heartbeat_s
        self.chunk_max = int(a.chunk_max_mb * (1 << 20))
        self.upload_max = int(a.upload_max_mb * (1 << 20)) if a.upload_max_mb else None
        self.spool_max = int(a.spool_max_gb * 1e9) if a.spool_max_gb else None
        self.refail_s = float(getattr(a, "refail_s", 3600))
        self.cap_x, self.cap_min_s = float(getattr(a, "lease_cap_x", 3.0)), float(getattr(a, "lease_cap_min_s", 7200))
        # bytes in the spool, cached: an os.walk per upload chunk grew with the orphans; uploads add, fail / complete
        # subtract, and the reaper's recount (sweep_spool, every tick) keeps it honest
        self.spool_lock = threading.Lock()
        self.spool_used = _dir_bytes(self.spool)
        self.admin_hash = self._token_hash(a.admin_token_file, "admin")
        self.search_hash = self._token_hash(a.search_token_file, "search")
        self.policy = V.Policy(a.spot_rate, a.spot_first, rates={"eval_mix": getattr(a, "eval_spot_rate", 0.08)},
                               docs=getattr(a, "eval_spot_docs", 3), data=self.data)
        self.store.reserve_max_s = a.reserve_max_s; self.store.reserve_idle_s = max(4 * a.lease_s, 60)
        self.piece_cache = {}
        self.up_memo = {}                                   # job id -> expected upload bytes (sched.Transfers)
        self.cache_reports = {}                             # worker id -> (rev, set of sha256[:16])
        self.page, self.page_t = None, 0.0
        self.campaign = a.campaign or os.path.join(os.path.dirname(os.path.abspath(a.db)), "campaign.json")
        self.probation = (int(getattr(a, "probation_pass", SCH.PROBATION_PASS)), int(getattr(a, "eval_unlock", SCH.EVAL_UNLOCK)))
        self.store.probation = self.probation                # the donor API reports the tiers with the same thresholds
        self.proxies = {x.strip() for x in (getattr(a, "trusted_proxy", None) or "").split(",") if x.strip()}
        self.worker_current_flag = getattr(a, "worker_current", None) or None
        self.worker_min = getattr(a, "worker_min", None) or WORKER_MIN
        self.release_path = getattr(a, "release_manifest", None) or os.path.join(os.path.dirname(os.path.abspath(a.db)), "release.json")
        self._rel, self._rel_key, self._meta_cur = None, None, None
        for k, v in (("worker-current", self.worker_current_flag), ("worker-min", self.worker_min)):
            if v is not None and not VERSION_RE.fullmatch(v): sys.exit(f"--{k}: X.Y.Z, not {v!r}")
        if vtuple(self.worker_min) > vtuple(self.worker_current): sys.exit("--worker-min is above the current worker release")
        self.store.meta_set("worker_min", self.worker_min); self.sync_meta()
        self.restart = self.store.after_restart(self.lease_s)

    # ------------------------------------------------------------ versions (0.5)
    @staticmethod
    def client_version(headers):
        """a worker's release: its X-RQAH-Version header, else (before 0.5) the X.Y of its user agent as X.Y.0 (0.3
        workers sent 0.2); None for anything else (the CLI, curl)"""
        v = (headers.get(VERSION_HEADER) or "").strip()
        if VERSION_RE.fullmatch(v): return v
        m = re.fullmatch(r"rad-quant-at-home-worker/(\d{1,4}\.\d{1,4})", (headers.get("User-Agent") or "").strip())
        return m.group(1) + ".0" if m else None

    def release(self):
        """the release manifest (common/release.py) or None; re-read when the file changes, the last good one kept
        when it turns unreadable (a warning in the log)"""
        try: st = os.stat(self.release_path); key = (st.st_mtime_ns, st.st_size)
        except OSError: key = None
        if key != self._rel_key:
            self._rel_key = key
            if key is None: self._rel = None
            else:
                try:
                    with open(self.release_path, "rb") as f: self._rel = RL.loads(f.read(1 << 20))
                    print(f"release manifest: {self._rel['version']} ({', '.join(sorted(self._rel['images']))})", flush=True)
                except (OSError, ValueError) as e:
                    print(f"release manifest {self.release_path} refused, keeping the previous one: {e}", flush=True)
        return self._rel

    @property
    def worker_current(self):
        """the release workers should run: --worker-current, else the release manifest's, else this server's"""
        if self.worker_current_flag: return self.worker_current_flag
        rel = self.release()
        return rel["version"] if rel else VERSION

    def sync_meta(self):
        """meta worker_current (the status page reads it) follows the manifest"""
        cur = self.worker_current
        if cur != self._meta_cur: self.store.meta_set("worker_current", cur); self._meta_cur = cur

    def update_info(self, ver=None):
        """what every lease answer, /status and /me carry: the release workers should run, the oldest still served,
        and (from the release manifest, when it is that release) its image digests per stack and its notes page"""
        cur = self.worker_current
        u = {"current": cur, "min": self.worker_min}
        rel = self.release()
        if rel and rel["version"] == cur: u.update(RL.advertised(rel))
        if ver:
            u["yours"] = ver
            u["available"] = vtuple(ver) < vtuple(cur)
            u["supported"] = vtuple(ver) >= vtuple(self.worker_min)
        return u

    def version_report(self):
        """queued / leased jobs whose version this server's code does not accept, or that are retired -> lines (startup
        warning; admin.py versions has the full table)"""
        ret, out = self.store.retired(), []
        for r in self.store.q("SELECT type, version, state, COUNT(*) n FROM jobs WHERE state IN ('queued','leased') "
                              "GROUP BY type, version, state ORDER BY type, version"):
            m = jobs.REGISTRY.get(r["type"])
            if m is None or r["version"] not in jobs.served(m) or r["version"] in ret.get(r["type"], ()):
                out.append(f"{r['n']} {r['state']} {r['type']} v{r['version']} jobs: this server "
                           + (f"runs {r['type']} v{m.VERSION} and does not accept v{r['version']}" if m else "does not know the type")
                           + (" (retired)" if m and r["version"] in ret.get(r["type"], ()) else "")
                           + " -- they are not leased (admin.py versions; retire --migrate)")
        return out

    def tier(self, q, wr):
        """(tier, passed quant results) of worker row wr (sched.tier)"""
        if not wr["probation"] or wr["verifier"]: return "full", None
        n = SCH.passed_quant(q, wr["id"])
        return SCH.tier(wr, n, *self.probation), n

    def register(self, b, addr):
        """POST /register: with an invite as before; without one (open registration) on probation, rate-limited"""
        name = str(b.get("name", ""))[:64]
        if not NAME_RE.match(name): raise HTTPError(400, "bad name")
        inv, lb = b.get("invite"), b.get("leaderboard") is True
        if inv:
            r = self.store.register(str(inv), name, leaderboard=lb)
            if r is None: raise HTTPError(403, "bad invite")
            return {"worker_id": r[0], "token": r[1], "probation": False}
        if not getattr(self.a, "open_registration", True):
            raise HTTPError(403, "registration needs an invite on this server (open registration is off)")
        wid, x = self.store.register_open(name, addr, getattr(self.a, "reg_per_ip_day", 5), getattr(self.a, "reg_global_hour", 60), lb)
        if wid is None: raise HTTPError(429, "too many registrations without an invite; try again later", retry_s=x)
        return {"worker_id": wid, "token": x, "probation": True}

    def build_page(self):
        """the status page from the database (kept in memory; / serves the last good one)"""
        page = SP.build(self.store, SP.load_campaign(self.campaign))
        self.page, self.page_t = page, time.time()
        if self.a.status_dir: SP.write(os.path.join(self.a.status_dir, "index.html"), page)
        return page

    @staticmethod
    def _token_hash(path, what):
        if not path: return None
        tok = open(path).read().strip()
        if len(tok) < 32: sys.exit(f"{path}: {what} token shorter than 32 characters")
        return hashlib.sha256(tok.encode()).digest()

    def manifest(self):
        p = os.path.join(self.data, "manifest.json")
        m = os.path.getmtime(p) if os.path.exists(p) else None
        if m != self._man_mtime:
            try:
                self._man = (read_json(p, {}) or {}).get("files", {}); self._man_mtime = m; self.piece_cache = {}; self.up_memo = {}
            except (OSError, ValueError) as e:              # e.g. written 0600 by another uid: keep the last good one
                print(f"manifest.json unreadable, keeping the previous one ({len(self._man)} files): {e!r}", flush=True)
        return self._man

    def data_path(self, rel):
        if not safe_relpath(rel) or rel not in self.manifest(): raise HTTPError(404, "not in manifest")
        return os.path.join(self.data, rel)

    def private(self, rel):
        """goldens/<g>/... and corpora/<c>/... are served only while their marker says public (golden_common)"""
        parts = rel.split("/")
        if parts[0] in ("goldens", "corpora") and len(parts) >= 3:
            mk = read_json(os.path.join(self.data, parts[0], parts[1], "golden.json" if parts[0] == "goldens" else "corpus.json"))
            ok, why = G.marker_ok(mk)
            return None if ok else why
        return None

    # ------------------------------------------------------------ lease
    def lease(self, w, caps, ver=None):
        """docs/SCHED.md: priority classes strictly in order; inside a class whole levels first and cached inputs
        first; only jobs that fit the worker's declared data budget; a gang job, or a higher-class job that has
        waited --class-wait-s, drains a busy worker that could run it. ver: the worker's release (client_version);
        below --worker-min: 426"""
        try: validate(caps, CAPS, "caps")
        except SchemaError as e: raise HTTPError(400, str(e))
        if ver:
            with self.store.tx() as db: db.execute("UPDATE workers SET version=?, last_seen=? WHERE id=?", (ver, time.time(), w["id"]))
            if vtuple(ver) < vtuple(self.worker_min):
                raise HTTPError(426, f"this worker runs rad-quant-at-home {ver}; the queue needs {self.worker_min} or newer "
                                f"(current {self.worker_current}). It gets no new jobs until updated: {UPDATE_HOW}",
                                update=self.update_info(ver), retry_s=3600)
        r = self._pick(w, caps)
        r["update"] = self.update_info(ver)
        return r

    def _pick(self, w, caps):
        cr = caps.get("cache")
        if cr is not None:
            if "shas" in cr: self.cache_reports[w["id"]] = (cr["rev"], frozenset(cr["shas"]))
            elif self.cache_reports.get(w["id"], (None,))[0] != cr["rev"]:
                return {"job": None, "retry_s": 0, "cache_unknown": True}           # e.g. after a server restart
        cached = self.cache_reports.get(w["id"], (None, frozenset()))[1] if cr is not None else frozenset()
        budget = caps.get("budget")
        stored = dict(caps)
        if cr is not None: stored["cache"] = {"rev": cr["rev"], "n": len(cached)}
        self.store.expire()
        cards = caps.get("cards", 1 if caps.get("gpu") else 0)
        slots = min(caps.get("slots", max(cards, 1)), self.a.max_leases)
        now = time.time()
        with self.store.tx() as db:
            db.execute("UPDATE workers SET last_seen=?, caps=? WHERE id=?", (now, json.dumps(stored), w["id"]))
            wr = db.execute("SELECT * FROM workers WHERE id=?", (w["id"],)).fetchone()
            if wr["quarantined"]:
                return {"job": None, "retry_s": 3600, "quarantined": True, "why": "this worker is quarantined (a result failed "
                        "verification); ask the operator"}
            act = db.execute("SELECT j.requires FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.worker_id=? AND l.state='active'",
                             (w["id"],)).fetchall()
            tr_, passed = self.tier(lambda sql, args: db.execute(sql, args).fetchall(), wr)
            if tr_ == "probation": slots = 1
            if len(act) >= slots:
                return {"job": None, "retry_s": 30, "why": "all slots busy" if tr_ != "probation" else
                        f"probation: one job at a time until {self.probation[0]} quantization results pass ({passed} so far)"}
            used = [norm_req(json.loads(r["requires"])) for r in act]
            free_cards = cards - sum(r["cards"] for r in used); free_ram = caps.get("ram_gb", 0) - sum(r["ram_gb"] for r in used)

            # verifiers are never excluded: a re-run every verifier failed once would stay queued forever (verify_state='pending')
            # one attempt per worker per job (#5), for --refail-s after that lease ended (donors switch machines off
            # and on; a transient error should not bench a worker from a job for good); 0 = until the job is requeued
            win = "" if self.refail_s <= 0 else f" AND COALESCE(l.ended, l.issued) > {time.time() - self.refail_s:.3f}"
            failed = frozenset() if wr["verifier"] else {r["job_id"] for r in db.execute(
                "SELECT DISTINCT l.job_id FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.worker_id=? AND j.state='queued' "
                "AND l.state IN ('failed','rejected','expired') AND l.issued > COALESCE(j.requeued, 0)" + win, (w["id"],))}
            if not wr["verifier"]:                          # #7: abandon + lease again does not restart the deadline
                failed = set(failed) | {jid for jid, t0 in self._held(db, w["id"]).items()
                                        if (c := self._cap_of(db, jid)) is not None and t0 + c <= now}
            eligible = job_filter(wr, caps, failed, SCH.donor_groups(db, wr), tr_, self.store.retired())
            tr = SCH.Transfers(self.manifest(), self.up_memo)

            def fits_now(r): return r["cards"] <= free_cards and r["ram_gb"] <= free_ram

            def affordable(j):
                if not budget and not cached: return j["input_bytes"], 0       # nothing to weigh: skip the JSON
                d, have = tr.down(j, cached)
                return (d, have) if SCH.fits_budget(budget, d, tr.up(j)) else None

            def drain(j, r, why):
                db.execute("INSERT INTO reservations VALUES(?,?,?)", (j["id"], w["id"], now))
                self.store.event(db, "reserve", j["id"], w["id"], detail={"cards": r["cards"], "free_cards": free_cards, "why": why})
                return {"job": None, "retry_s": min(self.a.retry_s, 10), "draining": j["id"],
                        "why": f"draining for {'gang ' if r['cards'] > 1 else ''}job {j['id']} ({r['cards']} cards, {r['ram_gb']:g} GB RAM)"}

            res = db.execute("SELECT * FROM reservations WHERE worker_id=?", (w["id"],)).fetchone()
            if res:
                j = db.execute("SELECT * FROM jobs WHERE id=? AND state='queued'", (res["job_id"],)).fetchone()
                r = eligible(j) if j else None
                if r is not None and affordable(j) is None: r = None
                if r is None: db.execute("DELETE FROM reservations WHERE job_id=?", (res["job_id"],))
                elif fits_now(r): return self._grant(db, w, j, r, now, tr, cached)
                else: return {"job": None, "retry_s": min(self.a.retry_s, 10), "draining": j["id"],
                              "why": f"draining for job {j['id']} ({r['cards']} cards)"}
            over, starving = 0, None
            reserved = {r[0] for r in db.execute("SELECT job_id FROM reservations")}
            reserved |= {r[0] for r in db.execute("SELECT job_id FROM reserve_lapsed WHERE worker_id=?", (w["id"],))}   # #8
            for (prio,) in db.execute("SELECT DISTINCT priority FROM jobs WHERE state='queued' ORDER BY priority DESC").fetchall():
                cands = []
                for j in db.execute("SELECT * FROM jobs WHERE state='queued' AND priority=? ORDER BY id LIMIT ?",
                                    (prio, self.a.class_scan)).fetchall():
                    r = eligible(j)
                    if r is None: continue
                    t = affordable(j)
                    if t is None: over += 1; continue
                    cands.append({"j": j, "r": r, "down": t[0], "cached": t[1], "prio": prio})
                if not cands: continue
                ranks = SCH.level_ranks(db, [c["j"]["level"] for c in cands])
                for c in cands: c["rank"] = ranks.get(c["j"]["level"], c["j"]["id"])
                cands.sort(key=SCH.order_key(bool(budget) and budget.get("down") is not None))
                for c in cands:
                    j, r = c["j"], c["r"]
                    if fits_now(r):
                        if starving is not None and starving["prio"] > prio: return drain(starving["j"], starving["r"], "class-wait")
                        return self._grant(db, w, j, r, now, tr, cached)
                    if j["id"] in reserved: continue
                    if r["cards"] > 1 and now - j["created"] >= self.a.gang_wait_s: return drain(j, r, "gang")
                    if starving is None and now - (j["updated"] or j["created"]) >= self.a.class_wait_s: starving = c
        if over:
            return {"job": None, "retry_s": max(self.a.retry_s, 60), "over_budget": over,
                    "why": f"{over} queued jobs this worker could run need more data than its budget has left"}
        return {"job": None, "retry_s": self.a.retry_s}

    def lease_cap(self, jtype, params):
        """#7: the longest a lease may be held, renewed or not: --lease-cap-x times the type's reference runtime
        (lumens(p) = R9700 card-minutes), at least --lease-cap-min-s; None = no cap (--lease-cap-x 0)"""
        if self.cap_x <= 0: return None
        try: lum = float(jobs.get(jtype).lumens(params))
        except Exception: lum = 0.0
        return max(self.cap_min_s, self.cap_x * 60 * lum)

    def _cap_of(self, db, job_id):
        j = db.execute("SELECT type, params FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self.lease_cap(j["type"], json.loads(j["params"])) if j else None

    @staticmethod
    def _held(db, wid, job_id=None):
        """{job id: when this worker first leased it} over its abandoned leases of still-queued jobs since their last
        requeue: a worker's leases of one job share one deadline"""
        q = ("SELECT l.job_id, MIN(l.issued) t0 FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.worker_id=? AND "
             "l.state='abandoned' AND l.issued > COALESCE(j.requeued, 0)")
        a = [wid]
        if job_id is not None: q += " AND l.job_id=?"; a.append(job_id)
        return {r["job_id"]: r["t0"] for r in db.execute(q + " GROUP BY l.job_id", a)}

    def _grant(self, db, w, j, r, now, tr=None, cached=frozenset()):
        lid = secrets.token_hex(16)
        cap = self.lease_cap(j["type"], json.loads(j["params"]))
        deadline = None if cap is None else min(now, self._held(db, w["id"], j["id"]).get(j["id"], now)) + cap
        exp = now + self.lease_s if deadline is None else min(now + self.lease_s, deadline)
        db.execute("DELETE FROM reservations WHERE job_id=?", (j["id"],))
        db.execute("INSERT INTO leases(id, job_id, worker_id, issued, expires, state, deadline) VALUES(?,?,?,?,?, 'active', ?)",
                   (lid, j["id"], w["id"], now, exp, deadline))
        db.execute("UPDATE jobs SET state='leased', attempts=attempts+1, lease_id=?, lease_owner=?, lease_expires=?, "
                   "updated=? WHERE id=?", (lid, w["id"], exp, now, j["id"]))
        self.store.event(db, "lease", j["id"], w["id"], lid)
        man = self.manifest(); params = json.loads(j["params"])
        tr = tr or SCH.Transfers(man, self.up_memo)
        return {"job": {"id": j["id"], "type": j["type"], "version": j["version"], "params": params,
                        "expect": {"down": tr.down(j, cached)[0], "up": tr.up(j)},
                        "inputs": [{"name": n, "sha256": man[n]["sha256"], "size": man[n]["size"]} for n in json.loads(j["inputs"])],
                        "outputs": jobs.get(j["type"]).outputs(params), "requires": r,
                        "lease_id": lid, "lease_expires": exp, "lease_deadline": deadline, "heartbeat_s": self.hb_s}}

    def _lease(self, db, w, job_id, lease_id):
        l = db.execute("SELECT * FROM leases WHERE id=? AND job_id=?", (str(lease_id), job_id)).fetchone()
        if l is None or l["worker_id"] != w["id"]: raise HTTPError(403, "not your lease")
        j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return l, j

    def heartbeat(self, w, b):
        now = time.time()
        with self.store.tx() as db:
            l, j = self._lease(db, w, int(b["job_id"]), b["lease_id"])
            db.execute("UPDATE workers SET last_seen=? WHERE id=?", (now, w["id"]))
            if l["state"] != "active" or j["state"] in ("accepted", "cancelled", "failed"):
                late_ok = l["state"] == "expired" and j["state"] in ("queued", "leased")
                return {"ok": False, "cancel": not late_ok, "state": l["state"], "job_state": j["state"]}
            if l["deadline"] is not None and now >= l["deadline"]:
                # #7: held as long as the job type may take (x3, >= 2 h): the job goes back to the queue. The worker may
                # still finish: its result counts if it arrives before another one is accepted
                self.store.expire_lease(db, l, now, why="past its deadline")
                return {"ok": False, "cancel": False, "state": "expired", "job_state": "queued", "deadline": True}
            exp = now + self.lease_s if l["deadline"] is None else min(now + self.lease_s, l["deadline"])
            db.execute("UPDATE leases SET expires=? WHERE id=?", (exp, l["id"]))
            db.execute("UPDATE jobs SET lease_expires=? WHERE id=?", (exp, j["id"]))
            return {"ok": True, "cancel": False, "lease_expires": exp, "lease_deadline": l["deadline"]}

    def fail(self, w, b):
        with self.store.tx() as db:
            l, j = self._lease(db, w, int(b["job_id"]), b["lease_id"])
            if l["state"] not in ("active", "expired"): return {"ok": True, "ignored": l["state"]}
            kind = "abandon" if b.get("abandon") else "reject" if b.get("reject") else "fail"
            err = f"{kind}: {str(b.get('error', ''))[:1500]}"
            self.store.end_lease(db, l, {"abandon": "abandoned", "reject": "rejected", "fail": "failed"}[kind],
                                 error=err, count_attempt=kind != "abandon")
            self.store.event(db, kind, j["id"], w["id"], l["id"], {"error": err[:500]})
        d = os.path.join(self.spool, str(int(b["job_id"])), str(b["lease_id"]))
        gone = _dir_bytes(d); shutil.rmtree(d, ignore_errors=True)
        with self.spool_lock: self.spool_used -= gone
        return {"ok": True}

    # ------------------------------------------------------------ uploads
    def _upload_target(self, w, job_id, lease_id, name):
        if not NAME_RE.match(name): raise HTTPError(400, "bad name")
        with self.store.tx() as db:
            l, j = self._lease(db, w, job_id, lease_id)
            if j["state"] == "accepted": raise HTTPError(409, "job already accepted", accepted=True)
            if j["state"] in ("cancelled", "failed") or l["state"] not in ("active", "expired"):
                raise HTTPError(409, f"lease {l['state']}, job {j['state']}")
            outs = jobs.get(j["type"]).outputs(json.loads(j["params"]))
        if name not in outs: raise HTTPError(403, f"{name} is not an output of {j['type']}")
        d = os.path.join(self.spool, str(job_id), lease_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, name), outs[name]

    def upload_size(self, w, job_id, lease_id, name):
        p, _ = self._upload_target(w, job_id, lease_id, name)
        return {"size": os.path.getsize(p) if os.path.exists(p) else 0}

    def upload(self, w, job_id, lease_id, name, offset, n, rfile):
        p, cap = self._upload_target(w, job_id, lease_id, name)
        cur = os.path.getsize(p) if os.path.exists(p) else 0
        if offset != cur: raise HTTPError(409, "offset mismatch", size=cur)
        if self.upload_max is not None: cap = min(cap, self.upload_max)
        if cur + n > cap: raise HTTPError(413, f"{name} over its cap of {cap} bytes")
        if self.spool_max is not None and self.spool_bytes() + n > self.spool_max:
            raise HTTPError(507, "server upload spool is full, retry later", size=cur)
        with open(p, "ab") as f:
            left = n
            try:
                while left:
                    b = rfile.read(min(left, 1 << 20))
                    if not b: break
                    f.write(b); left -= len(b)
            finally:
                with self.spool_lock: self.spool_used += n - left
        if left:
            os.truncate(p, cur)                            # #16: a short chunk is not kept
            with self.spool_lock: self.spool_used -= n - left   # #22's counter: those bytes are gone again
            raise HTTPError(400, "short body", size=cur)
        return {"size": cur + n}

    def spool_bytes(self): return self.spool_used

    def sweep_spool(self):
        """uploads that can no longer complete: whole jobs that are accepted / failed / cancelled, and inside open
        jobs the leases that ended other than by expiry (an expired lease may still finish: the late-result rule)"""
        try: jobs_ = os.listdir(self.spool)
        except OSError: return
        for jd in jobs_:
            if not jd.isdigit(): continue
            j = self.store.q("SELECT state FROM jobs WHERE id=?", (int(jd),))
            if not j or j[0]["state"] in ("accepted", "failed", "cancelled"):
                shutil.rmtree(os.path.join(self.spool, jd), ignore_errors=True); continue
            try: leases_ = os.listdir(os.path.join(self.spool, jd))
            except OSError: continue                       # removed by complete() or fail() meanwhile
            for ld in leases_:
                l = self.store.q("SELECT state FROM leases WHERE id=?", (ld,))
                if not l or l[0]["state"] not in ("active", "expired"):
                    shutil.rmtree(os.path.join(self.spool, jd, ld), ignore_errors=True)
        n = _dir_bytes(self.spool)
        with self.spool_lock: self.spool_used = n

    # ------------------------------------------------------------ completion
    def complete(self, w, b):
        job_id, lease_id = int(b["job_id"]), str(b["lease_id"])
        files, result = b.get("files"), b.get("result", {})
        if not isinstance(files, dict) or not isinstance(result, dict) or len(json.dumps(result)) > 64 << 10:
            raise HTTPError(400, "bad body")
        with self.store.tx() as db:
            l, j = self._lease(db, w, job_id, lease_id)
            if db.execute("SELECT quarantined FROM workers WHERE id=?", (w["id"],)).fetchone()[0]:
                raise HTTPError(403, "this worker is quarantined")
            if j["state"] == "accepted":
                self.store.event(db, "duplicate", job_id, w["id"], lease_id)
                return {"ok": True, "duplicate": True, "yours": j["accepted_lease"] == lease_id}
            if j["state"] in ("cancelled", "failed") or l["state"] not in ("active", "expired"):
                raise HTTPError(409, f"lease {l['state']}, job {j['state']}")
            params = json.loads(j["params"])
        m = jobs.get(j["type"])
        if j["version"] not in jobs.served(m) or j["version"] in self.store.retired().get(j["type"], ()):
            raise HTTPError(409, f"this server no longer accepts {j['type']} version {j['version']} results (it runs version "
                            f"{m.VERSION}); the operator retired or replaced it", retired=True)
        outs = m.outputs(params)
        if set(files) - set(outs): raise HTTPError(400, f"unexpected files {sorted(set(files) - set(outs))}")
        miss = m.required_outputs(params) - set(files)
        if miss: raise HTTPError(400, f"missing outputs {sorted(miss)}")
        d = os.path.join(self.spool, str(job_id), lease_id)
        bad = []
        for n, meta in files.items():                      # hash outside the lock (hundreds of MB)
            p = os.path.join(d, n)
            if not os.path.exists(p) or os.path.getsize(p) != meta.get("size") or sha256_file(p) != meta.get("sha256"):
                bad.append(n)
                if os.path.exists(p):
                    gone = os.path.getsize(p); os.remove(p)
                    with self.spool_lock: self.spool_used -= gone
        if bad:
            with self.store.tx() as db: self.store.event(db, "upload_bad", job_id, w["id"], lease_id, {"files": bad})
            raise HTTPError(422, "uploaded bytes do not match the declared sha256 / size: re-upload", bad=bad)
        check = None
        try:
            info = m.accept(params, {n: os.path.join(d, n) for n in files}, result, lambda rel: self.data_path(rel))
            if j["pred"] and hasattr(m, "plausible") and isinstance(info, dict):    # #4: vs the search's prediction
                verdict, det = m.plausible(json.loads(j["pred"]), info)
                if verdict == "reject" and not w["verifier"]:
                    raise ValueError(f"implausible: mean KL {det['kld_mean']:.6g} is more than {det['band']['worse']:g} x the "
                                     f"predicted {det['pred_kld']:.6g}")
                check = (verdict if verdict != "reject" else "pass", det)          # our own verifier: a fact, not a fake
        except Exception as e:
            with self.store.tx() as db:
                l2 = db.execute("SELECT * FROM leases WHERE id=?", (lease_id,)).fetchone()
                self.store.end_lease(db, l2, "rejected", error=f"result rejected: {e}")
                self.store.event(db, "result_rejected", job_id, w["id"], lease_id, {"error": str(e)[:500]})
                if str(e).startswith("implausible") and not j["needs"]:
                    # two donors agree the mix is far worse than predicted: the prediction is off, not both donors.
                    # Our verifiers settle it (an honest surprise must not fail the job for good)
                    n = db.execute("SELECT COUNT(DISTINCT worker_id) FROM events WHERE job_id=? AND kind='result_rejected' AND "
                                   "detail LIKE '%implausible%' AND t > ?", (job_id, j["requeued"] or 0)).fetchone()[0]
                    if n >= IMPLAUSIBLE_TO_VERIFIER:
                        db.execute("UPDATE jobs SET needs='verifier' WHERE id=?", (job_id,))
                        self.store.event(db, "implausible_to_verifier", job_id, detail={"donors": n})
            gone = _dir_bytes(d); shutil.rmtree(d, ignore_errors=True)
            with self.spool_lock: self.spool_used -= gone
            raise HTTPError(422, f"result rejected: {e}", rejected=True)
        with self.store.tx() as db:
            j = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if j["state"] == "accepted":                   # someone else won the race while we hashed
                self.store.event(db, "duplicate", job_id, w["id"], lease_id)
                return {"ok": True, "duplicate": True, "yours": False}
            if j["state"] in ("cancelled", "failed"): raise HTTPError(409, f"job {j['state']}")
            jd = os.path.join(self.spool, str(job_id)); gone = _dir_bytes(jd)   # this lease's files move out, the rest go
            dst = os.path.join(self.results, str(job_id))
            if os.path.exists(dst): shutil.rmtree(dst)
            try: os.replace(d, dst)
            except OSError as e:                           # spool and results on different filesystems
                if e.errno != errno.EXDEV: raise
                shutil.copytree(d, dst + ".part"); os.replace(dst + ".part", dst); shutil.rmtree(d)
            now = time.time()
            late = j["lease_id"] != lease_id
            db.execute("UPDATE leases SET state='done', ended=? WHERE id=?", (now, lease_id))
            if late and j["lease_id"]: db.execute("UPDATE leases SET state='superseded', ended=? WHERE id=?", (now, j["lease_id"]))
            db.execute("UPDATE jobs SET state='accepted', accepted_lease=?, accepted_by=?, accepted_at=?, result=?, files=?, "
                       "lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=? WHERE id=?",
                       (lease_id, w["id"], now, json.dumps(result), json.dumps(files), now, job_id))
            if check:
                db.execute("UPDATE jobs SET check_state=?, check_detail=? WHERE id=?", (check[0], json.dumps(check[1]), job_id))
                if check[0] == "suspect": self.store.event(db, "suspect", job_id, w["id"], lease_id, check[1])
            self.store.event(db, "accept_late" if late else "accept", job_id, w["id"], lease_id)
        shutil.rmtree(jd, ignore_errors=True)
        with self.spool_lock: self.spool_used -= gone
        try: V.after_accept(self.store, self.results, job_id, self.policy)        # spot-check draw / comparison, points
        except Exception: traceback.print_exc()
        return {"ok": True, "duplicate": False}

    def may_read(self, w, rel):
        """a lease that is active and not past its expiry (the reaper marks it expired up to a tick later)"""
        for r in self.store.q("SELECT j.inputs FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.worker_id=? AND l.state='active' "
                              "AND l.expires>?", (w["id"], time.time())):
            if rel in json.loads(r["inputs"]): return True
        return False

    def status(self, w=None):
        rows = self.store.q("SELECT type, state, COUNT(*) n FROM jobs GROUP BY type, state")
        out = {"jobs": [dict(r) for r in rows], "workers": self.store.q("SELECT COUNT(*) FROM workers")[0][0],
               "update": self.update_info(w["version"] if w is not None else None)}
        if w is not None:                                   # the asking worker's own standing (its points, flags)
            me = self.store.q("SELECT COALESCE(SUM(lumens), 0) FROM points WHERE worker_id=?", (w["id"],))[0][0]
            out["me"] = {"id": w["id"], "name": w["name"], "lumens": me, "trusted": bool(w["trusted"]),
                         "quarantined": bool(w["quarantined"]), "tier": self.tier(self.store.q, w)[0]}
        return out

    def version(self):
        """GET /api/v0/version (no auth): public, like the status page"""
        return {"api": API, "server": VERSION, "worker": self.update_info()}

    def health(self):
        self.store.q("SELECT 1")
        if not os.path.isdir(self.data) or not os.access(self.spool, os.W_OK) or not os.access(self.results, os.W_OK):
            raise HTTPError(503, "data / spool / results not usable")
        return {"ok": True}

    def admin_ok(self, token):
        return self.admin_hash is not None and token is not None and \
            hmac.compare_digest(hashlib.sha256(token.encode()).digest(), self.admin_hash)

    def admin_summary(self):
        """read-only: job counts per level and state, workers, the last 50 events (no tokens, no params)"""
        q = self.store.q
        return {"t": time.time(),
                "levels": [dict(r) for r in q("SELECT level, type, state, COUNT(*) n, SUM(check_state='pass') checked "
                                              "FROM jobs GROUP BY level, type, state ORDER BY level")],
                "workers": [{"id": r["id"], "name": r["name"], "last_seen": r["last_seen"], "disabled": r["disabled"],
                             "trusted": r["trusted"], "verifier": r["verifier"], "quarantined": r["quarantined"],
                             "gpu": (json.loads(r["caps"] or "{}")).get("gpu_name", ""),
                             "cards": (json.loads(r["caps"] or "{}")).get("cards")} for r in q("SELECT * FROM workers ORDER BY id")],
                "lumens": V.standings(self.store),
                "spot": [dict(r) for r in q("SELECT verify_state, COUNT(*) n FROM jobs WHERE verify_state IS NOT NULL GROUP BY verify_state")],
                "reservations": [dict(r) for r in q("SELECT * FROM reservations")],
                "active_leases": q("SELECT COUNT(*) FROM leases WHERE state='active'")[0][0],
                "spool_bytes": self.spool_bytes(),
                "events": [dict(r) for r in q("SELECT t, job_id, worker_id, kind FROM events ORDER BY id DESC LIMIT 50")]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"rad-quant-at-home-server/{VERSION}"
    Q: Queue = None

    def log_message(self, fmt, *args):
        if self.Q.a.verbose: sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, obj=None, body=None, headers=()):
        b = body if body is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json" if body is None else "application/octet-stream")
        self.send_header("Content-Length", str(len(b)))
        for k, v in headers: self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def _content_length(self, limit):
        """the request body's declared length; 0 when absent, refused (and the connection closed: the body stays
        unread) when not an integer, negative or over the cap"""
        try: n = int(self.headers.get("Content-Length") or 0)
        except ValueError: self.close_connection = True; raise HTTPError(400, "bad content-length")
        if n < 0: self.close_connection = True; raise HTTPError(400, "bad content-length")
        if n > limit: self.close_connection = True; raise HTTPError(413, f"body too large (max {limit} bytes)")
        return n

    def _json(self, limit=JSON_MAX):
        n = self._content_length(limit)
        try:
            b = json.loads(self.rfile.read(n) or b"{}")
        except ValueError: raise HTTPError(400, "bad json")
        if not isinstance(b, dict): raise HTTPError(400, "bad json")
        return b

    def _client_addr(self):
        """the peer's address; behind one of --trusted-proxy, the rightmost X-Forwarded-For entry that is not a trusted
        proxy (the one our proxy appended: entries further left are whatever the client sent)"""
        peer = self.client_address[0]
        if peer not in self.Q.proxies: return peer
        hops = [x.strip() for x in ",".join(self.headers.get_all("X-Forwarded-For") or []).split(",") if x.strip()]
        for x in reversed(hops):
            if x not in self.Q.proxies: return x
        return peer

    def _worker(self):
        a = self.headers.get("Authorization", "")
        w = self.Q.store.auth(a[7:] if a.startswith("Bearer ") else None)
        if w is None: raise HTTPError(401, "bad token")
        return w

    def _dispatch(self, method):
        try:
            u = urlsplit(self.path); parts = [unquote(x) for x in u.path.strip("/").split("/")]
            Q = self.Q
            if method == "GET" and parts == ["healthz"]: return self._send(200, Q.health())
            if method == "GET" and parts == ["api", API, "version"]: return self._send(200, Q.version())
            if method == "GET" and parts in ([""], ["index.html"]): return self._page()
            if method == "GET" and parts[:4] == ["api", API, "admin", "summary"] and len(parts) == 4:
                a = self.headers.get("Authorization", "")
                if Q.admin_hash is None: raise HTTPError(404, "no such endpoint")
                if not Q.admin_ok(a[7:] if a.startswith("Bearer ") else None): raise HTTPError(401, "bad admin token")
                return self._send(200, Q.admin_summary())
            if parts[:3] == ["api", API, "search"]:
                return self._search(method, parts[3:])
            if method == "GET" and parts[:1] == ["data"]:
                return self._data(self._worker(), "/".join(parts[1:]))
            if parts[:2] != ["api", API]: raise HTTPError(404, "no such endpoint")
            ep = parts[2:]
            if method == "POST" and ep == ["register"]: return self._send(200, Q.register(self._json(), self._client_addr()))
            w = self._worker()
            if method == "POST" and ep == ["lease"]:
                return self._send(200, Q.lease(w, self._json().get("caps", {}), Q.client_version(self.headers)))
            if method == "POST" and ep == ["heartbeat"]: return self._send(200, Q.heartbeat(w, self._json()))
            if method == "POST" and ep == ["complete"]: return self._send(200, Q.complete(w, self._json()))
            if method == "POST" and ep == ["fail"]: return self._send(200, Q.fail(w, self._json()))
            if method == "GET" and ep == ["status"]: return self._send(200, Q.status(w))
            if ep[:1] == ["me"]: return self._me(w, method, ep[1:], parse_qs(u.query))
            if ep[:1] == ["upload"] and len(ep) == 4:
                job_id, lease_id, name = int(ep[1]), ep[2], ep[3]
                if not re.fullmatch("[0-9a-f]{32}", lease_id): raise HTTPError(400, "bad lease id")
                if method == "GET": return self._send(200, Q.upload_size(w, job_id, lease_id, name))
                if method == "PUT":
                    off = int(parse_qs(u.query).get("offset", ["0"])[0])
                    return self._send(200, Q.upload(w, job_id, lease_id, name, off, self._content_length(Q.chunk_max), self.rfile))
            raise HTTPError(404, "no such endpoint")
        except HTTPError as e:
            if method == "PUT" and e.code != 200: self.close_connection = True    # unread body
            self._send(e.code, {"error": e.msg, **e.extra})
        except (KeyError, ValueError, TypeError, OverflowError) as e:     # OverflowError: an id beyond SQLite's 64 bits
            self.close_connection = True
            self._send(400, {"error": f"bad request: {e!r}"[:300]})
        except Exception as e:
            traceback.print_exc(); self.close_connection = True
            self._send(500, {"error": "internal"})

    def _page(self, head=False):
        b = self.Q.page
        if b is None: raise HTTPError(404, "no status page (--status-interval 0)")
        self.send_response(200)
        for k, v in (("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(b))),
                     ("Cache-Control", "public, max-age=30"),
                     ("Last-Modified", time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(self.Q.page_t))),
                     ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
                                                 "frame-ancestors 'none'"),
                     ("X-Content-Type-Options", "nosniff"), ("Referrer-Policy", "no-referrer"), ("X-Frame-Options", "DENY")):
            self.send_header(k, v)
        self.end_headers()
        if not head: self.wfile.write(b)

    def _me(self, w, method, ep, qs):
        """the donor's own view (server/donor.py); any of their worker tokens"""
        S = self.Q.store
        arg = lambda k, d: int(qs.get(k, [d])[0])
        try:
            if method == "GET" and ep == []: return self._send(200, {**D.me(S, w), "update": self.Q.update_info()})
            if method == "GET" and ep == ["jobs"]:
                wk = qs.get("worker", [None])[0]
                return self._send(200, D.my_jobs(S, w, arg("limit", 50), int(wk) if wk else None, qs.get("active", ["0"])[0] == "1"))
            if method == "GET" and ep == ["points"]: return self._send(200, D.my_points(S, w, arg("limit", 50)))
            if method == "GET" and ep == ["events"]: return self._send(200, D.my_events(S, w, arg("after", 0), arg("limit", 100)))
            if method == "POST" and ep == ["profile"]: return self._send(200, D.set_profile(S, w, self._json()))
            if ep == ["link"]: raise HTTPError(410, D.LINK_GONE)
        except PermissionError as e: raise HTTPError(403, str(e))
        except ValueError as e: raise HTTPError(400, str(e)[:300])
        raise HTTPError(404, "no such endpoint")

    def _search(self, method, ep):
        """the search's batch API: its own token, eval_mix only (server/batch.py validates everything)"""
        Q = self.Q
        if Q.search_hash is None: raise HTTPError(404, "no such endpoint")
        a = self.headers.get("Authorization", "")
        tok = a[7:] if a.startswith("Bearer ") else None
        if tok is None or not hmac.compare_digest(hashlib.sha256(tok.encode()).digest(), Q.search_hash):
            raise HTTPError(401, "bad search token")
        try:
            if method == "POST" and ep == ["batch"]:
                return self._send(200, B.submit(Q.store, Q.data, Q.manifest(), self._json(BATCH_MAX), Q.piece_cache))
            if method == "GET" and len(ep) == 2 and ep[0] == "batch":
                return self._send(200, B.status(Q.store, Q.results, int(ep[1])))
            if method == "POST" and len(ep) == 3 and ep[0] == "batch" and ep[2] == "adopt":
                return self._send(200, B.adopt(Q.store, int(ep[1]), self._json().get("candidates")))
            if method == "GET" and len(ep) == 4 and ep[0] == "batch" and ep[2] == "rows":
                it = [x for x in B.status(Q.store, Q.results, int(ep[1]))["items"] if x["id"] == ep[3]]
                if not it or it[0]["state"] != "accepted": raise HTTPError(404, "no finished candidate of that id")
                p = B.rows_path(Q.results, it[0]["job"]); n = os.path.getsize(p)
                self.send_response(200); self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(n)); self.end_headers(); self.wfile.flush()
                with open(p, "rb") as f: self.connection.sendfile(f, 0, n)
                return
        except B.BatchError as e: raise HTTPError(400, str(e))
        raise HTTPError(404, "no such endpoint")

    def _data(self, w, rel):
        p = self.Q.data_path(rel)
        if not self.Q.may_read(w, rel): raise HTTPError(403, "no active lease lists this file")
        why = self.Q.private(rel)
        if why: raise HTTPError(403, f"not public: {why}")
        size = os.path.getsize(p); a, b = 0, size - 1; code = 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
            if not m: raise HTTPError(416, "bad range")
            a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else size - 1
            b = min(b, size - 1)
            if a > b: raise HTTPError(416, "range not satisfiable")
            code = 206
        n = b - a + 1
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream"); self.send_header("Content-Length", str(n))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206: self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers(); self.wfile.flush()
        with open(p, "rb") as f:
            self.connection.sendfile(f, a, n)

    def do_GET(self): self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def do_PUT(self): self._dispatch("PUT")
    def do_HEAD(self):
        if urlsplit(self.path).path in ("/", "/index.html") and self.Q.page is not None: return self._page(head=True)
        self.send_response(405); self.send_header("Content-Length", "0"); self.end_headers()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=env("DB")); ap.add_argument("--data", default=env("DATA"))
    ap.add_argument("--spool", default=env("SPOOL")); ap.add_argument("--results", default=env("RESULTS"))
    ap.add_argument("--bind", default=env("BIND", "127.0.0.1")); ap.add_argument("--port", type=int, default=env("PORT", 8740, int))
    ap.add_argument("--lease-s", type=float, default=env("LEASE_S", 120, float), help="lease length; every heartbeat renews it")
    ap.add_argument("--heartbeat-s", type=float, default=env("HEARTBEAT_S", 30, float))
    ap.add_argument("--retry-s", type=float, default=env("RETRY_S", 30, float), help="what an idle worker is told to wait")
    ap.add_argument("--max-leases", type=int, default=env("MAX_LEASES", 16, int),
                    help="most active leases per worker (a worker gets min(this, its slots); slots default to its cards)")
    ap.add_argument("--gang-wait-s", type=float, default=env("GANG_WAIT_S", 0, float),
                    help="how long a gang job waits before a busy multi-card worker starts draining for it")
    ap.add_argument("--class-wait-s", type=float, default=env("CLASS_WAIT_S", 900, float),
                    help="a queued job a busy worker could run but keeps passing over for lower-priority work drains that "
                         "worker after waiting this long (docs/SCHED.md)")
    ap.add_argument("--class-scan", type=int, default=env("CLASS_SCAN", 5000, int),
                    help="queued jobs of one priority class looked at per lease request")
    ap.add_argument("--reserve-max-s", type=float, default=env("RESERVE_MAX_S", 6 * 3600, float),
                    help="a drain that has not freed the cards by then is given up; that worker may not reserve the same "
                         "job again for as long (#8), another worker may")
    ap.add_argument("--spot-rate", type=float, default=env("SPOT_RATE", 0.04, float),
                    help="fraction of each donor's accepted jobs re-run on a verifier (spot checks)")
    ap.add_argument("--spot-first", type=int, default=env("SPOT_FIRST", 1, int),
                    help="always re-run a donor's first N accepted jobs of each type")
    ap.add_argument("--eval-spot-rate", type=float, default=env("EVAL_SPOT_RATE", 0.08, float),
                    help="fraction of each donor's eval_mix results spot-checked (a few random documents re-scored on a verifier)")
    ap.add_argument("--eval-spot-docs", type=int, default=env("EVAL_SPOT_DOCS", 3, int),
                    help="documents an eval_mix spot check re-scores (0: the whole golden)")
    ap.add_argument("--search-token-file", default=env("SEARCH_TOKEN_FILE"),
                    help="enables /api/v0/search/... (eval_mix batches of the search) for this bearer token")
    ap.add_argument("--chunk-max-mb", type=float, default=env("CHUNK_MAX_MB", CHUNK_MAX >> 20, float), help="largest upload PUT")
    ap.add_argument("--upload-max-mb", type=float, default=env("UPLOAD_MAX_MB", 0, float),
                    help="ceiling per uploaded file on top of each job type's own caps (0: the type caps only)")
    ap.add_argument("--spool-max-gb", type=float, default=env("SPOOL_MAX_GB", 0, float),
                    help="refuse upload chunks (507, the worker retries) while the spool holds this much (0: no cap)")
    ap.add_argument("--refail-s", type=float, default=env("REFAIL_S", 3600, float),
                    help="a worker that failed / let expire a job may lease it again after this many seconds (0: only "
                         "after the job is requeued; default 1 h)")
    ap.add_argument("--lease-cap-x", type=float, default=env("LEASE_CAP_X", 3.0, float),
                    help="a lease expires after this many times its job type's reference runtime even while it heartbeats "
                         "(#7; the job is requeued, a late result still counts while the job is open; 0: no cap)")
    ap.add_argument("--lease-cap-min-s", type=float, default=env("LEASE_CAP_MIN_S", 7200, float),
                    help="... but never before this many seconds (default 2 h)")
    ap.add_argument("--admin-token-file", default=env("ADMIN_TOKEN_FILE"),
                    help="enables the read-only GET /api/v0/admin/summary for this bearer token")
    ap.add_argument("--status-interval", type=float, default=env("STATUS_INTERVAL", 60, float),
                    help="rebuild the public status page (served at /) every this many seconds (0: no page)")
    ap.add_argument("--status-dir", default=env("STATUS_DIR"), help="also write the page to DIR/index.html (any static host)")
    ap.add_argument("--campaign", default=env("CAMPAIGN"),
                    help="campaign.json for the page (totals, title; default: next to the database; re-read at every build)")
    ap.add_argument("--score-threads", type=int, default=env("SCORE_THREADS", 0, int),
                    help="score score-mode quant results in this process (docs/VERIFY.md; ~2.6 GB RAM each at the 27B); "
                         "0: run `admin.py score --loop` beside it instead")
    ap.add_argument("--score-poll-s", type=float, default=env("SCORE_POLL_S", 10, float))
    ap.add_argument("--open-registration", action=argparse.BooleanOptionalAction,
                    default=env("OPEN_REGISTRATION", True, lambda v: v.lower() not in ("0", "false", "no", "off")),
                    help="/register without an invite makes a worker on probation (default on; --no-open-registration: "
                         "invites only)")
    ap.add_argument("--reg-per-ip-day", type=int, default=env("REG_PER_IP_DAY", 5, int),
                    help="registrations without an invite per client address (IPv6: per /64) in 24 h (0: no limit)")
    ap.add_argument("--reg-global-hour", type=int, default=env("REG_GLOBAL_HOUR", 60, int),
                    help="registrations without an invite in total per hour (0: no limit)")
    ap.add_argument("--trusted-proxy", default=env("TRUSTED_PROXY", ""),
                    help="comma-separated addresses of the reverse proxy in front: only from these is X-Forwarded-For "
                         "read for the client address (in the container: the address the proxy reaches it from)")
    ap.add_argument("--probation-pass", type=int, default=env("PROBATION_PASS", SCH.PROBATION_PASS, int),
                    help="passed quant results after which a worker registered without an invite may hold more than one lease")
    ap.add_argument("--eval-unlock", type=int, default=env("EVAL_UNLOCK", SCH.EVAL_UNLOCK, int),
                    help="... and after which it may run eval_mix and every other type whose check takes our hardware")
    ap.add_argument("--worker-current", default=env("WORKER_CURRENT"),
                    help="the worker release donors should run, advertised in every lease answer (default: the release "
                         "manifest's, else this server's release)")
    ap.add_argument("--release-manifest", default=env("RELEASE_MANIFEST"),
                    help="release.json of the current worker release (its image digests; admin.py release-fetch writes it; "
                         "default: next to the database; re-read when it changes)")
    ap.add_argument("--worker-min", default=env("WORKER_MIN", WORKER_MIN),
                    help="the oldest worker release that is still leased jobs; older ones get 426 with the way to update "
                         "(default %(default)s; raise it once the previous release's window ends: RUNBOOK.md)")
    ap.add_argument("--verbose", action="store_true", default=env("VERBOSE", False, lambda v: v not in ("0", "false", "no")))
    a = ap.parse_args()
    miss = [f"--{k} (RQAH_{k.upper()})" for k in ("db", "data", "spool", "results") if not getattr(a, k)]
    if miss: ap.error("missing " + ", ".join(miss))
    try: Handler.Q = Queue(a)
    except SchemaTooNew as e: sys.exit(str(e))
    Q0 = Handler.Q
    if Q0.store.migrated: print(f"database migrated to schema {Q0.store.migrated[-1]} (backup next to it: *.bak)", flush=True)
    down, n = Q0.restart
    if n: print(f"restart: {n} active leases extended" + (f" by the downtime ({down:.0f} s)" if down is not None else ""), flush=True)
    for l in Q0.version_report(): print("WARNING " + l, flush=True)
    def reaper():
        while True:
            time.sleep(min(5.0, a.lease_s / 4))
            try: Handler.Q.store.expire(); Handler.Q.sweep_spool(); Handler.Q.store.alive(); Handler.Q.sync_meta()
            except Exception: traceback.print_exc()
    threading.Thread(target=reaper, daemon=True).start()
    if a.status_interval > 0:
        def pager():
            while True:
                try: Handler.Q.build_page()
                except Exception: traceback.print_exc()
                time.sleep(a.status_interval)
        try: Handler.Q.build_page()                         # the first page before the port opens
        except Exception: traceback.print_exc()
        threading.Thread(target=pager, daemon=True).start()
    if a.score_threads > 0:
        from server import score as SC
        def scorer():
            Q = Handler.Q
            while True:
                try: SC.run_pending(Q.store, Q.results, lambda rel: os.path.join(Q.data, rel))
                except Exception: traceback.print_exc()
                time.sleep(a.score_poll_s)
        threading.Thread(target=scorer, daemon=True).start()      # one is enough: numpy's BLAS is multi-threaded
    srv = ThreadingHTTPServer((a.bind, a.port), Handler); srv.daemon_threads = True
    # SIGTERM (docker stop, systemd): stop accepting, let serve_forever return. Every state change is its own SQLite
    # transaction and uploads resume from the spool, so even a SIGKILL loses nothing but in-flight requests.
    def on_term(signum, _):
        print(f"signal {signum}: shutting down", flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    reg = (f"open (probation {a.probation_pass}/{a.eval_unlock}, {a.reg_per_ip_day}/address/day, {a.reg_global_hour}/h)"
           if a.open_registration else "invites only")
    print(f"rad-quant-at-home queue on http://{a.bind}:{srv.server_address[1]}  db={a.db}  data={a.data}  results={a.results}  "
          f"admin summary {'on' if a.admin_token_file else 'off'}, search API {'on' if a.search_token_file else 'off'}, "
          f"registration {reg}, workers {Q0.worker_min}..{Q0.worker_current}"
          f"{' (release manifest)' if Q0.release() and not a.worker_current else ''}, "
          f"spot checks {a.spot_rate:.0%} (eval {a.eval_spot_rate:.0%}, {a.eval_spot_docs} docs) + first {a.spot_first}, scorer {'on' if a.score_threads > 0 else 'off'}, status page "
          f"{f'every {a.status_interval:g}s' + (' -> ' + a.status_dir if a.status_dir else '') if a.status_interval > 0 else 'off'}", flush=True)
    srv.serve_forever()
    srv.server_close()
    try: Handler.Q.store.alive()                             # the downtime a restart measures starts now
    except Exception: traceback.print_exc()
    with Handler.Q.store.lock: Handler.Q.store.db.close()
    print("stopped", flush=True)


if __name__ == "__main__":
    main()
