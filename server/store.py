# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The queue's state: one SQLite file (WAL), shared by the HTTP server and the admin CLI.

jobs.state: queued -> leased -> accepted, or failed (max_attempts used up / admin), or cancelled (admin).
A lease is one attempt by one worker; leases.state: active -> done (its result accepted) | expired (missed its
heartbeat) | abandoned (the worker gave it back) | failed (the worker reported an error) | rejected (the worker
refused the job, or the server refused its result) | superseded (another lease's result was accepted first) |
cancelled (admin, quarantine).

0.2 (server/verify.py): workers.trusted (may run TRUSTED_ONLY types), workers.verifier (runs spot-check re-runs; its
own results are never spot-checked), workers.quarantined; jobs.needs ('trusted' | 'verifier' | NULL),
jobs.verify_of (a spot-check re-run: the job it re-runs), jobs.verify_state on the original (NULL = not drawn,
pending, pass, fail, error), jobs.exclude_worker; points (lumens: grant / revoke rows per job); reservations (a gang
job a worker is draining for); batches + batch_items (eval_mix candidates of the search).

w25 (server/donor.py, server/statuspage.py): donors (one person's workers: workers.donor_id; display_name,
leaderboard = opt-in to the public status page, hidden = the operator took the name off it). Every worker has a donor;
a worker registered with a personal invite (invites.personal) joins the donor of the invite's first registration.

0.3 (server/score.py, docs/VERIFY.md): scores (one row per scored tensor of a score-mode quant result: series, key, K,
rel, codes sha256, kind donor | trusted | published, state pass | fail | provisional | suspect | void), score_den (tr(W H W^T)
per tensor, computed once).

0.4: jobs.requeued (the last admin requeue / invalidate: a worker's failed, rejected or expired leases before it no
longer exclude it, #5).

0.5: leases.deadline (#7: heartbeats renew a lease only up to it; then it expires like a missed heartbeat);
reserve_lapsed (#8: a gang / class-wait reservation that timed out without the worker draining; that worker may not
reserve the job again for one more reserve_max_s); jobs.grp (#11: the copies of one best-of-N job share it; a donor
holds at most one of them), scores.tiles (#11: fingerprints of sampled trellis tiles, for near-duplicate codes),
meta (server-local secrets, e.g. which tiles are sampled). #4 (eval_mix): jobs.pred (the search's predicted KL and
plausibility band; server side only, never sent to a worker), jobs.trust_state (a full re-run on a verifier: NULL,
pending, pass, fail, error; jobs.verify_state is the spot check of a few documents), batch_items.adopted (when the
search adopted the candidate).

0.5 (open registration, docs/VERIFY.md "Identity"): workers.probation (0: registered with an invite or by the
operator; 1: registered without an invite, on probation until enough quant results pass, server/sched.py tier();
2: a probation worker that was quarantined: it never unlocks, admin.py probation ID off clears it); reg_log (one
row per registration without an invite: time and a keyed hash of the client address or IPv6 /64, never the address
itself; rows older than a day are dropped), for the rate limits.

Schema versions (0.5, RUNBOOK.md "Upgrades"): the database's PRAGMA user_version is the last migration applied
(MIGRATIONS; 0 = a database from before 0.5, or a new one). Opening applies the missing ones in order, each in its own
transaction, after a consistent copy of the old database (FILE.schemaN.bak, once per version). A database from a
newer server (user_version > SCHEMA_VERSION) is refused: restore the backup or run the newer server. Migrations only
add (tables, columns, rows), so the previous release's server still opens a migrated database (one-release rollback).
  1  the 0.2-0.5 columns (MIGRATE) and one donor per worker from before donors
  2  workers.version (the worker software's release, from its X-RQAH-Version header or, before 0.5, its user agent);
     meta keys alive_t (the server's last reaper tick: a restart extends the leases by the downtime), worker_current /
     worker_min (what the server advertises, for the status page), retired (job versions taken out of service)"""
import hashlib, hmac, ipaddress, json, os, secrets, sqlite3, threading, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers(id INTEGER PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT UNIQUE NOT NULL,
  created REAL, last_seen REAL, caps TEXT, disabled INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS invites(code_hash TEXT PRIMARY KEY, uses_left INTEGER NOT NULL, created REAL, note TEXT);
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, type TEXT NOT NULL, version INTEGER NOT NULL, params TEXT NOT NULL,
  inputs TEXT NOT NULL, requires TEXT NOT NULL, input_bytes INTEGER NOT NULL, state TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 5, level TEXT, ukey TEXT UNIQUE,
  lease_id TEXT, lease_owner INTEGER, lease_expires REAL, created REAL, updated REAL,
  accepted_lease TEXT, accepted_by INTEGER, accepted_at REAL, result TEXT, files TEXT, error TEXT,
  check_state TEXT, check_detail TEXT);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs(state, priority, id);
CREATE INDEX IF NOT EXISTS jobs_level ON jobs(level);
CREATE TABLE IF NOT EXISTS leases(id TEXT PRIMARY KEY, job_id INTEGER NOT NULL, worker_id INTEGER NOT NULL, issued REAL,
  expires REAL, state TEXT NOT NULL, ended REAL, note TEXT);
CREATE INDEX IF NOT EXISTS leases_worker ON leases(worker_id, state);
CREATE INDEX IF NOT EXISTS leases_job ON leases(job_id);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, t REAL, job_id INTEGER, worker_id INTEGER, lease_id TEXT,
  kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS points(id INTEGER PRIMARY KEY, t REAL, worker_id INTEGER NOT NULL, job_id INTEGER NOT NULL,
  lumens REAL NOT NULL, kind TEXT NOT NULL, note TEXT);
CREATE INDEX IF NOT EXISTS points_worker ON points(worker_id);
CREATE TABLE IF NOT EXISTS reservations(job_id INTEGER PRIMARY KEY, worker_id INTEGER NOT NULL, t REAL);
CREATE TABLE IF NOT EXISTS reserve_lapsed(job_id INTEGER NOT NULL, worker_id INTEGER NOT NULL, t REAL, PRIMARY KEY(job_id, worker_id));
CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY, t REAL, name TEXT, spec TEXT, n INTEGER);
CREATE TABLE IF NOT EXISTS batch_items(batch_id INTEGER NOT NULL, cand TEXT NOT NULL, job_id INTEGER NOT NULL,
  PRIMARY KEY(batch_id, cand));
CREATE TABLE IF NOT EXISTS donors(id INTEGER PRIMARY KEY, display_name TEXT, leaderboard INTEGER NOT NULL DEFAULT 0,
  hidden INTEGER NOT NULL DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS scores(id INTEGER PRIMARY KEY, job_id INTEGER, grp TEXT, series TEXT NOT NULL, key TEXT NOT NULL,
  K REAL NOT NULL, rel REAL, num REAL, den REAL, codes_sha TEXT, worker_id INTEGER, donor_id INTEGER, kind TEXT NOT NULL,
  state TEXT NOT NULL, basis TEXT, bound REAL, t REAL, note TEXT, n INTEGER);
CREATE INDEX IF NOT EXISTS scores_tk ON scores(series, key, state);
CREATE INDEX IF NOT EXISTS scores_sha ON scores(codes_sha);
CREATE INDEX IF NOT EXISTS scores_job ON scores(job_id);
CREATE TABLE IF NOT EXISTS score_den(model TEXT NOT NULL, key TEXT NOT NULL, den REAL NOT NULL, PRIMARY KEY(model, key));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reg_log(t REAL NOT NULL, ipk TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS reg_log_ip ON reg_log(ipk, t);
"""
# columns added after v0: (table, column, declaration); added to an existing database on open
MIGRATE = [("workers", "trusted", "INTEGER NOT NULL DEFAULT 0"), ("workers", "verifier", "INTEGER NOT NULL DEFAULT 0"),
           ("workers", "quarantined", "INTEGER NOT NULL DEFAULT 0"), ("workers", "quarantine_reason", "TEXT"),
           ("jobs", "needs", "TEXT"), ("jobs", "verify_of", "INTEGER"), ("jobs", "verify_state", "TEXT"),
           ("jobs", "verify_detail", "TEXT"), ("jobs", "exclude_worker", "INTEGER"),
           ("workers", "donor_id", "INTEGER"), ("invites", "personal", "INTEGER NOT NULL DEFAULT 0"), ("invites", "donor_id", "INTEGER"),
           ("jobs", "requeued", "REAL"), ("leases", "deadline", "REAL"), ("jobs", "grp", "TEXT"), ("scores", "tiles", "BLOB"),
           ("jobs", "pred", "TEXT"), ("jobs", "trust_state", "TEXT"), ("batch_items", "adopted", "REAL"),
           ("workers", "probation", "INTEGER NOT NULL DEFAULT 0")]


def _add_columns(db, cols):
    for t, c, decl in cols:
        if c not in {r[1] for r in db.execute(f"PRAGMA table_info({t})")}:
            db.execute(f"ALTER TABLE {t} ADD COLUMN {c} {decl}")


def _migrate_1(db):
    _add_columns(db, MIGRATE)
    for w in db.execute("SELECT id, name, created FROM workers WHERE donor_id IS NULL").fetchall():   # workers from before donors
        d = db.execute("INSERT INTO donors(display_name, created) VALUES(?,?)", (w["name"], w["created"])).lastrowid
        db.execute("UPDATE workers SET donor_id=? WHERE id=?", (d, w["id"]))


def _migrate_2(db):
    _add_columns(db, [("workers", "version", "TEXT")])


# user_version -> the step that reaches it; only ever append (a step must be safe on a database the previous release
# wrote, and leave one that release can still open)
MIGRATIONS = {1: _migrate_1, 2: _migrate_2}
SCHEMA_VERSION = max(MIGRATIONS)


class SchemaTooNew(RuntimeError):
    pass


def h(s): return hashlib.sha256(s.encode()).hexdigest()


def new_token():
    """a worker token; the prefix keeps it from starting with "-" (token_urlsafe can), which a command line would read
    as a flag (`--token -x...`)"""
    return "rq" + secrets.token_urlsafe(32)


class Store:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA busy_timeout=30000")
        self.lock = threading.RLock()
        self.migrated = self.migrate()

    def migrate(self):
        """bring the schema to SCHEMA_VERSION (module docstring) -> the versions applied (empty: nothing to do)"""
        uv = self.db.execute("PRAGMA user_version").fetchone()[0]
        if uv > SCHEMA_VERSION:
            raise SchemaTooNew(f"{self.path}: database schema {uv} is newer than this server's {SCHEMA_VERSION} (written by a "
                               f"newer release): run that release, or restore the backup {self.path}.schema{SCHEMA_VERSION}.bak")
        if uv == SCHEMA_VERSION: return []
        old = self.db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone()[0]
        if old:                                              # a consistent copy first (WAL included), once per version
            bak = f"{self.path}.schema{uv}.bak"
            if not os.path.exists(bak):
                dst = sqlite3.connect(bak + ".part")
                try: self.db.backup(dst)
                finally: dst.close()
                os.replace(bak + ".part", bak)
        self.db.executescript(SCHEMA)
        done = []
        for v in range(uv + 1, SCHEMA_VERSION + 1):
            with self.tx() as db:
                if db.execute("PRAGMA user_version").fetchone()[0] >= v: continue      # another process got there first
                MIGRATIONS[v](db)
                db.execute(f"PRAGMA user_version={int(v)}")
            done.append(v)
        return done

    # ------------------------------------------------------------ plumbing
    def tx(self):
        store = self
        class T:
            def __enter__(s): store.lock.acquire(); store.db.execute("BEGIN IMMEDIATE"); return store.db
            def __exit__(s, et, ev, tb):
                try: store.db.execute("COMMIT" if et is None else "ROLLBACK")
                finally: store.lock.release()
        return T()

    def secret(self, name):
        """a random value made once per queue database (meta table) and never shown to anyone"""
        with self.tx() as db:
            r = db.execute("SELECT v FROM meta WHERE k=?", (name,)).fetchone()
            if r: return r[0]
            v = secrets.token_hex(32); db.execute("INSERT INTO meta(k, v) VALUES(?,?)", (name, v))
        return v

    def meta_get(self, k, default=None):
        r = self.q("SELECT v FROM meta WHERE k=?", (k,))
        return r[0][0] if r else default

    def meta_set(self, k, v):
        with self.tx() as db: db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?,?)", (k, str(v)))

    def after_restart(self, lease_s, now=None):
        """a server start: the leases that were active when it went down get a fresh lease_s (their workers retry and
        heartbeat once it answers again), and their hard deadlines move by the downtime (meta alive_t: the last reaper
        tick; unknown after an upgrade from 0.4: no move). -> (downtime s or None, leases extended)"""
        now = now or time.time()
        alive = self.meta_get("alive_t")
        down = max(0.0, now - float(alive)) if alive else None
        with self.tx() as db:
            n = db.execute("UPDATE leases SET expires=MAX(expires, ?), deadline=deadline + ? WHERE state='active'",
                           (now + lease_s, down or 0.0)).rowcount
            db.execute("UPDATE jobs SET lease_expires=(SELECT expires FROM leases l WHERE l.id=jobs.lease_id) WHERE state='leased'")
            self.event(db, "server_start", detail={"downtime_s": None if down is None else round(down, 1), "leases": n})
            db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('alive_t', ?)", (str(now),))
        return down, n

    def alive(self, now=None):
        with self.tx() as db: db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('alive_t', ?)", (str(now or time.time()),))

    def retired(self):
        """{job type: set of versions taken out of service} (admin.py retire)"""
        try: return {t: set(v) for t, v in json.loads(self.meta_get("retired") or "{}").items()}
        except (ValueError, TypeError, AttributeError): return {}

    def q(self, sql, args=()):
        with self.lock: return self.db.execute(sql, args).fetchall()

    def event(self, db, kind, job_id=None, worker_id=None, lease_id=None, detail=None):
        db.execute("INSERT INTO events(t, job_id, worker_id, lease_id, kind, detail) VALUES(?,?,?,?,?,?)",
                   (time.time(), job_id, worker_id, lease_id, kind, None if detail is None else json.dumps(detail)))

    # ------------------------------------------------------------ workers
    def add_invite(self, uses=1, note="", personal=False):
        """personal: every worker registered with it belongs to one donor (one person's boxes)"""
        code = "ri" + secrets.token_urlsafe(18)
        with self.tx() as db: db.execute("INSERT INTO invites(code_hash, uses_left, created, note, personal) VALUES(?,?,?,?,?)",
                                         (h(code), uses, time.time(), note, int(personal)))
        return code

    def new_donor(self, db, display_name, leaderboard=False):
        return db.execute("INSERT INTO donors(display_name, leaderboard, created) VALUES(?,?,?)",
                          (display_name, int(bool(leaderboard)), time.time())).lastrowid

    def register(self, invite, name, leaderboard=False):
        token = new_token()
        with self.tx() as db:
            r = db.execute("SELECT * FROM invites WHERE code_hash=?", (h(invite),)).fetchone() if invite else None
            if r is None or r["uses_left"] <= 0: return None
            db.execute("UPDATE invites SET uses_left=uses_left-1 WHERE code_hash=?", (h(invite),))
            did = r["donor_id"] if r["personal"] and r["donor_id"] else self.new_donor(db, name, leaderboard)
            if r["personal"] and not r["donor_id"]: db.execute("UPDATE invites SET donor_id=? WHERE code_hash=?", (did, h(invite)))
            cur = db.execute("INSERT INTO workers(name, token_hash, created, donor_id) VALUES(?,?,?,?)", (name, h(token), time.time(), did))
            self.event(db, "register", worker_id=cur.lastrowid, detail={"name": name, "donor": did})
        return cur.lastrowid, token

    def reg_key(self, addr):
        """the rate-limit key of a client address: a keyed hash (the address itself is never stored) of the IPv4
        address, or of the IPv6 /64 (one home connection usually has a whole /64)"""
        try:
            ip = ipaddress.ip_address(addr)
            if ip.version == 6 and ip.ipv4_mapped: ip = ip.ipv4_mapped
            k = str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else str(ip)
        except ValueError:
            k = str(addr)[:100]
        return hmac.new(self.secret("reg_ip_key").encode(), k.encode(), hashlib.sha256).hexdigest()[:32]

    def register_open(self, name, addr, per_ip_day=5, global_hour=60, leaderboard=False, now=None):
        """registration without an invite: a new donor and a worker on probation. -> (worker id, token), or
        (None, seconds until a registration from this address would be allowed) when a rate limit is reached
        (per_ip_day per client address / IPv6 /64 in 24 h, global_hour in total in one hour; 0 = no limit)"""
        now = now or time.time(); ipk = self.reg_key(addr)
        with self.tx() as db:
            db.execute("DELETE FROM reg_log WHERE t < ?", (now - 86400,))
            if per_ip_day > 0:
                r = db.execute("SELECT COUNT(*), MIN(t) FROM reg_log WHERE ipk=? AND t >= ?", (ipk, now - 86400)).fetchone()
                if r[0] >= per_ip_day: return None, max(1, int(r[1] + 86400 - now) + 1)
            if global_hour > 0:
                r = db.execute("SELECT COUNT(*), MIN(t) FROM reg_log WHERE t >= ?", (now - 3600,)).fetchone()
                if r[0] >= global_hour: return None, max(1, int(r[1] + 3600 - now) + 1)
            db.execute("INSERT INTO reg_log(t, ipk) VALUES(?,?)", (now, ipk))
            token = new_token()
            did = self.new_donor(db, name, leaderboard)
            cur = db.execute("INSERT INTO workers(name, token_hash, created, donor_id, probation) VALUES(?,?,?,?,1)",
                             (name, h(token), now, did))
            self.event(db, "register", worker_id=cur.lastrowid, detail={"name": name, "donor": did, "probation": True})
        return cur.lastrowid, token

    def add_worker(self, name, donor_id=None):
        token = new_token()
        with self.tx() as db:
            did = donor_id or self.new_donor(db, name)
            cur = db.execute("INSERT INTO workers(name, token_hash, created, donor_id) VALUES(?,?,?,?)", (name, h(token), time.time(), did))
        return cur.lastrowid, token

    def auth(self, token):
        if not token: return None
        r = self.q("SELECT * FROM workers WHERE token_hash=? AND disabled=0", (h(token),))
        return r[0] if r else None

    # ------------------------------------------------------------ jobs
    def enqueue(self, db, jtype, version, params, inputs, requires, input_bytes, level=None, ukey=None, priority=0,
                max_attempts=5, needs=None, verify_of=None, exclude_worker=None, grp=None):
        if ukey and db.execute("SELECT 1 FROM jobs WHERE ukey=?", (ukey,)).fetchone(): return None
        t = time.time()
        if needs is None and requires.get("trusted"): needs = "trusted"
        cur = db.execute("INSERT INTO jobs(type, version, params, inputs, requires, input_bytes, state, priority, "
                         "max_attempts, level, ukey, created, updated, needs, verify_of, exclude_worker, grp) "
                         "VALUES(?,?,?,?,?,?,'queued',?,?,?,?,?,?,?,?,?,?)",
                         (jtype, version, json.dumps(params, sort_keys=True), json.dumps(inputs), json.dumps(requires),
                          input_bytes, priority, max_attempts, level, ukey, t, t, needs, verify_of, exclude_worker, grp))
        self.event(db, "enqueue", job_id=cur.lastrowid)
        return cur.lastrowid

    def expire(self, now=None):
        """leases that missed their heartbeat: lease expired, job back to the queue (or failed: no attempts left)"""
        now = now or time.time(); n = 0
        with self.tx() as db:
            # gang reservations of workers that went away, or that waited too long
            db.execute("DELETE FROM reservations WHERE job_id NOT IN (SELECT id FROM jobs WHERE state='queued') OR "
                       "worker_id IN (SELECT id FROM workers WHERE last_seen < ? OR quarantined=1 OR disabled=1)",
                       (now - getattr(self, "reserve_idle_s", 300),))
            if getattr(self, "reserve_max_s", None):
                # a drain that never freed the cards: that worker may not reserve the job again for another
                # reserve_max_s, so another busy multi-card worker gets its turn (#8)
                db.execute("INSERT OR REPLACE INTO reserve_lapsed SELECT job_id, worker_id, ? FROM reservations WHERE t < ?",
                           (now, now - self.reserve_max_s))
                db.execute("DELETE FROM reservations WHERE t < ?", (now - self.reserve_max_s,))
                db.execute("DELETE FROM reserve_lapsed WHERE t < ? OR job_id NOT IN (SELECT id FROM jobs WHERE state='queued')",
                           (now - self.reserve_max_s,))
            for l in db.execute("SELECT * FROM leases WHERE state='active' AND expires<?", (now,)).fetchall():
                past = l["deadline"] is not None and l["expires"] >= l["deadline"]
                self.expire_lease(db, l, now, why="past its deadline" if past else "expired"); n += 1
        return n

    def expire_lease(self, db, l, now, why="expired"):
        """lease l ends as expired; its job goes back to the queue (or fails: no attempts left). The worker may still
        finish it: a late result counts while the job is open (Queue.complete)"""
        db.execute("UPDATE leases SET state='expired', ended=? WHERE id=?", (now, l["id"]))
        j = db.execute("SELECT * FROM jobs WHERE id=?", (l["job_id"],)).fetchone()
        if j and j["state"] == "leased" and j["lease_id"] == l["id"]:
            st = "failed" if j["attempts"] >= j["max_attempts"] else "queued"
            db.execute("UPDATE jobs SET state=?, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=?, "
                       "error=COALESCE(error, '') || ? WHERE id=?", (st, now, f"[lease {l['id'][:8]} {why}]", j["id"]))
        self.event(db, "expire", l["job_id"], l["worker_id"], l["id"], None if why == "expired" else {"why": why})

    def end_lease(self, db, lease, state, job_state=None, error=None, count_attempt=True):
        now = time.time()
        db.execute("UPDATE leases SET state=?, ended=?, note=? WHERE id=?", (state, now, error, lease["id"]))
        j = db.execute("SELECT * FROM jobs WHERE id=?", (lease["job_id"],)).fetchone()
        if j["state"] == "leased" and j["lease_id"] == lease["id"]:
            att = j["attempts"] - (0 if count_attempt else 1)
            st = job_state or ("failed" if att >= j["max_attempts"] else "queued")
            db.execute("UPDATE jobs SET state=?, attempts=?, lease_id=NULL, lease_owner=NULL, lease_expires=NULL, updated=?, "
                       "error=? WHERE id=?", (st, att, now, (error or "")[:2000], j["id"]))
