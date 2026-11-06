#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rad-quant-at-home (alias rqah) [--json] COMMAND -- the donor CLI (stdlib only): your workers, jobs, lumens and logs, read
from the queue server with your worker's own token (server/donor.py). Inside the worker container:
`docker exec rad-quant-at-home rqah status`; elsewhere: python3 client/rqah.py --server URL --token-file F ...

  status                         your donor profile, workers (caps, running jobs, flags), lumens, queue depth, and
                                 the jobs this machine's worker is running (from its state dir, if readable)
  jobs [--limit N] [--active] [--worker ID]     recent jobs of your workers and their verification state
  points [--limit N]             lumens per worker and type, the ledger, pending work, your leaderboard rank
  logs [--limit N] [--follow] [--local]         the server's log of your workers' events (--local: this worker's
                                 finished-job records, STATE/worker.jsonl)
  config [show]                  local settings (server, token fingerprint, caps from the environment) and your
                                 public profile
  config set [--display-name NAME] [--leaderboard on|off]   the leaderboard entry (opt-in; off by default)
  link --other-token-file F      join this worker's donor with the donor of another of your workers (both tokens
                                 prove they are yours): status, jobs and points then cover both

--json on every command (before or after it): one JSON object on stdout (logs --follow: one per line), keys as in
the server's answer plus "ok"; errors as {"ok": false, "error", "http"}. Exit codes: 0 ok, 1 server error, 2 usage /
configuration, 3 token refused, 4 server unreachable.

Settings (flags win): --server / $RQAH_SERVER; the token from --token-file, $RQAH_TOKEN_FILE, $RQAH_TOKEN,
else STATE/token; --state / $RQAH_STATE (the worker image sets these). The token is never printed: config shows a
fingerprint (the first 12 hex digits of its sha256)."""
import argparse, glob, hashlib, json, os, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import API, env

UA = "rqah-cli/0.4"
LOCAL_ENV = ("GPU_NAME", "GPUS", "CARDS", "VRAM_GB", "RAM_GB", "CACHE_GB", "WINDOWS", "ON_STOP", "JOB_TYPES", "NAME",
             "IMAGE_ID", "LEADERBOARD", "MAX_DOWN_MBPS", "MAX_UP_MBPS", "MAX_DOWN_GB_DAY", "MAX_DOWN_GB_MONTH", "MAX_UP_GB_DAY",
             "MAX_UP_GB_MONTH", "BUDGET_RESET_DAY")


class Fail(Exception):
    def __init__(self, msg, code=1, http=None): super().__init__(msg); self.code, self.http = code, http


def e(k, d=None): return env(k, d, side="worker")


def token(a):
    if a.token: return a.token
    for f in (a.token_file, e("TOKEN_FILE"), os.path.join(a.state, "token") if a.state else None):
        if f and os.path.exists(f):
            try: return open(f).read().strip()
            except OSError as x: raise Fail(f"token file {f}: {x.strerror}", 2)
    return e("TOKEN")


def call(a, method, ep, body=None, q=None):
    if not a.server: raise Fail("no server: --server URL or $RQAH_SERVER", 2)
    tok = token(a)
    if not tok: raise Fail("no token: --token-file F, $RQAH_TOKEN_FILE or STATE/token (the worker writes it at registration)", 2)
    url = a.server.rstrip("/") + f"/api/{API}/{ep}"
    if q: url += "?" + "&".join(f"{k}={v}" for k, v in q.items() if v is not None)
    h = {"User-Agent": UA, "Authorization": f"Bearer {tok}"}
    data = None
    if body is not None: data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h), timeout=a.timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as x:
        try: msg = json.loads(x.read() or b"{}").get("error", "")
        except ValueError: msg = ""
        raise Fail(f"server: HTTP {x.code} {msg}".strip(), 3 if x.code in (401, 403) and ep != "me/link" else 1, x.code)
    except (urllib.error.URLError, OSError) as x:
        raise Fail(f"server unreachable: {getattr(x, 'reason', x)}", 4)


# ---------------------------------------------------------------- formatting
def lt(t): return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def ago(t):
    if not t: return "never"
    d = time.time() - t
    return "just now" if d < 60 else f"{d / 60:.0f} min ago" if d < 3600 else f"{d / 3600:.1f} h ago" if d < 86400 else f"{d / 86400:.0f} d ago"


def table(rows, cols):
    """rows: list of dicts; cols: [(header, key or fn)]"""
    cells = [[str(c(r) if callable(c) else r.get(c, "")) for _, c in cols] for r in rows]
    w = [max([len(h)] + [len(x[i]) for x in cells]) for i, (h, _) in enumerate(cols)]
    out = ["  ".join(h.ljust(w[i]) for i, (h, _) in enumerate(cols)).rstrip()]
    out += ["  ".join(x[i].ljust(w[i]) for i in range(len(cols))).rstrip() for x in cells]
    return "\n".join(out)


def local_jobs(a):
    """the jobs this machine's worker is running (STATE/current/*.json), if the state dir is readable"""
    out = []
    for f in sorted(glob.glob(os.path.join(a.state or "/nonexistent", "current", "*.json"))):
        try: c = json.load(open(f))
        except (OSError, ValueError): continue
        j = c.get("job", {})
        out.append({"job": j.get("id"), "type": j.get("type"), "phase": c.get("phase"), "devices": c.get("devices"),
                    "since": c.get("t_lease")})
    return out


def gb(n):
    if n is None: return "unlimited"
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def budget_cell(w):
    """the DATA column: what the worker declared it has left (after its running jobs' reservations)"""
    b = w.get("budget")
    if not b: return "-"
    if "up" in b["spent"]: return "SPENT"
    return " / ".join(f"{d} " + ("SPENT" if d in b["spent"] else gb(b[d]) if b[d] is not None else "-") for d in ("down", "up"))


def local_budget(a):
    """this machine's data budget (STATE/bandwidth.json, written by the worker), if readable"""
    try: b = json.load(open(os.path.join(a.state or "/nonexistent", "bandwidth.json")))
    except (OSError, ValueError): return None
    lim = b.get("limits") or {}
    if not any(v is not None for v in lim.values()): return {"limited": False, **{k: b.get(k) for k in ("day", "month")}}
    return {"limited": True, "day": b.get("day"), "month": b.get("month"), "reset_day": b.get("reset_day"),
            "used": {k: b.get(k, 0) for k in lim}, "limits": lim,
            "spent": sorted({k.split("_")[0] for k, v in lim.items() if v is not None and v - b.get(k, 0) < (1 << 20)})}


# ---------------------------------------------------------------- commands
def cmd_status(a):
    m = call(a, "GET", "me"); m["local"] = local_jobs(a); m["local_budget"] = local_budget(a)
    if a.json: return m
    d = m["donor"]
    board = (f"on the leaderboard: #{d['rank']} of {d['on_board']}" if d["rank"] else
             "on the leaderboard once a job is verified" if d["leaderboard"] and not d["hidden"] else
             "hidden from the leaderboard by the operator" if d["hidden"] else
             "not on the leaderboard (rqah config set --leaderboard on)")
    print(f"rad-quant-at-home · donor {d['display_name'] or '-'!r} · {board}")
    print(f"lumens {m['lumens']:.2f} from {m['verified_jobs']} verified jobs · pending verification: {m['pending']['jobs']} jobs "
          f"(~{m['pending']['lumens']:.1f} lumens)\n")
    print(table(m["workers"], [("ID", "id"), ("NAME", "name"), ("GPU", lambda w: w["gpu_name"] or "-"),
                               ("CARDS", lambda w: w["cards"] if w["cards"] is not None else "-"),
                               ("VRAM", lambda w: f"{w['vram_gb']:g}" if w["vram_gb"] is not None else "-"),
                               ("RAM", lambda w: f"{w['ram_gb']:g}" if w["ram_gb"] is not None else "-"),
                               ("LAST SEEN", lambda w: ago(w["last_seen"])),
                               ("RUNNING", lambda w: ",".join(f"{r['job']}:{r['type']}" for r in w["running"]) or "-"),
                               ("COULD RUN", lambda w: w["could_run"] if w["could_run"] is not None else "-"),
                               ("LUMENS", lambda w: f"{w['lumens']:.2f}"),
                               ("DATA LEFT", lambda w: budget_cell(w)),
                               ("FLAGS", lambda w: ",".join(k for k in ("trusted", "verifier", "quarantined", "disabled") if w[k]) or "-")]))
    for w in m["workers"]:
        if w["quarantined"]: print(f"\nworker {w['id']} is QUARANTINED: {w['quarantine_reason']} (ask the operator)")
        b = w.get("budget")
        if b and b["spent"]:
            until = lt(b["reset"]) if b.get("reset") else "the next reset"
            print(f"\nworker {w['id']}: data budget spent ({', '.join('download' if d == 'down' else 'upload' for d in b['spent'])}): "
                  + ("only jobs whose inputs it already caches" if b["spent"] == ["down"] else "no new jobs") + f" until {until}; "
                  "running jobs finish")
    q = m["queue"]["by_type"]
    print("\nqueue: " + ("; ".join(f"{t} {v['queued']} queued, {v['running']} running" for t, v in sorted(q.items())) or "empty"))
    if m["local"]:
        print("this machine: " + "; ".join(f"job {j['job']} ({j['type']}) {j['phase']} since {lt(j['since'])}" for j in m["local"]))
    lb = m["local_budget"]
    if lb and lb["limited"]:
        parts = [f"{'download' if k.startswith('down') else 'upload'} {gb(lb['used'][k])} of {gb(v)} "
                 + ("today" if k.endswith("_day") else "this month")
                 for k, v in sorted(lb["limits"].items()) if v is not None]
        print(f"data budget (this machine): {'; '.join(parts)} (day {lb['day']}, month from {lb['month']})"
              + (f" -- SPENT: {', '.join(lb['spent'])}" if lb["spent"] else ""))
    return None


def cmd_jobs(a):
    r = call(a, "GET", "me/jobs", q={"limit": a.limit, "active": 1 if a.active else None, "worker": a.worker})
    if a.json: return r
    if not r["jobs"]: print("no jobs yet"); return None
    print(table(r["jobs"], [("JOB", "job"), ("TYPE", "type"), ("WHAT", "label"), ("WORKER", "worker_name"),
                            ("STARTED", lambda j: lt(j["issued"])), ("ENDED", lambda j: lt(j["ended"])),
                            ("STATE", "verification"),
                            ("LUMENS", lambda j: f"{j['lumens']:.2f}" if j["lumens"] else
                             f"(~{j['lumens_expected']:.1f})" if j.get("lumens_expected") else "-"),
                            ("NOTE", lambda j: (j["note"] or "")[:60])]))
    return None


def cmd_points(a):
    r = call(a, "GET", "me/points", q={"limit": a.limit})
    if a.json: return r
    rank = f"#{r['rank']} of {r['on_board']} on the leaderboard" if r["rank"] else "not on the leaderboard"
    print(f"{r['lumens']:.2f} lumens from {r['verified_jobs']} verified jobs ({rank}); pending: {r['pending']['jobs']} jobs, "
          f"~{r['pending']['lumens']:.1f} lumens\n")
    print(table(r["by_worker"], [("WORKER", "id"), ("NAME", "name"), ("LUMENS", lambda x: f"{x['lumens']:.2f}"), ("JOBS", "verified_jobs")]))
    if r["by_type"]:
        print("\n" + table([{"type": t, **v} for t, v in sorted(r["by_type"].items())],
                           [("TYPE", "type"), ("LUMENS", lambda x: f"{x['lumens']:.2f}"), ("JOBS", "jobs")]))
    if r["ledger"]:
        print("\n" + table(r["ledger"], [("WHEN", lambda x: lt(x["t"])), ("JOB", "job"), ("WHAT", "label"), ("WORKER", "worker"),
                                         ("LUMENS", lambda x: f"{x['lumens']:+.2f}"), ("KIND", "kind"), ("NOTE", lambda x: x["note"] or "")]))
    return None


def _fmt_event(ev):
    d = " ".join(f"{k}={v}" for k, v in ev["detail"].items())
    return (f"{lt(ev['t'])}  {ev['kind']:<16} " + (f"job {ev['job']} " if ev["job"] else "") +
            (f"worker {ev['worker_name'] or ev['worker']} " if ev["worker"] else "") + d).rstrip()


def cmd_logs(a):
    if a.local:
        p = os.path.join(a.state or "", "worker.jsonl")
        if not a.state or not os.path.exists(p): raise Fail(f"no local worker records ({p or 'no --state'})", 2)
        recs = [json.loads(l) for l in open(p) if l.strip()][-a.limit:]
        if a.json: return {"local": recs, "current": local_jobs(a)}
        for r in recs:
            print(f"{lt(r.get('t'))}  job {r.get('job')} {r.get('type')}: fetch {r.get('fetch_s', '-')} s, run {r.get('run_s', '-')} s, "
                  f"upload {r.get('upload_s', '-')} s, down {r.get('down_bytes', 0)} B, up {r.get('up_bytes', 0)} B")
        return None
    # the newest `limit` events: page forward from the start (the server answers oldest first after `after`)
    after, evs = 0, []
    while True:
        r = call(a, "GET", "me/events", q={"after": after, "limit": 1000})
        evs += r["events"]
        if len(r["events"]) < 1000: break
        after = r["last"]
    evs = evs[-a.limit:]; last = evs[-1]["id"] if evs else after
    if a.json and not a.follow: return {"events": evs, "last": last}
    for ev in evs: print(json.dumps(ev) if a.json else _fmt_event(ev), flush=True)
    if not a.follow: return None
    try:
        while True:
            time.sleep(a.poll)
            r = call(a, "GET", "me/events", q={"after": last, "limit": 1000})
            for ev in r["events"]: print(json.dumps(ev) if a.json else _fmt_event(ev), flush=True)
            last = r["last"]
    except KeyboardInterrupt:
        return None


def cmd_config(a):
    if a.action == "set":
        b = {}
        if a.display_name is not None: b["display_name"] = a.display_name
        if a.leaderboard is not None: b["leaderboard"] = a.leaderboard == "on"
        if not b: raise Fail("config set: give --display-name and/or --leaderboard on|off", 2)
        r = call(a, "POST", "me/profile", b)
        if a.json: return {**r, "set": b}
        print("set: " + ", ".join(f"{k} = {v}" for k, v in b.items())); return None
    tok = token(a)
    loc = {"server": a.server, "state": a.state,
           "token": {"source": "flag" if a.token else "file" if any(f and os.path.exists(f) for f in (a.token_file, e("TOKEN_FILE"),
                     os.path.join(a.state, "token") if a.state else None)) else "env" if tok else None,
                     "fingerprint": hashlib.sha256(tok.encode()).hexdigest()[:12] if tok else None},
           "env": {f"RQAH_{k}": e(k) for k in LOCAL_ENV if e(k) not in (None, "")}}
    out = {"local": loc}
    try:
        m = call(a, "GET", "me")
        me_w = [w for w in m["workers"] if w["id"] == m["worker"]["id"]][0]
        out["profile"] = m["donor"]
        out["worker"] = {k: me_w[k] for k in ("id", "name", "gpu_name", "cards", "vram_gb", "ram_gb", "disk_gb", "job_types", "image",
                                              "trusted", "verifier", "quarantined")}
    except Fail as x:
        out["server_error"] = str(x)
        if a.json: print(json.dumps({"ok": False, **out, "error": str(x), "http": x.http})); sys.exit(x.code)
        print_config(out); raise
    if a.json: return out
    print_config(out); return None


def print_config(o):
    L = o["local"]
    print(f"server       {L['server'] or '-'}\nstate        {L['state'] or '-'}\n"
          f"token        {('sha256 ' + L['token']['fingerprint'] + ' (from ' + L['token']['source'] + ')') if L['token']['fingerprint'] else 'none'}")
    for k, v in sorted(L["env"].items()): print(f"{k:<24} {v}")
    if "profile" in o:
        p, w = o["profile"], o["worker"]
        print(f"\ndisplay name {p['display_name']!r}\nleaderboard  {'on' if p['leaderboard'] else 'off'}"
              + (" (hidden by the operator)" if p["hidden"] else "") + (f", rank #{p['rank']}" if p["rank"] else ""))
        print(f"this worker  {w['id']} {w['name']!r}: {w['gpu_name'] or 'no GPU'}, cards {w['cards']}, VRAM {w['vram_gb']} GB/card, "
              f"RAM {w['ram_gb']} GB, cache {w['disk_gb']} GB, types {','.join(w['job_types']) or '-'}"
              + "".join(f", {k}" for k in ("trusted", "verifier", "quarantined") if w[k]))
    if "server_error" in o: print(f"\nserver: {o['server_error']}")


def cmd_link(a):
    try: other = open(a.other_token_file).read().strip()
    except OSError as x: raise Fail(f"{a.other_token_file}: {x.strerror}", 2)
    r = call(a, "POST", "me/link", {"token": other})
    if a.json: return r
    print(f"linked: donor {r['donor']} ({r['moved']} workers moved)"); return None


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    S = argparse.SUPPRESS                             # sub-command copies must not reset what was given before the command
    common.add_argument("--json", action="store_true", default=S, help="machine-readable output")
    common.add_argument("--server", default=S); common.add_argument("--token", default=S, help="(prefer --token-file)")
    common.add_argument("--token-file", default=S); common.add_argument("--state", default=S)
    common.add_argument("--timeout", type=float, default=S)
    ap = argparse.ArgumentParser(prog="rqah", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[common])
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("status", parents=[common])
    x = sp.add_parser("jobs", parents=[common]); x.add_argument("--limit", type=int, default=30)
    x.add_argument("--active", action="store_true"); x.add_argument("--worker", type=int)
    x = sp.add_parser("points", parents=[common]); x.add_argument("--limit", type=int, default=20)
    x = sp.add_parser("logs", parents=[common]); x.add_argument("--limit", type=int, default=50)
    x.add_argument("--follow", "-f", action="store_true"); x.add_argument("--local", action="store_true")
    x.add_argument("--poll", type=float, default=5)
    x = sp.add_parser("config", parents=[common]); x.add_argument("action", nargs="?", choices=("show", "set"), default="show")
    x.add_argument("--display-name"); x.add_argument("--leaderboard", choices=("on", "off"))
    x = sp.add_parser("link", parents=[common]); x.add_argument("--other-token-file", required=True)
    a = ap.parse_args(argv)
    for k, d in (("json", False), ("server", e("SERVER")), ("token", None), ("token_file", None), ("state", e("STATE")),
                 ("timeout", 30.0)):
        if not hasattr(a, k): setattr(a, k, d)
    try:
        out = {"status": cmd_status, "jobs": cmd_jobs, "points": cmd_points, "logs": cmd_logs, "config": cmd_config,
               "link": cmd_link}[a.cmd](a)
    except Fail as x:
        if a.json: print(json.dumps({"ok": False, "error": str(x), "http": x.http}))
        else: print(f"rqah: {x}", file=sys.stderr)
        return x.code
    if a.json and out is not None: print(json.dumps({"ok": True, **out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
