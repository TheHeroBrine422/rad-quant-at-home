#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""statuspage.py --db QUEUE.sqlite --out index.html [--campaign campaign.json] -- the public status page (stdlib only).

ONE static HTML file built from the queue database: campaign progress, jobs running / queued by type, the donor
leaderboard (opt-in display names only, one row per name), recent finished work, the update time. No JavaScript, no forms, no external
resources; a <meta refresh> reloads it every minute. server.py rebuilds it every --status-interval seconds (default
60), serves it at / and, with --status-dir, also writes DIR/index.html for any static host; this script builds it once
(cron, tests). It is polyglot XHTML: an XML parser must accept it (tests/statuspage_test.py).

0.5: the worker releases donors' workers seen in the last 24 h run, as shares of current / older (the version numbers
are public; nothing per worker), from workers.version and the release the server advertises (meta worker_current).

What it may show: counts, job types, ladder level names, public corpus names, layer numbers, lumens, opted-in donors'
display names (checked by donor.public_name, then escaped). What it never shows: worker names, tokens, invites,
addresses, hostnames, GPU names, job parameters beyond the above, errors, quarantine reasons, file paths.

Campaign file (optional; default: campaign.json next to the database; re-read at every build):
  {"title": "EXL3-V3 ladder for Qwen3.8-27B", "ladder": {"levels": 73}, "search": {"evals": 5000},
   "goldens": {"chunks": 120}, "note": "one line of text"}
ladder.levels may also be the list of level names (only those count). Without the file the total is the number of
levels with jobs in the queue."""
import argparse, html, json, os, re, sqlite3, sys, tempfile, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jobs
from server import donor as D

QUANT = jobs.QUANT
TYPE_ORDER = ("quant_layer", "quant_tensor", "rxq_quant", "quant_head", "eval_mix", "golden_record", "noop")


class RO:
    """read-only access to the queue database for a standalone build"""
    def __init__(self, path):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only=1")
    def q(self, sql, args=()): return self.db.execute(sql, args).fetchall()
    def close(self): self.db.close()


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0


def load_campaign(path):
    try:
        with open(path) as f: c = json.load(f)
        return c if isinstance(c, dict) else {}
    except (OSError, ValueError):
        return {}


def collect(S, campaign=None, now=None):
    """the page's numbers (a dict of public values only)"""
    now = now or time.time(); c = campaign or {}
    # ladder: a level is done when every one of its (not cancelled) quant jobs is accepted
    lv = {}
    for r in S.q("SELECT level, state, check_state, COUNT(*) n FROM jobs WHERE type IN (" + ",".join(f"'{t}'" for t in jobs.QUANT) + ") AND "
                 "verify_of IS NULL AND level IS NOT NULL AND state<>'cancelled' GROUP BY level, state, check_state"):
        if not D.ID_RE.fullmatch(r["level"] or ""): continue
        x = lv.setdefault(r["level"], {"jobs": 0, "accepted": 0, "checked": 0})
        x["jobs"] += r["n"]
        if r["state"] == "accepted": x["accepted"] += r["n"]; x["checked"] += r["n"] if r["check_state"] == "pass" else 0
    want = (c.get("ladder") or {}).get("levels") if isinstance(c.get("ladder"), dict) else None
    if isinstance(want, list):
        names = [x for x in want if isinstance(x, str) and D.ID_RE.fullmatch(x)]
        lv = {k: v for k, v in lv.items() if k in names}; total = len(names)
    else:
        total = int(want) if _num(want) else len(lv)
    done = sorted(k for k, v in lv.items() if v["jobs"] and v["accepted"] == v["jobs"])
    verified = [k for k in done if lv[k]["checked"] == lv[k]["jobs"]]
    active = sorted(((k, v) for k, v in lv.items() if v["accepted"] < v["jobs"]), key=lambda kv: -kv[1]["accepted"] / kv[1]["jobs"])

    def count(sql, args=()): return S.q(sql, args)[0][0] or 0
    ev_done = count("SELECT COUNT(*) FROM jobs WHERE type='eval_mix' AND verify_of IS NULL AND state='accepted'")
    ev_all = count("SELECT COUNT(*) FROM jobs WHERE type='eval_mix' AND verify_of IS NULL AND state<>'cancelled'")
    gd_done = count("SELECT COUNT(*) FROM jobs WHERE type='golden_record' AND verify_of IS NULL AND state='accepted'")
    gd_all = count("SELECT COUNT(*) FROM jobs WHERE type='golden_record' AND verify_of IS NULL AND state<>'cancelled'")
    # spot checks from the audit log: a failed one quarantines the donor, which resets the job (verify_state NULL)
    spot = {r["kind"][5:]: r["n"] for r in S.q("SELECT kind, COUNT(*) n FROM events WHERE kind IN ('spot_pass','spot_fail') GROUP BY kind")}
    spot["pending"] = count("SELECT COUNT(*) FROM jobs WHERE verify_of IS NULL AND verify_state='pending'")
    # jobs by type
    qd = D.queue_depth(S)
    day = {r["type"]: r["n"] for r in S.q("SELECT type, COUNT(*) n FROM jobs WHERE state='accepted' AND verify_of IS NULL AND "
                                          "accepted_at>=? GROUP BY type", (now - 86400,))}
    tot = {r["type"]: r["n"] for r in S.q("SELECT type, COUNT(*) n FROM jobs WHERE state='accepted' AND verify_of IS NULL GROUP BY type")}
    types = [t for t in TYPE_ORDER if t in qd or t in tot] + sorted(t for t in set(qd) | set(tot) if t not in TYPE_ORDER)
    types = [t for t in types if D.ID_RE.fullmatch(t.replace("_", "-"))]
    by_type = [{"type": t, "what": D.LABEL_TYPES.get(t, t), "running": qd.get(t, {}).get("running", 0),
                "queued": qd.get(t, {}).get("queued", 0), "reruns": qd.get(t, {}).get("reruns", 0),
                "day": day.get(t, 0), "total": tot.get(t, 0)} for t in types]
    busy = S.q("SELECT j.requires, l.worker_id FROM leases l JOIN jobs j ON j.id=l.job_id WHERE l.state='active'")
    # workers seen in the last 15 min whose data budget is spent (a count only: no names, no amounts)
    paused = sum(1 for r in S.q("SELECT caps FROM workers WHERE last_seen>=? AND disabled=0", (now - 900,))
                 if D.budget_paused(_caps(r["caps"])))
    cards = sum(max(0, int(json.loads(r["requires"]).get("cards", 0) or 0)) for r in busy)
    # leaderboard
    rows, others = D.board(S)
    donors_day = count("SELECT COUNT(DISTINCT w.donor_id) FROM workers w WHERE w.verifier=0 AND w.last_seen>=?", (now - 86400,))
    lumens_all = round(sum(r["lumens"] for r in rows) + others[1], 2)
    # recent finished work
    named = {did: r["name"] for r in rows for did in r["donors"]}
    wk = {r["id"]: r for r in S.q("SELECT id, donor_id, verifier, quarantined FROM workers")}
    net = {(r["worker_id"], r["job_id"]): r["n"] for r in S.q("SELECT worker_id, job_id, SUM(lumens) n FROM points GROUP BY worker_id, job_id")}
    recent = []
    for j in S.q("SELECT * FROM jobs WHERE state='accepted' AND verify_of IS NULL ORDER BY accepted_at DESC, id DESC LIMIT 15"):
        w = wk.get(j["accepted_by"])
        by = ("our hardware" if w is not None and w["verifier"] else
              named.get(w["donor_id"]) if w is not None and not w["quarantined"] else None)
        if by is None: by = "a donor"
        if net.get((j["accepted_by"], j["id"]), 0) > 1e-9 or (w is not None and w["verifier"] and
                                                              (j["type"] not in QUANT or j["check_state"] == "pass")):
            st = "verified"
        elif j["verify_state"] == "pending": st = "spot check pending"
        elif j["type"] in QUANT and j["check_state"] != "pass": st = "awaiting check"
        else: st = "accepted"
        try: p = json.loads(j["params"])
        except ValueError: p = {}
        recent.append({"t": j["accepted_at"], "type": j["type"], "label": D.label(j["type"], p, j["level"]), "by": by,
                       "state": st, "listed": by not in ("a donor", "our hardware")})
    title = c.get("title") if isinstance(c.get("title"), str) else None
    try:                                                     # a database a 0.5 server has not migrated yet has neither
        cur = (S.q("SELECT v FROM meta WHERE k='worker_current'") or [[None]])[0][0]
        seen = Counter(r[0] for r in S.q("SELECT version FROM workers WHERE verifier=0 AND disabled=0 AND last_seen>=?", (now - 86400,)))
    except sqlite3.OperationalError:
        cur, seen = None, Counter()
    vt = lambda v: tuple(int(x) for x in v.split(".")) if isinstance(v, str) and re.fullmatch(r"\d{1,4}\.\d{1,4}\.\d{1,4}", v) else None
    versions = None
    if cur and vt(cur) and seen:
        n = sum(seen.values())
        older = Counter({v: k for v, k in seen.items() if vt(v) and vt(v) < vt(cur)})
        unknown = sum(k for v, k in seen.items() if not vt(v))
        versions = {"current": cur, "workers": n, "on_current": sum(k for v, k in seen.items() if vt(v) and vt(v) >= vt(cur)),
                    "older": sorted(older.items(), key=lambda kv: vt(kv[0]), reverse=True)[:4], "older_n": sum(older.values()),
                    "unknown": unknown}
    note = c.get("note") if isinstance(c.get("note"), str) else None
    tgt = lambda k, f: (c.get(k) or {}).get(f) if isinstance(c.get(k), dict) and _num((c.get(k) or {}).get(f)) else None
    return {"t": now, "title": title, "note": note,
            "ladder": {"done": len(done), "total": max(total, len(done)), "verified": len(verified),
                       "active": [{"level": k, "accepted": v["accepted"], "jobs": v["jobs"]} for k, v in active[:8]],
                       "active_more": max(0, len(active) - 8)},
            "search": {"done": ev_done, "queued": ev_all - ev_done, "target": tgt("search", "evals")},
            "goldens": {"done": gd_done, "total": max(gd_all, tgt("goldens", "chunks") or 0)},
            "spot": {"pass": spot.get("pass", 0), "fail": spot.get("fail", 0), "pending": spot.get("pending", 0)},
            "by_type": by_type, "cards_busy": cards, "workers_busy": len({r["worker_id"] for r in busy}), "budget_paused": paused,
            "donors_day": donors_day, "board": rows[:50], "board_more": max(0, len(rows) - 50), "others": others,
            "lumens_all": lumens_all, "recent": recent, "versions": versions}


CSS = """
:root{color-scheme:light dark;--fg:#1d1d1f;--bg:#fbfbfa;--mut:#6b6b70;--line:#d9d9de;--acc:#b15d00;--ok:#2a7a3b}
@media (prefers-color-scheme:dark){:root{--fg:#e8e8ea;--bg:#16161a;--mut:#9a9aa3;--line:#33333a;--acc:#ffae4a;--ok:#6fcf86}}
body{font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);background:var(--bg);margin:0}
main{max-width:58rem;margin:0 auto;padding:1.2rem 1rem 2rem}
h1{font-size:1.6rem;margin:.2rem 0}h1 span{color:var(--acc)}h2{font-size:1.1rem;margin:1.6rem 0 .5rem;border-bottom:1px solid var(--line)}
.sub,.mut,footer{color:var(--mut)}.big{font-size:1.25rem;font-weight:600}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:.25rem .5rem;border-bottom:1px solid var(--line)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}progress{width:100%;height:.8rem}
.ok{color:var(--ok)}ul.kpi{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:.5rem 2rem}
"""


def _caps(s):
    try: c = json.loads(s or "{}")
    except ValueError: return {}
    return c if isinstance(c, dict) else {}


def _e(x): return html.escape(str(x), quote=True)
def _n(x): return f"{x:,}" if isinstance(x, int) else f"{x:,.1f}"
def _utc(t): return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(t))


def _ago(t, now):
    if not t: return "—"
    d = now - t
    return "now" if d < 900 else "today" if d < 86400 else f"{int(d // 86400)} d ago"


def render(d):
    L, S_, G_ = d["ladder"], d["search"], d["goldens"]
    o = ['<!DOCTYPE html>', '<html lang="en">', '<head>',
         '<meta charset="utf-8" />', '<meta name="viewport" content="width=device-width, initial-scale=1" />',
         '<meta http-equiv="refresh" content="60" />', '<meta name="robots" content="noindex" />',
         '<title>rad-quant-at-home · status</title>', f'<style>{CSS}</style>', '</head>', '<body>', '<main>',
         '<h1>rad-quant<span>@home</span></h1>',
         f'<p class="sub">{_e(d["title"] or "Donated compute for EXL3 quantization")} · updated {_e(_utc(d["t"]))}</p>']
    if d["note"]: o.append(f'<p>{_e(d["note"][:300])}</p>')
    o += ['<section id="campaign">', '<h2>Campaign</h2>', '<ul class="kpi">',
          f'<li><span class="big">ladder {_n(L["done"])} / {_n(L["total"])} levels</span> '
          f'<span class="mut">({_n(L["verified"])} checked)</span></li>',
          f'<li><span class="big">search {_n(S_["done"])} evals</span> <span class="mut">'
          + (f'of {_n(int(S_["target"]))} planned, ' if S_["target"] else "") + f'{_n(S_["queued"])} queued</span></li>',
          f'<li><span class="big">goldens {_n(G_["done"])} / {_n(G_["total"])} chunks</span></li>',
          f'<li><span class="big">spot checks {_n(d["spot"]["pass"])} passed</span> <span class="mut">'
          f'{_n(d["spot"]["fail"])} failed, {_n(d["spot"]["pending"])} pending</span></li>', '</ul>']
    if L["total"]:
        o.append(f'<p><progress max="{L["total"]}" value="{L["done"]}">{_n(L["done"])} of {_n(L["total"])}</progress></p>')
    if L["active"]:
        o += ['<table>', '<thead><tr><th>level in progress</th><th class="n">jobs done</th><th>progress</th></tr></thead>', '<tbody>']
        for a in L["active"]:
            o.append(f'<tr><td>{_e(a["level"])}</td><td class="n">{_n(a["accepted"])} / {_n(a["jobs"])}</td>'
                     f'<td><progress max="{a["jobs"]}" value="{a["accepted"]}">{_n(a["accepted"])}</progress></td></tr>')
        o += ['</tbody>', '</table>']
        if L["active_more"]: o.append(f'<p class="mut">and {_n(L["active_more"])} more levels</p>')
    o += ['</section>', '<section id="jobs">', '<h2>Jobs</h2>',
          f'<p>{_n(d["cards_busy"])} cards busy on {_n(d["workers_busy"])} workers · {_n(d["donors_day"])} donors active in the last 24 h'
          + (f' · {_n(d["budget_paused"])} workers at their data cap until it resets' if d.get("budget_paused") else '') + '</p>',
          '<table>', '<thead><tr><th>job type</th><th class="n">running</th><th class="n">queued</th>'
          '<th class="n">done (24 h)</th><th class="n">done (all)</th></tr></thead>', '<tbody>']
    for t in d["by_type"]:
        o.append(f'<tr><td>{_e(t["what"])}</td><td class="n">{_n(t["running"])}</td><td class="n">{_n(t["queued"])}</td>'
                 f'<td class="n">{_n(t["day"])}</td><td class="n">{_n(t["total"])}</td></tr>')
    if not d["by_type"]: o.append('<tr><td colspan="5" class="mut">no jobs yet</td></tr>')
    o += ['</tbody>', '</table>']
    V_ = d.get("versions")
    if V_:
        pct = lambda k: f"{100 * k / V_['workers']:.0f}%"
        o.append(f'<p class="mut">worker release {_e(V_["current"])} (current): {pct(V_["on_current"])} of the {_n(V_["workers"])} '
                 'workers seen in the last 24 h' + "".join(f' · {_e(v)}: {pct(k)}' for v, k in V_["older"])
                 + (f' · older than 0.5: {pct(V_["unknown"])}' if V_["unknown"] else '')
                 + (' · donors: <code>rqah status</code> says how to update' if V_["older_n"] or V_["unknown"] else '') + '</p>')
    rr = sum(t["reruns"] for t in d["by_type"])
    if rr: o.append(f'<p class="mut">plus {_n(rr)} verification re-runs on our own hardware</p>')
    o += ['</section>', '<section id="donors">', '<h2>Donors</h2>',
          f'<p>{_n(d["lumens_all"])} lumens of verified work so far. A lumen is one R9700 card-minute of work that passed '
          'verification. Donors appear here only if they opt in (<code>rqah config set --leaderboard on</code>), one row per '
          'display name: every machine under the same name (case does not matter) adds to it. A name is a label, not an '
          'account: anyone can use it.</p>',
          '<table>', '<thead><tr><th class="n">#</th><th>name</th><th class="n">lumens</th><th class="n">verified jobs</th>'
          '<th>last active</th></tr></thead>', '<tbody>']
    for r in d["board"]:
        o.append(f'<tr><td class="n">{r["rank"]}</td><td>{_e(r["name"])}</td><td class="n">{_n(float(r["lumens"]))}</td>'
                 f'<td class="n">{_n(r["jobs"])}</td><td>{_e(_ago(r["last_seen"], d["t"]))}</td></tr>')
    if not d["board"]: o.append('<tr><td colspan="5" class="mut">nobody listed yet</td></tr>')
    o += ['</tbody>', '</table>']
    if d["board_more"] or d["others"][0]:
        o.append(f'<p class="mut">' + (f'{_n(d["board_more"])} more listed names; ' if d["board_more"] else "") +
                 f'{_n(d["others"][0])} donors not listed: {_n(float(d["others"][1]))} lumens</p>')
    o += ['</section>', '<section id="recent">', '<h2>Recent work</h2>', '<table>',
          '<thead><tr><th>finished</th><th>job</th><th>by</th><th>state</th></tr></thead>', '<tbody>']
    for r in d["recent"]:
        o.append(f'<tr><td>{_e(_utc(r["t"]))}</td><td>{_e(r["label"])}</td><td>{_e(r["by"])}</td>'
                 f'<td{" class=" + chr(34) + "ok" + chr(34) if r["state"] == "verified" else ""}>{_e(r["state"])}</td></tr>')
    if not d["recent"]: o.append('<tr><td colspan="4" class="mut">nothing finished yet</td></tr>')
    o += ['</tbody>', '</table>', '</section>',
          f'<footer><p>Last update {_e(_utc(d["t"]))}; rebuilt about every minute. Every result is checked, and a share '
          'is re-run on our own hardware before it counts.</p></footer>', '</main>', '</body>', '</html>', '']
    return "\n".join(o)


def build(S, campaign=None, now=None):
    return render(collect(S, campaign, now)).encode("utf-8")


def write(path, page):
    """atomic: a static host never serves half a file"""
    d = os.path.dirname(os.path.abspath(path)); os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".index-", dir=d)
    with os.fdopen(fd, "wb") as f: f.write(page)
    os.chmod(tmp, 0o644); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--campaign", help="campaign.json (default: next to the database)")
    a = ap.parse_args()
    S = RO(a.db)
    page = build(S, load_campaign(a.campaign or os.path.join(os.path.dirname(os.path.abspath(a.db)), "campaign.json")))
    S.close()
    write(a.out, page)
    print(f"{a.out}: {len(page)} bytes")


if __name__ == "__main__":
    main()
