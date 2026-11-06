#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""statuspage_test.py [WORKDIR] -- the public status page from a fixture database (no server, a few seconds).

The fixture holds what must never reach the page: worker names of our machines, tokens, an IP in a job error and a
quarantine reason, paths in job parameters and in a bogus level name, a donor who did not opt in, a quarantined donor
who did, a display name that is an address, one with markup (written straight into the database, past the API's
check), operator text with markup in campaign.json. Checks: the page builds (in process and with the standalone
script on a read-only connection); it is well-formed XHTML (xml.etree) with only an allow-listed set of elements, no
script / form / input / external resource / event-handler attribute (html.parser; xmllint --html too if present);
no token, IP, e-mail, path or hostname pattern and none of the fixture's secrets; everything donor- or
operator-supplied escaped; the numbers right (ladder, search, goldens, jobs by type, leaderboard order and opt-in,
recent work); and the donor display-name rules (server/donor.py)."""
import html.parser, json, os, re, shutil, sqlite3, subprocess, sys, time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server.store import Store
from server import statuspage as SP, donor as D

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/w25-statuspage")
RESULTS = []
NOW = 1793900000.0                                                       # 2026-11-05 ~ 20:13 UTC: a fixed clock


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


# patterns no public page may contain (the test's own view of "private")
LEAKS = {"token": re.compile(r"\b(?:rq|ri)[A-Za-z0-9_-]{20,}"), "hex id": re.compile(r"\b[0-9a-f]{24,}\b"),
         "IPv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IPv6": re.compile(r"(?i)\b(?:[0-9a-f]{1,4}:){4,7}[0-9a-f]{1,4}\b"),
         "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "url": re.compile(r"(?i)\b(?:https?|ftp|file)://"),
         "abs path": re.compile(r"(?<![\w.<])/(?:root|data|srv|home|workspace|tmp|var|etc|mnt|opt)\b"),
         "bearer": re.compile(r"(?i)bearer"), "hostname": re.compile(r"(?i)\b(?:gpuhost|queuehost|localhost)\b|\.(?:lan|local|internal)\b")}
ALLOWED = {"html", "head", "meta", "title", "style", "body", "main", "h1", "h2", "p", "span", "ul", "li", "section", "table",
           "thead", "tbody", "tr", "th", "td", "progress", "code", "footer"}
VOID = {"meta"}


class Shape(html.parser.HTMLParser):
    """element allow-list, nesting, and attributes: no on*, no href / src / action / style attributes"""
    def __init__(self):
        super().__init__(convert_charrefs=True); self.stack, self.errors, self.tags, self.text = [], [], set(), []
        self.in_style = False
    def handle_starttag(self, t, attrs):
        self.tags.add(t)
        if t not in ALLOWED: self.errors.append(f"element <{t}>")
        for k, v in attrs:
            if k.startswith("on") or k in ("href", "src", "action", "formaction", "style", "srcset", "background", "ping"):
                self.errors.append(f"attribute {k} on <{t}>")
        if t not in VOID: self.stack.append(t)
        self.in_style = t == "style"
    def handle_startendtag(self, t, attrs):
        self.handle_starttag(t, attrs)
        if t not in VOID: self.stack.pop()
    def handle_endtag(self, t):
        if not self.stack or self.stack[-1] != t: self.errors.append(f"</{t}> closes <{self.stack[-1] if self.stack else None}>")
        else: self.stack.pop()
        self.in_style = False
    def handle_data(self, d):
        if not self.in_style: self.text.append(d)


def fixture(db):
    S = Store(db)
    ids, toks = {}, {}
    for name in ("ada-box", "zoe-box", "gpuhost-verifier", "home-4070", "cheater-box", "ipname-box", "markup-box", "quiet-box"):
        ids[name], toks[name] = S.add_worker(name)
    with S.tx() as d:
        don = lambda n: d.execute("SELECT donor_id FROM workers WHERE id=?", (ids[n],)).fetchone()[0]
        d.execute("UPDATE donors SET display_name='Ada Lovelace', leaderboard=1 WHERE id=?", (don("ada-box"),))
        d.execute("UPDATE donors SET display_name='Zoë 🚀 Team', leaderboard=1 WHERE id=?", (don("zoe-box"),))
        d.execute("UPDATE donors SET display_name='Cheaty McCheat', leaderboard=1 WHERE id=?", (don("cheater-box"),))
        d.execute("UPDATE donors SET display_name='10.20.30.40', leaderboard=1 WHERE id=?", (don("ipname-box"),))
        d.execute("UPDATE donors SET display_name=?, leaderboard=1 WHERE id=?", ('<script>alert(1)</script>"&', don("markup-box")))
        d.execute("UPDATE donors SET display_name='Quiet Person', leaderboard=0 WHERE id=?", (don("quiet-box"),))
        d.execute("UPDATE workers SET verifier=1 WHERE id=?", (ids["gpuhost-verifier"],))
        d.execute("UPDATE workers SET quarantined=1, quarantine_reason='spot check of job 3 failed from 10.20.30.40' WHERE id=?",
                  (ids["cheater-box"],))
        for n in ids: d.execute("UPDATE workers SET last_seen=?, caps=? WHERE id=?",
                                (NOW - 600, json.dumps({"gpu_name": "AMD Radeon on home.lan", "cards": 1}), ids[n]))
        jid = [0]

        def job(jtype, params, state, by=None, level=None, check=None, verify=None, cards=1, t=NOW - 3600, err=None, needs=None):
            jid[0] += 1
            d.execute("INSERT INTO jobs(id, type, version, params, inputs, requires, input_bytes, state, level, accepted_by, "
                      "accepted_at, check_state, verify_state, created, updated, error, needs, accepted_lease) "
                      "VALUES(?,?,1,?,?,?,0,?,?,?,?,?,?,?,?,?,?,?)",
                      (jid[0], jtype, json.dumps(params), json.dumps(["models/q@1/weights/L00.safetensors"]),
                       json.dumps({"gpu": True, "cards": cards}), state, level, ids.get(by), t if state == "accepted" else None,
                       check, verify, t - 600, t, err, needs, f"{jid[0]:032x}" if state == "accepted" else None))
            if state == "accepted":
                d.execute("INSERT INTO leases VALUES(?,?,?,?,?, 'done', ?, NULL)", (f"{jid[0]:032x}", jid[0], ids[by], t - 300, t + 60, t))
            if state == "leased":
                d.execute("INSERT INTO leases VALUES(?,?,?,?,?, 'active', NULL, NULL)", (f"{jid[0]:032x}", jid[0], ids[by], t, NOW + 60))
            return jid[0]
        qp = lambda lvl, l: {"model": "q", "revision": "1", "level": lvl, "layer": l, "K": 3, "codebook": "i8",
                             "settings": {"hout": "/srv/secret/hout/private-snap"}, "stamp": "x"}
        done = [job("quant_layer", qp("k3.25-i8", l), "accepted", "ada-box", "k3.25-i8", "pass", t=NOW - 2 * 86400 + l) for l in range(3)]
        job("quant_head", {**qp("k3.25-i8", 0), "layer": None}, "accepted", "zoe-box", "k3.25-i8", "pass", t=NOW - 2 * 86400 + 9)
        job("quant_layer", qp("k2.5-i8", 0), "accepted", "zoe-box", "k2.5-i8", None, t=NOW - 100)
        job("quant_layer", qp("k2.5-i8", 1), "leased", "home-4070", "k2.5-i8", cards=2)
        job("quant_layer", qp("k2.5-i8", 2), "queued", None, "k2.5-i8", err="worker at 10.20.30.40 failed: /srv/secret/challenge/x")
        job("quant_layer", qp("k4-i8", 0), "queued", None, "k4-i8")
        job("quant_layer", qp("k4-i8", 1), "queued", None, "/srv/secret/challenge/secret-level")           # not a level name
        ev = [job("eval_mix", {"base": "k3.25-i8", "golden": "xcorp-s07", "mix": "/home/someone/x"}, "accepted",
                  ["ada-box", "ipname-box", "ada-box", "markup-box", "gpuhost-verifier"][i], t=NOW - 100000 * (i % 2) - 60 * i)
              for i in range(5)]
        job("eval_mix", {"base": "k3.25-i8"}, "queued", None); job("eval_mix", {"base": "k3.25-i8"}, "queued", None)
        rr = job("eval_mix", {"base": "k3.25-i8"}, "queued", None, needs="verifier")         # a spot-check re-run
        d.execute("UPDATE jobs SET verify_of=? WHERE id=?", (ev[1], rr))
        g = job("golden_record", {"corpus": "xcorp-web", "chunk": "c0001"}, "accepted", "ada-box", verify="pass")
        job("golden_record", {"corpus": "xcorp-web", "chunk": "c0002"}, "queued", None)
        cj = job("eval_mix", {"base": "k3.25-i8"}, "queued", None)            # the cheater's, invalidated and requeued
        nq = job("noop", {"seconds": 1}, "accepted", "quiet-box")
        ni = job("noop", {"seconds": 1}, "accepted", "ipname-box", t=NOW - 4000)
        pts = [("ada-box", j, 12.0) for j in done] + [("ada-box", ev[0], 3.0), ("ada-box", g, 30.0), ("zoe-box", 4, 20.0),
                                                      ("markup-box", ev[3], 5.0), ("quiet-box", nq, 0.5),
                                                      ("cheater-box", cj, 4.0), ("cheater-box", cj, -4.0), ("ipname-box", ni, 1.0)]
        for kind, j in (("spot_pass", g), ("spot_fail", cj)):
            d.execute("INSERT INTO events(t, job_id, worker_id, kind, detail) VALUES(?,?,?,?,?)",
                      (NOW - 400, j, None, kind, json.dumps({"error": "rows differ on 10.20.30.40"})))
        for n, j, l in pts:
            d.execute("INSERT INTO points(t, worker_id, job_id, lumens, kind, note) VALUES(?,?,?,?,?,?)",
                      (NOW - 500, ids[n], j, l, "grant" if l > 0 else "revoke", "revoked: 10.20.30.40" if l < 0 else None))
        d.execute("UPDATE jobs SET verify_state='pending' WHERE id=?", (ev[1],))
    return S, ids, toks


def main():
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(W)
    db = f"{W}/queue.sqlite"
    S, ids, toks = fixture(db)
    camp = {"title": 'EXL3-V3 ladder <b>Qwen3.8-27B</b> & "friends"', "ladder": {"levels": 73}, "search": {"evals": 5000},
            "note": "operator note </p><script>x()</script>"}
    json.dump(camp, open(f"{W}/campaign.json", "w"))
    data = SP.collect(S, camp, NOW)
    page = SP.render(data)
    open(f"{W}/index.html", "w").write(page)
    # 1. well-formed, allow-listed, no active content
    try: ET.fromstring(page.split("\n", 1)[1]); wf = True; why = ""
    except ET.ParseError as e: wf, why = False, str(e)
    ok("page is well-formed XHTML (xml.etree parses it)", wf and page.startswith("<!DOCTYPE html>\n"), why)
    sh = Shape(); sh.feed(page); sh.close()
    ok("only allow-listed elements, balanced; no script / form / input / iframe / img / link; no on*, href, src, style attributes",
       not sh.errors and not sh.stack and not {"script", "form", "input", "iframe", "img", "link", "object", "embed"} & sh.tags,
       f"{sh.errors[:3]} open {sh.stack}")
    ok("meta refresh (no JavaScript needed), charset, title", '<meta http-equiv="refresh" content="60" />' in page and
       '<meta charset="utf-8" />' in page and "<title>" in page)
    xl = shutil.which("xmllint")
    if xl:
        r = subprocess.run([xl, "--noout", f"{W}/index.html"], capture_output=True, text=True)
        r2 = subprocess.run([xl, "--html", "--noout", f"{W}/index.html"], capture_output=True, text=True)
        errs = [l for l in r2.stderr.splitlines() if "error" in l.lower() and "invalid" not in l.lower()]   # HTML4 parser: HTML5 tags "invalid"
        ok("xmllint: XML well-formed, and its HTML parser reports no errors (other than HTML5 element names)",
           r.returncode == 0 and not errs, (r.stderr + "\n".join(errs))[:300])
    # 2. nothing private
    hits = {k: rx.findall(page) for k, rx in LEAKS.items() if rx.search(page)}
    ok("no token / IP / e-mail / URL / path / hostname patterns", not hits, str(hits)[:300])
    secrets = list(toks.values()) + list(ids) + ["10.20.30.40", "/srv/secret/challenge", "private-snap", "/home/someone", "Quiet Person",
                                                 "Cheaty", "spot check of job 3", "revoked", "home.lan", "secret-level", "xcorp-s07"]
    found = [s for s in secrets if s in page]
    ok("none of the fixture's secrets: tokens, worker names, the error, the quarantine reason, params, the opted-out donor, "
       "the quarantined donor", not found, str(found))
    # 3. escaping
    ok("operator text escaped (title and note with markup)",
       "EXL3-V3 ladder &lt;b&gt;Qwen3.8-27B&lt;/b&gt; &amp; &quot;friends&quot;" in page and "&lt;/p&gt;&lt;script&gt;x()&lt;/script&gt;" in page
       and "<b>" not in page)
    hostile = dict(data, board=[{"rank": 1, "name": '<img src=x onerror=alert(1)>"\'&', "lumens": 1.0, "jobs": 1, "last_seen": NOW}],
                   recent=[{"t": NOW, "type": "x", "label": "<i>l</i>", "by": "</td><script>", "state": "<b>", "listed": True}])
    hp = SP.render(hostile)
    sh2 = Shape(); sh2.feed(hp); sh2.close()
    ok("render() escapes whatever reaches it (a hostile name past the filter, label, state)",
       "&lt;img src=x onerror=alert(1)&gt;&quot;&#x27;&amp;" in hp and "<img" not in hp and "<script" not in hp and
       not sh2.errors and "img" not in sh2.tags, str(sh2.errors[:2]))
    ok("display names that are markup or an address are not shown (counted with the unlisted donors)",
       "alert(1)" not in page and "10.20.30" not in page)
    # 4. the numbers
    txt = " ".join(" ".join(sh.text).split())
    ok("campaign: ladder 1 / 73 levels (1 checked), search 5 evals of 5,000 planned (3 queued), goldens 1 / 2, spot checks 1 / 1 / 1",
       "ladder 1 / 73 levels (1 checked)" in txt and "search 5 evals of 5,000 planned, 3 queued" in txt and "goldens 1 / 2 chunks" in txt
       and "spot checks 1 passed 1 failed, 1 pending" in txt, txt[:400])
    ok("levels in progress: k2.5-i8 1 / 3, k4-i8 0 / 1; the bogus level name never shown",
       "k2.5-i8 1 / 3" in txt and "k4-i8 0 / 1" in txt and "secret-level" not in txt)
    ok("jobs by type: running / queued / 24 h / all (re-runs apart); cards busy",
       "quantize a layer 1 3 1 4" in txt and "quantize the LM head 0 0 0 1" in txt and "score a candidate mix 0 3 3 5" in txt and "record reference outputs 0 1 1 1" in txt and
       "plus 1 verification re-runs" in txt and "2 cards busy on 1 workers" in txt, txt[txt.find("Jobs"):txt.find("Jobs") + 400])
    b = data["board"]
    ok("leaderboard: opted-in donors only, by lumens (Ada 69, Zoë 20); not: opted-out, quarantined, address / markup names, verifier",
       [r["name"] for r in b] == ["Ada Lovelace", "Zoë 🚀 Team"] and b[0]["lumens"] == 69.0 and b[0]["jobs"] == 5 and
       data["others"] == (3, 6.5) and "1 Ada Lovelace 69.0 5 now" in txt, f"{[(r['name'], r['lumens'], r['jobs']) for r in b]} others {data['others']}")
    rec = data["recent"]
    ok("recent work: newest first, labels without paths, 'by' = listed name / a donor / our hardware, states",
       [r["t"] for r in rec] == sorted((r["t"] for r in rec), reverse=True) and any(r["label"] == "k3.25-i8 · layer 2" and r["by"] == "Ada Lovelace" and r["state"] == "verified" for r in rec)
       and any(r["by"] == "our hardware" for r in rec) and any(r["label"] == "k2.5-i8 · layer 0" and r["state"] == "awaiting check" for r in rec)
       and any(r["state"] == "spot check pending" and r["by"] == "a donor" for r in rec) and any(r["label"] == "golden · xcorp-web c0001" for r in rec)
       and not any(r["by"] == "Cheaty McCheat" for r in rec), str([(r["label"], r["by"], r["state"]) for r in rec[:6]]))
    ok("last update time on the page (UTC)", "updated 2026-11-05" in txt and "Last update 2026-11-05" in txt)
    # 5. standalone: a read-only connection, campaign.json next to the db
    S.db.close()
    import hashlib
    fp = lambda: [hashlib.sha256(open(f, "rb").read()).hexdigest() for f in (db, db + "-wal")    # (a reader may create an
                  if os.path.exists(f) and os.path.getsize(f)]                                     # empty -wal / -shm)
    before = fp()
    r = subprocess.run([sys.executable, "-I", f"{HERE}/server/statuspage.py", "--db", db, "--out", f"{W}/static/index.html"],
                       capture_output=True, text=True)
    sp = open(f"{W}/static/index.html").read() if os.path.exists(f"{W}/static/index.html") else ""
    ok("statuspage.py --db --out: builds from the database alone (read-only), campaign.json found next to it",
       r.returncode == 0 and "ladder 1 / 73 levels" in sp and oct(os.stat(f"{W}/static/index.html").st_mode & 0o777) == "0o644",
       (r.stdout + r.stderr).strip()[-200:])
    ok("building the page wrote nothing to the database (file hashes unchanged)", fp() == before)
    # 6. display-name rules (the API's check)
    good, bad = ["Ada Lovelace", "Zoë 🚀 Team", "team-42_x", "李雷"], ["", " ", "x" * 33, "10.0.0.1", "me@example.org", "http://x.y",
                                                                 "a<b", "rq" + "A" * 30, "tab\there\u200b", "deadbeef" * 4, "a&b"]
    gres = [D.clean_display_name(x) for x in good]
    bres = []
    for x in bad:
        try: D.clean_display_name(x); bres.append(x)
        except ValueError: pass
    ok("display names: printable, 1-32 chars; addresses, e-mails, URLs, tokens, long hex, markup and control characters refused",
       gres == good and not bres, f"accepted bad {bres}")
    ok("scrub(): paths, IPs, tokens, e-mails out of donor-facing text",
       D.scrub("fail at /root/x/y.py from 10.1.2.3:8740 tok rq" + "Z" * 30 + " a@b.cd https://h/x") ==
       "fail at <path> from <ip> tok <token> <email> <url>", D.scrub("fail at /root/x/y.py from 10.1.2.3:8740 tok rq" + "Z" * 30 + " a@b.cd https://h/x"))
    n_ok = sum(r[1] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
