#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""searchq.py --server URL --token-file F COMMAND -- the search's side of rad-quant-at-home's batch API (stdlib only).

  submit SPEC.json            post a batch of eval_mix candidates (server/batch.py has the spec); prints {batch, ...}
  status ID                   per-candidate state, KL mean, spot-check state
  wait ID [--poll S] [--timeout S]   until every candidate is accepted (exit 1 on timeout)
  collect ID OUT              OUT/<candidate>.rows ([rows, 4] f32 {kl, nll_q, nll_p, top1}, radiance's FILE.rows) for
                              every finished candidate + OUT/results.jsonl; rows already there (same size) are kept,
                              rows of candidates no longer accepted (invalidated) are deleted
  adopt ID C [C ...] [--wait] the search acts on these candidates: each gets a full re-run on a trusted worker; --wait
                              until all are "confirmed" (exit 1 if one fails, i.e. stops being accepted)

Each candidate in status / results.jsonl has a "trust": confirmed (a trusted worker ran or fully re-ran it),
rerun-pending, spot-pending / spot-pass (a few documents re-scored), unverified. A candidate may carry "pred_kld" in the
spec (the predicted mean KL): far worse results are refused, far better ones are "suspect" and re-run in full.

The token is the server's --search-token-file token: it can enqueue eval_mix jobs on public goldens and read their
results, nothing else."""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

CAND_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def call(a, method, path, body=None, raw=False):
    tok = open(a.token_file).read().strip()
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(a.server.rstrip("/") + "/api/v0/search" + path, data=data, method=method,
                               headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                                        "User-Agent": "rqah-searchq/0.3"})
    try:
        resp = urllib.request.urlopen(r, timeout=300)
        return resp.read() if raw else json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True); ap.add_argument("--token-file", required=True)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("submit").add_argument("spec")
    sp.add_parser("status").add_argument("id", type=int)
    x = sp.add_parser("wait"); x.add_argument("id", type=int); x.add_argument("--poll", type=float, default=30)
    x.add_argument("--timeout", type=float, default=0)
    x = sp.add_parser("collect"); x.add_argument("id", type=int); x.add_argument("out")
    x = sp.add_parser("adopt"); x.add_argument("id", type=int); x.add_argument("cands", nargs="+")
    x.add_argument("--wait", action="store_true"); x.add_argument("--poll", type=float, default=30)
    a = ap.parse_args()
    if a.cmd == "submit": print(json.dumps(call(a, "POST", "/batch", json.load(open(a.spec)))))
    elif a.cmd == "status": print(json.dumps(call(a, "GET", f"/batch/{a.id}"), indent=1))
    elif a.cmd == "wait":
        t0 = time.time()
        while True:
            st = call(a, "GET", f"/batch/{a.id}")
            print(time.strftime("%H:%M:%S"), json.dumps(st["counts"]), flush=True)
            if st["done"]: break
            if a.timeout and time.time() - t0 > a.timeout: sys.exit(1)
            time.sleep(a.poll)
    elif a.cmd == "collect":
        st = call(a, "GET", f"/batch/{a.id}")
        os.makedirs(a.out, exist_ok=True); n = 0
        for it in st["items"]:
            if not CAND_RE.fullmatch(str(it["id"])): continue          # the server's id becomes a file name
            dst = os.path.join(a.out, f"{it['id']}.rows")
            if it["state"] != "accepted":
                if os.path.exists(dst): os.remove(dst)              # invalidated since an earlier collect
                continue
            b = call(a, "GET", f"/batch/{a.id}/rows/{it['id']}", raw=True)
            if not (os.path.exists(dst) and os.path.getsize(dst) == len(b)):
                with open(dst + ".part", "wb") as f: f.write(b)
                os.replace(dst + ".part", dst)
            n += 1
        with open(os.path.join(a.out, "results.jsonl.part"), "w") as f:
            for it in st["items"]: f.write(json.dumps(it) + "\n")
        os.replace(os.path.join(a.out, "results.jsonl.part"), os.path.join(a.out, "results.jsonl"))
        print(json.dumps({"batch": a.id, "collected": n, "of": len(st["items"])}))
    elif a.cmd == "adopt":
        print(json.dumps(call(a, "POST", f"/batch/{a.id}/adopt", {"candidates": a.cands})), flush=True)
        while a.wait:
            st = {it["id"]: it for it in call(a, "GET", f"/batch/{a.id}")["items"]}
            gone = [c for c in a.cands if st[c]["state"] != "accepted"]
            if gone: sys.exit(f"no longer accepted (failed its trusted re-run?): {gone}")
            if all(st[c].get("trust") == "confirmed" for c in a.cands): print("confirmed"); break
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
