#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""score_real.py --ckpt CKPT --hess HESSDIR --runs RUNS --layers A-B [--model M] [--work DIR] [--only-spread]
-- verification by score on REAL quantizer output (quantize/ourq.py runs; CPU only, numpy).

RUNS holds ourq output dirs (q-LNN.safetensors + proxy.jsonl), named
  k<K>            honest, seed 0, at each K of a ladder (e.g. k2 k2.25 ... k4)
  k<K>-s<N>       honest repeats with --seed-offset N
  att-retarget-k2 an attack: K3 codes of the dequantized K2 weights (lower-K quality in K3 codes)
  att-mul1        an attack: K3 codes made with the mul1 codebook, submitted as i8
  att-nohess      an attack: K3 with the Hessian damped away (sigma_reg 1.0: no LDLQ feedback to speak of)
and the test plays a campaign through the real scorer (server/score.py) in a fresh queue database: the K3 level first
(a donor's result, provisional, anchored by a verifier re-run = another seed), honest repeats by other donors, then
the other levels in ladder order (each judged on the curve of the earlier ones), then the attacks, plus synthetic
attacks on real codes (1% of one tensor's words changed, random codes, K2 codes padded to K3, a K2.75 shard declared
K3). It also checks the scorer's error against ourq's own log (same quantity) and times the scoring.
--only-spread: just the honest seed spread and the scorer-vs-log agreement for the runs given (e.g. 27B)."""
import argparse, glob, json, math, os, re, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server.store import Store
from server import score as SC
from jobs import qwen, quant_common as Q
from checker.check import ST

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--hess", required=True); ap.add_argument("--runs", required=True)
ap.add_argument("--layers", required=True); ap.add_argument("--model", default="m"); ap.add_argument("--work")
ap.add_argument("--only-spread", action="store_true")
a = ap.parse_args()
W = a.work or os.path.join(a.runs, "_score_real"); shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
DATA, RES = f"{W}/data", f"{W}/results"; os.makedirs(RES)
lo, hi = map(int, a.layers.split("-")); LAYERS = list(range(lo, hi + 1))
RESULTS = []


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


subprocess.run([sys.executable, "-I", f"{HERE}/server/admin.py", "--db", f"{W}/q.sqlite", "--data", DATA, "datapack", a.model, "r0",
                a.ckpt, a.hess, "--layers", a.layers, "--no-head", "--n-params", "1000000"], check=True, stdout=subprocess.DEVNULL)
dp = lambda rel: os.path.join(DATA, rel)
S = Store(f"{W}/q.sqlite")
SETTINGS = {"sigma_reg": 0.025, "sigma_reg_in": 0.005, "sigma_reg_out": None, "scaled_hessian": True, "qtf2": True, "hout": None,
            "seed_offset": 0}
WK = {}


def worker(name, verifier=False):
    if name not in WK:
        WK[name] = S.add_worker(name)[0]
        if verifier: S.q("UPDATE workers SET verifier=1 WHERE id=?", (WK[name],))
    return WK[name]


def params(K, layer, seed=0):
    p = {"model": a.model, "revision": "r0", "arch": "qwen3_5", "layer": layer, "K": K, "codebook": "i8", "level": f"k{K:g}",
         "settings": dict(SETTINGS, seed_offset=seed), "verify": "score"}
    p["stamp"] = Q.settings_stamp(p); return p


def result_dir(src, layer, d, K=None, transform=None):
    """a worker's upload from an ourq output dir: the layer's shard, its proxy lines, job.json, stamp.json"""
    os.makedirs(d, exist_ok=True)
    shard = f"q-L{layer:02d}.safetensors"
    if transform is None: shutil.copyfile(f"{src}/{shard}", f"{d}/{shard}")
    else: transform(f"{src}/{shard}", f"{d}/{shard}")
    keys = set(qwen.keys(qwen.layer_plan("qwen3_5", layer)))
    with open(f"{d}/proxy.jsonl", "w") as f:
        for l in open(f"{src}/proxy.jsonl"):
            r = json.loads(l)
            if r["key"] in keys:
                if K is not None: r["K"] = K
                if r.get("codebook") != "i8": r["codebook"] = "i8"          # (the mul1 attack: the worker claims i8)
                f.write(json.dumps(r) + "\n")
    json.dump([], open(f"{d}/stamp.json", "w"))


def submit(src, K, layer, who, seed=0, verify_of=None, transform=None, Klog=None):
    p = params(K, layer, seed)
    wid = worker(who, who.startswith("ver"))
    with S.tx() as db:
        jid = S.enqueue(db, "quant_layer", 3, p, [], {"gpu": True}, 0, level=p["level"], verify_of=verify_of,
                        needs="verifier" if verify_of else None)
        db.execute("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=? WHERE id=?", (wid, time.time(), jid))
    result_dir(src, layer, f"{RES}/{jid}", Klog, transform)
    json.dump({"settings_stamp": p["stamp"]}, open(f"{RES}/{jid}/job.json", "w"))
    t0 = time.time(); st = SC.score_job(S, RES, dp, jid); dt = time.time() - t0
    return jid, st, dt


def job(jid): return S.q("SELECT * FROM jobs WHERE id=?", (jid,))[0]


def rels(jid): return {r["key"]: r["rel"] for r in S.q("SELECT key, rel FROM scores WHERE job_id=?", (jid,))}


def logged(src):
    return {json.loads(l)["key"]: json.loads(l)["proxy_err"] for l in open(f"{src}/proxy.jsonl")}


def unquarantine():
    S.q("UPDATE workers SET quarantined=0")


runs = {os.path.basename(d): d for d in glob.glob(f"{a.runs}/*") if os.path.isdir(d) and glob.glob(f"{d}/q-L*.safetensors")}
print("runs:", " ".join(sorted(runs)))
times = []

# ------------------------------------------------------------------ the scorer = ourq's own logged error (honest runs)
worst = 0.0
seeds = sorted(r for r in runs if re.fullmatch(r"k[0-9.]+(-s\d+)?", r))
# ------------------------------------------------------------------ campaign
if a.only_spread:
    by = {}
    for name in seeds:
        K = float(re.match(r"k([0-9.]+)", name).group(1)); s = int(name.split("-s")[1]) if "-s" in name else 0
        for L in LAYERS:
            jid, st, dt = submit(runs[name], K, L, f"donor-{name}", seed=s); times.append(dt)
            lg = logged(runs[name])
            for k, r in rels(jid).items():
                worst = max(worst, abs(r / lg[k] - 1)); by.setdefault((K, k), []).append(r)
    ok("the scorer's error equals ourq's logged proxy_err on every honest tensor (|ratio - 1| <= 2e-3)", worst <= 2e-3, f"worst {worst:.2e}")
    for (K, k), v in sorted(by.items()):
        if len(v) > 1: print(f"  spread K{K:g} {k.replace('model.language_model.', ''):45s} max/min {max(v) / min(v) - 1:+.4f}  ({len(v)} seeds)")
    sp = max(max(v) / min(v) - 1 for v in by.values() if len(v) > 1) if any(len(v) > 1 for v in by.values()) else float("nan")
    ok(f"honest seed spread at full size below the repeat tolerance ({SC.TOL['repeat']:.0%})", sp < SC.TOL["repeat"], f"max {sp:+.4f}")
    print(f"scoring: {np.mean(times):.1f} s per layer (mean of {len(times)}), max {max(times):.1f} s")
    print(f"\n{sum(c for _, c in RESULTS)}/{len(RESULTS)} checks passed"); sys.exit(0 if all(c for _, c in RESULTS) else 1)

L0 = LAYERS
# 1. the K3 level: donor A, provisional; the anchor (a verifier, seed 1) decides it
ja = {L: submit(runs["k3"], 3, L, "donor-a") for L in L0}
ok("K3 level, first result per layer: provisional (no reference yet), an anchor re-run queued",
   all(st == "provisional" for _, st, _ in ja.values()) and all(S.q("SELECT 1 FROM jobs WHERE verify_of=?", (j,)) for j, _, _ in ja.values()))
for L, (j, _, dt) in ja.items():
    times.append(dt)
    rr = S.q("SELECT id FROM jobs WHERE verify_of=?", (j,))[0]["id"]
    S.q("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=? WHERE id=?", (worker("ver", True), time.time(), rr))
    result_dir(runs["k3-s1"], L, f"{RES}/{rr}"); json.dump({"settings_stamp": params(3, L, 0)["stamp"]}, open(f"{RES}/{rr}/job.json", "w"))
    SC.score_job(S, RES, dp, rr)
ok("anchored by the verifier's re-run (another seed): every K3 result passes on the repeat basis",
   all(job(j)["check_state"] == "pass" for j, _, _ in ja.values()), str([job(j)["check_state"] for j, _, _ in ja.values()]))
for L, (j, _, _) in ja.items():
    lg = logged(runs["k3"])
    for k, r in rels(j).items(): worst = max(worst, abs(r / lg[k] - 1))
# 2. honest repeats by other donors (best-of-N copies): seeds 2-4
reps = [submit(runs[n], 3, L, f"donor-{n}", seed=int(n[-1])) for n in ("k3-s2", "k3-s3", "k3-s4") if n in runs for L in L0]
ok("honest repeats (seeds 2-4, other donors): all pass", all(st == "pass" for _, st, _ in reps), str([st for _, st, _ in reps]))
spread = {}
for n in ("k3", "k3-s1", "k3-s2", "k3-s3", "k3-s4"):
    if n in runs:
        for k, v in logged(runs[n]).items(): spread.setdefault(k, []).append(v)
sp = {k: max(v) / min(v) - 1 for k, v in spread.items()}
print("  honest K3 spread over seeds (max/min - 1): " + ", ".join(f"{k.split('.')[-1]}@L{k.split('.')[3]} {v:+.4f}" for k, v in sorted(sp.items())))
# 3. the other levels in ladder order, each on the curve of the ones before
order = [K for K in (2.5, 3.5, 2, 4, 2.25, 2.75, 3.25) if f"k{K:g}" in runs]
lv = {}
for K in order:
    lv[K] = [submit(runs[f"k{K:g}"], K, L, f"donor-k{K:g}") for L in L0]
    for (j, st, dt), L in zip(lv[K], L0):
        times.append(dt); lg = logged(runs[f"k{K:g}"])
        for k, r in rels(j).items(): worst = max(worst, abs(r / lg[k] - 1))
bases = {K: sorted({json.loads(job(j)["check_detail"])["tensors"][0].get("basis") or "-" for j, _, _ in v}) for K, v in lv.items()}
ok("honest levels in ladder order (K " + ", ".join(f"{K:g}" for K in order) + ") all pass on the curve",
   all(st == "pass" for v in lv.values() for _, st, _ in v), str({K: [st for _, st, _ in v] for K, v in lv.items()}) + f" bases {bases}")
for n in ("k2.5-s1", "k2.5-s2"):
    if n in runs:
        r = [submit(runs[n], 2.5, L, f"donor-{n}", seed=int(n[-1])) for L in L0]
        ok(f"honest repeat {n}: pass", all(st == "pass" for _, st, _ in r), str([st for _, st, _ in r]))
ok("the scorer's error equals ourq's logged proxy_err on every honest tensor (|ratio - 1| <= 2e-3)", worst <= 2e-3, f"worst {worst:.2e}")

# 4. attacks at K3 (each from a new donor; a fail quarantines it)
def attack(name, src, transform=None, K=3, expect="fail"):
    r = [submit(src, K, L, f"mallory-{name}", transform=transform, Klog=K) for L in L0]
    det = []
    for j, st, _ in r:
        ev = S.q("SELECT detail FROM events WHERE job_id=? AND kind LIKE 'score_%' ORDER BY id DESC LIMIT 1", (j,))
        det.append(json.loads(ev[0]["detail"] or "{}").get("total_rel") if ev else None)
    ok(f"attack {name}: {expect}", all(st == expect for _, st, _ in r), f"{[st for _, st, _ in r]} total rel {det}")
    unquarantine()
    return r
honest = {L: json.loads(job(j)["check_detail"])["total_rel"] for L, (j, _, _) in ja.items()}
print("  honest K3 total rel per layer:", {L: round(v, 5) for L, v in honest.items()})
if "att-retarget-k2" in runs: attack("retarget-k2 (K3 codes of the K2 dequantized weights)", runs["att-retarget-k2"])
if "att-mul1" in runs:     # i8 is the int8-rounded mul1 codebook: mul1 codes decoded as i8 are an i8 quant of nearly the same
    r = [submit(runs["att-mul1"], 3, L, "donor-mul1", Klog=3) for L in L0]   # quality, judged exactly as the kernel decodes them
    tot = [json.loads(job(j)["check_detail"] or "{}").get("total_rel") for j, _, _ in r]
    print(f"  (info) mul1 codes declared i8: {[st for _, st, _ in r]}, total rel {tot} vs honest {list(honest.values())}: judged on "
          "their quality as i8 codes, which is what the kernel will decode"); unquarantine()
if "att-nohess" in runs: attack("Hessian ignored (sigma_reg 1.0)", runs["att-nohess"])


def perturb(src, dst, frac=0.01, seed=5):
    st = ST(src); keys = st.keys(); out = {}
    first = sorted(k for k in keys if k.endswith(".trellis"))[0]
    rng = np.random.default_rng(seed)
    for k in keys:
        x = st.get(k).copy()
        if k == first:
            f = x.reshape(-1); i = rng.choice(f.size, max(1, int(f.size * frac)), replace=False)
            f[i] = rng.integers(-32768, 32768, i.size).astype(np.int16)
        out[k] = x
    write(dst, out, st)


def randomize(src, dst):
    st = ST(src); rng = np.random.default_rng(9)
    write(dst, {k: (rng.integers(-32768, 32768, st.get(k).shape).astype(np.int16) if k.endswith(".trellis") else st.get(k))
                for k in st.keys()}, st)


def pad_to(words):
    def f(src, dst):
        st = ST(src); out = {}
        for k in st.keys():
            x = st.get(k)
            if k.endswith(".trellis"): x = np.concatenate([x, np.zeros(x.shape[:2] + (words - x.shape[2],), np.int16)], -1)
            out[k] = x
        write(dst, out, st)
    return f


DT = {np.dtype(np.int16): "I16", np.dtype(np.float16): "F16", np.dtype(np.int32): "I32"}


def write(dst, tensors, like):
    import struct
    hdr, off, blobs = {"__metadata__": like.hdr.get("__metadata__", {})}, 0, []
    for k, x in tensors.items():
        x = np.ascontiguousarray(x); b = x.tobytes()
        hdr[k] = {"dtype": DT[x.dtype], "shape": list(like.hdr[k]["shape"]) if not k.endswith(".trellis") else list(x.shape),
                  "data_offsets": [off, off + len(b)]}; off += len(b); blobs.append(b)
    hb = json.dumps(hdr).encode(); hb += b" " * (-len(hb) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs: f.write(b)


attack("1% of one tensor's trellis words changed", runs["k3"], perturb)
attack("random codes", runs["k3"], randomize)
if "k2" in runs: attack("K2 codes padded to the K3 shape", runs["k2"], pad_to(48))
if "k2.75" in runs: attack("a K2.75 shard declared K3 (lower K, honest codes)", runs["k2.75"])
ok("every K3 result of the honest donors and the verifier is still accepted after the attacks",
   all(job(j)["state"] == "accepted" and job(j)["check_state"] == "pass" for j, _, _ in list(ja.values()) + reps))
print(f"scoring: {np.mean(times):.1f} s per layer (mean of {len(times)}), max {max(times):.1f} s")
print(f"\n{sum(c for _, c in RESULTS)}/{len(RESULTS)} checks passed")
sys.exit(0 if all(c for _, c in RESULTS) else 1)
