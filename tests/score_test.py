#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""score_test.py [WORKDIR] -- verification by score (server/score.py, docs/VERIFY.md) on a synthetic layer, no GPU,
in process (~1 min): a tiny qwen3_5 GDN layer (every linear 256 x 256), random PSD input Hessians, and EXL3 codes
made in numpy. The "honest" result is a code set C0 with the bf16 weights defined as decode(C0) + noise, so its error
is known; an honest repeat changes two trellis words (another valid quant of nearly the same quality).

Checks: the band logic (repeat / interp / extrap, one-sided, the trusted bound, family multipliers); the job flow
(no reference -> provisional -> anchor re-run on a verifier -> trusted rows -> the original passes and earns; an
honest repeat by another donor passes on the repeat basis); attacks that must fail (perturbed codes, random codes,
lower-K codes padded to the declared K's shape, a lower-K shape, a wrong codebook marker, a non-finite scale, a
published result replayed, a result 1/8 bit worse than the curve); a duplicate of another donor's codes passes but
earns nothing; a failed score quarantines and invalidates that donor's unconfirmed results but leaves its
score-verified ones; re-checks fail an earlier extrapolated pass once neighbours arrive; #12: the all band's
center is the upper median, an honest result after a better-than-honest one is suspect (not failed) and its anchor
passes it, a trusted-band fail still fails, an outlier-better result logs an event; best of N at export; #11:
tile fingerprints, perturbed best-of-N copies and another donor's perturbed copy earn nothing, a perturbed published
result is a replay."""
import json, math, os, shutil, struct, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from server.store import Store
from server import score as SC, verify as V
from jobs import qwen, quant_common as Q
from checker import exl3dec

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="rqah-score-"))
RESULTS = []
N = 256                                                     # every linear N x N
SIGMA = 0.3                                                 # noise of the "honest" quant (rel ~ 0.08)


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def write_st(path, tensors, meta=None):
    """{name: (dtype str, np array)} -> safetensors"""
    hdr, off, blobs = {}, 0, []
    if meta: hdr["__metadata__"] = meta
    for k, (dt, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes(); hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b); blobs.append(b)
    hb = json.dumps(hdr).encode(); hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs: f.write(b)


def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


rng = np.random.default_rng(1)
P = {"model": "tiny", "revision": "r1", "arch": "qwen3_5", "layer": 0, "K": 3, "codebook": "i8", "level": "k3",
     "settings": {"sigma_reg": 0.025, "sigma_reg_in": 0.005, "sigma_reg_out": None, "scaled_hessian": True, "qtf2": True,
                  "hout": None, "seed_offset": 0}, "verify": "score"}
P["stamp"] = Q.settings_stamp(P)
PLAN = qwen.layer_plan("qwen3_5", 0)
KEYS = qwen.keys(PLAN)
DATA = f"{W}/data"; RES = f"{W}/results"
ROOT = f"{DATA}/models/tiny@r1"
dp = lambda rel: os.path.join(DATA, rel)


def codes(K, seed):
    r = np.random.default_rng(seed)
    t = {"trellis": r.integers(-32768, 32768, (N // 16, N // 16, round(16 * K)), dtype=np.int64).astype(np.int16),
         "suh": (r.choice([-1.0, 1.0], N) * 0.05).astype(np.float16), "svh": (r.choice([-1.0, 1.0], N) * 0.05).astype(np.float16),
         "mul1": np.array(exl3dec.MUL1, dtype=np.uint32).view(np.int32).reshape(())}
    return t


def setup():
    """the data root (weights = decode(C0) + noise, Hessians) and the honest code sets C0 per tensor"""
    for d in ("weights", "hessians"): os.makedirs(f"{ROOT}/{d}", exist_ok=True)
    C0, wts = {}, {}
    for i, key in enumerate(KEYS):
        C0[key] = codes(3, 100 + i)
        D, _ = exl3dec.decode(C0[key], 3, "i8")                       # [k, n]
        Wt = D + SIGMA * D.std() * rng.standard_normal(D.shape).astype(np.float32)
        wts[f"{key}.weight"] = ("BF16", bf16(Wt.T))
    write_st(f"{ROOT}/weights/L00.safetensors", wts)
    for hname, _ in PLAN:
        X = rng.standard_normal((N, 4 * N)).astype(np.float32) * np.linspace(0.2, 2, N, dtype=np.float32)[:, None]
        H = X @ X.T / X.shape[1]
        up = np.concatenate([H[r, r:] for r in range(N)]).astype(np.float32)
        write_st(f"{ROOT}/hessians/{hname}.safetensors", {"hessian_upper": ("F32", up)}, {"k": str(N)})
    return C0


def make_result(d, cs, p=None, K=None):
    """a result dir as a worker uploads it: the shard, proxy.jsonl (informational), job.json, stamp.json"""
    p = p or P; K = p["K"] if K is None else K
    os.makedirs(d, exist_ok=True)
    t = {}
    for key, c in cs.items():
        t[f"{key}.trellis"] = ("I16", c["trellis"]); t[f"{key}.suh"] = ("F16", c["suh"]); t[f"{key}.svh"] = ("F16", c["svh"])
        t[f"{key}.mul1"] = ("I32", c["mul1"])
    write_st(f"{d}/q-L{p['layer']:02d}.safetensors", t, {"ourq_stamp": "0123456789abcdef"})
    with open(f"{d}/proxy.jsonl", "w") as f:
        for key in cs: f.write(json.dumps({"key": key, "K": K, "codebook": p["codebook"], "scaled_hessian": True, "proxy_err": 0.01}) + "\n")
    json.dump({"settings_stamp": p["stamp"]}, open(f"{d}/job.json", "w")); json.dump([], open(f"{d}/stamp.json", "w"))


def mutate(cs, n_words, seed, keys=None):
    r = np.random.default_rng(seed); out = {}
    for key, c in cs.items():
        c = {k: v.copy() for k, v in c.items()}
        if keys is None or key in keys:
            flat = c["trellis"].reshape(-1)
            idx = r.choice(flat.size, n_words, replace=False)
            flat[idx] = r.integers(-32768, 32768, n_words).astype(np.int16)
        out[key] = c
    return out


S = None
WK = {}


def accept_job(p, cs, worker, verify_of=None):
    """a job accepted from `worker` with these codes (as server.complete does), returns its id"""
    with S.tx() as db:
        jid = S.enqueue(db, "quant_layer", 3, p, [], {"gpu": True, "cards": 1, "vram_gb": 4, "ram_gb": 4}, 0, level=p["level"],
                        verify_of=verify_of, needs="verifier" if verify_of else None)
        now = time.time()
        db.execute("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=?, accepted_lease=? WHERE id=?",
                   (WK[worker], now, f"lease-{jid}", jid))
    make_result(f"{RES}/{jid}", cs, p)
    return jid


def job(jid): return S.q("SELECT * FROM jobs WHERE id=?", (jid,))[0]


def lumens(worker): return S.q("SELECT COALESCE(SUM(lumens), 0) FROM points WHERE worker_id=?", (WK[worker],))[0][0]


def lumens_of(jid): return S.q("SELECT COALESCE(SUM(lumens), 0) FROM points WHERE job_id=?", (jid,))[0][0]


def run():
    global S
    SC.NOISE, SC.NOISE_DEFAULT = {}, 0.0        # the logic, without the seed-noise term (calibrated on real runs: score_real.py)
    # the synthetic "honest repeats" below are C0 with two words changed (the only good codes this fixture has): exactly
    # what near-duplicate detection (#11) calls a copy, so it is off until its own section
    near_dup = SC.NEAR_DUP; SC.NEAR_DUP = 1.01
    os.makedirs(RES, exist_ok=True)
    C0 = setup()
    S = Store(f"{W}/q.sqlite")
    for name in ("alice", "bob", "carol", "mallory", "trent", "ver"):
        WK[name] = S.add_worker(name)[0]
    with S.tx() as db: db.execute("UPDATE workers SET verifier=1 WHERE id=?", (WK["ver"],))

    # ---------------------------------------------------------------- band logic
    b = SC.bound([(2.0, 0.08), (4.0, 0.08 * math.exp(-1.37 * 2))], 3.0, "i8", "x.mlp.up_proj")
    ok("band: interp between K2 and K4 = the geometric mean, tol 4%", b[2] == "interp" and abs(math.exp(b[0]) - 0.08 * math.exp(-1.37)) < 1e-12 and b[1] == 0.04)
    b = SC.bound([(2.0, 0.08)], 3.0, "i8", "x.mlp.up_proj")
    ok("band: extrap one bit along the slope (1.37 / bit), tol 4% + 3%/bit", b[2] == "extrap" and abs(b[0] - (math.log(0.08) - 1.37)) < 1e-12 and abs(b[1] - 0.07) < 1e-12)
    b = SC.bound([(3.0, 0.02), (3.0, 0.019), (2.0, 0.08)], 3.0, "i8", "x.self_attn.k_proj")
    ok("band: repeat = the minimum at that K, tol 2%", b[2] == "repeat" and abs(math.exp(b[0]) - 0.019) < 1e-12 and b[1] == 0.02)
    ok("band: one-sided (far better passes, 1/8 bit worse fails, within tol passes)",
       SC.judge(0.001, b)[0] and not SC.judge(0.019 * math.exp(1.37 / 8), b)[0] and SC.judge(0.019 * 1.015, b)[0])
    b = SC.bound([(2.0, 0.08)], 4.0, "i8", "x.self_attn.k_proj")
    ok("band: k_proj has its own wider curve table (extrap 8% + 12%/bit)", abs(b[1] - 0.32) < 1e-12, f"{b[1]:.3f}")
    SC.NOISE, SC.NOISE_DEFAULT = {"in_proj_z": 30.0}, 8.0
    ok("band: + 4 x the seed noise NOISE / sqrt(in x out) (30 for the input group, 8 for the rest)",
       abs(SC.tol("x.linear_attn.in_proj_z", "repeat", 31_457_280) - (0.02 + 120 / math.sqrt(31_457_280))) < 1e-12 and
       abs(SC.tol("x.mlp.down_proj", "interp", 89_128_960) - (0.04 + 32 / math.sqrt(89_128_960))) < 1e-12)
    SC.NOISE, SC.NOISE_DEFAULT = {}, 0.0
    ok("band: no rows -> None (provisional)", SC.bound([], 3.0, "i8", "x.mlp.up_proj") is None)

    # ---------------------------------------------------------------- cold start: provisional -> anchor -> pass
    ja = accept_job(P, C0, "alice")
    st = SC.score_job(S, RES, dp, ja)
    reruns = S.q("SELECT * FROM jobs WHERE verify_of=?", (ja,))
    ok("cold start: no reference -> provisional, one anchor re-run queued for verifiers (verify_state untouched)",
       st == "provisional" and len(reruns) == 1 and reruns[0]["needs"] == "verifier" and job(ja)["verify_state"] is None, st)
    ok("cold start: no lumens before the anchor", lumens("alice") == 0)
    ok("the scores recompute the error from the codes (rel ~ noise^2, not the logged 0.01)",
       all(0.03 < r["rel"] < 0.2 for r in S.q("SELECT rel FROM scores WHERE job_id=?", (ja,))))
    # the verifier's re-run: another honest quant (two words differ)
    S.q("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=? WHERE id=?", (WK["ver"], time.time(), reruns[0]["id"]))
    make_result(f"{RES}/{reruns[0]['id']}", mutate(C0, 2, 7))
    st2 = SC.score_job(S, RES, dp, reruns[0]["id"])
    ok("anchor: the verifier's re-run is scored as trusted rows (pass)", st2 == "pass" and
       all(r["kind"] == "trusted" and r["state"] == "pass" for r in S.q("SELECT * FROM scores WHERE job_id=?", (reruns[0]["id"],))))
    ok("anchor: the provisional original is then decided against it (repeat basis): pass, lumens granted",
       job(ja)["check_state"] == "pass" and lumens("alice") > 0 and
       all(r["basis"] == "repeat" for r in S.q("SELECT * FROM scores WHERE job_id=?", (ja,))), job(ja)["check_state"])
    ok("verifier earns nothing", lumens("ver") == 0)

    # ---------------------------------------------------------------- honest repeat (another donor, other codes)
    jb = accept_job(P, mutate(C0, 2, 11), "bob")
    ok("honest repeat by another donor (other codes, same quality): pass on the repeat basis, earns",
       SC.score_job(S, RES, dp, jb) == "pass" and lumens("bob") > 0)

    # ---------------------------------------------------------------- attacks
    def attack(name, cs, worker="mallory", expect="fail", p=None):
        j = accept_job(p or P, cs, worker)
        st = SC.score_job(S, RES, dp, j)
        fail_ev = S.q("SELECT detail FROM events WHERE job_id=? AND kind='quarantine' OR (job_id=? AND kind LIKE 'score_%') ORDER BY id DESC LIMIT 1", (j, j))
        ok(name, st == expect, f"{st}: {fail_ev[0]['detail'][:150] if fail_ev and fail_ev[0]['detail'] else ''}")
        return j
    jm_ok = accept_job(P, mutate(C0, 2, 12), "mallory")            # mallory's first, honest result: score-verified
    SC.score_job(S, RES, dp, jm_ok)
    jm_open = accept_job({**P, "verify": "replicate"}, C0, "mallory")   # and an unconfirmed replicate-mode result
    attack("perturbed codes (1% of the trellis words of one tensor) -> fail", mutate(C0, 123, 13, keys={KEYS[0]}))
    ok("a failed score quarantines the donor", S.q("SELECT quarantined FROM workers WHERE id=?", (WK["mallory"],))[0][0] == 1)
    ok("quarantine: the donor's score-verified result stands; its unconfirmed one is invalidated and requeued",
       job(jm_ok)["state"] == "accepted" and job(jm_ok)["check_state"] == "pass" and job(jm_open)["state"] == "queued")
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    attack("random codes -> fail", {k: codes(3, 900 + i) for i, k in enumerate(KEYS)})
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    low = {}
    for i, k in enumerate(KEYS):                                    # K2 codes padded to K3's 48 words per tile
        c = codes(2, 300 + i); c["trellis"] = np.concatenate([c["trellis"], np.zeros((N // 16, N // 16, 16), np.int16)], -1); low[k] = c
    attack("lower-K codes (K2) padded to the K3 shape -> fail", low)
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    attack("lower-K shape (K2 codes declared as K3) -> fail (structure)", {k: codes(2, 400 + i) for i, k in enumerate(KEYS)})
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    bad = mutate(C0, 2, 14); bad[KEYS[1]]["mul1"] = np.array(12345, np.int32).reshape(())
    attack("wrong codebook marker (mul1) -> fail (format)", bad)
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    bad = mutate(C0, 2, 15); bad[KEYS[2]]["svh"][3] = np.float16("nan")
    attack("a non-finite scale -> fail (format)", bad)
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))

    # replay of a published result: import it as a guard, then submit it
    pub = mutate(C0, 3, 21); pd = f"{W}/published"; make_result(pd, pub)
    SC.import_reference(S, dp, "quant_layer", P, pd, kind="published", note="released v2 level")
    attack("replay of a published model's codes -> fail", pub)
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    ok("a published import is a replay guard only, never a band reference",
       all(r["state"] == "guard" for r in S.q("SELECT state FROM scores WHERE kind='published'")))

    # duplicate of another donor's codes (e.g. a best-of-N copy answered with someone else's result)
    p2 = dict(P, settings=dict(P["settings"], seed_offset=1)); p2["stamp"] = Q.settings_stamp(p2)
    before = lumens("carol")
    jd = accept_job(p2, mutate(C0, 2, 11), "carol")                 # = bob's codes
    st = SC.score_job(S, RES, dp, jd)
    ok("duplicate of another donor's codes: passes (valid result) but earns nothing", st == "pass" and lumens("carol") == before and
       json.loads(job(jd)["check_detail"]).get("duplicate_of") == jb)

    # 1/8 bit worse than the curve: references at K2.875 and K3.125 placed around this K3 result's true quality
    p3 = dict(P, K=3.0, level="k3-cheat"); p3["stamp"] = Q.settings_stamp(p3)
    cs = mutate(C0, 2, 31)
    make_result(f"{W}/probe", cs, p3)
    rows, _ = SC.score_files(S, "quant_layer", p3, f"{W}/probe", dp)
    rel = {r["key"]: r["rel"] for r in rows}
    ser2 = "curve-test-" + SC.series(p3)
    # a series of its own: honest neighbours as if this result were quantized at 1/8 bit less than declared
    for key in KEYS:
        honest_k3 = rel[key] * math.exp(-1.37 / 8)
        for k in (2.875, 3.125):
            S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t) VALUES(?,?,?,?,?,?,?,?)",
                (ser2, key, k, honest_k3 * math.exp(-1.37 * (k - 3)), "trusted", "pass", "import", time.time()))
    states = [SC.decide(S.db, ser2, key, 3.0, "i8", rel[key])[0] for key in KEYS]
    ok("a result 1/8 bit worse than its interpolated neighbours fails every tensor (interp, tol 4%)", states == ["fail"] * len(KEYS), str(states))
    states = [SC.decide(S.db, ser2, key, 3.0, "i8", rel[key] * math.exp(-1.37 / 8) * 1.02)[0] for key in KEYS]
    ok("... while one 2% off the curve passes", states == ["pass"] * len(KEYS), str(states))

    # re-check: a result 6% above the curve passes while the only reference is 1 bit away (extrap: tol 7%), and fails
    # once neighbours on both sides arrive (interp: tol 4%)
    ser3 = "recheck-test"
    pr = dict(P, level="k3-rc"); pr["stamp"] = Q.settings_stamp(pr)
    jr = accept_job(pr, mutate(C0, 2, 41), "trent")
    sres = {r["key"]: r["rel"] for r in SC.score_files(S, "quant_layer", pr, f"{RES}/{jr}", dp)[0]}
    def ref(k, key): return sres[key] / 1.06 * math.exp(-1.37 * (k - 3))
    for key in KEYS:
        S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t) VALUES(?,?,?,?,?,?,?,?)",
            (ser3, key, 2.0, ref(2.0, key), "trusted", "pass", "import", time.time()))
    series_orig = SC.series
    SC.series = lambda p: ser3 if p.get("level") == "k3-rc" else series_orig(p)
    try:
        st = SC.score_job(S, RES, dp, jr)
        ok("re-check: 6% above the curve, only a reference 1 bit away -> pass (extrap, tol 7%)", st == "pass" and lumens("trent") > 0, st)
        for key in KEYS:
            for k in (2.75, 3.25):
                S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t) VALUES(?,?,?,?,?,?,?,?)",
                    (ser3, key, k, ref(k, key), "trusted", "pass", "import", time.time()))
        SC.reevaluate(S, RES, ser3, KEYS)
        ev = [r["kind"] for r in S.q("SELECT kind FROM events WHERE job_id=? ORDER BY id", (jr,))]
        ok("re-check: neighbours on both sides arrive -> re-decided on interp (tol 4%): fail, donor quarantined, lumens "
           "revoked, the job requeued for anyone else", "score_recheck_fail" in ev and job(jr)["state"] == "queued" and
           job(jr)["exclude_worker"] == WK["trent"] and S.q("SELECT quarantined FROM workers WHERE id=?", (WK["trent"],))[0][0] == 1
           and lumens("trent") == 0 and all(r["state"] in ("fail", "void") for r in S.q("SELECT state FROM scores WHERE job_id=?", (jr,))),
           str(ev[-4:]))
    finally:
        SC.series = series_orig

    # #12: the all band's center is the upper median, and the trusted band is the authority
    b = SC.bound([(3.0, 0.02), (3.0, 0.019), (3.0, 0.025), (2.0, 0.08)], 3.0, "i8", "x.mlp.up_proj", center="median")
    b2 = SC.bound([(3.0, 0.02), (3.0, 0.0192)], 3.0, "i8", "x.mlp.up_proj", center="median")
    ok("band (#12): the all bound's center is the upper median of the rows at K (min stays for trusted rows)",
       abs(math.exp(b[0]) - 0.02) < 1e-12 and abs(math.exp(b2[0]) - 0.02) < 1e-12)
    ser4 = "griefing-test"
    pg = dict(P, level="k3-gr"); pg["stamp"] = Q.settings_stamp(pg)
    cs_c = mutate(C0, 2, 61)
    make_result(f"{W}/probe-gr", cs_c, pg)
    crel = {r["key"]: r["rel"] for r in SC.score_files(S, "quant_layer", pg, f"{W}/probe-gr", dp)[0]}
    for key in KEYS:                    # honest = carol's quality / 1.01; one passed donor row 4% better than honest at K3;
        h = crel[key] / 1.01            # our only trusted rows one bit away (extrap)
        S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t) VALUES(?,?,?,?,?,?,?,?)",
            (ser4, key, 2.0, h * math.exp(1.37), "trusted", "pass", "import", time.time()))
        S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t, worker_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (ser4, key, 3.0, h * 0.96, "donor", "pass", "repeat", time.time(), WK["mallory"]))
    SC.series = lambda p: ser4 if p.get("level") == "k3-gr" else series_orig(p)
    try:
        jc = accept_job(pg, cs_c, "carol")
        before = lumens("carol")
        st = SC.score_job(S, RES, dp, jc)
        rr = S.q("SELECT * FROM jobs WHERE verify_of=?", (jc,))
        ok("#12: after one donor row 4% better than honest, an honest +1% repeat (trusted band passes, all band fails) is "
           "suspect: not quarantined, no points yet, an anchor re-run queued",
           st == "suspect" and S.q("SELECT quarantined FROM workers WHERE id=?", (WK["carol"],))[0][0] == 0 and lumens("carol") == before
           and len(rr) == 1 and rr[0]["needs"] == "verifier" and job(jc)["state"] == "accepted", st)
        S.q("UPDATE jobs SET state='accepted', accepted_by=?, accepted_at=? WHERE id=?", (WK["ver"], time.time(), rr[0]["id"]))
        make_result(f"{RES}/{rr[0]['id']}", mutate(C0, 2, 62), pg)
        SC.score_job(S, RES, dp, rr[0]["id"])
        ok("#12: the anchor at that K (trusted, repeat basis) then decides it: pass, lumens granted",
           job(jc)["check_state"] == "pass" and lumens("carol") > before, job(jc)["check_state"])
        S.q("UPDATE scores SET state='void' WHERE series=? AND kind='trusted' AND K=3.0", (ser4,))
        jm = accept_job(pg, mutate(C0, 123, 63, keys={KEYS[0]}), "mallory")
        ok("#12: a result that fails the trusted band still fails (and quarantines)", SC.score_job(S, RES, dp, jm) == "fail" and
           S.q("SELECT quarantined FROM workers WHERE id=?", (WK["mallory"],))[0][0] == 1)
        S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))
    finally:
        SC.series = series_orig
    ser5 = "outlier-test"
    po = dict(P, level="k3-out"); po["stamp"] = Q.settings_stamp(po)
    for key in KEYS:                    # our own K3 rows 20% worse than this donor's
        S.q("INSERT INTO scores(series, key, K, rel, kind, state, basis, t) VALUES(?,?,?,?,?,?,?,?)",
            (ser5, key, 3.0, crel[key] * 1.2, "trusted", "pass", "import", time.time()))
    SC.series = lambda p: ser5 if p.get("level") == "k3-out" else series_orig(p)
    try:
        jo = accept_job(po, cs_c, "trent")
        S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["trent"],))
        st = SC.score_job(S, RES, dp, jo)
        ev = S.q("SELECT detail FROM events WHERE job_id=? AND kind='score_outlier'", (jo,))
        ok("#12: a result more than the repeat tolerance below our trusted center passes and logs a score_outlier event",
           st == "pass" and ev and len(json.loads(ev[0][0])["tensors"]) == len(KEYS), st)
    finally:
        SC.series = series_orig

    # best of N at export
    from server import admin
    pb = dict(P, level="k3-best"); pb["stamp"] = Q.settings_stamp(pb)
    js = []
    for c, n_words in enumerate((4, 1, 3)):
        pc = dict(pb, settings=dict(pb["settings"], seed_offset=c)); pc["stamp"] = Q.settings_stamp(pc)
        j = accept_job(pc, mutate(C0, n_words, 50 + c), "bob"); SC.score_job(S, RES, dp, j); js.append(j)
    best = SC.best_of_group(S, [job(j) for j in js])
    ok("best of N: export keeps the passed copy with the lowest total error", best is not None and best["id"] == js[1],
       f"{[(j, job(j)['check_state'], round(json.loads(job(j)['check_detail'] or '{}').get('total_rel', 0), 5)) for j in js]}")

    # #11: near-duplicate codes (on from here; the fingerprints of everything above are forgotten: a fresh campaign)
    SC.NEAR_DUP = near_dup
    S.q("UPDATE scores SET tiles=NULL")
    big = codes(3, 77)["trellis"]; big = np.concatenate([big] * 16, 0).reshape(64, 64, 48)      # 4096 tiles: sampled
    f0, f1 = SC.tiles_fp(big, "k", "seed-a"), SC.tiles_fp(mutate({"k": {"trellis": big}}, 1, 5)["k"]["trellis"], "k", "seed-a")
    fr = SC.tiles_fp(codes(3, 78)["trellis"], "k", "seed-a"), SC.tiles_fp(codes(3, 79)["trellis"], "k", "seed-a")
    ok("#11 fingerprints: 256 sampled tiles; one changed word keeps >= 255 of them, independent codes share none, and "
       "which tiles are sampled depends on the queue's secret",
       len(f0) == 4 * SC.TILES and SC.tiles_same(f0, f1) >= 255 / 256 and SC.tiles_same(*fr) == 0 and
       SC.tiles_fp(big, "k", "seed-b") != f0 and S.secret("tiles") == S.secret("tiles"), f"{SC.tiles_same(f0, f1):.4f}")
    pn = dict(P, level="k3-nd")
    def copy(c, who, base):
        pc = dict(pn, settings=dict(pn["settings"], seed_offset=c)); pc["stamp"] = Q.settings_stamp(pc)
        j = accept_job(pc, base, who); return j, SC.score_job(S, RES, dp, j)
    WK["dave"] = S.add_worker("dave")[0]; WK["erin"] = S.add_worker("erin")[0]
    q0 = mutate(C0, 2, 91)
    l0 = lumens("dave")
    got = [copy(c, "dave", mutate(q0, 1, 92 + c) if c else q0) for c in range(3)]
    l1 = lumens("dave")
    dups = [json.loads(job(j)["check_detail"]).get("duplicate_of") for j, _ in got]
    ok("#11: one quant submitted as 3 best-of-N copies with one trellis word changed per tensor: all pass (valid codes), "
       "only the first earns", [st for _, st in got] == ["pass"] * 3 and dups == [None, got[0][0], got[0][0]] and
       abs((l1 - l0) - lumens_of(got[0][0])) < 1e-9 and lumens_of(got[1][0]) == lumens_of(got[2][0]) == 0, f"{dups}, +{l1 - l0}")
    je, st = copy(3, "erin", mutate(q0, 3, 96))
    nd = [t.get("near_duplicate") for t in json.loads(job(je)["check_detail"])["tensors"]]
    ok("#11: another donor's copy of those codes with words changed: a duplicate (no points), near_duplicate on every tensor",
       st == "pass" and lumens("erin") == 0 and json.loads(job(je)["check_detail"]).get("duplicate_of") == got[0][0] and
       all(x and x > 0.98 for x in nd), str(nd[:3]))
    pub2 = mutate(C0, 3, 97); make_result(f"{W}/published2", pub2)
    SC.import_reference(S, dp, "quant_layer", P, f"{W}/published2", kind="published", note="released v3 level")
    jr_, st = copy(4, "mallory", mutate(pub2, 2, 98))
    ok("#11: a published result with words changed is a replay (fail), not a duplicate, though it matches donor rows too",
       st == "fail" and {r[0] for r in S.q("SELECT basis FROM scores WHERE job_id=?", (jr_,))} == {"replay"}, st)
    S.q("UPDATE workers SET quarantined=0 WHERE id=?", (WK["mallory"],))

    # quant_tensor export: a layer's groups merged into one shard = the whole-layer shard, tensor for tensor
    from common.st import merge_shards, subset_shard
    from checker.check import ST as _ST
    whole = f"{RES}/{ja}/q-L00.safetensors"; parts = []
    for g, (_, ks) in enumerate(PLAN):
        parts.append(f"{W}/g{g}.safetensors")
        subset_shard(whole, [f"{k}.{x}" for k in ks for x in ("trellis", "suh", "svh", "mul1")], parts[-1])
    merge_shards(parts[::-1], f"{W}/merged.safetensors")
    A_, B_ = _ST(whole), _ST(f"{W}/merged.safetensors")
    ok("group shards (subset) merged back = the whole-layer shard, every tensor byte for byte",
       sorted(A_.keys()) == sorted(B_.keys()) and all(np.array_equal(A_.get(k), B_.get(k)) for k in A_.keys()))

    n_ok = sum(1 for _, c in RESULTS if c)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return n_ok == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
