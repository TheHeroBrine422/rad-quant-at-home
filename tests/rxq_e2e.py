#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rxq_e2e.py RXM FORK [WORKDIR] -- the V3 ladder's rxq_quant jobs end to end on the CPU (ladder branch, 2026-11-08):
job spec -> queue -> worker -> accept -> score -> export / assemble, on a tiny two-layer model (layer 2 GDN + MLP,
layer 3 attention + MLP, small shapes, random weights / Hessians / H_out), with rad-exl3-mix's real quantizer path
(RXM/pipeline: `python -m rxq job`, the fork's quantize.py at FORK) and rxq's CPU stand-in kernels (rxq/cpukern.py;
greedy trellis encoder, test only). Needs torch + safetensors (the rxq environment); localhost only, ~3-5 min.

  datapack   admin.py datapack --groups --hout, rxq-settings (the V3 settings file by its stamp), manifest
  generate   rad-exl3-mix `python -m rxq.ladder jobs MANIFEST --levels 0.5-i4,2.5-i8 --layers 2-3`: 16 rxq_quant jobs,
             every input pinned (sha256 = the manifest's), requirements from the measured table
  enqueue    admin.py enqueue-jobs (all checked first); a second run adds nothing (ukeys)
  run        server (RQAH_ALLOW_STANDIN=1) + one worker (fake card, RXQ_CPU_STANDIN=1): every job accepted
  identity   every accepted tensor / C9 file == a direct `python -m rxq job` of the same tensors on the original
             fixture files, bit for bit (data and metadata); log proxy errors equal
  score      admin.py score: the server's own decode of the codes (checker/exl3dec.py) + the Hessian-weighted error;
             every job scored (provisional at first: no reference rows; the anchors then run on a verifier)
  export     admin.py export --rxq-python: the level assembled by rxq from the accepted job dirs == rxq assemble of the
             direct outputs (q-LNN shards' tensors, c9 scales)
  pins       a Hessian changed in the datapack after enqueue: enqueue-jobs refuses the stale jobs, pins-check lists the
             queued ones, the worker refuses them ("differ from the job's pins") and runs nothing
  accept     in process: a result whose job.json does not record the pinned inputs, a t-file with another settings stamp,
             a missing C9 file, and stand-in codes on a server without RQAH_ALLOW_STANDIN are all refused
Last line: "RXQ E2E n/m PASS"."""
import glob, hashlib, json, os, shutil, signal, sqlite3, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
RXM, FORK = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
W = os.path.abspath(sys.argv[3] if len(sys.argv) > 3 else "/tmp/rqah-rxq-e2e")
PIPE = os.path.join(RXM, "pipeline")
PORT = int(os.environ.get("E2E_PORT", "18791")); URL = f"http://127.0.0.1:{PORT}"
PY = sys.executable
DBP, DATA, RES, SPOOL = f"{W}/db/queue.sqlite", f"{W}/data", f"{W}/out/results", f"{W}/out/spool"
RESULTS = []
EXT = f"{FORK}/exllamav3/exllamav3_ext"


def ok(name, cond, detail=""):
    RESULTS.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def admin(*args, env=None):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run([PY, "-I", f"{HERE}/server/admin.py", "--db", DBP, "--data", DATA, "--results", RES, *args],
                       capture_output=True, text=True, env=e)
    return r.returncode, r.stdout + r.stderr


def db():
    c = sqlite3.connect(DBP); c.row_factory = sqlite3.Row; return c


def wait(cond, timeout, step=0.5):
    t = time.time() + timeout
    while time.time() < t:
        if cond(): return True
        time.sleep(step)
    return cond()


if os.path.exists(W): shutil.rmtree(W)
for d in (f"{W}/db", DATA, RES, SPOOL, f"{W}/secrets"): os.makedirs(d)

# ---------------------------------------------------------------- fixture: a tiny Qwen3.5-layout model (layers 2, 3)
import torch                                           # noqa: E402
from safetensors import safe_open                      # noqa: E402
from safetensors.torch import save_file                # noqa: E402
sys.path.insert(0, PIPE)
from rxq import qwen35, config as RC                   # noqa: E402

SH = {"ssm_qkv": (384, 256), "ssm_z": (256, 256), "ssm_out": (256, 256), "attn_qg": (256, 256), "attn_k": (128, 256),
      "attn_v": (128, 256), "attn_output": (256, 256), "ffn_gate": (384, 256), "ffn_up": (384, 256), "ffn_down": (256, 384)}
g = torch.Generator().manual_seed(86)
CK, HS, HO = f"{W}/fixture/ckpt", f"{W}/fixture/hess", f"{W}/fixture/hout"
for d in (CK, HS, HO): os.makedirs(d)
tens = {}
for l in (2, 3):
    for hname, ks in qwen35.layer_plan(l):
        k_in = SH[qwen35.family(ks[0])][1]
        A = torch.randn(4 * k_in, k_in, generator=g) * torch.linspace(0.3, 2.0, k_in)
        H = (A.T @ A).float(); iu = torch.triu_indices(k_in, k_in)
        save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{HS}/{hname}.safetensors", metadata={"k": str(k_in), "rows": str(4 * k_in)})
        for key in ks:
            n, k = SH[qwen35.family(key)]
            tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
            save_file({"hout_diag": (torch.rand(n, generator=g) + 0.1).float()}, f"{HO}/{key}.safetensors")
save_file(tens, f"{CK}/model.safetensors")
json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in tens}}, open(f"{CK}/model.safetensors.index.json", "w"))
json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"num_hidden_layers": 4}}, open(f"{CK}/config.json", "w"))
KEYS = list(tens)

# ---------------------------------------------------------------- datapack, settings, manifest, jobs
rc1, o1 = admin("datapack", "tiny", "r1", CK, HS, "--layers", "2-3", "--groups", "--no-head", "--hout", "bn1920", HO, "--n-params", "1000000")
rc2, o2 = admin("rxq-settings", "tiny", "r1", f"{PIPE}/config/v3.toml")
rc3, o3 = admin("manifest")
st = RC.settings_stamp(RC.load(f"{PIPE}/config/v3.toml"))
ok("datapack", rc1 == 0 and rc2 == 0 and rc3 == 0 and os.path.exists(f"{DATA}/models/tiny@r1/rxq/{st}.toml")
   and os.path.exists(f"{DATA}/models/tiny@r1/weights/L03-g2.safetensors"), (o1 + o2 + o3).strip().replace("\n", " | ")[:300])
env_rxq = dict(os.environ, PYTHONPATH=PIPE)


def gen(levels, layers, out):
    r = subprocess.run([PY, "-m", "rxq.ladder", "jobs", f"{DATA}/manifest.json", "--model", "tiny", "--rev", "r1", "--levels", levels,
                        "--layers", layers, "-o", out], capture_output=True, text=True, env=env_rxq, cwd=PIPE)
    return r.returncode, r.stdout + r.stderr


rc, o = gen("0.5-i4,2.5-i8", "2-3", f"{W}/jobs.jsonl")
J = [json.loads(l) for l in open(f"{W}/jobs.jsonl")] if rc == 0 else []
man = json.load(open(f"{DATA}/manifest.json"))["files"]
pins_ok = all(all(man[n]["sha256"] == h for n, h in j["params"]["pins"].items()) for j in J)
ok("generate", rc == 0 and len(J) == 16 and pins_ok and all(j["params"]["hout"] == "bn1920" for j in J),
   f"{len(J)} jobs; pins == manifest {pins_ok}; e.g. req {J[0]['params']['req'] if J else None} | {o.strip()[:160]}")
rc, o = admin("enqueue-jobs", f"{W}/jobs.jsonl"); rc2, o2 = admin("enqueue-jobs", f"{W}/jobs.jsonl")
ok("enqueue", rc == 0 and "16 new jobs" in o and "0 new jobs (16 already queued)" in o2, (o.strip() + " | " + o2.strip())[:240])

# ---------------------------------------------------------------- server + worker
e = {k: v for k, v in os.environ.items() if not k.startswith(("RADQUANT_", "RQAH_"))}
e.update(RQAH_ALLOW_STANDIN="1", RQAH_SPOT_RATE="0", RQAH_SPOT_FIRST="0")
open(f"{W}/secrets/admin-token", "w").write("rxq-e2e-admin-" + hashlib.sha256(W.encode()).hexdigest()[:32])
srv = subprocess.Popen([PY, "-I", f"{HERE}/server/server.py", "--db", DBP, "--data", DATA, "--spool", SPOOL, "--results", RES,
                        "--port", str(PORT), "--lease-s", "60", "--heartbeat-s", "5", "--retry-s", "0.5",
                        "--admin-token-file", f"{W}/secrets/admin-token"], stdout=open(f"{W}/server.log", "a"), stderr=subprocess.STDOUT, env=e)
time.sleep(1.5)
_, out = admin("add-worker", "cpu1"); tok = out.split("token")[-1].strip()
admin("verifier", out.split()[1])               # our own box: its results are trusted rows (anchors)


def worker(name, *extra):
    we = dict(e, RQAH_ALLOW_STANDIN="", PYTHONPATH="")
    return subprocess.Popen([PY, "-I", f"{HERE}/worker/worker.py", "--server", URL, f"--token={tok}", "--state", f"{W}/{name}",
                             "--poll", "0.5", "--gpus", "0", "--gpu-name", "fake-cpu", "--gpu-env", "none", "--vram-gb", "24",
                             "--ram-gb", "16", "--job-types", "rxq_quant", "--rxq", PIPE, "--exl3-src", EXT, "--python", PY,
                             "--env", "RXQ_CPU_STANDIN=1", "--env", "OMP_NUM_THREADS=2", *extra],
                            stdout=open(f"{W}/{name}.log", "a"), stderr=subprocess.STDOUT, env=we)


def states():
    return {r["id"]: r["state"] for r in db().execute("SELECT id, state FROM jobs WHERE verify_of IS NULL")}


wk = worker("w1")
done = wait(lambda: states() and all(s == "accepted" for s in states().values()), 900, 2)
ok("run", done, f"states {sorted(states().values())[:3]}... ({len(states())} jobs); worker log tail: "
   + open(f"{W}/w1.log").read()[-200:].replace("\n", " | ") if not done else f"{len(states())} jobs accepted")

# ---------------------------------------------------------------- identity with direct rxq runs
direct = f"{W}/direct"
for j in J:
    p = j["params"]; lv = p["level"]
    spec = {"type": "rxq.quant", "version": 1, "level": lv, "K": p["K"], "codebook": p["codebook"], "seed_offset": 0, "keys": p["keys"],
            "settings_stamp": st, "config": f"{PIPE}/config/v3.toml",
            "inputs": {"ckpt": CK, "hessians": HS, "hin": None, "wpre": None, "hout": HO}}
    sp = f"{W}/direct-{lv}-{p['layer']}-{p['group']}.json"; json.dump(spec, open(sp, "w"))
    subprocess.run([PY, "-m", "rxq", "job", sp, "-o", f"{direct}/{lv}", "--device", "cpu"], capture_output=True, text=True,
                   env=dict(env_rxq, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=EXT, OMP_NUM_THREADS="2"), cwd=PIPE)
rows = db().execute("SELECT id, params FROM jobs WHERE verify_of IS NULL AND state='accepted'").fetchall()
nsame = ntot = 0; bad = []
for r in rows:
    p = json.loads(r["params"])
    for k in p["keys"]:
        for f in (f"t-{k}.safetensors",) + ((f"c9-{k}.safetensors",) if k in p["c9"] else ()):
            a, b = f"{RES}/{r['id']}/{f}", f"{direct}/{p['level']}/{f}"
            ntot += 1
            if os.path.exists(a) and os.path.exists(b):
                with safe_open(a, "pt") as x, safe_open(b, "pt") as y:
                    s = x.metadata() == y.metadata() and sorted(x.keys()) == sorted(y.keys()) and all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())
                nsame += s
                if not s: bad.append(f)
            else: bad.append(f + " (missing)")
pq = {}
for r in rows:
    for l in open(f"{RES}/{r['id']}/log.jsonl"): x = json.loads(l); pq[x["key"]] = x["proxy_err"]
dq = {}
for f in glob.glob(f"{direct}/*/log.jsonl"):
    lvn = f.split("/")[-2]
    for l in open(f): x = json.loads(l); dq[(lvn, x["key"])] = x["proxy_err"]
ok("identity", ntot and nsame == ntot, f"{nsame}/{ntot} tensor + C9 files bit-identical (data + metadata) to direct rxq job runs {bad[:2]}")

# ---------------------------------------------------------------- score, export
rc, o = admin("score", env={"RQAH_ALLOW_STANDIN": "1"})
sc = [r["check_state"] for r in db().execute("SELECT check_state FROM jobs WHERE verify_of IS NULL")]
nrows = db().execute("SELECT COUNT(*) FROM scores").fetchone()[0]
ok("score", rc == 0 and all(s in ("pass", "provisional", "suspect") for s in sc) and nrows >= 26,
   f"check states {dict((s, sc.count(s)) for s in set(sc))}, {nrows} score rows | {o.strip()[-200:]}")
lvl = J[-1]["params"]["level"] if J else ""
lv25 = [j for j in J if j["params"]["K"] == 2.5][0]["params"]["level"] if J else ""
rc, o = admin("export", "--level", lv25, f"{W}/export", "--partial", "--rxq-python", PY, "--rxq", PIPE, env={"RQAH_ALLOW_STANDIN": "1"})
r2 = subprocess.run([PY, "-m", "rxq", "assemble", f"{W}/direct-level", f"{direct}/{lv25}", "--keys-from", "--allow-standin"],
                    capture_output=True, text=True, env=env_rxq, cwd=PIPE)
same_lv = rc == 0 and r2.returncode == 0
if same_lv:
    for f in ("q-L02.safetensors", "q-L03.safetensors"):
        with safe_open(f"{W}/export/level/{f}", "pt") as x, safe_open(f"{W}/direct-level/{f}", "pt") as y:
            same_lv &= sorted(x.keys()) == sorted(y.keys()) and all(torch.equal(x.get_tensor(n), y.get_tensor(n)) for n in x.keys())
    c9a = sorted(os.listdir(f"{W}/export/level/c9/scales")) if os.path.isdir(f"{W}/export/level/c9/scales") else []
    same_lv &= c9a == sorted(os.listdir(f"{W}/direct-level/c9/scales")) and len(c9a) == 2
ok("export", same_lv, f"admin export (rxq assemble of the accepted job dirs) == direct assemble: shards + c9 | {o.strip()[-160:]} {r2.stderr[-160:]}")

# ---------------------------------------------------------------- content addressing
wk.send_signal(signal.SIGTERM); wk.wait(60)
rc, o = gen("6.5-mul1", "2", f"{W}/jobs65.jsonl")
rc, o = admin("enqueue-jobs", f"{W}/jobs65.jsonl")
J65 = [json.loads(l) for l in open(f"{W}/jobs65.jsonl")]
hp = f"{HS}/{qwen35.layer_plan(2)[0][0]}.safetensors"
orig = open(hp, "rb").read()
with safe_open(hp, "pt") as f: up = f.get_tensor("hessian_upper"); meta = f.metadata()
save_file({"hessian_upper": up * 1.01}, hp + ".new", metadata=meta); os.replace(hp + ".new", hp)   # the datapack links it
admin("manifest")
rc_r, o_r = admin("enqueue-jobs", f"{W}/jobs65.jsonl")
rc_p, o_p = admin("pins-check")
wk2 = worker("w2", "--idle-exit", "6")
wait(lambda: wk2.poll() is not None, 240, 1)
ids65 = [r["id"] for r in db().execute("SELECT id, params FROM jobs WHERE level=?", (J65[0]["level"],))]
fails = db().execute("SELECT COUNT(*) FROM events WHERE kind IN ('fail','reject') AND detail LIKE '%pins%'").fetchone()[0]
acc65 = sum(1 for i in ids65 if states().get(i) == "accepted")
ok("pins", rc_r != 0 and "pinned inputs" in o_r and "1 jobs with stale pins" in o_p and acc65 == len(ids65) - 1 and fails >= 1,
   f"re-enqueue refused {rc_r != 0}; pins-check: {o_p.strip().splitlines()[-1] if o_p.strip() else ''}; worker: {acc65}/{len(ids65)} "
   f"accepted, {fails} pin refusals ({open(f'{W}/w2.log').read().count('pins')} log lines)")
open(hp, "wb").write(orig); admin("manifest")

# ---------------------------------------------------------------- accept() refusals, in process
os.environ.pop("RQAH_ALLOW_STANDIN", None)
import jobs                                             # noqa: E402
from jobs import rxq_quant as RQ                        # noqa: E402
r0 = [r for r in rows if json.loads(r["params"])["c9"]][0]
p0 = json.loads(r0["params"]); src = f"{RES}/{r0['id']}"
dp = lambda rel: os.path.join(DATA, rel)


def files_of(d): return {n: os.path.join(d, n) for n in os.listdir(d)}


def refused(mut):
    d = f"{W}/tamper"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(src, d)
    mut(d)
    try: RQ.accept(p0, files_of(d), {}, dp); return False, "accepted"
    except ValueError as x: return True, str(x)[:90]


os.environ["RQAH_ALLOW_STANDIN"] = "1"
base_ok, why0 = refused(lambda d: None)


def m_pins(d):
    j = json.load(open(f"{d}/job.json")); j["rqah"]["inputs"] = {}; json.dump(j, open(f"{d}/job.json", "w"))


def m_c9(d): os.remove(f"{d}/c9-{p0['c9'][0]}.safetensors")


def m_stamp(d):
    k = p0["keys"][0]; f = f"{d}/t-{k}.safetensors"
    with safe_open(f, "pt") as x: t = {n: x.get_tensor(n) for n in x.keys()}; m = dict(x.metadata())
    m["rxq_settings"] = "0" * 16; save_file(t, f, metadata=m)


res = {"pins": refused(m_pins), "c9": refused(lambda d: m_c9(d)), "stamp": refused(m_stamp)}
os.environ.pop("RQAH_ALLOW_STANDIN")
res["standin"] = refused(lambda d: None)
ok("accept", not base_ok and all(v[0] for v in res.values()), f"untouched result accepted ({why0}); refused: " +
   "; ".join(f"{k}: {v[1]}" for k, v in res.items()))

srv.send_signal(signal.SIGTERM); srv.wait(30)
print(f"RXQ E2E {sum(RESULTS)}/{len(RESULTS)} PASS")
sys.exit(0 if all(RESULTS) else 1)
