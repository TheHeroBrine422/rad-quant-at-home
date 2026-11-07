#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""setup_test.py [WORKDIR] -- `rqah setup` (client/setup.py) against fake machines, no GPU, no network, a few seconds.

Each machine is a fake root (sysfs: kfd topology, drm cards and connectors, PCI; /proc; /dev/kfd and render nodes;
/etc/group, /etc/localtime) plus canned output of nvidia-smi, rocminfo and docker. Checks: detection and defaults on
a 4 / 8 / 12 / 16 / 32 GB card matrix (NVIDIA and AMD, display or headless), a 2-card box with an iGPU, unsupported
cards (gfx1100, cc 7.5, old driver), a no-Docker machine, Docker without permission, NVIDIA without the Container
Toolkit; the files (contents, modes, the invite never in setup.json / stdout / stderr), `docker compose config` on
them when the plugin is installed, re-runs (edits keep the rest and the invite, .prev backups), `--check` (ok,
loose modes, hand edits, missing files, no config), the plain prompts driven by a script, the curses form rendered as
text and driven through a pseudo-terminal, the offline smoke check's commands, no writes outside the config dir, and
Python 3.10 syntax. Writes text screenshots of the main screens to WORKDIR/screens/."""
import ast, io, json, os, pty, re, select, shutil, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from client import setup as S

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="rqah-setup-"))
os.makedirs(W, exist_ok=True)
shutil.rmtree(os.path.join(W, "cfg"), ignore_errors=True)
RESULTS = []
INVITE = "inv-Zq8xK2pLm4Nw7Rt9"
GIB = 1 << 30


def ok(name, cond, detail=""):
    RESULTS.append((name, bool(cond))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


# ---------------------------------------------------------------- fake machines
def wr(root, path, text):
    p = os.path.join(root, path.lstrip("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)


def link(root, path, target):
    p = os.path.join(root, path.lstrip("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    os.symlink(os.path.relpath(os.path.join(root, target.lstrip("/")), os.path.dirname(p)), p)


def machine(tag, amd=(), nvidia=(), ram_gb=32, docker="ok", toolkit=True, driver="575.64", free_gb=500, fs="xfs",
            rocminfo=False, kfd=True, threads=16):
    """amd: [(gfx_target_version, vram GiB, pci id, display)]; nvidia: [(name, MiB, cc, display)];
    docker: ok | missing | denied | down | podman. Returns (root, cmds json path)"""
    root = os.path.join(W, "m", tag)
    shutil.rmtree(root, ignore_errors=True)
    wr(root, "/proc/meminfo", f"MemTotal:       {ram_gb * 1024 * 1024} kB\nMemAvailable:   {ram_gb * 800 * 1024} kB\n")
    wr(root, "/proc/cpuinfo", "".join(f"processor\t: {i}\nmodel name\t: Fake CPU 9000\n\n" for i in range(threads)))
    wr(root, "/proc/mounts", f"/dev/sda1 / ext4 rw 0 0\n/dev/sdb1 /var/lib/docker {fs} rw 0 0\n/dev/sdc1 /srv btrfs rw 0 0\n")
    wr(root, "/etc/group", "root:x:0:\nvideo:x:39:\nrender:x:105:\ndocker:x:990:me\n")
    wr(root, "/usr/share/zoneinfo/America/Chicago", "TZif")
    link(root, "/etc/localtime", "/usr/share/zoneinfo/America/Chicago")
    os.makedirs(os.path.join(root, "var/lib/docker/volumes"), exist_ok=True)
    os.makedirs(os.path.join(root, "srv"), exist_ok=True)
    if amd:
        wr(root, "/sys/class/kfd/kfd/topology/nodes/0/properties", "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\n")
        if kfd: wr(root, "/dev/kfd", "")
    names = []
    for i, (gv, vram, pid, disp) in enumerate(amd):
        dev = f"/sys/devices/pci0000:00/0000:0{3 + i}:00.0"
        wr(root, dev + "/vendor", "0x1002\n"); wr(root, dev + "/device", pid + "\n"); wr(root, dev + "/class", "0x030000\n")
        wr(root, dev + "/mem_info_vram_total", f"{int(vram * GIB)}\n")
        card, minor = f"card{i + 1}", 128 + i
        link(root, f"/sys/class/drm/{card}/device", dev)
        link(root, f"/sys/class/drm/renderD{minor}/device", dev)
        wr(root, f"/sys/class/drm/{card}-DP-1/status", "connected\n" if disp else "disconnected\n")
        wr(root, f"/dev/dri/renderD{minor}", ""); wr(root, f"/dev/dri/{card}", "")
        if kfd:
            wr(root, f"/sys/class/kfd/kfd/topology/nodes/{i + 1}/properties",
               f"simd_count 128\ngfx_target_version {gv}\ndrm_render_minor {minor}\nvendor_id 4098\ndevice_id {int(pid, 16)}\n")
        names.append((S.gfx_name(gv), {"0x7551": "AMD Radeon AI PRO R9700", "0x7550": "AMD Radeon RX 9070 XT"}.get(pid, "AMD Radeon Graphics")))
    cmds = {}
    if nvidia:
        rows = [f"{i}, {n}, {mib}, {cc}, {'Enabled' if d else 'Disabled'}, {driver}, 00000000:0{1 + i}:00.0"
                for i, (n, mib, cc, d) in enumerate(nvidia)]
        cmds["nvidia-smi"] = [["--query-gpu=index,name,memory.total,compute_cap,display_active,driver_version,pci.bus_id", 0,
                               "\n".join(rows) + "\n", ""]]
        if toolkit: cmds["nvidia-ctk"] = [[0, "NVIDIA Container Toolkit CLI version 1.17.8", ""]]
    if rocminfo and amd:
        out = "".join(f"*******\nAgent {k + 2}\n*******\n  Name:                    {g}\n  Marketing Name:          {m}\n"
                      "  Device Type:             GPU\n" for k, (g, m) in enumerate(names))
        cmds["rocminfo"] = [[0, "*******\nAgent 1\n*******\n  Name: Fake CPU\n  Device Type: CPU\n" + out, ""]]
    info = {"ServerVersion": "29.8.2", "DockerRootDir": "/var/lib/docker", "Runtimes": {"runc": {}},
            "SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"]}
    if nvidia and toolkit: info["Runtimes"]["nvidia"] = {}
    if docker != "missing":
        cmds["docker"] = [["--version", 0, "podman version 5.6.0\n" if docker == "podman" else "Docker version 29.8.2, build 2\n", ""],
                          ["compose", "version", 0, "5.6.0\n", ""],
                          ["image", "inspect", 1, "", "Error: No such image\n"]]
        if docker in ("ok", "podman"): cmds["docker"].insert(0, ["info", 0, json.dumps(info) + "\n", ""])
        elif docker == "denied":
            cmds["docker"].insert(0, ["info", 1, "", "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock\n"])
        elif docker == "down":
            cmds["docker"].insert(0, ["info", 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?\n"])
    else:
        cmds["docker"] = "missing"
    for c in ("nvidia-smi", "rocminfo", "nvidia-ctk"):
        cmds.setdefault(c, "missing")
    cmds["_statvfs"] = {"/var/lib/docker": int(free_gb * 1e9), "/srv": int(2000e9), "/": int(30e9)}
    spec = os.path.join(W, "m", tag + ".json")
    json.dump(cmds, open(spec, "w"))
    return root, spec


def host_of(root, spec): return S.fake_host_from_file(root, spec)


def run_setup(args, root, spec, cfgdir, stdin=None, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("RQAH_")}; e.update(env or {})
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "client", "setup.py"), "--fake-root", root, "--fake-cmds", spec,
                        "--config-dir", cfgdir] + args, input=stdin, capture_output=True, text=True, env=e, timeout=60)
    return r.returncode, r.stdout, r.stderr


def tree(d):
    out = {}
    for b, ds, fs in os.walk(d):
        for f in fs:
            p = os.path.join(b, f)
            try: out[p] = (os.lstat(p).st_mtime_ns, os.lstat(p).st_size)
            except OSError: pass
    return out


SCREENS = os.path.join(W, "screens"); os.makedirs(SCREENS, exist_ok=True)


def shot(name, text):
    open(os.path.join(SCREENS, name + ".txt"), "w").write(text if text.endswith("\n") else text + "\n")


# ================================================================ 1. the card matrix (detection + defaults + summary)
SRV = ["--server", "https://queue.example.org"]
MATRIX = [  # tag, machine kwargs, expected (vendor, gpus, vram_gb, ram_gb, writable, a job line that must be there)
    ("nv4", dict(nvidia=[("NVIDIA GeForce RTX 3050 Laptop GPU", 4096, "8.6", False)], ram_gb=16), ("nvidia", [0], 4, 12, False, "nothing")),
    ("nv8", dict(nvidia=[("NVIDIA GeForce RTX 4060 Ti", 8188, "8.9", False)], ram_gb=32), ("nvidia", [0], 8, 24, True, "up to K6")),
    ("nv12d", dict(nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", True)], ram_gb=32), ("nvidia", [0], 11, 24, True, "up to K6")),
    ("nv12", dict(nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", False)], ram_gb=16), ("nvidia", [0], 12, 12, True, "at every K")),
    ("amd16", dict(amd=[(120001, 15.92, "0x7550", False)], ram_gb=32), ("amd", [0], 16, 24, True, "up to 4.5 bpw")),
    ("amd32", dict(amd=[(120001, 31.86, "0x7551", False)], ram_gb=64, rocminfo=True), ("amd", [0], 32, 48, True, "LM head")),
]
for tag, kw, (ven, gpus, vram, ram, writable, job) in MATRIX:
    root, spec = machine(tag, **kw)
    hw = S.detect(host_of(root, spec))
    r = S.recommend(hw)
    ok(f"{tag}: defaults", (r["vendor"], r["gpus"], r["vram_gb"], r["ram_gb"]) == (ven, gpus, vram, ram), str(r))
    cfgdir = os.path.join(W, "cfg", tag)
    rc, out, err = run_setup(["--yes"] + SRV + ["--invite-file", "-"], root, spec, cfgdir, stdin=INVITE + "\n")
    ok(f"{tag}: --yes {'writes' if writable else 'refuses'}", (rc == 0) == writable and os.path.exists(cfgdir) == writable, f"rc={rc} {err.strip()[-200:]}")
    ok(f"{tag}: summary lists '{job}'", job in out, "")
    ok(f"{tag}: invite never printed", INVITE not in out + err)
    shot(f"summary-{tag}", out + err)
    if writable:
        comp = open(os.path.join(cfgdir, "compose.yaml")).read()
        envt = open(os.path.join(cfgdir, "worker.env")).read()
        img = f"{S.IMAGE_REPO}:{S.VERSION}-" + (S.ROCM_TAG if ven == "amd" else S.CUDA_TAG)
        ok(f"{tag}: image {img.split(':')[-1]}", img in comp and img in open(os.path.join(cfgdir, "docker-run.sh")).read())
        if ven == "amd":
            ok(f"{tag}: ROCm devices, groups, seccomp", "devices: [/dev/kfd, /dev/dri]" in comp and '"39", "105"' in comp
               and "seccomp=unconfined" in comp and "driver: nvidia" not in comp)
        else:
            ok(f"{tag}: NVIDIA reservation, no kfd", "driver: nvidia" in comp and "/dev/kfd" not in comp and "seccomp" not in comp
               and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in envt)
        ok(f"{tag}: worker.env values", f"RQAH_VRAM_GB={vram}\n" in envt and f"RQAH_RAM_GB={ram}\n" in envt
           and f"RQAH_GPUS={','.join(map(str, gpus))}\n" in envt and f"RQAH_INVITE={INVITE}\n" in envt)

# the gpu-name slugs, the gfx decode
ok("gpu slugs", [S.gpu_slug(n) for n in ("AMD Radeon AI PRO R9700", "NVIDIA GeForce RTX 4070", "NVIDIA GeForce RTX 4060 Ti",
                                         "AMD Radeon RX 9070 XT", "NVIDIA RTX A6000")]
   == ["r9700", "rtx4070", "rtx4060ti", "rx9070xt", "rtxa6000"])
ok("gfx names", [S.gfx_name(v) for v in (120001, 110000, 100306, 90010)] == ["gfx1201", "gfx1100", "gfx1036", "gfx90a"])

# ================================================================ 2. a 2-card box with an iGPU, unsupported cards
root, spec = machine("amd2igpu", amd=[(100306, 0.5, "0x164e", True), (120001, 31.86, "0x7551", False), (120001, 31.86, "0x7551", False)],
                     ram_gb=64, rocminfo=True)
hw = S.detect(host_of(root, spec))
ok("igpu box: three ROCm indices, the iGPU unusable", [g["index"] for g in hw["gpus"]] == [0, 1, 2]
   and not S.gpu_supported(hw["gpus"][0], hw)[0] and S.gpu_supported(hw["gpus"][1], hw)[0], str([(g["index"], g["arch"]) for g in hw["gpus"]]))
r = S.recommend(hw)
ok("igpu box: defaults skip the iGPU", r["gpus"] == [1, 2] and r["vram_gb"] == 32 and r["gpu_name"] == "r9700", str(r))
cfg = dict(S.DEFAULTS); cfg.update(r); cfg["server"] = "https://q.example.org"
E, Wn = S.validate(cfg, hw)
ok("igpu box: valid; RAM warning for 2 cards absent at 48 GB", not E and not any("cards need" in w for w in Wn), str(E + Wn))
jl = S.jobs_for("amd", 32, 48, 2)
ok("igpu box: TP2 evals offered", any("TP2" in t for t, _ in jl))
cfg2 = dict(cfg, gpus=[0, 1])
E, _ = S.validate(cfg2, hw)
ok("igpu box: picking the iGPU is an error", any("integrated GPU" in e for e in E), str(E))
cfg3 = dict(cfg, ram_gb=16, mem_limit_gb=20)
_, Wn = S.validate(cfg3, hw)
ok("igpu box: 2 cards on 16 GB RAM warn", any("2 cards need 24 GB" in w for w in Wn), str(Wn))
shot("hardware-amd2igpu", "\n".join(S.hardware_lines(hw)))

root, spec = machine("gfx1100", amd=[(110000, 23.98, "0x744c", True)])
hw = S.detect(host_of(root, spec))
ok("gfx1100: not usable, says why", not S.gpu_supported(hw["gpus"][0], hw)[0] and "gfx1100" in S.gpu_supported(hw["gpus"][0], hw)[1])
rc, out, err = run_setup(["--yes", "--gpus", "0", "--vram-gb", "23"] + SRV, root, spec, os.path.join(W, "cfg", "gfx1100"))
ok("gfx1100: --yes refuses", rc == 1 and "gfx1100" in out, out[-300:])

root, spec = machine("nvold", nvidia=[("NVIDIA GeForce GTX 1080", 8192, "6.1", False), ("NVIDIA GeForce RTX 3090", 24576, "8.6", False)],
                     driver="550.120")
hw = S.detect(host_of(root, spec))
ok("old driver / cc 6.1: both cards refused", [S.gpu_supported(g, hw)[1][:24] for g in hw["gpus"]]
   == ["compute capability 6.1 <", "NVIDIA driver 550.120 < "], str([S.gpu_supported(g, hw) for g in hw["gpus"]]))

root, spec = machine("amdnokfd", amd=[(120001, 31.86, "0x7551", False)], kfd=False)
hw = S.detect(host_of(root, spec))
ok("amdgpu without KFD: listed, not usable", len(hw["gpus"]) == 1 and hw["gpus"][0].get("no_kfd") and not hw["kfd"])

# ================================================================ 3. Docker states
root, spec = machine("nodocker", nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", False)], docker="missing", toolkit=False)
cfgdir = os.path.join(W, "cfg", "nodocker")
rc, out, err = run_setup(["--yes", "--smoke"] + SRV, root, spec, cfgdir)
ok("no Docker: files written, warned, no toolkit error (nothing to check it with)", rc == 0 and "Docker isn't installed" in out
   and "Container Toolkit" not in out and os.path.exists(os.path.join(cfgdir, "compose.yaml")), out[-400:] + err[-300:])
ok("no Docker: smoke says so, start hint is docker-run.sh", "not installed; skipping" in out and "docker-run.sh" in out)
shot("summary-nodocker", out)
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("no Docker: --check passes with the warning", rc == 0 and "Docker isn't installed" in out, out[-300:])
shot("check-nodocker", out)

root, spec = machine("denied", amd=[(120001, 31.86, "0x7551", False)], docker="denied")
hw = S.detect(host_of(root, spec))
ok("Docker denied: explained", "docker group" in (hw["docker"]["error"] or "") and not hw["docker"]["access"])
root, spec = machine("down", amd=[(120001, 31.86, "0x7551", False)], docker="down")
ok("Docker down: explained", "not running" in (S.detect(host_of(root, spec))["docker"]["error"] or ""))
root, spec = machine("podman", amd=[(120001, 31.86, "0x7551", False)], docker="podman")
hw = S.detect(host_of(root, spec))
cfg = dict(S.DEFAULTS); cfg.update(S.recommend(hw)); cfg["server"] = "https://q.example.org"
ok("Podman: warned", any("Podman" in w for w in S.validate(cfg, hw)[1]))

root, spec = machine("notoolkit", nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", False)], toolkit=False)
rc, out, err = run_setup(["--yes"] + SRV, root, spec, os.path.join(W, "cfg", "notoolkit"))
ok("NVIDIA without the Container Toolkit: refused", rc == 1 and "Container Toolkit is missing" in out, out[-300:])

# ================================================================ 4. files: modes, secrets, re-runs, compose validity
root, spec = machine("r9700", amd=[(120001, 31.86, "0x7551", True)], ram_gb=64, rocminfo=True)
cfgdir = os.path.join(W, "cfg", "r9700")
ans = os.path.join(W, "answers.json")
json.dump({"server": "https://queue.example.org", "name": "Ada Lovelace", "leaderboard": True, "windows": "22:00-08:00",
           "max_down_gb_month": 300, "max_up_gb_month": 100, "budget_reset_day": 15, "max_up_mbps": 10}, open(ans, "w"))
before = {p: v for p, v in tree(W).items() if not p.startswith(cfgdir) and "/screens/" not in p}
rc, out, err = run_setup(["--yes", "--answers", ans], root, spec, cfgdir, env={"RQAH_INVITE": INVITE})
ok("answers file + $RQAH_INVITE: written", rc == 0, err[-300:])
shot("summary-r9700", out)
after = {p: v for p, v in tree(W).items() if not p.startswith(cfgdir) and "/screens/" not in p}
ok("nothing written outside the config dir", before == after, str(set(after) ^ set(before))[:300])
md = lambda n: os.stat(os.path.join(cfgdir, n)).st_mode & 0o777
ok("modes: dir 0700, worker.env 0600, setup.json 0600, compose 0644, run 0700",
   (os.stat(cfgdir).st_mode & 0o777, md("worker.env"), md("setup.json"), md("compose.yaml"), md("docker-run.sh")) == (0o700, 0o600, 0o600, 0o644, 0o700))
sj = open(os.path.join(cfgdir, "setup.json")).read()
ok("the invite is only in worker.env", INVITE not in sj + open(os.path.join(cfgdir, "compose.yaml")).read()
   + open(os.path.join(cfgdir, "docker-run.sh")).read() + out + err and INVITE in open(os.path.join(cfgdir, "worker.env")).read())
envt = open(os.path.join(cfgdir, "worker.env")).read()
ok("display (1 GB off), leaderboard, windows, budgets in worker.env",
   all(x in envt for x in ("RQAH_VRAM_GB=31\n", "RQAH_LEADERBOARD=1\n", "RQAH_NAME=Ada Lovelace\n", "RQAH_WINDOWS=22:00-08:00\n",
                            "RQAH_MAX_DOWN_GB_MONTH=300\n", "RQAH_MAX_UP_MBPS=10\n", "RQAH_BUDGET_RESET_DAY=15\n")), envt)
ok("env lines parse the same in docker --env-file and compose (no quotes, $, #)",
   all(re.fullmatch(r"[A-Z_][A-Z0-9_]*=[^\"'$#\\]*", l) for l in envt.splitlines() if l and not l.startswith("#")))
run = open(os.path.join(cfgdir, "docker-run.sh")).read()
ok("docker-run.sh: env-file, devices, volume, localtime; no secret on the line",
   all(x in run for x in ("--env-file", "--device /dev/kfd", "--group-add 39", "rqah-state:/var/lib/rqah", "/etc/localtime:/etc/localtime:ro",
                          "--cap-drop ALL", "--memory 56g", "--stop-timeout 900")) and "RQAH_INVITE" not in run, run)
rc2 = subprocess.run(["sh", "-n", os.path.join(cfgdir, "docker-run.sh")]).returncode
ok("docker-run.sh: valid sh", rc2 == 0)
if shutil.which("docker") and subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
    for d in ("r9700", "nv8"):
        r = subprocess.run(["docker", "compose", "--project-directory", os.path.join(W, "cfg", d), "-f",
                            os.path.join(W, "cfg", d, "compose.yaml"), "config", "--format", "json"], capture_output=True, text=True)
        try: c = json.loads(r.stdout)["services"]["worker"]
        except (ValueError, KeyError): c = {}
        envc = c.get("environment", {})
        ok(f"docker compose config accepts {d}/compose.yaml and reads worker.env", r.returncode == 0 and envc.get("RQAH_SERVER")
           and (d != "r9700" or envc.get("RQAH_NAME") == "Ada Lovelace" and envc.get("RQAH_INVITE") == INVITE), r.stderr[-300:])
else:
    print("[SKIP] docker compose config (no compose plugin here)")

# re-run: a flag changes one value, everything else and the invite stay; .prev keeps the old one
rc, out, err = run_setup(["--yes", "--cache-gb", "120"], root, spec, cfgdir)
envt2 = open(os.path.join(cfgdir, "worker.env")).read()
ok("re-run: cache changed, the rest kept", rc == 0 and "RQAH_CACHE_GB=120\n" in envt2 and "RQAH_NAME=Ada Lovelace\n" in envt2
   and "RQAH_WINDOWS=22:00-08:00\n" in envt2 and f"RQAH_INVITE={INVITE}\n" in envt2, err[-200:])
ok("re-run: .prev backup, mode 0600", os.path.exists(os.path.join(cfgdir, "worker.env.prev")) and md("worker.env.prev") == 0o600
   and "RQAH_CACHE_GB=200\n" in open(os.path.join(cfgdir, "worker.env.prev")).read())
rc, out, err = run_setup(["--yes", "--no-invite"], root, spec, cfgdir)
ok("re-run --no-invite drops the invite", rc == 0 and "RQAH_INVITE" not in open(os.path.join(cfgdir, "worker.env")).read()
   and "no invite stored" in out)
rc, out, err = run_setup(["--yes", "--invite-file", "-"], root, spec, cfgdir, stdin="bad invite with spaces\n")
ok("a malformed invite is refused (and not printed)", rc == 1 and "bad invite" not in out + err, out[-200:])

# ================================================================ 5. --check
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("--check: ok", rc == 0 and "OK" in out, out[-300:])
shot("check-r9700", out)
rc, out, err = run_setup(["--check", "--json"], root, spec, cfgdir)
j = json.loads(out)
ok("--check --json", rc == 0 and j["ok"] and isinstance(j["warnings"], list))
os.chmod(os.path.join(cfgdir, "worker.env"), 0o644)
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("--check: loose worker.env mode is an error", rc == 1 and "readable by others" in out)
os.chmod(os.path.join(cfgdir, "worker.env"), 0o600)
c = os.path.join(cfgdir, "compose.yaml"); keep = open(c).read()
open(c, "a").write("# my edit\n")
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("--check: a hand edit is a warning", rc == 0 and "compose.yaml differs" in out)
open(c, "w").write(keep)
with open(os.path.join(cfgdir, "worker.env"), "a") as f: f.write('RQAH_NAME="quoted"\n')
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("--check: a quoted env line is an error", rc == 1 and "read differently" in out)
rc, out, err = run_setup(["--yes"], root, spec, cfgdir)                  # rewrite: clean again
os.rename(os.path.join(cfgdir, "docker-run.sh"), os.path.join(W, "x.sh"))
rc, out, err = run_setup(["--check"], root, spec, cfgdir)
ok("--check: a missing file is an error", rc == 1 and "docker-run.sh is missing" in out)
os.rename(os.path.join(W, "x.sh"), os.path.join(cfgdir, "docker-run.sh"))
rc, out, err = run_setup(["--check"], root, spec, os.path.join(W, "cfg", "nothing-here"))
ok("--check: no config -> exit 2", rc == 2 and not os.path.exists(os.path.join(W, "cfg", "nothing-here")))
# the same config on a machine whose card is gone / smaller
root_b, spec_b = machine("r9700-swapped", amd=[(120001, 15.92, "0x7550", False)], ram_gb=32)
rc, out, err = run_setup(["--check"], root_b, spec_b, cfgdir)
ok("--check: a config for a 32 GB card on a 16 GB card / 32 GB RAM box fails", rc == 1 and "more than the card has" in out
   and "more than this machine has" in out, out[-400:])
before = tree(cfgdir)
run_setup(["--check"], root, spec, cfgdir); run_setup(["--show"], root, spec, cfgdir)
ok("--check / --show change nothing", tree(cfgdir) == before)

# ================================================================ 6. validation rules
for bad in ("a@b.example", "http://x", "Ada#1", "cost $5", "<b>", "x" * 33, " lead", "deadbeef" * 4):
    ok(f"name refused: {bad[:20]!r}", S.check_name(bad) is not None)
ok("names accepted", all(S.check_name(n) is None for n in ("Ada", "Zoë 🚀 Team", "team-42_x.y")))
try:
    from server.donor import NAME_BAD as SRV_BAD
    ok("display-name rule in sync with server/donor.py", SRV_BAD.pattern == S.NAME_BAD.pattern)
except ImportError as x:
    print(f"[SKIP] server/donor.py: {x}")
ok("server URL rules", S.check_server("https://q.example.org") is None and S.check_server("http://127.0.0.1:8740") is None
   and S.check_server("http://q.example.org") and S.check_server("ftp://x") and S.check_server(""))
hw = S.detect(host_of(*machine("v", amd=[(120001, 31.86, "0x7551", False)], ram_gb=64)))
base = dict(S.DEFAULTS); base.update(S.recommend(hw)); base["server"] = "https://q.example.org"
for k, v, msg in (("windows", ["25:00-08:00"], "window"), ("budget_reset_day", 29, "1..28"), ("cache_gb", 5, "at least 10"),
                  ("cache_gb", 900, "free at"), ("vram_cap_gb", 40, "not above"), ("mem_limit_gb", 49, "a few GB above"),
                  ("state", "relative/path", "absolute"), ("gpus", [3], "no amd GPU with index 3"), ("max_up_mbps", -1, "≥ 0")):
    E, _ = S.validate(dict(base, **{k: v}), hw)
    ok(f"validate {k}={v!r}", any(msg in e for e in E), str(E))
E, Wn = S.validate(dict(base, state="/srv/rqah"), hw)
ok("a host path for the state: no error, warns to create it for uid 10001 (btrfs: reflinks ok)", not E and any("uid 10001" in w for w in Wn)
   and not any("reflinks" in w for w in Wn), str(E + Wn))
_, Wn = S.validate(dict(base, state="/srv/rqah"), dict(hw, selinux_enforcing=True))
ok("SELinux: the bind mount gets :z", S.state_mount(dict(base, state="/srv/rqah"), dict(hw, selinux_enforcing=True)).endswith(":z"))
hwe = S.detect(host_of(*machine("ext4", amd=[(120001, 31.86, "0x7551", False)], ram_gb=64, fs="ext4")))
_, Wn = S.validate(dict(base), hwe)
ok("ext4 cache on an eval card: reflink note", any("no reflinks" in w for w in Wn))
_, Wd = S.validate(dict(base, leaderboard=True), hw)
_, Wo = S.validate(dict(base, leaderboard=True, name="Ada"), hw)
_, Wf = S.validate(dict(base, leaderboard=False), hw)
ok("leaderboard with the default name 'donor': warned (the board credits by name); own name or not listed: no warning",
   any("default name" in w for w in Wd) and not any("default name" in w for w in Wo + Wf), str(Wd))

# ================================================================ 7. plain prompts (scripted)
root, spec = machine("plain", nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", True)], ram_gb=32)
hw = S.detect(host_of(root, spec))
cfg = dict(S.DEFAULTS); cfg.update(S.recommend(hw))
answers = iter(["n",                          # advanced? no
                "",                           # server: empty is not accepted while there is none
                "http://insecure.example", "https://queue.example.org",
                "Grace", "y",                 # name, leaderboard
                "0", "",                      # cards, vram (default 11)
                "20", "", "0", "8",           # ram, cache, down Mbps, up Mbps
                "0", "50", "3",               # down month, up month, reset day (asked: an up budget is set)
                "23:00-07:00"])
asked, pw_asked, outl = [], [], []


def inp(p):
    asked.append(p); return next(answers)


def getpw(p):
    pw_asked.append(p); return INVITE


res = S.plain_ui(cfg, None, hw, "/tmp/x", False, inp=inp, getpw=getpw, out=outl.append)
c, sec, ss = res
ok("plain: answers land", (c["server"], c["name"], c["leaderboard"], c["ram_gb"], c["max_up_mbps"], c["max_up_gb_month"],
                          c["budget_reset_day"], c["windows"], c["vram_gb"]) == ("https://queue.example.org", "Grace", True, 20, 8, 50, 3,
                                                                                  ["23:00-07:00"], 11), str(c))
ok("plain: ram change moved the container limit", c["mem_limit_gb"] == 28, str(c["mem_limit_gb"]))
ok("plain: the invite through getpass only", sec == INVITE and ss == "new" and len(pw_asked) == 1
   and not any(INVITE in p for p in asked) and not any(INVITE in l for l in outl))
ok("plain: a bad URL re-asked with a reason", any("https://" in l and "localhost" in l for l in outl))
E, Wn = S.validate(c, hw, sec)
shot("plain-4070", "\n".join(outl) + "\n" + "\n".join(f"{p}<answer>" for p in asked[:6]) + "\n...\n\n"
     + "\n".join(S.summary_lines(c, hw, ss, os.path.expanduser("~/.config/rad-quant-at-home"), E, Wn)))

# ================================================================ 8. the curses form: rendered, then driven through a pty
root, spec = machine("form", amd=[(120001, 31.86, "0x7551", False), (120001, 31.86, "0x7551", False)], ram_gb=128, rocminfo=True)
hw = S.detect(host_of(root, spec))
cfg = dict(S.DEFAULTS); cfg.update(S.recommend(hw)); cfg["server"] = "https://queue.example.org"; cfg["name"] = "Ada"
f = S.Form(cfg, INVITE, hw, "~/.config/rad-quant-at-home", True)
txt = "\n".join(t for t, _ in f.lines(100, 34))
ok("form: labels, values, keys; the invite hidden", "Queue URL" in txt and "(stored, hidden)" in txt and INVITE not in txt
   and "[s] summary" in txt and "0,1" in txt, "")
shot("form-2xr9700", txt)
f.pos = 5; txt = "\n".join(t for t, _ in f.lines(100, 34)); shot("form-2xr9700-cards", txt)
f.adv = True; f.pos = 9; txt = "\n".join(t for t, _ in f.lines(100, 34))
ok("form: advanced fields shown, scrolls", "Container memory limit" in txt and "›" in txt)
shot("form-2xr9700-advanced", txt)
f.mode = "summary"; txt = "\n".join(t for t, _ in f.lines(100, 40))
ok("form: summary screen, wrapped to the width", "What will be written" in txt and "[w] write" in txt and "TP2" in txt
   and all(len(l) <= 100 for l in txt.splitlines()))
shot("form-2xr9700-summary", txt)
ok("form: 80x24 fits", all(len(t) <= 80 for t, _ in f.lines(80, 24)) and len(f.lines(80, 24)) == 24)
f.mode = "form"
ok("form: a tiny terminal gets a message, not a crash", "too small" in "".join(t for t, _ in f.lines(40, 10)))


def pty_run(args, keys, cfgdir, env_extra=None, timeout=20):
    """run setup.py on a pseudo-terminal, typing `keys` (list of (delay, bytes)); returns (rc, all output)"""
    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update({"TERM": "xterm-256color", "LINES": "34", "COLUMNS": "100", **(env_extra or {})})
        os.environ.pop("RQAH_INVITE", None)
        os.execv(sys.executable, [sys.executable, "-I", os.path.join(HERE, "client", "setup.py"), "--fake-root", root,
                                  "--fake-cmds", spec, "--config-dir", cfgdir] + args)
    import fcntl, struct, termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 34, 100, 0, 0))
    buf, t0, ki = b"", time.time(), 0
    nxt = time.time() + 1.0
    while time.time() - t0 < timeout:
        if ki < len(keys) and time.time() >= nxt:
            os.write(fd, keys[ki][1]); nxt = time.time() + keys[ki][0]; ki += 1
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try: d = os.read(fd, 65536)
            except OSError: d = b""                               # EIO: the child closed the terminal
            if not d:
                _, st = os.waitpid(pid, 0); os.close(fd)
                return os.waitstatus_to_exitcode(st), buf.decode(errors="replace")
            buf += d
        pid_, st = os.waitpid(pid, os.WNOHANG)
        if pid_:
            try:
                while True:
                    d = os.read(fd, 65536)
                    if not d: break
                    buf += d
            except OSError: pass
            return os.waitstatus_to_exitcode(st), buf.decode(errors="replace")
    os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0)
    return None, buf.decode(errors="replace")


cfgdir = os.path.join(W, "cfg", "pty")
json.dump({"server": "https://queue.example.org", "name": "Ada"}, open(os.path.join(W, "pty-ans.json"), "w"))
# down to the invite (row 2), Enter, type it hidden, Enter; [s] summary, [w] write; then the post-write questions: n, n, n
keys = [(0.3, b"j"), (0.3, b"\r")] + [(0.02, bytes([c])) for c in INVITE.encode()] + [(0.3, b"\r"), (0.5, b"s"), (0.5, b"w"),
                                                                                    (0.5, b"n\r"), (0.5, b"n\r"), (0.5, b"n\r")]
rc, outp = pty_run(["--curses", "--answers", os.path.join(W, "pty-ans.json")], keys, cfgdir)
ok("pty curses: writes the config", rc == 0 and os.path.exists(os.path.join(cfgdir, "worker.env")), f"rc={rc} {outp[-300:]!r}")
ok("pty curses: the typed invite is stored and never echoed", os.path.exists(os.path.join(cfgdir, "worker.env"))
   and f"RQAH_INVITE={INVITE}\n" in open(os.path.join(cfgdir, "worker.env")).read() and INVITE not in outp
   and INVITE[:8] not in outp)
rc, outp = pty_run(["--curses"], [(0.5, b"q")], os.path.join(W, "cfg", "pty-quit"))
ok("pty curses: q quits without writing", rc == 1 and not os.path.exists(os.path.join(W, "cfg", "pty-quit")), f"rc={rc}")
rc, outp = pty_run(["--plain"], [(0.3, b"\x03")], os.path.join(W, "cfg", "pty-plain"))
ok("pty plain: Ctrl-C quits without writing", rc == 1 and "nothing written" in outp and not os.path.exists(os.path.join(W, "cfg", "pty-plain")))
# the plain prompts on a real terminal: a 4070 that drives the display, all defaults but the account and a window
root4, spec4 = machine("plain-pty", nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", True)], ram_gb=32)
root_save, spec_save = root, spec
root, spec = root4, spec4
typed = ["n", "https://queue.example.org", INVITE, "Grace", "y", "", "", "", "", "", "", "", "", "22:00-08:00", "y", "n", "n", "n"]
rc, outp = pty_run(["--plain"], [(0.35, (t + "\r").encode()) for t in typed], os.path.join(W, "cfg", "pty-plain-ok"), timeout=30)
envp = os.path.join(W, "cfg", "pty-plain-ok", "worker.env")
ok("pty plain: written; the invite stored, never echoed", rc == 0 and os.path.exists(envp) and f"RQAH_INVITE={INVITE}\n" in open(envp).read()
   and "RQAH_WINDOWS=22:00-08:00\n" in open(envp).read() and INVITE not in outp, f"rc={rc} {outp[-300:]!r}")
ok("pty plain: the 12 GB hint for a display card", "declaring 12 GB would add more jobs" in outp)
shot("plain-pty-4070", outp.replace("\r\n", "\n").replace("\r", ""))
root, spec = root_save, spec_save
r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "client", "setup.py"), "--fake-root", root, "--fake-cmds", spec,
                    "--config-dir", os.path.join(W, "cfg", "notty")], stdin=subprocess.DEVNULL, capture_output=True, text=True)
ok("no terminal and no --yes: exit 2 with a hint", r.returncode == 2 and "--yes" in r.stderr)

# ================================================================ 9. the smoke check's commands (fake docker)
root, spec = machine("smoke", nvidia=[("NVIDIA GeForce RTX 4070", 12282, "8.9", False)])
host = host_of(root, spec)
calls = []
orig = host.cmds["docker"]


def fake_docker(argv):
    calls.append(argv)
    if argv[1:3] == ["image", "inspect"]: return 0, "sha256:abc\n", ""
    if argv[1] == "compose" and "config" in argv: return 0, "", ""
    if argv[1] == "run": return 0, json.dumps([["NVIDIA GeForce RTX 4070", 12878086144]]) + "\n", ""
    return orig(argv)


host.cmds["docker"] = fake_docker
hw = S.detect(host)
cfg = dict(S.DEFAULTS); cfg.update(S.recommend(hw)); cfg["server"] = "https://q.example.org"
said = []
res = S.smoke(host, hw, cfg, "/cfg", pull=False, gpu_test=True, say=said.append)
runc = [c for c in calls if c[1] == "run"]
ok("smoke: compose validated, image looked up, GPU probe offline without the env file",
   res and any(c[1:3] == ["compose", "--project-directory"] for c in calls) and runc and "--network" in runc[0]
   and runc[0][runc[0].index("--network") + 1] == "none" and "--env-file" not in runc[0] and "--gpus" in runc[0]
   and not any(c[1] == "pull" for c in calls), str(said))
ok("smoke: reports the cards", any("RTX 4070 12 GB" in s for s in said), str(said))

# ================================================================ 10. Python 3.10
src = open(os.path.join(HERE, "client", "setup.py")).read()
try:
    ast.parse(src, feature_version=(3, 10)); ok("Python 3.10 grammar", True)
except SyntaxError as x:
    ok("Python 3.10 grammar", False, str(x))
nested = [l for l in src.splitlines() if re.search(r'f"[^"]*\{[^}]*"[^"]*"[^}]*\}', l)]
ok("no 3.12-only nested same-quote f-strings", not nested, str(nested[:3]))
imports = set(re.findall(r"^import (.+)$", src, re.M)[0].replace(" ", "").split(","))
ok("stdlib only", imports <= {"argparse", "datetime", "getpass", "glob", "json", "os", "re", "shlex", "shutil", "stat",
                              "subprocess", "sys", "tempfile", "textwrap", "unicodedata"}, str(imports))

n_fail = sum(1 for _, c in RESULTS if not c)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed; screens in {SCREENS}")
sys.exit(1 if n_fail else 0)
