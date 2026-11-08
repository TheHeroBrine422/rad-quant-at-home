#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rqah setup -- the donor's setup wizard (stdlib only, Python 3.10+): looks at this machine, asks what to donate,
and writes the worker's configuration into one directory (default ~/.config/rad-quant-at-home):

  worker.env      every worker setting as RQAH_* (and the invite): mode 0600, read by both ways of running it
  compose.yaml    `docker compose up -d` in that directory (the right image: ROCm gfx1201 or CUDA)
  docker-run.sh   the same as one `docker run` line, for hosts without the compose plugin
  setup.json      the answers (no secret), so a re-run starts from them

  rqah setup                       interactive: a full-screen form when the terminal allows it, else plain prompts
  rqah setup --plain               plain prompts (any terminal, also over SSH without a usable TERM)
  rqah setup --yes [flags]         no questions: flags, then --answers FILE, then the current config, then defaults
  rqah setup --check               validate the current config against this machine; changes nothing (exit 0 ok,
                                   1 problems, 2 no config)
  rqah setup --show                print the hardware it detects and the defaults it would offer
  rqah setup --update [--to X.Y.Z] [--yes] [--release] [--drain-timeout MIN] [--dry-run]     (alias `rqah update`)
                                   move the worker to the release the queue advertises (or --to): pull the new
                                   image's tag and check its digest (a moved or tampered tag is refused), drain the
                                   running container (its jobs finish; --release hands them back), rewrite the files
                                   with the new pin (TAG@DIGEST), recreate the container on the same state volume
                                   (token, input cache, budget counters stay)
  rqah setup --auto-update on|off  the hourly timer (systemd --user, asked before it is installed) that runs
                                   `--update --auto`: newer releases only, digest-checked, a day after publication,
                                   once the running jobs are done

What it does on its own: reads /proc, /sys, /dev and /etc, and runs `nvidia-smi`, `rocminfo` and `docker info` /
`docker compose version` / `docker image inspect` if they are installed (all read-only, all local). It writes only
inside the config directory and never runs sudo. Network only when asked: `--pull` (docker pull of the image),
`--start` (starts the worker, which registers with the invite) and `--update` (asks the queue for its release, pulls). `--smoke` runs the image offline (--network none) to
check it sees the GPUs. The invite is read without echo (or from --invite-file / $RQAH_INVITE) and never printed."""
import argparse, datetime, getpass, glob, json, os, re, shlex, shutil, stat, subprocess, sys, tempfile, textwrap, time, unicodedata
import urllib.error, urllib.request

SETUP_FORMAT = 1
IMAGE_REPO = "ghcr.io/theherobrine422/rad-quant-at-home-worker"
RELEASES = "https://github.com/TheHeroBrine422/rad-quant-at-home/releases"
ROCM_TAG, CUDA_TAG = "rocm7.14-gfx1201", "cu128"
ROCM_ARCHS = ("gfx1201",)                     # the arch the ROCm image is built for (Dockerfile.rocm GPU_ARCH)
CUDA_MIN_CC, NV_MIN_DRIVER = (8, 0), 570
STATE_IN = "/var/lib/rqah"                    # the image's state dir (token, cache, running jobs)
VOLUME = "rqah-state"
FILES = ("worker.env", "compose.yaml", "docker-run.sh", "setup.json")
GB = 1e9
GIB = 1 << 30


def repo_version():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
    try:
        v = open(p).read().strip()
        if re.fullmatch(r"\d+\.\d+\.\d+", v): return v
    except OSError:
        pass
    return "0.4.0"


VERSION = repo_version()


# ================================================================ the host (real, or a fake root + fake commands)
class Host:
    """everything detection reads goes through here, so tests can point it at a fake sysfs and canned command output.
    root: prefix for /sys, /proc, /dev, /etc; cmds: None = run the real commands, else {name: fn(argv) -> (rc, out, err)}"""

    def __init__(self, root="/", cmds=None, statvfs=None, euid=None):
        self.root, self.cmds = root, cmds
        self._statvfs = statvfs
        self.euid = os.geteuid() if euid is None else euid

    def p(self, path): return os.path.join(self.root, path.lstrip("/"))

    def read(self, path, default=None):
        try:
            with open(self.p(path), errors="replace") as f: return f.read()
        except OSError:
            return default

    def readf(self, real, default=None):
        """read a path that glob() already resolved under root"""
        try:
            with open(real, errors="replace") as f: return f.read()
        except OSError:
            return default

    def which(self, name):
        if self.cmds is not None: return name in self.cmds
        return shutil.which(name) is not None

    def run(self, argv, timeout=15):
        """(rc, stdout, stderr); 127 if the command is missing. Never through a shell."""
        if self.cmds is not None:
            fn = self.cmds.get(argv[0])
            return fn(argv) if fn else (127, "", f"{argv[0]}: not found")
        if not shutil.which(argv[0]): return 127, "", f"{argv[0]}: not found"
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {timeout} s"
        except OSError as x:
            return 126, "", f"{argv[0]}: {x.strerror}"

    def statvfs(self, path):
        """free space at a path as the host sees it (not prefixed: the fake takes logical paths)"""
        if self._statvfs: return self._statvfs(path)
        return os.statvfs(self.p(path))


def fake_host_from_file(root, spec_file):
    """testing only (--fake-root / --fake-cmds): a JSON {cmd: [[argv-prefix..., rc, out, err], ...] | "missing"},
    {"_statvfs": {"/path": free_bytes}}"""
    spec = json.load(open(spec_file))
    sv = spec.pop("_statvfs", {})
    euid = spec.pop("_euid", 1000)

    def mk(rules):
        def fn(argv):
            for r in rules:
                pre, (rc, out, err) = r[:-3], r[-3:]
                if list(argv[1:1 + len(pre)]) == pre: return rc, out, err
            return 1, "", "fake: no rule"
        return fn

    class V:
        def __init__(self, b): self.f_bavail, self.f_frsize = b // 4096, 4096

    def statvfs(path):
        best = max((k for k in sv if path == k or path.startswith(k.rstrip("/") + "/")), key=len, default=None)
        if best is None: raise FileNotFoundError(path)
        return V(sv[best])
    return Host(root, {k: mk(v) for k, v in spec.items() if v != "missing"}, statvfs=statvfs if sv else None, euid=euid)


# ================================================================ detection (read-only)
AMD_IDS = {"0x7551": "AMD Radeon AI PRO R9700", "0x7550": "AMD Radeon RX 9070 / 9070 XT", "0x7590": "AMD Radeon RX 9060 XT"}


def kv_lines(text):
    out = {}
    for l in (text or "").splitlines():
        p = l.split(None, 1)
        if len(p) == 2: out[p[0]] = p[1].strip()
    return out


def gfx_name(v):
    """kfd's gfx_target_version (major*10000 + minor*100 + stepping) -> gfx1201"""
    v = int(v)
    return f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}" if v else ""


def rocminfo_names(host):
    """[(gfx name, marketing name)] of GPU agents in rocminfo's order, if rocminfo runs"""
    rc, out, _ = host.run(["rocminfo"])
    if rc != 0: return []
    res, cur = [], {}
    for l in out.splitlines() + ["*******"]:
        if l.strip().startswith("*******") or l.startswith("Agent "):
            if cur.get("Device Type") == "GPU": res.append((cur.get("Name", ""), cur.get("Marketing Name", "")))
            cur = {}
            continue
        m = re.match(r"\s*([A-Za-z ]+?):\s+(.*)$", l)
        if m and m.group(1) not in cur: cur[m.group(1)] = m.group(2).strip()
    return res


def drm_cards(host):
    """{realpath of the PCI device: {"card": cardN, "vram": bytes, "display": bool, "product": str}}"""
    out = {}
    for c in sorted(glob.glob(host.p("/sys/class/drm/card[0-9]*"))):
        n = os.path.basename(c)
        if "-" in n: continue
        dev = os.path.realpath(os.path.join(c, "device"))
        vram = (host.read(f"/sys/class/drm/{n}/device/mem_info_vram_total") or "").strip()
        con = [host.read(f"/sys/class/drm/{os.path.basename(s)}/status", "").strip()
               for s in glob.glob(host.p(f"/sys/class/drm/{n}-*"))]
        out[dev] = {"card": n, "vram": int(vram) if vram.isdigit() else None, "display": "connected" in con,
                    "product": (host.read(f"/sys/class/drm/{n}/device/product_name") or "").strip(),
                    "vendor": (host.read(f"/sys/class/drm/{n}/device/vendor") or "").strip(),
                    "device": (host.read(f"/sys/class/drm/{n}/device/device") or "").strip()}
    return out


def detect_amd(host):
    gpus, cards = [], drm_cards(host)
    nodes = sorted(glob.glob(host.p("/sys/class/kfd/kfd/topology/nodes/[0-9]*")), key=lambda p: int(os.path.basename(p)))
    names = rocminfo_names(host)
    seen = set()
    for nd in nodes:
        pr = kv_lines(host.readf(nd + "/properties"))
        gv = int(pr.get("gfx_target_version", "0") or 0)
        if not gv: continue                                       # a CPU node
        idx = len(gpus)
        minor = pr.get("drm_render_minor", "")
        dev = os.path.realpath(host.p(f"/sys/class/drm/renderD{minor}/device")) if minor else None
        c = cards.get(dev, {})
        seen.add(dev)
        vram = c.get("vram")
        if vram is None:
            mb = kv_lines(host.readf(nd + "/mem_banks/0/properties"))
            vram = int(mb.get("size_in_bytes", 0)) or None
        did = "0x%04x" % int(pr.get("device_id", "0")) if pr.get("device_id", "").isdigit() else c.get("device", "")
        arch = gfx_name(gv)
        name = (names[idx][1] if idx < len(names) and names[idx][1] and names[idx][0] == arch else "") or c.get("product") \
            or AMD_IDS.get(did, "") or f"AMD GPU {did or '?'}"
        gpus.append({"vendor": "amd", "index": idx, "name": name, "arch": arch, "vram": vram, "display": c.get("display", False),
                     "card": c.get("card"), "render": f"renderD{minor}" if minor else None, "pci_id": did,
                     "igpu": bool(vram is not None and vram < 3 * GIB and arch not in ROCM_ARCHS)})
    # amdgpu cards without a kfd node (no compute: old kernel, kfd off): listed, not usable
    for dev, c in cards.items():
        if c["vendor"] == "0x1002" and dev not in seen:
            gpus.append({"vendor": "amd", "index": None, "name": c["product"] or AMD_IDS.get(c["device"], f"AMD GPU {c['device']}"),
                         "arch": "", "vram": c["vram"], "display": c["display"], "card": c["card"], "render": None,
                         "pci_id": c["device"], "igpu": False, "no_kfd": True})
    return gpus


def detect_nvidia(host):
    """(gpus, driver version or None, problem or None)"""
    q = "index,name,memory.total,compute_cap,display_active,driver_version,pci.bus_id"
    rc, out, err = host.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if rc == 127:
        nv = [d for d in glob.glob(host.p("/sys/bus/pci/devices/*")) if (host.readf(d + "/vendor") or "").strip() == "0x10de"
              and (host.readf(d + "/class") or "").startswith("0x03")]
        return [], None, ("an NVIDIA GPU is on the PCI bus but nvidia-smi is missing (no NVIDIA driver?)" if nv else None)
    if rc != 0:
        return [], None, f"nvidia-smi failed: {(err or out).strip()[:200]}"
    gpus, drv = [], None
    for l in out.strip().splitlines():
        f = [x.strip() for x in l.split(",")]
        if len(f) < 7: continue
        try: cc = tuple(int(x) for x in f[3].split("."))
        except ValueError: cc = None
        drv = f[5]
        gpus.append({"vendor": "nvidia", "index": int(f[0]), "name": f[1],
                     "vram": int(float(f[2])) * (1 << 20) if re.fullmatch(r"[\d.]+", f[2]) else None,
                     "arch": f"cc {f[3]}" if cc else "", "cc": cc, "display": f[4].lower() == "enabled", "pci_id": f[6],
                     "igpu": False})
    return gpus, drv, None


def meminfo(host):
    m = {}
    for l in (host.read("/proc/meminfo") or "").splitlines():
        p = l.split()
        if len(p) >= 2 and p[1].isdigit(): m[p[0].rstrip(":")] = int(p[1]) * 1024
    return m


def detect_cpu(host):
    model = ""
    for l in (host.read("/proc/cpuinfo") or "").splitlines():
        if l.startswith("model name"): model = l.split(":", 1)[1].strip(); break
    n = sum(1 for l in (host.read("/proc/cpuinfo") or "").splitlines() if l.startswith("processor"))
    return {"model": model, "threads": n or (os.cpu_count() if host.root == "/" else 0)}


def groups(host):
    g = {}
    for l in (host.read("/etc/group") or "").splitlines():
        p = l.split(":")
        if len(p) >= 3 and p[2].isdigit(): g[p[0]] = int(p[2])
    if host.root == "/":
        import grp
        for n in ("video", "render"):
            if n not in g:
                try: g[n] = grp.getgrnam(n).gr_gid
                except KeyError: pass
    return g


def detect_tz(host):
    t = (host.read("/etc/timezone") or "").strip()
    if t: return t
    try:
        l = os.readlink(host.p("/etc/localtime"))
        if "zoneinfo/" in l: return l.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return None


def detect_docker(host):
    d = {"present": host.which("docker"), "podman": False, "access": False, "error": None, "version": None, "root_dir": None,
         "runtimes": [], "rootless": False, "selinux": False, "compose": None, "nvidia_toolkit": False}
    d["nvidia_toolkit"] = any(host.which(x) for x in ("nvidia-ctk", "nvidia-container-cli", "nvidia-container-runtime-hook")) \
        or bool(glob.glob(host.p("/etc/cdi/nvidia*")) + glob.glob(host.p("/var/run/cdi/nvidia*")))
    if not d["present"]: return d
    rc, out, err = host.run(["docker", "--version"])
    d["podman"] = "podman" in (out + err).lower()
    rc, out, err = host.run(["docker", "info", "--format", "{{json .}}"])
    if rc == 0:
        try: info = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError): info = {}
        d["access"] = bool(info.get("ServerVersion"))
        d["version"] = info.get("ServerVersion")
        d["root_dir"] = info.get("DockerRootDir")
        d["runtimes"] = sorted((info.get("Runtimes") or {}).keys())
        so = " ".join(info.get("SecurityOptions") or [])
        d["rootless"], d["selinux"] = "rootless" in so, "selinux" in so
        if "nvidia" in d["runtimes"]: d["nvidia_toolkit"] = True
        if not d["access"]: d["error"] = "docker info gave no server version"
    else:
        e = (err or out).strip()
        d["error"] = ("this user can't reach the Docker daemon (permission denied on its socket): add yourself to the "
                      "docker group (then log in again), or run the docker commands with sudo") if "permission denied" in e.lower() \
            else ("the Docker daemon is not running" if "cannot connect" in e.lower() or "is the docker daemon running" in e.lower()
                  else f"docker info failed: {e[:200]}")
    rc, out, _ = host.run(["docker", "compose", "version", "--short"])
    d["compose"] = out.strip() if rc == 0 and out.strip() else None
    return d


def disk_at(host, path):
    """(free bytes, fstype, the existing directory measured) at path or its nearest existing parent"""
    p, free = os.path.abspath(path), None
    while True:
        try:
            s = host.statvfs(p); free = s.f_bavail * s.f_frsize; break
        except OSError:
            if p == "/": break
            p = os.path.dirname(p)
    fs, best = None, -1
    for l in (host.read("/proc/mounts") or "").splitlines():
        f = l.split()
        if len(f) >= 3:
            mp = f[1].replace("\\040", " ")
            if (p == mp or p.startswith(mp.rstrip("/") + "/")) and len(mp) > best: fs, best = f[2], len(mp)
    return free, fs, p


def detect(host):
    amd = detect_amd(host)
    nv, drv, nv_problem = detect_nvidia(host)
    m = meminfo(host)
    dk = detect_docker(host)
    g = groups(host)
    return {"gpus": amd + nv, "nvidia_driver": drv, "nvidia_problem": nv_problem, "ram": m.get("MemTotal"),
            "ram_avail": m.get("MemAvailable"), "cpu": detect_cpu(host), "docker": dk,
            "kfd": os.path.exists(host.p("/dev/kfd")),
            "kfd_access": os.access(host.p("/dev/kfd"), os.R_OK | os.W_OK) if os.path.exists(host.p("/dev/kfd")) else False,
            "dri": sorted(os.path.basename(x) for x in glob.glob(host.p("/dev/dri/renderD*"))),
            "video_gid": g.get("video"), "render_gid": g.get("render"), "tz": detect_tz(host),
            "localtime": os.path.exists(host.p("/etc/localtime")),
            "selinux_enforcing": (host.read("/sys/fs/selinux/enforce") or "").strip() == "1",
            "euid": host.euid, "_host": host}


# ================================================================ policy: what a card / host can do, defaults
def gpu_supported(g, hw):
    """(ok, why not)"""
    if g.get("no_kfd"): return False, "no /dev/kfd compute node for it (amdgpu without KFD?)"
    if g["vendor"] == "amd":
        if g.get("igpu"): return False, "integrated GPU"
        if g["arch"] not in ROCM_ARCHS: return False, f"{g['arch'] or 'unknown arch'}: the ROCm image is built for {', '.join(ROCM_ARCHS)} only"
        return True, ""
    if g.get("cc") is None: return False, "compute capability unknown (driver too old?)"
    if g["cc"] < CUDA_MIN_CC: return False, f"compute capability {g['cc'][0]}.{g['cc'][1]} < 8.0"
    try:
        if hw.get("nvidia_driver") and int(hw["nvidia_driver"].split(".")[0]) < NV_MIN_DRIVER:
            return False, f"NVIDIA driver {hw['nvidia_driver']} < {NV_MIN_DRIVER}"
    except ValueError:
        pass
    return True, ""


def gib(b): return None if b is None else b / GIB


def card_gb(g):
    """a card's VRAM as the README counts it (12 GB for a 4070's 12282 MiB, 32 for an R9700)"""
    return int(round(gib(g["vram"]))) if g.get("vram") else 0


def jobs_for(vendor, vram, ram, cards=1):
    """what the server would send a worker with these declared caps (README job table, docs/SMALL-CARDS.md);
    [(text, ok)] for the summary"""
    out = []
    if vram < 6 or ram < 4:
        return [("nothing: every 27B job needs at least 6 GB of VRAM and 4 GB of RAM (only test jobs)", False)]
    if vram < 8 or ram < 12:
        out.append(("27B Hessian groups except down_proj (quant_tensor, only for levels split by group)", True))
    elif vram < 12:
        out.append(("27B layers up to K6 (quant_layer), and every Hessian group but down_proj above K6 (quant_tensor)", True))
    else:
        out.append(("every 27B layer and Hessian group at every K (quant_layer, quant_tensor)", True))
    if vram >= 32 and ram >= 48: out.append(("the LM head (quant_head: 32 GB VRAM, 48 GB RAM)", True))
    elif vram >= 32: out.append(("not the LM head (needs 48 GB of RAM)", False))
    if vendor == "amd":
        if vram >= 32 and ram >= 24: out.append(("mix evaluations up to 7 bpw (eval_mix)", True))
        elif vram >= 16 and ram >= 16: out.append(("mix evaluations up to 4.5 bpw (eval_mix)", True))
        if cards >= 2 and vram >= 32: out.append(("two-card (TP2) evaluations above 7 bpw", True))
    else:
        out.append(("no mix evaluations (the CUDA image has quantization jobs only)", False))
    return out


def gpu_slug(name):
    """the --gpu-name default: 'NVIDIA GeForce RTX 4070' -> rtx4070, 'AMD Radeon AI PRO R9700' -> r9700"""
    drop = {"amd", "radeon", "nvidia", "geforce", "ai", "pro", "graphics", "gpu", "laptop", "/"}
    t = [w for w in re.split(r"[\s_]+", (name or "").lower()) if w and w not in drop and w != "/"]
    s = re.sub(r"[^a-z0-9]", "", "".join(t))[:32]
    return s or "gpu"


def recommend(hw, vendor=None, gpus=None, state=None):
    """defaults for this machine (the questions start from these unless a config exists)"""
    ok = [g for g in hw["gpus"] if gpu_supported(g, hw)[0]]
    if vendor is None:
        vs = sorted({g["vendor"] for g in ok}, key=lambda v: -sum(card_gb(g) for g in ok if g["vendor"] == v))
        vendor = vs[0] if vs else ("amd" if any(g["vendor"] == "amd" for g in hw["gpus"]) or hw["kfd"] else "nvidia")
    mine = [g for g in ok if g["vendor"] == vendor]
    sel = gpus if gpus is not None else [g["index"] for g in mine]
    chosen = [g for g in hw["gpus"] if g["vendor"] == vendor and g["index"] in sel]
    vram = min((card_gb(g) - (1 if g.get("display") else 0) for g in chosen), default=0)
    ram_total = int((hw["ram"] or 0) / GIB + 0.5)
    ram = max(0, min(int(ram_total * 0.75), ram_total - 4))
    mem = max(ram + 2, min(ram + 8, ram_total - 1)) if ram_total else 0
    eval_ok = vendor == "amd" and vram >= 16
    want_cache = 200 if eval_ok else 50
    st = state or "volume"
    free, _, _ = disk_at(hw["_host"], state_dir_for_disk(st, hw))
    cache = want_cache if free is None else max(0, min(want_cache, int(free / GB) - 25))
    return {"vendor": vendor, "gpus": sel, "gpu_name": gpu_slug(chosen[0]["name"]) if chosen else "", "vram_gb": max(vram, 0),
            "vram_cap_gb": 0, "ram_gb": ram, "mem_limit_gb": mem, "state": st, "cache_gb": cache}


def state_dir_for_disk(state, hw):
    if state == "volume": return (hw["docker"].get("root_dir") or "/var/lib/docker") + "/volumes"
    return state


# ================================================================ the configuration
DEFAULTS = {"server": "", "name": "donor", "leaderboard": False, "vendor": None, "gpus": None, "gpu_name": "", "vram_gb": 0,
            "vram_cap_gb": 0, "ram_gb": 0, "mem_limit_gb": 0, "state": "volume", "cache_gb": 0, "windows": [],
            "max_down_mbps": 0, "max_up_mbps": 0, "max_down_gb_day": 0, "max_down_gb_month": 0, "max_up_gb_day": 0,
            "max_up_gb_month": 0, "budget_reset_day": 1, "on_stop": "finish", "container": "rad-quant-at-home", "image": "",
            "version": "", "digest": "", "auto_update": False}
# version / digest: the worker release and image digest pinned by `--update` ("" = this setup's own VERSION, no digest);
# auto_update: the hourly timer may run `--update --auto` (AUTO_UPDATE_DEFAULT is what the question offers)
NUM = ("vram_gb", "vram_cap_gb", "ram_gb", "mem_limit_gb", "cache_gb", "max_down_mbps", "max_up_mbps", "max_down_gb_day",
       "max_down_gb_month", "max_up_gb_day", "max_up_gb_month")
# display names: the server's rule (server/donor.py clean_display_name: tests/setup_test.py keeps them in sync), plus
# `#` and `$`, which a compose env file would read as a comment or an interpolation
NAME_BAD = re.compile(r"(?i)(?:\b(?:\d{1,3}\.){3}\d{1,3}\b|@|://|\b(?:rq|ri)[A-Za-z0-9_-]{16,}|[0-9a-f]{24,}|[<>&\"'`\\])")
WINDOW_RE = re.compile(r"([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d")
CONTAINER_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}")
IMAGE_RE = re.compile(r"[a-z0-9][a-z0-9._/:@-]{0,254}")
INVITE_RE = re.compile(r"[\x21-\x7e]{4,512}")                   # printable, no spaces: safe in an env file


def check_name(s):
    if not isinstance(s, str) or not 1 <= len(s) <= 32: return "1..32 characters"
    if s != s.strip(): return "no leading or trailing spaces"
    if any(unicodedata.category(c)[0] == "C" for c in s): return "no control characters"
    if NAME_BAD.search(s): return "no addresses, e-mails, URLs, tokens or <>&\"'`\\ characters"
    if "#" in s or "$" in s: return "no # or $"
    return None


def check_server(s):
    if not s: return "required: the queue URL you were given"
    m = re.fullmatch(r"(https?)://([A-Za-z0-9.-]+|\[[0-9a-fA-F:]+\])(:\d{1,5})?(/[A-Za-z0-9._~/-]*)?", s)
    if not m: return "a URL like https://queue.example.org"
    if m.group(1) == "http" and m.group(2) not in ("localhost", "127.0.0.1", "[::1]"):
        return "https:// (plain http only for localhost tests)"
    return None


def parse_windows(v):
    if isinstance(v, list): ws = v
    else: ws = [x for x in str(v or "").replace(" ", ",").split(",") if x and x.lower() not in ("always", "-", "none")]
    return ws


VERSION_RE = re.compile(r"\d{1,4}\.\d{1,4}\.\d{1,4}")


def vtuple(v): return tuple(int(x) for x in v.split(".")) if isinstance(v, str) and VERSION_RE.fullmatch(v) else None


def image_for(cfg, version=None, digest=None):
    """the image reference compose.yaml / docker-run.sh run: REPO:VERSION-STACK, pinned @DIGEST once `--update` checked
    one (docker then pulls and runs exactly that image, whatever the tag points at later)"""
    if cfg.get("image"): return cfg["image"]
    d = digest if version else cfg.get("digest")
    return (f"{IMAGE_REPO}:{version or cfg.get('version') or VERSION}-{ROCM_TAG if cfg['vendor'] == 'amd' else CUDA_TAG}"
            + (f"@{d}" if d else ""))


def validate(cfg, hw, secret=None):
    """(errors, warnings): errors stop a write (unless --force); warnings are shown in the summary"""
    E, W = [], []
    x = check_server(cfg["server"])
    if x: E.append(f"server: {x}")
    x = check_name(cfg["name"])
    if x: E.append(f"name: {x}")
    if secret and not INVITE_RE.fullmatch(secret): E.append("invite: printable characters without spaces")
    if cfg["vendor"] not in ("amd", "nvidia"): E.append("vendor: amd or nvidia")
    for k in NUM:
        v = cfg[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v or v < 0: E.append(f"{k}: a number ≥ 0")
    if E: return E, W
    if not isinstance(cfg["budget_reset_day"], int) or not 1 <= cfg["budget_reset_day"] <= 28: E.append("budget_reset_day: 1..28")
    for w in cfg["windows"]:
        if not WINDOW_RE.fullmatch(w): E.append(f"window {w!r}: HH:MM-HH:MM (24 h, local time)")
    if cfg["on_stop"] not in ("finish", "abandon"): E.append("on_stop: finish or abandon")
    if not CONTAINER_RE.fullmatch(cfg["container"]): E.append("container: letters, digits, _ . -")
    if cfg["image"] and not IMAGE_RE.fullmatch(cfg["image"]): E.append("image: a lower-case image reference")
    if cfg.get("version") and not VERSION_RE.fullmatch(str(cfg["version"])): E.append("version: X.Y.Z (set by --update)")
    if cfg.get("digest") and not re.fullmatch(r"sha256:[0-9a-f]{64}", str(cfg["digest"])): E.append("digest: sha256:<64 hex> (set by --update)")
    if not isinstance(cfg.get("auto_update", False), bool): E.append("auto_update: true or false")
    if cfg["gpu_name"] and not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,31}", cfg["gpu_name"]): E.append("gpu_name: [a-z0-9._-], ≤ 32")
    st = cfg["state"]
    if st != "volume" and (not st.startswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]*", st) or "/../" in st + "/"):
        E.append("state: 'volume' or an absolute path ([A-Za-z0-9._/-])"); return E, W
    # GPUs
    gs = cfg["gpus"] or []
    if not gs: E.append("gpus: pick at least one card")
    byidx = {g["index"]: g for g in hw["gpus"] if g["vendor"] == cfg["vendor"] and g["index"] is not None}
    sel = []
    for i in gs:
        g = byidx.get(i)
        if g is None: E.append(f"gpus: no {cfg['vendor']} GPU with index {i} on this machine"); continue
        sel.append(g)
        ok, why = gpu_supported(g, hw)
        if not ok: E.append(f"GPU {i} ({g['name']}): {why}")
    if len({g["name"] for g in sel}) > 1:
        W.append("the selected cards are different models: the worker declares one VRAM size for all (the smallest); "
                 "consider one worker per model (a second config dir)")
    if sel:
        lo = min(card_gb(g) for g in sel)
        if cfg["vram_gb"] > lo: E.append(f"vram_gb: {cfg['vram_gb']:g} is more than the card has ({lo} GB)")
        if any(g.get("display") for g in sel) and cfg["vram_gb"] > lo - 1:
            W.append("a selected card drives a display: declare ~1 GB less than it has (and/or set vram_cap_gb)")
    if cfg["vram_cap_gb"] and cfg["vram_cap_gb"] > cfg["vram_gb"]: E.append("vram_cap_gb: not above vram_gb")
    if cfg["vram_gb"] < 6: E.append(
        f"vram_gb {cfg['vram_gb']:g}: no 27B job fits below 6 GB (docs/SMALL-CARDS.md); the worker would only get test jobs")
    ram_total = (hw["ram"] or 0) / GIB
    if ram_total and cfg["ram_gb"] > ram_total: E.append(f"ram_gb: {cfg['ram_gb']:g} is more than this machine has ({ram_total:.0f} GB)")
    elif ram_total and cfg["ram_gb"] > ram_total - 2: W.append("ram_gb leaves less than 2 GB to the rest of the machine")
    if cfg["ram_gb"] < 12: W.append(f"ram_gb {cfg['ram_gb']:g}: a 27B layer job needs 12 GB of RAM; below that only small Hessian groups (4 GB)")
    elif len(gs) > 1 and cfg["ram_gb"] < 12 * len(gs):
        W.append(f"ram_gb {cfg['ram_gb']:g}: {len(gs)} cards need {12 * len(gs)} GB to run a layer job on each at once; "
                 f"{int(cfg['ram_gb'] // 12)} will be busy at a time")
    if cfg["mem_limit_gb"] and cfg["mem_limit_gb"] < cfg["ram_gb"] + 2:
        E.append("mem_limit_gb: the container limit must be a few GB above ram_gb")
    if ram_total and cfg["mem_limit_gb"] > ram_total: W.append("mem_limit_gb is above this machine's RAM")
    if cfg["leaderboard"] and cfg["name"].casefold() == DEFAULTS["name"]:
        W.append(f"leaderboard with the default name {cfg['name']!r}: the board credits by name, so you would share a row "
                 "with everyone who kept it; pick your own")
    # disk
    free, fs, where = disk_at(hw["_host"], state_dir_for_disk(st, hw))
    if cfg["cache_gb"] < 10: E.append("cache_gb: at least 10 (a 27B layer's inputs are ~1.5 GB, an eval's up to 17 GB)")
    if free is not None:
        need = cfg["cache_gb"] + (25 if cfg["vendor"] == "amd" and cfg["vram_gb"] >= 16 else 2)
        if need * GB > free: E.append(f"cache_gb: {cfg['cache_gb']:g} GB + staging needs ~{need:g} GB, {free / GB:.0f} GB free at {where}")
        elif (need + 20) * GB > free: W.append(f"the disk at {where} is nearly full ({free / GB:.0f} GB free for ~{need:g} GB)")
    else:
        W.append(f"couldn't measure the free disk at {where}")
    if cfg["vendor"] == "amd" and cfg["vram_gb"] >= 16 and fs and fs not in ("xfs", "btrfs"):
        W.append(f"the cache is on {fs}: eval mixes are copied (no reflinks), which needs up to ~25 GB of extra disk per eval")
    if st != "volume":
        d = hw["_host"].p(st)
        if not os.path.isdir(d): W.append(f"{st} doesn't exist yet: create it for uid 10001 (sudo install -d -o 10001 -g 10001 -m 0700 {st})")
        else:
            s = os.stat(d)
            if s.st_uid != 10001 and not s.st_mode & 0o002:
                W.append(f"{st} isn't owned by uid 10001 (the container's user): sudo chown 10001:10001 {st}")
        if hw["docker"].get("rootless"): W.append("rootless Docker maps uid 10001 elsewhere: prefer the named volume")
    # bandwidth
    for d in ("down", "up"):
        if cfg[f"max_{d}_gb_day"] and cfg[f"max_{d}_gb_month"] and cfg[f"max_{d}_gb_day"] > cfg[f"max_{d}_gb_month"]:
            W.append(f"max_{d}_gb_day is above the monthly budget")
    if cfg["max_down_gb_month"] and cfg["max_down_gb_month"] < 20 or cfg["max_down_gb_day"] and cfg["max_down_gb_day"] < 2:
        W.append("a download budget this small fits few jobs (a 27B layer downloads ~1.5 GB the first time)")
    if cfg["max_up_mbps"] and cfg["max_up_mbps"] < 2: W.append("under 2 Mbit/s up, a 300 MB shard takes 20+ minutes")
    # docker / devices
    dk = hw["docker"]
    if not dk["present"]: W.append("Docker isn't installed: install it (docs.docker.com/engine/install) before starting the worker")
    elif dk["error"]: W.append(f"Docker: {dk['error']}")
    if dk.get("podman"): W.append("`docker` here is Podman: untested; the files are written for Docker")
    if cfg["vendor"] == "amd":
        if not hw["kfd"]: E.append("/dev/kfd is missing: the amdgpu driver with KFD (ROCm compute) is needed")
        if not hw["dri"]: E.append("no /dev/dri/renderD* nodes")
        if hw["video_gid"] is None or hw["render_gid"] is None: W.append("no video / render group on this host: the container may not open the GPU")
    else:
        if dk["present"] and not dk["nvidia_toolkit"]:
            E.append("the NVIDIA Container Toolkit is missing (docker run --gpus needs it): "
                     "docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
        if hw.get("nvidia_problem"): W.append(hw["nvidia_problem"])
    if cfg["windows"] and not hw["localtime"]: W.append("no /etc/localtime: the windows will be read as UTC")
    if hw.get("euid") == 0: W.append("running as root: run setup as your own user (it needs no sudo)")
    return E, W


# ================================================================ rendering the files
def env_lines(cfg, secret):
    L = [f"RQAH_SERVER={cfg['server']}"]
    if secret: L.append(f"RQAH_INVITE={secret}")
    L += [f"RQAH_NAME={cfg['name']}", f"RQAH_LEADERBOARD={1 if cfg['leaderboard'] else 0}",
          f"RQAH_GPU_NAME={cfg['gpu_name']}", f"RQAH_GPUS={','.join(str(i) for i in cfg['gpus'])}",
          f"RQAH_VRAM_GB={cfg['vram_gb']:g}", f"RQAH_RAM_GB={cfg['ram_gb']:g}", f"RQAH_CACHE_GB={cfg['cache_gb']:g}"]
    if cfg["vram_cap_gb"]: L.append(f"RQAH_VRAM_CAP_GB={cfg['vram_cap_gb']:g}")
    L += [f"RQAH_WINDOWS={','.join(cfg['windows'])}", f"RQAH_ON_STOP={cfg['on_stop']}"]
    for k in ("max_down_mbps", "max_up_mbps", "max_down_gb_day", "max_down_gb_month", "max_up_gb_day", "max_up_gb_month",
              "budget_reset_day"):
        L.append(f"RQAH_{k.upper()}={cfg[k]:g}")
    if cfg["vendor"] == "nvidia": L.append("CUDA_DEVICE_ORDER=PCI_BUS_ID")    # job indices = nvidia-smi's
    return L


def render_env(cfg, secret, stamp):
    head = ["# rad-quant-at-home worker settings, written by `rqah setup` " + stamp + ".",
            "# Change them by running `rqah setup` again (it starts from these), then recreate the container.",
            "# PRIVATE: this file holds your invite (mode 0600). Read by `docker compose` (compose.yaml) and by",
            "# `docker run --env-file` (docker-run.sh): plain KEY=VALUE, no quotes. 0 = unlimited / off."]
    return "\n".join(head + env_lines(cfg, secret)) + "\n"


def state_mount(cfg, hw):
    if cfg["state"] == "volume": return f"{VOLUME}:{STATE_IN}"
    return f"{cfg['state']}:{STATE_IN}" + (":z" if hw["selinux_enforcing"] else "")


def y(s): return json.dumps(str(s))         # a YAML double-quoted scalar


def render_compose(cfg, hw, stamp):
    amd = cfg["vendor"] == "amd"
    L = [f"# rad-quant-at-home worker ({'AMD, ROCm' if amd else 'NVIDIA, CUDA'}), written by `rqah setup` {stamp}.",
         "# In this directory:  docker compose up -d      (first start registers with the invite in worker.env)",
         "#                     docker compose logs -f    docker compose stop   (finishes the running jobs, up to 15 min)",
         f"#                     docker exec {cfg['container']} rqah status",
         "# Re-run `rqah setup` to change settings, then `docker compose up -d` again (recreates the container).",
         "name: rad-quant-at-home-worker", "", "services:", "  worker:",
         f"    image: {y(image_for(cfg))}", f"    container_name: {y(cfg['container'])}", "    restart: unless-stopped",
         "    env_file: [worker.env]"]
    if amd:
        L += ["    devices: [/dev/kfd, /dev/dri]",
              f"    group_add: [{y(hw['video_gid'] if hw['video_gid'] is not None else 'video')}, "
              f"{y(hw['render_gid'] if hw['render_gid'] is not None else 'render')}]   # the host's video, render",
              "    # ROCm's NUMA calls need seccomp off (AMD's documented container setup); the rest stays locked down",
              '    security_opt: ["seccomp=unconfined", "no-new-privileges:true"]']
    else:
        L += ["    deploy:", "      resources:", "        reservations:", "          devices:",
              "            - {driver: nvidia, count: all, capabilities: [gpu]}   # = docker run --gpus all",
              '    security_opt: ["no-new-privileges:true"]']
    L += ["    cap_drop: [ALL]", f"    mem_limit: {cfg['mem_limit_gb']:g}g", "    stop_grace_period: 15m", "    volumes:",
          f"      - {y(state_mount(cfg, hw))}   # token, running jobs, input cache"]
    if hw["localtime"]: L.append('      - "/etc/localtime:/etc/localtime:ro"   # the time windows use your local time')
    L += ["    logging:", "      driver: json-file", '      options: {max-size: "20m", max-file: "3"}']
    if cfg["state"] == "volume": L += ["", "volumes:", f"  {VOLUME}:", f"    name: {VOLUME}   # the same volume docker-run.sh uses"]
    return "\n".join(L) + "\n"


def run_argv(cfg, hw, cfgdir):
    a = ["docker", "run", "-d", "--name", cfg["container"], "--restart", "unless-stopped",
         "--env-file", os.path.join(cfgdir, "worker.env")]
    if cfg["vendor"] == "amd":
        a += ["--device", "/dev/kfd", "--device", "/dev/dri",
              "--group-add", str(hw["video_gid"] if hw["video_gid"] is not None else "video"),
              "--group-add", str(hw["render_gid"] if hw["render_gid"] is not None else "render"),
              "--security-opt", "seccomp=unconfined"]
    else:
        a += ["--gpus", "all"]
    a += ["--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--memory", f"{cfg['mem_limit_gb']:g}g",
          "--stop-timeout", "900", "-v", state_mount(cfg, hw)]
    if hw["localtime"]: a += ["-v", "/etc/localtime:/etc/localtime:ro"]
    a += ["--log-opt", "max-size=20m", "--log-opt", "max-file=3", image_for(cfg)]
    return a


def render_run(cfg, hw, cfgdir, stamp):
    a = [shlex.quote(x) for x in run_argv(cfg, hw, cfgdir)]
    lines, cur = [], "exec"
    for x in a:
        if len(cur) + len(x) + 1 > 110: lines.append(cur + " \\"); cur = "  " + x
        else: cur += " " + x
    lines.append(cur)
    return "\n".join(["#!/bin/sh", f"# rad-quant-at-home worker as one docker run line, written by `rqah setup` {stamp}.",
                      "# Creates and starts the container (registers with the invite in worker.env on the first start).",
                      f"# After a re-run of setup: docker stop {cfg['container']} && docker rm {cfg['container']} && sh {shlex.quote(os.path.join(cfgdir, 'docker-run.sh'))}",
                      "set -e"] + lines) + "\n"


def summary_lines(cfg, hw, secret_state, cfgdir, E=(), W=()):
    sel = [g for g in hw["gpus"] if g["vendor"] == cfg["vendor"] and g["index"] in (cfg["gpus"] or [])]
    cards = ", ".join(f"{g['index']}: {g['name']}" for g in sel)
    rate = lambda v: "unlimited" if not v else f"{v:g} Mbit/s"
    bud = lambda v, p: "" if not v else f"{v:g} GB/{p}"
    def budget(d):
        x = [s for s in (bud(cfg[f'max_{d}_gb_day'], 'day'), bud(cfg[f'max_{d}_gb_month'], 'month')) if s]
        return ", ".join(x) or "no cap"
    L = ["What will be written (in " + cfgdir + "):",
         "  worker.env      the settings below + " + {"new": "your invite", "kept": "your invite (kept)", "none": "no invite (a first registration starts on probation)"}[secret_state]
         + "  (mode 0600)",
         f"  compose.yaml    image {image_for(cfg)}",
         "  docker-run.sh   the same as one docker run line",
         "  setup.json      these answers (no invite), for the next run", "",
         "What the worker may use:",
         f"  queue           {cfg['server']}",
         f"  cards           {cards or '-'}",
         f"  VRAM            {cfg['vram_gb']:g} GB per card" + (f", allocator capped at {cfg['vram_cap_gb']:g} GB" if cfg["vram_cap_gb"] else "")
         + f"; reported as {cfg['gpu_name'] or '-'}",
         f"  host RAM        {cfg['ram_gb']:g} GB for all jobs; container limit {cfg['mem_limit_gb']:g} GB",
         f"  disk            input cache ≤ {cfg['cache_gb']:g} GB in " + (f"the Docker volume {VOLUME}" if cfg["state"] == "volume" else cfg["state"]),
         f"  network         down {rate(cfg['max_down_mbps'])}, up {rate(cfg['max_up_mbps'])}; budget down {budget('down')}, "
         f"up {budget('up')}" + (f" (month from day {cfg['budget_reset_day']})" if cfg["max_down_gb_month"] or cfg["max_up_gb_month"] else ""),
         f"  when            {', '.join(cfg['windows']) + ' (' + (hw['tz'] or 'local time') + ')' if cfg['windows'] else 'always'}"
         + f"; on stop: {cfg['on_stop']} the running jobs",
         "  updates         " + (f"automatic: an hourly timer (systemd --user) takes a new release a day after it is published, "
                                 "digest-checked, between jobs" if cfg.get("auto_update") else "by hand: `rqah update` when the worker says so"),
         f"  leaderboard     " + (f"listed as {cfg['name']!r} (with every opted-in machine of that name)" if cfg["leaderboard"]
                                  else f"not listed (worker name {cfg['name']!r})"), "",
         "Jobs it will get" + (" (small-card mode, docs/SMALL-CARDS.md):" if 0 < cfg["vram_gb"] < 12 else ":")]
    for t, ok in jobs_for(cfg["vendor"], cfg["vram_gb"], cfg["ram_gb"], len(cfg["gpus"] or [])):
        L.append(f"  {'+' if ok else '-'} {t}")
    lo = min((card_gb(g) for g in sel), default=0)
    for edge in (12, 16, 32):
        if cfg["vram_gb"] < edge <= lo:
            L.append(f"  (declaring {edge} GB would add more jobs: only if nothing else, e.g. your desktop, uses the card)"); break
    L += ["", "It runs only the job types built into the image, makes only outgoing HTTPS requests to the queue,",
          "opens no port, and reads and writes only its state volume. Your token stays in that volume."]
    if E: L += [""] + [f"ERROR    {x}" for x in E]
    if W: L += [""] + [f"WARNING  {x}" for x in W]
    return L


def hardware_lines(hw):
    L = []
    ram = hw["ram"] or 0
    L.append(f"CPU     {hw['cpu']['model'] or '?'} ({hw['cpu']['threads']} threads)   RAM {ram / GIB:.0f} GB"
             + (f" ({(hw['ram_avail'] or 0) / GIB:.0f} GB available now)" if hw["ram_avail"] else ""))
    if not hw["gpus"]: L.append("GPUs    none found" + (f" ({hw['nvidia_problem']})" if hw["nvidia_problem"] else ""))
    for g in hw["gpus"]:
        ok, why = gpu_supported(g, hw)
        L.append(f"GPU {g['vendor'].upper():<6} {'#' + str(g['index']) if g['index'] is not None else '-':<3} {g['name'][:34]:<34} "
                 f"{(card_gb(g) and str(card_gb(g)) + ' GB') or '? GB':>6}  {g['arch']:<8} {'display' if g.get('display') else '':<7} "
                 + ("ok" if ok else "not usable: " + why))
    dk = hw["docker"]
    if not dk["present"]: L.append("Docker  not installed")
    else:
        L.append("Docker  " + (f"{dk['version']}" if dk["access"] else "installed") + (" (Podman)" if dk["podman"] else "")
                 + (f", compose {dk['compose']}" if dk["compose"] else ", no compose plugin")
                 + (", rootless" if dk["rootless"] else "") + (f", data in {dk['root_dir']}" if dk["root_dir"] else "")
                 + (f" -- {dk['error']}" if dk["error"] else ""))
    amd = any(g["vendor"] == "amd" and (g["arch"] in ROCM_ARCHS or (g.get("vram") or 0) >= 4 * GIB) for g in hw["gpus"])
    nv = any(g["vendor"] == "nvidia" for g in hw["gpus"])
    if amd or hw["kfd"]:
        L.append(f"ROCm    /dev/kfd {'ok' if hw['kfd'] else 'MISSING'}, /dev/dri {' '.join(hw['dri']) or 'none'}, "
                 f"groups video {hw['video_gid']} render {hw['render_gid']} (no host ROCm needed)")
    if nv:
        L.append(f"NVIDIA  driver {hw['nvidia_driver'] or '?'}, Container Toolkit {'found' if dk['nvidia_toolkit'] else 'NOT found'}")
    L.append(f"Time    {hw['tz'] or 'unknown zone'}" + ("" if hw["localtime"] else " (no /etc/localtime)"))
    return L


# ================================================================ loading and writing the config dir
def default_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "rad-quant-at-home")


def read_env_file(path):
    out = {}
    try:
        for l in open(path):
            l = l.rstrip("\n")
            if not l or l.lstrip().startswith("#") or "=" not in l: continue
            k, v = l.split("=", 1); out[k.strip()] = v
    except OSError:
        pass
    return out


def load(cfgdir):
    """(cfg or None, stored invite or None)"""
    s = None
    try: s = json.load(open(os.path.join(cfgdir, "setup.json")))
    except (OSError, ValueError): pass
    env = read_env_file(os.path.join(cfgdir, "worker.env"))
    if not s: return None, env.get("RQAH_INVITE") or None
    cfg = dict(DEFAULTS); cfg.update({k: v for k, v in (s.get("settings") or {}).items() if k in DEFAULTS})
    return cfg, env.get("RQAH_INVITE") or None


def write_atomic(path, text, mode):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        if os.path.exists(path) and not os.path.islink(path):
            try:
                if open(path).read() != text: shutil.copyfile(path, path + ".prev"); os.chmod(path + ".prev", mode)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def stamp_now(): return datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def render_all(cfg, hw, secret, cfgdir, stamp):
    js = {"format": SETUP_FORMAT, "version": VERSION, "written": stamp, "settings": cfg,
          "hardware": {"gpus": [{k: g.get(k) for k in ("vendor", "index", "name", "arch", "vram")} for g in hw["gpus"]],
                       "ram": hw["ram"]}}
    return {"worker.env": (render_env(cfg, secret, stamp), 0o600), "compose.yaml": (render_compose(cfg, hw, stamp), 0o644),
            "docker-run.sh": (render_run(cfg, hw, cfgdir, stamp), 0o700),
            "setup.json": (json.dumps(js, indent=1, sort_keys=True) + "\n", 0o600)}


def write_all(cfgdir, cfg, hw, secret):
    os.makedirs(cfgdir, mode=0o700, exist_ok=True)
    os.chmod(cfgdir, 0o700)
    files = render_all(cfg, hw, secret, cfgdir, stamp_now())
    for n, (text, mode) in files.items(): write_atomic(os.path.join(cfgdir, n), text, mode)
    return [os.path.join(cfgdir, n) for n in files]


def strip_stamp(t):
    return re.sub(r"(written by `rqah setup` )[^.\n]*(\.?)", r"\1-\2", t) if t else t


def check_dir(cfgdir, hw):
    """`--check`: (errors, warnings, info) for an existing config, against this machine; changes nothing"""
    E, W, I = [], [], []
    cfg, secret = load(cfgdir)
    if cfg is None: return None, None, [f"no configuration in {cfgdir} (run `rqah setup`)"]
    for n in FILES:
        p = os.path.join(cfgdir, n)
        if not os.path.exists(p): E.append(f"{n} is missing"); continue
        st = os.stat(p)
        if n in ("worker.env", "setup.json") and st.st_mode & 0o077: E.append(f"{n} is readable by others (mode {stat.S_IMODE(st.st_mode):04o}; want 0600)")
        if st.st_uid != os.geteuid() and hw.get("euid") is not None and hw["euid"] != 0: W.append(f"{n} belongs to another user (uid {st.st_uid})")
    for l in open(os.path.join(cfgdir, "worker.env")) if os.path.exists(os.path.join(cfgdir, "worker.env")) else []:
        l = l.rstrip("\n")
        if l and not l.startswith("#") and not re.fullmatch(r"[A-Z_][A-Z0-9_]*=[^\"'$#]*", l):
            E.append(f"worker.env: a line docker and compose would read differently: {l.split('=', 1)[0]}=...")
    e2, w2 = validate(cfg, hw, secret)
    E += e2; W += w2
    if not secret: I.append("no invite stored: fine once the worker has registered (its token is in the state volume); "
                            "before that it registers without one, on probation")
    want = render_all(cfg, hw, secret, cfgdir, "-")
    for n in ("worker.env", "compose.yaml", "docker-run.sh"):
        try: have = open(os.path.join(cfgdir, n)).read()
        except OSError: continue
        if strip_stamp(have) != strip_stamp(want[n][0]):
            W.append(f"{n} differs from what setup would write for setup.json (edited by hand, or written for other hardware)")
    I.append(f"image {image_for(cfg)}")
    return E, W, I


# ================================================================ local checks after writing (no network unless asked)
def smoke(host, hw, cfg, cfgdir, pull=False, gpu_test=False, say=print):
    """(ok, lines). Local only, except docker pull when pull=True."""
    dk, ok = hw["docker"], True
    img = image_for(cfg)
    if not dk["present"] or not dk["access"]:
        say("docker: " + ("not installed" if not dk["present"] else dk["error"] or "no access") + "; skipping the container checks")
        return False
    if dk["compose"]:
        rc, out, err = host.run(["docker", "compose", "--project-directory", cfgdir, "-f", os.path.join(cfgdir, "compose.yaml"),
                                 "config", "--quiet"])
        say(f"compose.yaml: {'valid' if rc == 0 else 'INVALID: ' + (err or out).strip()[:300]}")
        ok &= rc == 0
    if pull:
        say(f"pulling {img} (several GB) ...")
        rc, out, err = host.run(["docker", "pull", img], timeout=7200)
        say("pulled" if rc == 0 else f"pull failed: {(err or out).strip()[-300:]}")
        ok &= rc == 0
    rc, out, _ = host.run(["docker", "image", "inspect", "--format", "{{.Id}}", img])
    have = rc == 0
    say(f"image {img}: " + ("present locally" if have else "not pulled yet (rqah setup --pull, or the first start pulls it)"))
    if gpu_test and have:
        py = "/usr/local/bin/python" if cfg["vendor"] == "amd" else "/opt/venv/bin/python"
        a = run_argv(cfg, hw, cfgdir)
        dev = []
        i = 0
        while i < len(a):
            if a[i] in ("--device", "--group-add", "--gpus") or (a[i] == "--security-opt" and a[i + 1].startswith("seccomp")):
                dev += a[i:i + 2]; i += 2
            else: i += 1
        code = ("import json,torch;n=torch.cuda.device_count();print(json.dumps([[torch.cuda.get_device_name(i),"
                "torch.cuda.get_device_properties(i).total_memory] for i in range(n)]))")
        rc, out, err = host.run(["docker", "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--security-opt",
                                 "no-new-privileges"] + dev + ["--entrypoint", py, img, "-I", "-c", code], timeout=600)
        try: seen = json.loads(out.strip().splitlines()[-1]) if rc == 0 else None
        except (ValueError, IndexError): seen = None
        if seen is None:
            say(f"GPU check in the image FAILED: {(err or out).strip()[-400:]}"); ok = False
        else:
            say(f"GPU check in the image (offline container): {len(seen)} card(s): "
                + "; ".join(f"{n} {m / GIB:.0f} GB" for n, m in seen))
            miss = [i for i in cfg["gpus"] if i >= len(seen)]
            if miss: say(f"  the container doesn't see card(s) {miss}"); ok = False
    return ok


def container_state(host, name):
    """(exists, running, image reference) of the worker container"""
    rc, out, _ = host.run(["docker", "inspect", "--format", "{{.State.Running}} {{.Config.Image}}", name])
    if rc != 0 or not out.strip(): return False, False, None
    run, _, img = out.strip().partition(" ")
    return True, run == "true", img or None


def drain(host, name, timeout_s=3600, release=False, say=print, sleep=time.sleep, clock=time.time):
    """stop the running worker without losing work. Its restart policy goes to "no" first (an exit must not bring it
    back), then SIGTERM: it finishes its running jobs and exits (--release, or after timeout_s: a second SIGTERM,
    which hands them back to the queue at once, not counted against them). -> True once it has stopped"""
    ex, run, _ = container_state(host, name)
    if not run: return True
    host.run(["docker", "update", "--restart", "no", name])
    host.run(["docker", "kill", "--signal", "TERM", name])
    if release:
        sleep(2); host.run(["docker", "kill", "--signal", "TERM", name])
        say(f"draining {name}: its running jobs are handed back to the queue now")
    else:
        say(f"draining {name}: it finishes its running jobs, then stops (up to {timeout_s / 60:g} min; "
            "--release hands them back instead)")
    t_end, said = clock() + (120 if release else timeout_s), clock()
    while clock() < t_end:
        if not container_state(host, name)[1]: return True
        if clock() - said >= 300: say("  still finishing ..."); said = clock()
        sleep(5)
    if not release:
        say(f"still running after {timeout_s / 60:g} min: handing its jobs back to the queue")
        host.run(["docker", "kill", "--signal", "TERM", name])
        t_end = clock() + 120
        while clock() < t_end:
            if not container_state(host, name)[1]: return True
            sleep(5)
    say("it did not stop by itself: docker stop (a job it still held resumes in the new container if its lease is alive, "
        "else the queue hands it to another worker)")
    host.run(["docker", "stop", "--time", "30", name], timeout=120)
    return not container_state(host, name)[1]


def start(host, hw, cfg, cfgdir, say=print, drain_s=3600, release=False, sleep=time.sleep, clock=time.time):
    """opt-in: create and start the worker (it registers with the queue on its first start). A worker container of
    that name that is running is drained first (drain()), then removed (never its volume) and created anew"""
    ex, run, _ = container_state(host, cfg["container"])
    if run and not drain(host, cfg["container"], drain_s, release, say, sleep, clock):
        say("start FAILED: the running worker did not stop"); return False
    if ex:
        rc, out, err = host.run(["docker", "rm", cfg["container"]])
        if rc != 0: say(f"start FAILED: docker rm {cfg['container']}: {(err or out).strip()[-300:]}"); return False
    if hw["docker"]["compose"]:
        argv = ["docker", "compose", "--project-directory", cfgdir, "-f", os.path.join(cfgdir, "compose.yaml"), "up", "-d"]
    else:
        argv = ["sh", os.path.join(cfgdir, "docker-run.sh")]
    rc, out, err = host.run(argv, timeout=7200)
    say(("started: " if rc == 0 else "start FAILED: ") + (out + err).strip()[-400:])
    return rc == 0


# ================================================================ updating (--update, --auto)
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
AUTO_UPDATE_DEFAULT = False      # the wizard's default answer (README "Updates and versions": the trade-off)
AUTO_DELAY_H = 24                # --auto takes a release only once it is this old (time to pull a bad one)
AUTO_DRAIN_MIN = 360             # --auto lets the running jobs finish for up to this long


def stack_of(cfg): return ROCM_TAG if cfg["vendor"] == "amd" else CUDA_TAG


def fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": f"rqah-setup/{VERSION}"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read(1 << 20) or b"{}")


def fetch_version(server, timeout=15):
    """the queue's GET /api/v0/version (no token): {"api", "server", "worker": {"current", "min", "images", ...}}"""
    return fetch_json(server.rstrip("/") + "/api/v0/version", timeout)


def fetch_release(version, timeout=30):
    """release.json of a GitHub Release (common/release.py; only for --to a release the queue does not advertise)"""
    return fetch_json(f"{RELEASES}/download/v{version}/release.json", timeout)


def pick_image(images, cfg, version):
    """(digest, why not) of this config's stack in an advertised {stack: {image, digest}} or a manifest's
    {stack: {repo, tag, digest}}: the image must be IMAGE_REPO:VERSION-STACK, the digest sha256:<64 hex>"""
    im = images.get(stack_of(cfg)) if isinstance(images, dict) else None
    if not isinstance(im, dict): return None, f"no {stack_of(cfg)} image advertised for {version}"
    ref = im.get("image") or (f"{im.get('repo')}:{im.get('tag')}" if im.get("repo") else None)
    if ref != f"{IMAGE_REPO}:{version}-{stack_of(cfg)}":
        return None, f"the advertised image {str(ref)[:120]!r} is not {IMAGE_REPO}:{version}-{stack_of(cfg)}"
    d = im.get("digest")
    if not isinstance(d, str) or not DIGEST_RE.fullmatch(d): return None, "the advertised digest is malformed"
    return d, None


def repo_digests(host, ref):
    rc, out, _ = host.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", ref])
    try: return json.loads(out) if rc == 0 else []
    except ValueError: return []


def published_age_h(published, now):
    try: t = datetime.datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError): return None
    return (now - t) / 3600


def update(host, hw, cfg, cfgdir, secret, a, say=print, ask=None, fetch=fetch_version, fetch_rel=fetch_release,
           sleep=time.sleep, clock=time.time):
    """--update: the advertised (or --to) worker release -> pull its tag, check the digest, drain, rewrite the pin
    (REPO:TAG@DIGEST), recreate. Exit code: 0 done or nothing to do, 1 failed, 2 refused / usage.

    What it trusts: the repository is IMAGE_REPO (fixed here); the release (X.Y.Z) and its digest come from the queue
    (or, for --to another release, from that GitHub Release's release.json, or --digest); the pulled tag must carry
    that digest, so a tag that was moved or tampered with on the registry is refused. --auto (the timer): only when
    setup.json has auto_update on, only to a newer release, only one published at least AUTO_DELAY_H ago, never
    without a digest; the running jobs finish first (a job boundary)."""
    auto = bool(getattr(a, "auto", False))
    if auto and not cfg.get("auto_update"): say("auto-update is off in setup.json (rqah setup --auto-update on)"); return 0
    if cfg.get("image"):
        say(f"setup.json overrides the image ({cfg['image']}): set the new one with `rqah setup --image REF --start`"); return 2
    have = cfg.get("version") or VERSION
    digest, why_d, info = getattr(a, "digest", None), None, {}
    if digest and not DIGEST_RE.fullmatch(digest): say("--digest: sha256:<64 hex>"); return 2
    try: v = fetch(cfg["server"])
    except (urllib.error.URLError, OSError, ValueError) as x:
        if not a.to:
            say(f"could not ask the queue ({cfg['server']}) for its release: {getattr(x, 'reason', x)}; try again, or give --to X.Y.Z")
            return 1
        v = {}
    w = v.get("worker") if isinstance(v, dict) and isinstance(v.get("worker"), dict) else {}
    if a.to:
        target = a.to
        if not VERSION_RE.fullmatch(target): say(f"--to {target!r}: X.Y.Z"); return 2
        why = "--to"
    else:
        target = w.get("current")
        if not isinstance(target, str) or not VERSION_RE.fullmatch(target):
            say("the queue's answer has no worker release (a server before 0.5?): give --to X.Y.Z"); return 1
        why = f"the queue advertises {target}" + (f" (oldest still served: {w.get('min')})" if isinstance(w.get("min"), str) else "")
    if w.get("current") == target: info = w
    if not digest and not getattr(a, "no_digest_check", False):
        if info: digest, why_d = pick_image(info.get("images"), cfg, target)
        else:
            try:
                rel = fetch_rel(target)
                digest, why_d = pick_image(rel.get("images") if isinstance(rel, dict) and rel.get("version") == target else None, cfg, target)
                info = {"notes": rel.get("notes_url"), "published": rel.get("created"), "setup_min": rel.get("setup_min")} if digest else {}
            except (urllib.error.URLError, OSError, ValueError) as x:
                why_d = f"no release.json for {target} on GitHub ({getattr(x, 'reason', x)})"
        if not digest:
            say(f"no digest to check {target} against: {why_d}. Refused (give --digest sha256:... from the release notes, "
                "or --no-digest-check for an image you built yourself)"); return 1
    if auto and not digest: say("--auto never runs without a digest"); return 2
    sm = info.get("setup_min")
    if isinstance(sm, str) and VERSION_RE.fullmatch(sm) and vtuple(sm) > vtuple(VERSION):
        say(f"release {target} needs rqah setup {sm} or newer (this is {VERSION}): update your rad-quant-at-home checkout "
            "(git pull) first; nothing changed"); return 2
    new_img, old_img = image_for(cfg, target, digest), image_for(cfg)
    ex, run, cur_img = container_state(host, cfg["container"])
    if target == have and (not ex or cur_img == new_img) and (digest or "") == (cfg.get("digest") or ""):
        if not auto or ex: say(f"up to date: {new_img}" + ("" if ex else " (no worker container yet: rqah setup --start)"))
        return 0
    if vtuple(target) < vtuple(have) and (not a.to or auto):
        say(f"the queue advertises {target}, older than the pinned {have}: nothing done (a downgrade needs --to {target})"); return 2
    if auto:
        if vtuple(target) == vtuple(have): return 0                  # same release, other digest: a person decides
        age = published_age_h(info.get("published"), clock())
        delay = getattr(a, "auto_delay_h", None) or cfg.get("auto_delay_h") or AUTO_DELAY_H
        if age is None or age < delay:
            say(f"release {target} is available; --auto takes it once it is {delay:g} h old" +
                (f" ({age:.1f} h now)" if age is not None else " (no publication time advertised)")); return 0
    drain_min = a.drain_timeout if a.drain_timeout is not None else (AUTO_DRAIN_MIN if auto else 60)
    say(f"update: {old_img} -> {new_img} ({why})")
    if isinstance(info.get("notes"), str) and info["notes"].startswith(RELEASES): say(f"  release notes: {info['notes']}")
    say(f"  1. docker pull {IMAGE_REPO}:{target}-{stack_of(cfg)}" + (f", check that it is {digest}" if digest else " (NO digest check)"))
    say(f"  2. drain {cfg['container']}: " + ("hand its running jobs back now" if a.release else
        f"let its running jobs finish (up to {drain_min:g} min, then hand them back)") if run else
        f"  2. {cfg['container']} is not running: nothing to drain")
    say(f"  3. rewrite compose.yaml / docker-run.sh / setup.json with {target}; recreate the container on the same state "
        + ("volume " + VOLUME if cfg["state"] == "volume" else "directory " + cfg["state"]) + " (token, cache, budget counters stay)")
    if a.dry_run: say("dry run: nothing done"); return 0
    if not (a.yes or auto):
        if ask is None:
            if not sys.stdin.isatty(): say("not a terminal: add --yes to go ahead"); return 2
            ask = lambda q: ask_yes(input, q, True)
        if not ask("Go ahead?"): say("cancelled: nothing changed"); return 1
    if not hw["docker"]["present"] or not hw["docker"]["access"]:
        say("docker: " + ("not installed" if not hw["docker"]["present"] else hw["docker"]["error"] or "no access")); return 1
    tag_ref = f"{IMAGE_REPO}:{target}-{stack_of(cfg)}"
    say(f"pulling {tag_ref} ...")
    rc, out, err = host.run(["docker", "pull", tag_ref], timeout=7200)
    if rc != 0: say(f"pull failed, nothing changed: {(err or out).strip()[-300:]}"); return 1
    if digest:
        got = repo_digests(host, tag_ref)
        if f"{IMAGE_REPO}@{digest}" not in got:
            say(f"REFUSED: {tag_ref} on the registry is not the image the release published (expected {digest}, got "
                f"{', '.join(x.split('@')[-1] for x in got if '@' in x) or 'no digest'}): the tag was moved or tampered with. "
                "Nothing changed; tell the operator"); return 1
        say(f"digest checked: {digest}")
    if run and not drain(host, cfg["container"], drain_min * 60, a.release, say, sleep, clock):
        say(f"the old worker did not stop; nothing else changed (docker update --restart unless-stopped {cfg['container']} "
            "restores its restart policy)"); return 1
    cfg = dict(cfg, version=target if target != VERSION or digest else "", digest=digest or "")
    write_all(cfgdir, cfg, hw, secret)
    say(f"written: compose.yaml, docker-run.sh, setup.json in {cfgdir} (the previous ones as *.prev)")
    if not start(host, hw, cfg, cfgdir, say, drain_min * 60, a.release, sleep, clock):
        say(f"the old container is stopped and the files name {target}: fix the error above, then `rqah setup --start` "
            f"(or go back: copy the *.prev files over and run `docker compose up -d`)"); return 1
    ex, run, img = container_state(host, cfg["container"])
    if not run or img not in (new_img, f"{IMAGE_REPO}@{digest}" if digest else new_img):
        say(f"after the update the container is {'running ' + str(img) if run else 'not running'}: check `docker logs {cfg['container']}`")
        return 1
    say(f"updated to {target}: {cfg['container']} runs {new_img}; `docker exec {cfg['container']} rqah status` shows it")
    return 0


# ---------------------------------------------------------------- the auto-update timer (systemd --user, opt-in)
UNIT = "rqah-update"


def render_units(cfgdir, python=None, script=None):
    """a oneshot service running --update --auto and an hourly timer for it; written into the config dir and linked
    into systemd's user manager only when the donor agrees (install_timer)"""
    py, sc = python or sys.executable or "/usr/bin/python3", script or os.path.abspath(__file__)
    svc = "\n".join([
        "# rad-quant-at-home: automatic worker updates (rqah setup --auto-update on). Written by rqah setup; it runs",
        "# `setup.py --update --auto`, which does nothing unless setup.json says auto_update and the queue advertises a",
        "# newer release (digest-checked, at least a day old). Off: rqah setup --auto-update off (or disable the timer).",
        "[Unit]", "Description=rad-quant-at-home worker auto-update", "After=network-online.target docker.service", "",
        "[Service]", "Type=oneshot", f"ExecStart={shlex.quote(py)} -I {shlex.quote(sc)} --update --auto --config-dir {shlex.quote(cfgdir)}",
        f"TimeoutStartSec={AUTO_DRAIN_MIN + 60}min", ""])
    tmr = "\n".join([
        "# rad-quant-at-home: when to look for a worker update (see rqah-update.service)",
        "[Unit]", "Description=rad-quant-at-home worker auto-update (hourly)", "",
        "[Timer]", "OnBootSec=20min", "OnUnitInactiveSec=1h", "RandomizedDelaySec=20min", "Persistent=true", "",
        "[Install]", "WantedBy=timers.target", ""])
    return {f"{UNIT}.service": svc, f"{UNIT}.timer": tmr}


def install_timer(host, cfgdir, on, say=print):
    """link and enable (on) or disable (off) the timer in the donor's systemd user manager; True if it worked"""
    if not host.which("systemctl"):
        if on: say(f"no systemctl: run `python3 {os.path.abspath(__file__)} --update --auto --config-dir {cfgdir}` hourly from cron instead")
        return False
    if not on:
        host.run(["systemctl", "--user", "disable", "--now", f"{UNIT}.timer"]); return True
    for argv in (["systemctl", "--user", "link", os.path.join(cfgdir, f"{UNIT}.service")],
                 ["systemctl", "--user", "enable", "--now", os.path.join(cfgdir, f"{UNIT}.timer")]):
        rc, out, err = host.run(argv)
        if rc != 0: say(f"{' '.join(argv[:3])} failed: {(err or out).strip()[-200:]}"); return False
    say(f"auto-update timer on ({UNIT}.timer, hourly; `systemctl --user list-timers` shows it). For it to run while you are "
        f"logged out: `loginctl enable-linger {os.environ.get('USER', '$USER')}` (once)")
    return True


# ================================================================ the questions
class Q:
    def __init__(self, key, label, kind, help_, section, advanced=False, when=None, choices=None):
        self.key, self.label, self.kind, self.help, self.section, self.advanced = key, label, kind, help_, section, advanced
        self.when, self.choices = when or (lambda c: True), choices


QUESTIONS = [
    Q("server", "Queue URL", "text", "the https:// URL of the queue (from us)", "Account"),
    Q("invite", "Invite code", "secret", "optional: the one-time code from us skips probation; leave empty to start on probation "
      "(quantization jobs only until a few results pass). Hidden while you type, stored only in worker.env (0600); "
      "Enter keeps the stored one, '-' removes it", "Account"),
    Q("name", "Display name", "text", "worker name; also your public name if you opt in below: 1-32 characters, no e-mail or URL. "
      "Use the same name on every machine to be credited together", "Account"),
    Q("leaderboard", "On the public leaderboard", "bool", "opt in to be listed by display name on the status page (one row per "
      "name: every opted-in machine with this name adds to it; anyone can use any name)", "Account"),
    Q("vendor", "GPU vendor", "choice", "AMD (ROCm image) or NVIDIA (CUDA image): one worker per vendor", "GPUs",
      choices=("amd", "nvidia")),
    Q("gpus", "Cards to donate", "gpus", "device indices, e.g. 0 or 0,1; all = every usable card", "GPUs"),
    Q("vram_gb", "VRAM per card (GB)", "num", "what the worker may use on each card; ~1 GB less if it drives your display", "GPUs"),
    Q("vram_cap_gb", "Cap the allocator at (GB)", "num", "0 = off; set it if you game or work on the card while it runs", "GPUs", True),
    Q("gpu_name", "GPU model name", "text", "reported with results (e.g. r9700, rtx4070)", "GPUs", True),
    Q("ram_gb", "Host RAM for jobs (GB)", "num", "all running jobs together; a 27B layer job needs 12, the LM head 48", "Memory and disk"),
    Q("mem_limit_gb", "Container memory limit (GB)", "num", "hard limit for the container; a few GB above the RAM for jobs", "Memory and disk", True),
    Q("state", "State and cache location", "text", "'volume' (a Docker volume, recommended) or an absolute host path "
      "(must belong to uid 10001)", "Memory and disk", True),
    Q("cache_gb", "Input cache (GB)", "num", "disk cap of the downloaded inputs; 50 is plenty for quantization, 200 for evaluations",
      "Memory and disk"),
    Q("max_down_mbps", "Download speed cap (Mbit/s)", "num", "0 = unlimited; all jobs together", "Network"),
    Q("max_up_mbps", "Upload speed cap (Mbit/s)", "num", "0 = unlimited; all jobs together", "Network"),
    Q("max_down_gb_month", "Download budget per month (GB)", "num", "0 = no cap; 1 GB = 10^9 bytes; leave your ISP cap some margin", "Network"),
    Q("max_up_gb_month", "Upload budget per month (GB)", "num", "0 = no cap", "Network"),
    Q("budget_reset_day", "Month starts on day", "int", "1-28, e.g. your ISP's billing day", "Network",
      when=lambda c: bool(c["max_down_gb_month"] or c["max_up_gb_month"])),
    Q("max_down_gb_day", "Download budget per day (GB)", "num", "0 = no cap", "Network", True),
    Q("max_up_gb_day", "Upload budget per day (GB)", "num", "0 = no cap", "Network", True),
    Q("windows", "Run only during", "windows", "local time, e.g. 22:00-08:00 or 22:00-08:00,12:00-13:00; 'always' = no limit", "Schedule"),
    Q("auto_update", "Update automatically", "bool", "an hourly timer on this machine (systemd --user) updates the worker to a "
      "new release we publish: only after the release is a day old, only if the image digest matches what the queue "
      "advertises, and only between jobs. Off: you update with `rqah update` when the worker tells you", "Schedule"),
    Q("on_stop", "On stop", "choice", "finish the running jobs (up to 15 min) or hand them back at once", "Schedule", True,
      choices=("finish", "abandon")),
    Q("container", "Container name", "text", "docker exec <name> rqah status", "Advanced", True),
    Q("image", "Image (empty = default)", "text", "override the worker image (e.g. a locally built tag)", "Advanced", True),
]


def fmt(cfg, q, secret_state=None):
    v = cfg.get(q.key) if q.key != "invite" else None
    if q.kind == "secret": return {"new": "(entered, hidden)", "kept": "(stored, hidden)", "none": "(none)"}[secret_state or "none"]
    if q.kind == "bool": return "yes" if v else "no"
    if q.kind == "gpus": return ",".join(str(i) for i in (v or [])) or "-"
    if q.kind == "windows": return ",".join(v) if v else "always"
    if q.kind in ("num", "int"): return f"{v:g}"
    if q.key == "image" and not v: return f"(default) {image_for(cfg)}"
    return str(v if v is not None else "")


def parse_answer(cfg, q, s, hw):
    """the typed text -> value; raises ValueError with a message"""
    s = s.strip()
    if q.kind == "bool":
        if s.lower() in ("y", "yes", "on", "1", "true"): return True
        if s.lower() in ("n", "no", "off", "0", "false"): return False
        raise ValueError("yes or no")
    if q.kind == "num":
        try: v = float(s)
        except ValueError: raise ValueError("a number") from None
        if v < 0 or v != v or v == float("inf"): raise ValueError("a number ≥ 0")
        return int(v) if v == int(v) else v
    if q.kind == "int":
        if not s.isdigit(): raise ValueError("a whole number")
        return int(s)
    if q.kind == "choice":
        if s.lower() not in q.choices: raise ValueError(" or ".join(q.choices))
        return s.lower()
    if q.kind == "gpus":
        if s.lower() == "all":
            return [g["index"] for g in hw["gpus"] if g["vendor"] == cfg["vendor"] and gpu_supported(g, hw)[0]]
        if not re.fullmatch(r"\d+(\s*,\s*\d+)*", s): raise ValueError("indices like 0 or 0,1, or 'all'")
        return sorted({int(x) for x in s.split(",")})
    if q.kind == "windows":
        ws = parse_windows(s)
        for w in ws:
            if not WINDOW_RE.fullmatch(w): raise ValueError(f"{w!r}: HH:MM-HH:MM, 24 h")
        return ws
    if q.key == "server":
        x = check_server(s)
        if x: raise ValueError(x)
    if q.key == "name":
        x = check_name(s)
        if x: raise ValueError(x)
    return s


def follow_up(cfg, q, old, hw):
    """values that follow another answer unless the user set them (vendor -> cards -> vram, ram -> mem limit)"""
    if q.key == "vendor" and cfg["vendor"] != old:
        r = recommend(hw, vendor=cfg["vendor"], state=cfg["state"])
        for k in ("gpus", "gpu_name", "vram_gb", "cache_gb"): cfg[k] = r[k]
    elif q.key == "gpus" and cfg["gpus"] != old:
        r = recommend(hw, vendor=cfg["vendor"], gpus=cfg["gpus"], state=cfg["state"])
        for k in ("gpu_name", "vram_gb"): cfg[k] = r[k]
    elif q.key == "ram_gb" and cfg["ram_gb"] != old:
        tot = int((hw["ram"] or 0) / GIB + 0.5)
        cfg["mem_limit_gb"] = max(cfg["ram_gb"] + 2, min(cfg["ram_gb"] + 8, tot - 1)) if tot else cfg["ram_gb"] + 4


# ---------------------------------------------------------------- plain prompts
def plain_ui(cfg, secret, hw, cfgdir, existing, inp=input, getpw=getpass.getpass, out=print):
    """returns (cfg, secret, secret_state) or None (cancelled)"""
    out("rad-quant-at-home setup " + VERSION + (" -- editing " + cfgdir if existing else " -- new config in " + cfgdir))
    out("")
    for l in hardware_lines(hw): out(l)
    out("\nEnter keeps the value in [brackets]. Ctrl-C quits without writing anything.\n")
    adv = ask_yes(inp, "Show the advanced settings too (allocator cap, container memory, state path, image ...)?", False)
    sstate = "kept" if secret else "none"
    section = None
    for q in QUESTIONS:
        if q.advanced and not adv or not q.when(cfg): continue
        if q.key == "vendor" and len({g["vendor"] for g in hw["gpus"] if gpu_supported(g, hw)[0]}) < 2: continue
        if q.section != section:
            section = q.section; out(f"-- {section}")
        if q.key == "gpus":
            for g in hw["gpus"]:
                if g["vendor"] == cfg["vendor"] and g["index"] is not None:
                    ok, why = gpu_supported(g, hw)
                    out(f"   {g['index']}: {g['name']} {card_gb(g)} GB" + ("" if ok else f"  (not usable: {why})"))
        out(f"   ({q.help})")
        while True:
            if q.kind == "secret":
                s = getpw(f"{q.label} [{fmt(cfg, q, sstate)}]: ")
                if s.strip() == "": break
                if s.strip() == "-": secret, sstate = None, "none"; break
                if INVITE_RE.fullmatch(s.strip()): secret, sstate = s.strip(), "new"; break
                out("  invite: printable characters, no spaces"); continue
            s = inp(f"{q.label} [{fmt(cfg, q)}]: ")
            if s.strip() == "" and not (q.key == "server" and not cfg["server"]): break
            if q.key == "image" and s.strip() == "-": cfg["image"] = ""; break
            try:
                old = cfg.get(q.key); cfg[q.key] = parse_answer(cfg, q, s, hw); follow_up(cfg, q, old, hw); break
            except ValueError as x:
                out(f"  {q.label}: {x}")
    return cfg, secret, sstate


def ask_yes(inp, prompt, default):
    while True:
        s = inp(f"{prompt} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        if not s: return default
        if s in ("y", "yes"): return True
        if s in ("n", "no"): return False


# ---------------------------------------------------------------- full-screen form (curses)
class Form:
    """the curses screen as plain data: lines() renders it to text (tests, screenshots), the curses loop only paints"""

    def __init__(self, cfg, secret, hw, cfgdir, existing):
        self.cfg, self.secret, self.hw, self.cfgdir, self.existing = cfg, secret, hw, cfgdir, existing
        self.sstate = "kept" if secret else "none"
        self.adv, self.pos, self.top, self.msg, self.mode, self.sum_top = False, 0, 0, "", "form", 0

    def items(self):
        two = len({g["vendor"] for g in self.hw["gpus"] if gpu_supported(g, self.hw)[0]}) > 1
        return [q for q in QUESTIONS if (self.adv or not q.advanced) and q.when(self.cfg) and (q.key != "vendor" or two or self.adv)]

    def lines(self, w, h):
        """[(text, style)] with style in '', 'title', 'sel', 'sec', 'err', 'dim'"""
        L = [(f" rad-quant-at-home setup {VERSION} · {'editing ' if self.existing else 'new config in '}{self.cfgdir}"[:w], "title")]
        if w < 60 or h < 16:
            return L + [(" the terminal is too small (60 x 16): enlarge it, or q and run with --plain"[:w], "err")]
        if self.mode == "summary":
            E, W = validate(self.cfg, self.hw, self.secret)
            body = []
            for l in summary_lines(self.cfg, self.hw, self.sstate, self.cfgdir, E, W):
                ind = " " * (18 if l.startswith("  ") and l[2:3] != " " and len(l) > 18 and l[17] == " " else 9 if l[:1] in "EW" else 4)
                body += textwrap.wrap(l, w - 1, subsequent_indent=ind, drop_whitespace=False) or [""]
            n = h - 3
            self.sum_top = max(0, min(self.sum_top, len(body) - n))
            for l in body[self.sum_top:self.sum_top + n]:
                L.append((l[:w], "err" if l.startswith("ERROR") else "dim" if l.startswith("WARNING") else ""))
            while len(L) < h - 1: L.append(("", ""))
            keys = (" [w] write the files   [b] back to the form   ↑↓ scroll   [q] quit without writing" if not E else
                    " fix the ERRORs first: [b] back to the form   ↑↓ scroll   [q] quit without writing")
            L.append((keys[:w], "title"))
            return L
        hwl = hardware_lines(self.hw)
        for l in hwl[:4 if h < 30 else 6]: L.append((" " + l[:w - 1], "dim"))
        L.append(("", ""))
        rows, sec = [], None
        its = self.items()
        for i, q in enumerate(its):
            if q.section != sec:
                sec = q.section; rows.append((None, f" {sec}"))
            rows.append((i, q))
        avail = h - len(L) - 3
        cur = next(k for k, r in enumerate(rows) if r[0] == min(self.pos, len(its) - 1))
        if cur < self.top: self.top = max(0, cur - 1)
        if cur >= self.top + avail: self.top = cur - avail + 1
        lw = max(len(q.label) for q in QUESTIONS) + 2
        for i, r in rows[self.top:self.top + avail]:
            if i is None: L.append((r[:w], "sec")); continue
            mark = "›" if i == self.pos else " "
            L.append((f" {mark} {r.label:<{lw}} {fmt(self.cfg, r, self.sstate)}"[:w], "sel" if i == self.pos else ""))
        while len(L) < h - 2: L.append(("", ""))
        q = its[min(self.pos, len(its) - 1)]
        L.append(((" " + (self.msg or q.help))[:w], "err" if self.msg else "dim"))
        L.append((" ↑↓ move  Enter edit  Space toggle  [a] " + ("hide" if self.adv else "show") + " advanced  [s] summary & save  [q] quit"
                  [:w], "title"))
        return L


def curses_ui(cfg, secret, hw, cfgdir, existing):
    import curses
    f = Form(cfg, secret, hw, cfgdir, existing)
    res = {}

    def edit_line(scr, prompt, value, hidden):
        h, w = scr.getmaxyx()
        buf = "" if hidden else value
        curses.curs_set(1)
        try:
            while True:
                shown = ("(typing hidden, Enter when done)" if hidden else buf)
                scr.move(h - 2, 0); scr.clrtoeol()
                scr.addnstr(h - 2, 0, f" {prompt}: {shown}", w - 1, curses.A_BOLD)
                if not hidden: scr.move(h - 2, min(w - 2, len(prompt) + 3 + len(buf)))
                ch = scr.get_wch()
                if ch in ("\n", "\r", curses.KEY_ENTER): return buf
                if ch == "\x1b": return None
                if ch in (curses.KEY_BACKSPACE, "\x7f", "\b", "\x08"): buf = buf[:-1]
                elif ch == "\x15": buf = ""                                   # Ctrl-U
                elif isinstance(ch, str) and ch.isprintable() and len(buf) < 512: buf += ch
        finally:
            curses.curs_set(0)

    def paint(scr):
        scr.erase()
        h, w = scr.getmaxyx()
        st = {"": 0, "title": curses.A_REVERSE, "sel": curses.A_REVERSE | curses.A_BOLD, "sec": curses.A_BOLD,
              "err": curses.A_BOLD, "dim": curses.A_DIM}
        for y_, (t, s) in enumerate(f.lines(w, h)[:h]):
            try: scr.addnstr(y_, 0, t.ljust(w - 1) if s in ("title", "sel") else t, w - 1, st[s])
            except curses.error: pass
        scr.refresh()

    def loop(scr):
        curses.curs_set(0); scr.keypad(True)
        try: curses.use_default_colors()
        except curses.error: pass
        while True:
            paint(scr)
            ch = scr.get_wch()
            if f.mode == "summary":
                E, _ = validate(f.cfg, f.hw, f.secret)
                if ch in ("b", "B", "\x1b"): f.mode = "form"
                elif ch in ("w", "W") and not E: res["ok"] = True; return
                elif ch in ("q", "Q"): return
                elif ch in (curses.KEY_DOWN, "j"): f.sum_top += 1
                elif ch in (curses.KEY_UP, "k"): f.sum_top = max(0, f.sum_top - 1)
                elif ch == curses.KEY_NPAGE: f.sum_top += 10
                elif ch == curses.KEY_PPAGE: f.sum_top = max(0, f.sum_top - 10)
                continue
            its = f.items(); q = its[min(f.pos, len(its) - 1)]; f.msg = ""
            if ch in (curses.KEY_DOWN, "j", "\t"): f.pos = min(len(its) - 1, f.pos + 1)
            elif ch in (curses.KEY_UP, "k", curses.KEY_BTAB): f.pos = max(0, f.pos - 1)
            elif ch in ("a", "A"):
                f.adv = not f.adv; f.pos = min(f.items().index(q) if q in f.items() else 0, len(f.items()) - 1)
            elif ch in ("s", "S", curses.KEY_F2): f.mode, f.sum_top = "summary", 0
            elif ch in ("q", "Q", "\x1b"): return
            elif ch == " " and q.kind in ("bool", "choice"):
                old = f.cfg[q.key]
                f.cfg[q.key] = (not old) if q.kind == "bool" else q.choices[(q.choices.index(old) + 1) % len(q.choices)] \
                    if old in q.choices else q.choices[0]
                follow_up(f.cfg, q, old, f.hw)
            elif ch in ("\n", "\r", curses.KEY_ENTER):
                if q.kind == "bool": f.cfg[q.key] = not f.cfg[q.key]; continue
                cur = "" if q.kind == "secret" else fmt(f.cfg, q).replace("(default) ", "") if q.key != "image" else f.cfg["image"]
                if q.kind == "windows" and not f.cfg["windows"]: cur = ""
                s = edit_line(scr, q.label, cur, q.kind == "secret")
                if s is None: continue
                if q.kind == "secret":
                    if s.strip() == "": continue
                    if s.strip() == "-": f.secret, f.sstate = None, "none"
                    elif INVITE_RE.fullmatch(s.strip()): f.secret, f.sstate = s.strip(), "new"
                    else: f.msg = "invite: printable characters, no spaces"
                    continue
                if q.key == "image" and s.strip() in ("", "-"): f.cfg["image"] = ""; continue
                try:
                    old = f.cfg.get(q.key); f.cfg[q.key] = parse_answer(f.cfg, q, s, f.hw); follow_up(f.cfg, q, old, f.hw)
                except ValueError as x:
                    f.msg = f"{q.label}: {x}"

    curses.wrapper(loop)
    return (f.cfg, f.secret, f.sstate) if res.get("ok") else None


def curses_usable(out=sys.stdout):
    if not (sys.stdin.isatty() and out.isatty()): return False
    if os.environ.get("TERM", "dumb") in ("dumb", "unknown", ""): return False
    try:
        import curses
        sz = shutil.get_terminal_size((0, 0))
        return sz.columns >= 80 and sz.lines >= 24
    except ImportError:
        return False


# ================================================================ main
def apply_flags(cfg, a, hw):
    """flags and --answers FILE on top of cfg; returns the keys set"""
    got = {}
    if a.answers:
        try: ans = json.load(open(a.answers))
        except (OSError, ValueError) as x: raise SystemExit(f"rqah setup: --answers {a.answers}: {x}")
        ans = ans.get("settings", ans)
        bad = sorted(set(ans) - set(DEFAULTS))
        if bad: raise SystemExit(f"rqah setup: --answers: unknown keys {bad}")
        got.update(ans)
    for k in DEFAULTS:
        v = getattr(a, k, None)
        if v is not None: got[k] = v
    if "gpus" in got and isinstance(got["gpus"], str):
        q = next(q for q in QUESTIONS if q.key == "gpus")
        cfg_v = dict(cfg); cfg_v["vendor"] = got.get("vendor") or cfg["vendor"]
        got["gpus"] = parse_answer(cfg_v, q, got["gpus"], hw)
    if "windows" in got: got["windows"] = parse_windows(got["windows"])
    if "leaderboard" in got and isinstance(got["leaderboard"], str): got["leaderboard"] = got["leaderboard"] == "on"
    if "auto_update" in got and isinstance(got["auto_update"], str): got["auto_update"] = got["auto_update"] == "on"
    cfg.update(got)
    return set(got)


def read_invite(a):
    if a.invite_file:
        try:
            s = (sys.stdin.read() if a.invite_file == "-" else open(a.invite_file).read()).strip()
        except OSError as x:
            raise SystemExit(f"rqah setup: --invite-file: {x.strerror}")
        return s or None
    return os.environ.get("RQAH_INVITE") or None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rqah setup", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=None, help="where the files go (default $XDG_CONFIG_HOME/rad-quant-at-home)")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--check", action="store_true", help="validate the existing config against this machine; change nothing")
    m.add_argument("--show", action="store_true", help="print what it detects and the defaults it would offer; change nothing")
    m.add_argument("--yes", "-y", action="store_true", help="no questions: flags / --answers / current config / defaults")
    ui = ap.add_mutually_exclusive_group()
    ui.add_argument("--plain", action="store_true", help="plain prompts instead of the full-screen form")
    ui.add_argument("--curses", action="store_true", help="the full-screen form even if the terminal looks small")
    ap.add_argument("--json", action="store_true", help="--check / --show: one JSON object on stdout")
    ap.add_argument("--answers", metavar="FILE", help="JSON answers (keys as in setup.json's settings; a setup.json works)")
    ap.add_argument("--invite-file", metavar="F", help="read the invite from F ('-' = stdin); also $RQAH_INVITE. Never a flag value")
    ap.add_argument("--no-invite", action="store_true", help="drop a stored invite (the worker is registered already); "
                    "by default a stored invite is kept")
    ap.add_argument("--force", action="store_true", help="write even with errors (e.g. preparing a config on another machine)")
    ap.add_argument("--smoke", action="store_true", help="after writing: validate compose.yaml, look for the image, and if it is "
                    "there, run it offline (--network none) to check it sees the GPUs")
    ap.add_argument("--pull", action="store_true", help="after writing: docker pull the image (network, several GB)")
    ap.add_argument("--start", action="store_true", help="after writing: start the worker (it registers with the queue); a "
                    "running worker of that name is drained first (its jobs finish)")
    u = ap.add_argument_group("updating (--update, alias `rqah update`; README \"Updating\")")
    u.add_argument("--update", action="store_true", help="move the worker to the release the queue advertises: pull, drain, "
                   "rewrite the pin, recreate on the same state volume")
    u.add_argument("--to", metavar="X.Y.Z", help="--update to this release instead (also a downgrade)")
    u.add_argument("--release", action="store_true", help="drain by handing the running jobs back at once (default: let them finish)")
    u.add_argument("--drain-timeout", type=float, default=None, metavar="MIN",
                   help="--update / --start: how long the running jobs may take to finish before they are handed back "
                        f"(60; --auto {AUTO_DRAIN_MIN})")
    u.add_argument("--dry-run", action="store_true", help="--update: say what it would do; change nothing")
    u.add_argument("--digest", metavar="sha256:...", help="--update: the digest to check the pulled tag against (default: "
                   "the one the queue advertises, or the release's release.json for --to)")
    u.add_argument("--no-digest-check", action="store_true", help="--update without a digest check (an image you built "
                   "yourself; never with --auto)")
    u.add_argument("--auto", action="store_true", help="--update as the timer runs it: only if setup.json has auto_update on, "
                   f"only a newer release published at least {AUTO_DELAY_H} h ago, digest-checked, no questions")
    s_auto = ap.add_argument_group("automatic updates")
    s_auto.add_argument("--auto-update", dest="auto_update", choices=("on", "off"),
                        help="the hourly update timer (setting auto_update; on asks before installing the timer, --yes installs it)")
    s = ap.add_argument_group("settings (each also a key of --answers)")
    s.add_argument("--server"); s.add_argument("--name"); s.add_argument("--leaderboard", choices=("on", "off"))
    s.add_argument("--vendor", choices=("amd", "nvidia")); s.add_argument("--gpus", help="0,1 or all")
    s.add_argument("--gpu-name", dest="gpu_name")
    for k in NUM: s.add_argument("--" + k.replace("_", "-"), dest=k, type=float)
    s.add_argument("--state", help="'volume' or an absolute host path for the state (token + cache)")
    s.add_argument("--window", dest="windows", help="22:00-08:00[,HH:MM-HH:MM]; 'always' for none")
    s.add_argument("--budget-reset-day", dest="budget_reset_day", type=int)
    s.add_argument("--on-stop", dest="on_stop", choices=("finish", "abandon"))
    s.add_argument("--container"); s.add_argument("--image")
    ap.add_argument("--fake-root", help=argparse.SUPPRESS); ap.add_argument("--fake-cmds", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    for k in NUM:
        v = getattr(a, k)
        if v is not None and v == int(v): setattr(a, k, int(v))

    host = fake_host_from_file(a.fake_root or "/", a.fake_cmds) if a.fake_cmds else Host(a.fake_root or "/")
    hw = detect(host)
    cfgdir = os.path.abspath(os.path.expanduser(a.config_dir or default_dir()))

    if a.check:
        E, W, I = check_dir(cfgdir, hw)
        if E is None:
            print(json.dumps({"ok": False, "error": I[0]}) if a.json else f"rqah setup --check: {I[0]}")
            return 2
        if a.json:
            print(json.dumps({"ok": not E, "errors": E, "warnings": W, "info": I, "config_dir": cfgdir})); return 1 if E else 0
        print(f"checking {cfgdir}")
        for l in hardware_lines(hw): print("  " + l)
        for x in I: print(f"info     {x}")
        for x in W: print(f"WARNING  {x}")
        for x in E: print(f"ERROR    {x}")
        print("OK: the config fits this machine" + (f" ({len(W)} warnings)" if W else "") if not E else f"{len(E)} problem(s)")
        return 1 if E else 0

    cfg, secret = load(cfgdir)
    existing = cfg is not None
    if a.auto: a.update = True
    if a.update:
        if cfg is None:
            print(f"rqah update: no configuration in {cfgdir}: it updates a worker set up with `rqah setup` (else README "
                  "\"Updating\" has the docker commands)"); return 2
        if a.auto:                                   # one at a time (the timer and a person may overlap)
            import fcntl
            lk = open(os.path.join(cfgdir, ".update.lock"), "a")
            try: fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: print("another update is running"); return 0
        return update(host, hw, cfg, cfgdir, secret, a)
    rec = recommend(hw)
    if cfg is None:
        cfg = dict(DEFAULTS); cfg.update(rec)
    if a.show:
        if a.json:
            hwj = {k: v for k, v in hw.items() if not k.startswith("_")}
            print(json.dumps({"ok": True, "hardware": hwj, "defaults": rec, "config": cfg if existing else None})); return 0
        for l in hardware_lines(hw): print(l)
        print("\ndefaults for this machine: " + ", ".join(f"{k}={v}" for k, v in rec.items()))
        for t, ok in jobs_for(rec["vendor"], rec["vram_gb"], rec["ram_gb"], len(rec["gpus"])): print(f"  {'+' if ok else '-'} {t}")
        return 0

    before = dict(cfg)
    got = apply_flags(cfg, a, hw)
    # a new vendor / card choice from flags brings its own defaults for what wasn't given
    if "vendor" in got and cfg["vendor"] != before["vendor"] or "gpus" in got and cfg["gpus"] != before["gpus"]:
        r = recommend(hw, vendor=cfg["vendor"], gpus=cfg["gpus"] if "gpus" in got else None, state=cfg["state"])
        for k in ("gpus", "gpu_name", "vram_gb"):
            if k not in got: cfg[k] = r[k]
    if "ram_gb" in got and "mem_limit_gb" not in got: follow_up(cfg, next(q for q in QUESTIONS if q.key == "ram_gb"), None, hw)
    new_inv = read_invite(a)
    sstate = "new" if new_inv else ("kept" if secret else "none")
    if new_inv: secret = new_inv
    if a.no_invite: secret, sstate = None, "none"

    if a.yes:
        E, W = validate(cfg, hw, secret)
        for l in summary_lines(cfg, hw, sstate, cfgdir, E, W): print(l)
        print("")
    else:
        if not sys.stdin.isatty():
            print("rqah setup: stdin is not a terminal: use --yes with flags or --answers FILE (see --help)", file=sys.stderr)
            return 2
        try:
            if (a.curses or curses_usable()) and not a.plain:
                res = curses_ui(cfg, secret, hw, cfgdir, existing)       # its summary screen is the confirmation
                if res is None: print("cancelled: nothing written"); return 1
                cfg, secret, sstate = res
                E, W = validate(cfg, hw, secret)
                for x in W: print(f"WARNING  {x}")
            else:
                cfg, secret, sstate = plain_ui(cfg, secret, hw, cfgdir, existing)
                E, W = validate(cfg, hw, secret)
                print("")
                for l in summary_lines(cfg, hw, sstate, cfgdir, E, W): print(l)
                print("")
                if E and not a.force: print("not written: fix the errors above (run setup again)"); return 1
                if not ask_yes(input, "Write these files?", True): print("cancelled: nothing written"); return 1
        except (KeyboardInterrupt, EOFError):
            print("\ncancelled: nothing written"); return 1
    if E and not a.force:
        print(f"not written: {len(E)} error(s) (--force writes anyway)", file=sys.stderr); return 1
    files = write_all(cfgdir, cfg, hw, secret)
    print("written: " + ", ".join(os.path.basename(f) for f in files) + f" in {cfgdir}")
    if cfg.get("auto_update"):
        for n, text in render_units(cfgdir).items(): write_atomic(os.path.join(cfgdir, n), text, 0o644)
        if a.yes or ask_yes(input, f"Install the hourly update timer now (systemctl --user, links {UNIT}.timer)?", True):
            install_timer(host, cfgdir, True)
        else: print(f"timer not installed: systemctl --user link {cfgdir}/{UNIT}.service && systemctl --user enable --now {cfgdir}/{UNIT}.timer")
    elif before.get("auto_update") and os.path.exists(os.path.join(cfgdir, f"{UNIT}.timer")):
        install_timer(host, cfgdir, False); print("auto-update off: the timer is disabled")
    if not secret: print("note: no invite stored: the worker registers without one and starts on probation (quantization jobs "
                         "only until a few results pass), unless its state volume has a token already")
    interactive = not a.yes
    do_smoke = a.smoke or (interactive and ask_yes(input, "Run the local checks now (no network: compose file, image, GPU access in the image)?", True))
    do_pull = a.pull or (interactive and hw["docker"]["access"] and ask_yes(input, f"Pull the image now ({image_for(cfg)}, several GB download)?", False))
    if do_smoke or do_pull: smoke(host, hw, cfg, cfgdir, pull=do_pull, gpu_test=do_smoke)
    if a.start or (interactive and hw["docker"]["access"] and ask_yes(input, "Start the worker now (it registers with the queue)?", False)):
        start(host, hw, cfg, cfgdir, drain_s=(a.drain_timeout or 60) * 60, release=a.release)
    else:
        how = f"cd {shlex.quote(cfgdir)} && docker compose up -d" if hw["docker"]["compose"] else f"sh {shlex.quote(os.path.join(cfgdir, 'docker-run.sh'))}"
        print(f"start it with:  {how}\nthen:           docker exec {cfg['container']} rqah status")
        if existing: print(f"(already running? recreate it so the new settings apply: {how}"
                           + ("" if hw["docker"]["compose"] else f" after docker stop {cfg['container']} && docker rm {cfg['container']}") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
