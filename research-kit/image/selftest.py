#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# selftest.py -- run INSIDE the research container (run.sh selftest does it for you). Proves what the volunteer was
# promised and prints a PASS / FAIL table they can paste back to us:
#   sandbox   not root, no capabilities, no-new-privileges, seccomp on, no docker socket, no disks, limits set
#   blocked   the host (bridge gateway and the host addresses given with --host-ip), the LAN router (--lan-gw), common
#             home-router addresses, cloud metadata (169.254.169.254), CGNAT / Tailscale ranges, a sample of RFC1918,
#             IPv6 ULA / link-local: none may answer
#   internet  DNS through the public resolver, HTTPS to Hugging Face, PyPI, GitHub; Tailscale logged in
#   gpu       rocminfo / nvidia-smi, then torch: a bf16 matmul checked against fp32
# How "blocked" is judged: a TCP connect that succeeds or is refused (RST), or a DNS answer / ICMP port-unreachable to a
# UDP query, means the address answered: FAIL. Rejected by the firewall (ICMP admin-prohibited -> EHOSTUNREACH), no
# route, or no answer within the timeout: PASS. Any address that answers fails the whole test. (The kernel sends about
# one ICMP error per second to a given address, so with many parallel probes most rejected ones show as "no answer".)
# Options: --host-ip IP (repeatable; run.sh passes the host's own addresses), --lan-gw IP (repeatable; the host's default
# gateways), --no-gpu, --speed (time a 100 MB Hugging Face download), --show-ips (print the host addresses; by default
# they are masked so the table is safe to paste), --json, --guard (only the quick host check the entrypoint runs before
# going online: exit 1 if the gateway answers). Exit 0 = PASS (warnings allowed), 1 = FAIL.
import argparse
import errno
import json
import os
import random
import socket
import stat
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

K = os.path.dirname(os.path.abspath(__file__))
HOST_PORTS = [22, 53, 80, 111, 139, 443, 445, 631, 2049, 2375, 3000, 5000, 8000, 8080, 8443, 9090]
LAN_PORTS = [53, 80, 443, 22]
SAMPLE_PORTS = [80, 443]
ROUTERS = ["192.168.0.1", "192.168.1.1", "192.168.1.254", "192.168.0.254", "192.168.2.1", "192.168.10.1",
           "192.168.50.1", "192.168.86.1", "192.168.88.1", "192.168.100.1", "192.168.178.1", "10.0.0.1",
           "10.0.0.138", "10.0.1.1", "10.1.1.1", "172.16.0.1"]
BLOCKED_ERRNO = {errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EACCES, errno.EPERM, errno.ETIMEDOUT,
                 errno.EADDRNOTAVAIL, errno.ENETDOWN, errno.EHOSTDOWN}


def tcp(ip, port, timeout):
    """'open' | 'refused' (both: something answered) | 'rejected' | 'timeout' | 'noroute'"""
    try:
        fam, _, _, _, sa = socket.getaddrinfo(ip, port, 0, socket.SOCK_STREAM)[0]   # handles fe80::1%eth0
    except OSError:
        return "noroute"
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t = time.monotonic()
    try:
        s.connect(sa)
        return "open"
    except ConnectionRefusedError:
        return "refused"
    except socket.timeout:
        return "timeout"
    except OSError as e:
        if e.errno in (errno.ENETUNREACH, errno.EADDRNOTAVAIL, errno.ENETDOWN):
            return "noroute"
        if e.errno in BLOCKED_ERRNO:
            return "rejected" if time.monotonic() - t < 0.5 else "timeout"
        return f"error:{errno.errorcode.get(e.errno, e.errno)}"
    finally:
        s.close()


def dns_query(name="huggingface.co"):
    q = struct.pack(">HHHHHH", random.randrange(65536), 0x0100, 1, 0, 0, 0)
    for part in name.split("."):
        q += bytes([len(part)]) + part.encode()
    return q + b"\x00" + struct.pack(">HH", 1, 1)


def udp_dns(ip, timeout):
    """'answered' | 'refused' (ICMP port unreachable: the host is there) | 'rejected' | 'timeout' | 'noroute'"""
    try:
        fam, _, _, _, sa = socket.getaddrinfo(ip, 53, 0, socket.SOCK_DGRAM)[0]
    except OSError:
        return "noroute"
    s = socket.socket(fam, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.connect(sa)
        s.send(dns_query())
        s.recv(512)
        return "answered"
    except ConnectionRefusedError:
        return "refused"
    except socket.timeout:
        return "timeout"
    except OSError as e:
        if e.errno in (errno.ENETUNREACH, errno.EADDRNOTAVAIL, errno.ENETDOWN):
            return "noroute"
        return "rejected" if e.errno in BLOCKED_ERRNO else f"error:{errno.errorcode.get(e.errno, e.errno)}"
    finally:
        s.close()


REACHED = {"open", "refused", "answered"}


def default_gateway():
    with open("/proc/net/route") as f:
        for line in f.readlines()[1:]:
            p = line.split()
            if p[1] == "00000000" and int(p[3], 16) & 2:
                return socket.inet_ntoa(struct.pack("<L", int(p[2], 16)))
    return None


def has_ipv6_route():
    try:
        with open("/proc/net/ipv6_route") as f:
            for line in f:
                p = line.split()
                # any route through a non-loopback device that is not the multicast / local table
                if p[9] != "lo" and not p[0].startswith("ff"):
                    return True
    except OSError:
        pass
    return False


class Report:
    def __init__(self):
        self.rows = []

    def add(self, section, name, status, detail=""):
        self.rows.append({"section": section, "check": name, "status": status, "detail": detail})

    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def probe_targets(rep, section, targets, timeout):
    """targets: [(label, ip, [ports], udp_dns?)]; one row per target"""
    jobs = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for label, ip, ports, udp in targets:
            fs = [(f"tcp/{p}", ex.submit(tcp, ip, p, timeout)) for p in ports]
            if udp:
                fs.append(("udp/53", ex.submit(udp_dns, ip, timeout)))
            jobs.append((label, ip, fs))
        for label, ip, fs in jobs:
            res = {k: f.result() for k, f in fs}
            hit = sorted(k for k, v in res.items() if v in REACHED)
            if hit:
                rep.add(section, label, "FAIL", "ANSWERED on " + ", ".join(f"{k} ({res[k]})" for k in hit))
            else:
                kinds = sorted(set(res.values()))
                how = {"rejected": "rejected by the firewall", "timeout": "no answer", "noroute": "no route"}
                rep.add(section, label, "PASS", "blocked: " + ", ".join(how.get(k, k) for k in kinds)
                        + f" ({len(res)} probes)")


def sandbox_checks(rep):
    st = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            st[k] = v.strip()
    rep.add("sandbox", "not root", "PASS" if os.getuid() != 0 else "FAIL", f"uid {os.getuid()}")
    cap = int(st.get("CapEff", "1"), 16)
    rep.add("sandbox", "no capabilities", "PASS" if cap == 0 else "FAIL", f"CapEff {st.get('CapEff')}, CapBnd "
            f"{st.get('CapBnd')}")
    rep.add("sandbox", "no-new-privileges", "PASS" if st.get("NoNewPrivs") == "1" else "FAIL",
            f"NoNewPrivs {st.get('NoNewPrivs')}")
    rep.add("sandbox", "seccomp filter", "PASS" if st.get("Seccomp") == "2" else "FAIL",
            f"Seccomp mode {st.get('Seccomp')} (2 = filter)")
    socks = [p for p in ("/var/run/docker.sock", "/run/docker.sock", "/run/containerd/containerd.sock",
                         "/run/podman/podman.sock") if os.path.exists(p)]
    rep.add("sandbox", "no container-engine socket", "FAIL" if socks else "PASS", ", ".join(socks) or "none")
    blk = []
    for root, dirs, files in os.walk("/dev"):
        for n in files + dirs:
            p = os.path.join(root, n)
            try:
                if stat.S_ISBLK(os.lstat(p).st_mode):
                    blk.append(p)
            except OSError:
                pass
    rep.add("sandbox", "no block devices (host disks)", "FAIL" if blk else "PASS", ", ".join(blk[:5]) or "none")
    gpu = sorted(n for n in os.listdir("/dev") if n in ("kfd", "dri") or n.startswith("nvidia"))
    rep.add("sandbox", "devices", "INFO", "GPU nodes: " + (" ".join(gpu) or "none"))

    def cg(name):
        try:
            return open(f"/sys/fs/cgroup/{name}").read().strip()
        except OSError:
            return "?"
    mem, cpu, pids = cg("memory.max"), cg("cpu.max"), cg("pids.max")
    swap = cg("memory.swap.max")
    lim_ok = mem not in ("max", "?") and pids not in ("max", "?")
    memg = f"{int(mem) / 2**30:.1f} GiB" if mem.isdigit() else mem
    cpus = cpu.split()
    cpus = f"{int(cpus[0]) / int(cpus[1]):g} CPUs" if len(cpus) == 2 and cpus[0].isdigit() else "no CPU cap"
    rep.add("sandbox", "resource limits", "PASS" if lim_ok else "WARN",
            f"memory {memg}, swap {swap}, {cpus}, pids {pids}")
    try:
        with open("/opt/donor-kit/.rw-test", "w"):
            pass
        os.unlink("/opt/donor-kit/.rw-test")
        rep.add("sandbox", "read-only image", "WARN", "the image is writable (run without --read-only)")
    except OSError as e:
        rep.add("sandbox", "read-only image", "PASS" if e.errno in (errno.EROFS, errno.EACCES) else "WARN",
                os.strerror(e.errno))


def blocked_checks(rep, args):
    gw = default_gateway()
    t = args.timeout
    tg = []
    if gw:
        tg.append((f"host via the bridge gateway ({gw})", gw, HOST_PORTS, True))
    for i, ip in enumerate(args.host_ip, 1):
        tg.append((f"host address {i}" + (f" ({ip})" if args.show_ips else ""), ip, HOST_PORTS, True))
    for i, ip in enumerate(args.lan_gw, 1):
        tg.append((f"LAN gateway {i}" + (f" ({ip})" if args.show_ips else ""), ip, LAN_PORTS, True))
    for ip in ROUTERS:
        if ip not in args.lan_gw:
            tg.append((f"common router address {ip}", ip, [80, 443, 53], False))
    tg.append(("cloud metadata 169.254.169.254", "169.254.169.254", [80, 443], False))
    tg.append(("link-local 169.254.1.1", "169.254.1.1", SAMPLE_PORTS, False))
    tg.append(("CGNAT / tailnet 100.64.0.1", "100.64.0.1", SAMPLE_PORTS, False))
    tg.append(("Tailscale quad100 100.100.100.100", "100.100.100.100", [53, 80], True))
    rnd = random.Random()
    sample = ["10.255.255.254", "172.16.0.2", "172.31.255.254", "192.168.255.254", "198.18.0.1"]
    sample += [f"10.{rnd.randrange(256)}.{rnd.randrange(256)}.{rnd.randrange(1, 255)}",
               f"172.{rnd.randrange(16, 32)}.{rnd.randrange(256)}.{rnd.randrange(1, 255)}",
               f"192.168.{rnd.randrange(256)}.{rnd.randrange(1, 255)}"]
    for ip in sample:
        tg.append((f"private sample {ip}", ip, SAMPLE_PORTS, False))
    probe_targets(rep, "blocked", tg, t)
    if has_ipv6_route():
        probe_targets(rep, "blocked", [("IPv6 ULA fd00::1", "fd00::1", SAMPLE_PORTS, False),
                                       ("IPv6 ULA fdff:ffff::1", "fdff:ffff::1", SAMPLE_PORTS, False)], t)
    else:
        rep.add("blocked", "IPv6 (ULA, link-local, the LAN's prefix)", "PASS", "the container has no IPv6 route")


def http(url, timeout=15, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "donor-kit-selftest", **(headers or {})})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return r.status, body, time.monotonic() - t


def internet_checks(rep, args):
    try:
        with open("/etc/resolv.conf") as f:
            ns = [ln.split()[1] for ln in f if ln.startswith("nameserver")]
        rep.add("internet", "resolver", "INFO", " ".join(ns) + " (Docker's embedded DNS forwards to the public "
                "resolvers the container was started with)")
    except OSError:
        pass
    try:
        a = sorted({x[4][0] for x in socket.getaddrinfo("huggingface.co", 443, socket.AF_INET)})
        rep.add("internet", "DNS (huggingface.co)", "PASS", ", ".join(a[:3]))
    except OSError as e:
        rep.add("internet", "DNS (huggingface.co)", "FAIL", str(e))
    r = tcp("1.1.1.1", 443, 5)
    rep.add("internet", "TCP to the internet (1.1.1.1:443)", "PASS" if r == "open" else "FAIL", r)
    for name, url in [("Hugging Face API", "https://huggingface.co/api/models?limit=1"),
                      ("Hugging Face file download", "https://huggingface.co/openai-community/gpt2/resolve/main/config.json"),
                      ("PyPI", "https://pypi.org/simple/pip/"),
                      ("GitHub", "https://github.com/")]:
        try:
            code, body, dt = http(url)
            rep.add("internet", name, "PASS" if code == 200 else "FAIL", f"HTTP {code}, {len(body)} B, {dt:.2f} s")
        except Exception as e:  # noqa: BLE001 -- any failure is the answer
            rep.add("internet", name, "FAIL", f"{type(e).__name__}: {e}"[:120])
    if args.speed:
        url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
        try:
            code, body, dt = http(url, 120, {"Range": "bytes=0-104857599"})
            rep.add("internet", "download speed (100 MB from HF)", "INFO", f"{len(body) / dt / 1e6:.0f} MB/s")
        except Exception as e:  # noqa: BLE001
            rep.add("internet", "download speed", "WARN", str(e)[:120])
    try:
        out = subprocess.run(["tailscale", "--socket=/run/tailscale/tailscaled.sock", "status", "--json"],
                             capture_output=True, text=True, timeout=10).stdout
        st = json.loads(out)
        me = st.get("Self") or {}
        tags = ",".join(me.get("Tags") or []) or "untagged"
        state = st.get("BackendState")
        rep.add("internet", "Tailscale node", "PASS" if state == "Running" else "WARN",
                f"{state}, {me.get('HostName', '?')}, {tags}" + ("" if state == "Running" else
                                                                 " (no auth key yet? see VOLUNTEER.md)"))
    except Exception as e:  # noqa: BLE001
        rep.add("internet", "Tailscale node", "WARN", f"tailscaled not answering: {e}"[:120])


def gpu_checks(rep):
    try:
        img = open("/opt/donor-kit/IMAGE").read().split()
    except OSError:
        img = []
    if os.path.exists("/dev/kfd"):
        try:
            out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60).stdout
            names = [ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("Name:")
                     and "gfx" in ln and "amdgcn" not in ln]
            rep.add("gpu", "rocminfo", "PASS" if names else "FAIL", " ".join(names) or "no GPU agent")
        except Exception as e:  # noqa: BLE001
            rep.add("gpu", "rocminfo", "FAIL", str(e)[:120])
    elif os.path.exists("/dev/nvidiactl") or "cuda" in img:
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60)
            rep.add("gpu", "nvidia-smi", "PASS" if p.returncode == 0 else "FAIL",
                    (p.stdout.strip().replace("\n", "; ") or p.stderr.strip())[:150])
        except Exception as e:  # noqa: BLE001
            rep.add("gpu", "nvidia-smi", "FAIL", str(e)[:120])
    else:
        rep.add("gpu", "GPU device nodes", "FAIL", "neither /dev/kfd nor /dev/nvidia*: started without the GPU flags?")
    py = "/opt/venv/bin/python" if os.path.exists("/opt/venv/bin/python") else sys.executable
    try:
        p = subprocess.run([py, "-I", os.path.join(K, "gpucheck.py")], capture_output=True, text=True, timeout=600)
        res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"ok": False, "detail":
                                                                                         p.stderr.strip()[-200:]}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}
    rep.add("gpu", "torch matmul", "PASS" if res.get("ok") else "FAIL", res.get("detail", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-ip", action="append", default=[])
    ap.add_argument("--lan-gw", action="append", default=[])
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--show-ips", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--guard", action="store_true")
    args = ap.parse_args()

    if args.guard:
        gw = default_gateway()
        if not gw:
            return 0
        res = [tcp(gw, p, 1.5) for p in (22, 53, 80, 443, 445, 8080)] + [udp_dns(gw, 1.5)]
        return 1 if any(r in REACHED for r in res) else 0

    rep = Report()
    t0 = time.time()
    sandbox_checks(rep)
    blocked_checks(rep, args)
    internet_checks(rep, args)
    if not args.no_gpu:
        gpu_checks(rep)
    bad = rep.failed()
    warn = [r for r in rep.rows if r["status"] == "WARN"]
    try:
        image = open("/opt/donor-kit/IMAGE").read().strip()
    except OSError:
        image = "unknown image"
    if args.json:
        print(json.dumps({"image": image, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                          "pass": not bad, "rows": rep.rows}, indent=1))
        return 1 if bad else 0
    print(f"donor-kit selftest  {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(t0))}  {image}")
    w = max(len(r["check"]) for r in rep.rows)
    sec = None
    for r in rep.rows:
        if r["section"] != sec:
            sec = r["section"]
            print(f"\n[{sec}]")
        print(f"  {r['status']:<4}  {r['check']:<{w}}  {r['detail']}")
    n = sum(r["status"] == "PASS" for r in rep.rows)
    print(f"\nSELFTEST {'FAIL' if bad else 'PASS'}: {n} passed, {len(bad)} failed, {len(warn)} warnings "
          f"({time.time() - t0:.0f} s)")
    for r in bad:
        print(f"  FAILED: [{r['section']}] {r['check']}: {r['detail']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
