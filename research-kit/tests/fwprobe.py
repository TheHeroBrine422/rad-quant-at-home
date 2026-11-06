#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# fwprobe.py EXPECT -- run in the fake container namespace of netns-firewall.sh: probe every target with the
# selftest's own probe functions and compare with EXPECT ("fw" = firewall installed, "open" = no firewall).
# Prints one line per probe and "FWPROBE PASS|FAIL n/m". The targets live in the fake LAN / host / internet namespaces.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image"))
from selftest import REACHED, tcp, udp_dns  # noqa: E402

# (label, kind, ip, port, reachable_with_firewall)
T = [
    ("host via bridge gateway", "tcp", "172.31.254.1", 22, False),
    ("host DNS on the gateway", "udp", "172.31.254.1", 53, False),
    ("host LAN address", "tcp", "192.168.77.2", 22, False),
    ("host public address", "tcp", "203.0.113.5", 22, False),
    ("LAN router web UI", "tcp", "192.168.77.1", 80, False),
    ("LAN router DNS", "udp", "192.168.77.1", 53, False),
    ("LAN NAS (SMB)", "tcp", "192.168.77.10", 445, False),
    ("other private 10/8", "tcp", "10.20.0.5", 80, False),
    ("other private 172.16/12", "tcp", "172.20.0.5", 80, False),
    ("cloud metadata", "tcp", "169.254.169.254", 80, False),
    ("CGNAT / tailnet", "tcp", "100.64.1.1", 80, False),
    ("benchmark 198.18/15", "tcp", "198.18.0.5", 80, False),
    ("neighbour on the host's public subnet", "tcp", "203.0.113.9", 80, False),
    ("IPv6 ULA on the LAN", "tcp", "fd77::1", 80, False),
    ("IPv6 host LAN address", "tcp", "fd77::2", 22, False),
    ("IPv6 LAN device, public prefix", "tcp", "2001:db8:77::9", 80, False),
    ("IPv6 host public address", "tcp", "2001:db8:77::2", 22, False),
    ("IPv6 link-local gateway", "tcp", "fe80::1%c0", 22, False),
    ("internet 1.1.1.1:443", "tcp", "1.1.1.1", 443, True),
    ("internet DNS 9.9.9.9", "udp", "9.9.9.9", 53, True),
    ("internet web 93.184.215.14:80", "tcp", "93.184.215.14", 80, True),
    ("internet IPv6 2001:db8::80", "tcp", "2001:db8::80", 80, True),
]


def main():
    expect = sys.argv[1]
    bad = 0
    for label, kind, ip, port, fw_ok in T:
        r = tcp(ip, port, 1.5) if kind == "tcp" else udp_dns(ip, 1.5)
        reached = r in REACHED
        want = fw_ok if expect == "fw" else True
        ok = reached == want
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'}  {label:<38} {kind}/{port:<4} {ip:<16} {r:<9} "
              f"(want {'reachable' if want else 'blocked'})")
    print(f"FWPROBE {'PASS' if not bad else 'FAIL'} {len(T) - bad}/{len(T)} ({expect})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
