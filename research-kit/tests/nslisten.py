#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# nslisten.py ADDR:PORT[/udp] ... -- tiny listeners for netns-firewall.sh: TCP accepts and closes, UDP answers any
# datagram (a stand-in for a DNS server). Runs until killed.
import socket
import sys
import threading


def tcp(addr, port):
    fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((addr, port))
    s.listen(64)
    while True:
        c, _ = s.accept()
        c.close()


def udp(addr, port):
    fam = socket.AF_INET6 if ":" in addr else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_DGRAM)
    s.bind((addr, port))
    while True:
        data, peer = s.recvfrom(2048)
        s.sendto(data[:2] + b"\x81\x80" + data[4:], peer)


for spec in sys.argv[1:]:
    proto = "udp" if spec.endswith("/udp") else "tcp"
    hp = spec.removesuffix("/udp")
    addr, port = hp.rsplit(":", 1)
    addr = addr.strip("[]")
    threading.Thread(target=udp if proto == "udp" else tcp, args=(addr, int(port)), daemon=True).start()
threading.Event().wait()
