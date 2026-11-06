#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# netns-firewall.sh [nft] [iptables] -- test firewall/donor-firewall.sh without a real donor host (as root, any Linux
# box; it touches only three network namespaces it creates and deletes). Topology:
#
#   ns C "container"  c0 172.31.254.2 / fd31::2 --- bridge donor0 172.31.254.1 / fd31::1 / fe80::1  in ns H
#   ns H "host"       hup 192.168.77.2 / fd77::2 (LAN uplink), hpub 203.0.113.5/24 (a public address), forwarding on,
#                     listeners on :22 tcp and :53 udp, and Docker-like iptables rules that ACCEPT everything to and
#                     from donor0 (so only our rules can block)
#   ns R "LAN + internet"  router 192.168.77.1, NAS 192.168.77.10, fd77::1, a LAN device with a public IPv6 address
#                     2001:db8:77::9 (the host has 2001:db8:77::2/64), a neighbour 203.0.113.9 on the host's
#                     public subnet; stand-ins for 10/8, 172.16/12, 169.254.169.254, 100.64/10, 198.18/15 and for the
#                     internet (1.1.1.1, 9.9.9.9, 93.184.215.14, 2001:db8::80); listeners on :80 :443 :445 tcp, :53 udp
#
# Per backend: without rules everything must answer (the probe can see a hole), the entrypoint's guard must refuse;
# installed: LAN / host / special ranges blocked (selftest's own probes and its full [blocked] section), internet
# open, nothing opens a connection into the container, the guard passes; installed twice: the same rules; a simulated
# Docker restart (FORWARD rebuilt with DOCKER-USER first): still blocked; switching backend leaves one rule set;
# uninstalled: everything answers again and nothing of ours is left. Output: one line per check, "NETNS-FIREWALL PASS".
set -u
T=$(cd "$(dirname "$0")" && pwd); K=$(cd "$T/.." && pwd); FW=$K/firewall/donor-firewall.sh
P=${NS_PREFIX:-dk}; H=${P}h C=${P}c R=${P}r
BACKENDS=("$@"); [ ${#BACKENDS[@]} -gt 0 ] || BACKENDS=(nft iptables)
FAILS=0
check() { local name=$1; shift; if "$@" > /tmp/$P-check.out 2>&1; then echo "  ok    $name"
  else echo "  FAIL  $name"; sed 's/^/        /' /tmp/$P-check.out | tail -25; FAILS=$((FAILS + 1)); fi; }
not() { ! "$@"; }
inH() { ip netns exec "$H" "$@"; }
inC() { ip netns exec "$C" "$@"; }
inR() { ip netns exec "$R" "$@"; }

cleanup() {
  for n in $H $C $R; do
    for p in $(ip netns pids "$n" 2> /dev/null); do kill "$p" 2> /dev/null; done
    ip netns del "$n" 2> /dev/null
  done
}
docker_sim() {   # what Docker leaves in FORWARD for a bridge: DOCKER-USER first, then accepts
  for t in iptables ip6tables; do
    $t -w -N DOCKER-USER 2> /dev/null || true
    $t -w -D FORWARD -j DOCKER-USER 2> /dev/null || true
    $t -w -I FORWARD 1 -j DOCKER-USER
    $t -w -C FORWARD -i donor0 -j ACCEPT 2> /dev/null || $t -w -A FORWARD -i donor0 -j ACCEPT
    $t -w -C FORWARD -o donor0 -j ACCEPT 2> /dev/null || $t -w -A FORWARD -o donor0 -j ACCEPT
  done
}
setup() {
  cleanup
  ip netns add "$H" && ip netns add "$C" && ip netns add "$R" || return 1
  for n in $H $C $R; do ip -n "$n" link set lo up; done
  ip link add c0 netns "$C" type veth peer name hc0 netns "$H"
  ip -n "$H" link add donor0 type bridge
  ip -n "$H" link set hc0 master donor0; ip -n "$H" link set hc0 up; ip -n "$H" link set donor0 up
  ip -n "$H" addr add 172.31.254.1/24 dev donor0
  ip -n "$H" addr add fd31::1/64 dev donor0 nodad; ip -n "$H" addr add fe80::1/64 dev donor0 nodad
  ip -n "$C" link set c0 up; ip -n "$C" addr add 172.31.254.2/24 dev c0; ip -n "$C" addr add fd31::2/64 dev c0 nodad
  ip -n "$C" route add default via 172.31.254.1; ip -n "$C" -6 route add default via fd31::1

  ip link add hup netns "$H" type veth peer name rlan netns "$R"
  ip -n "$H" addr add 192.168.77.2/24 dev hup; ip -n "$H" addr add fd77::2/64 dev hup nodad; ip -n "$H" link set hup up
  ip -n "$H" addr add 2001:db8:77::2/64 dev hup nodad
  ip -n "$R" addr add 192.168.77.1/24 dev rlan; ip -n "$R" addr add 192.168.77.10/24 dev rlan
  ip -n "$R" addr add fd77::1/64 dev rlan nodad; ip -n "$R" addr add 2001:db8:77::9/64 dev rlan nodad
  ip -n "$R" link set rlan up
  ip -n "$H" route add default via 192.168.77.1; ip -n "$H" -6 route add default via fd77::1
  ip link add hpub netns "$H" type veth peer name rpub netns "$R"
  ip -n "$H" addr add 203.0.113.5/24 dev hpub; ip -n "$H" link set hpub up
  ip -n "$R" addr add 203.0.113.9/24 dev rpub; ip -n "$R" link set rpub up
  ip -n "$R" link add inet0 type dummy; ip -n "$R" link set inet0 up
  for a in 1.1.1.1 9.9.9.9 93.184.215.14 10.20.0.5 172.20.0.5 169.254.169.254 100.64.1.1 198.18.0.5; do
    ip -n "$R" addr add $a/32 dev inet0; done
  ip -n "$R" addr add 2001:db8::80/128 dev inet0 nodad
  ip -n "$R" route add 172.31.254.0/24 via 192.168.77.2; ip -n "$R" -6 route add fd31::/64 via fd77::2
  inH sysctl -qw net.ipv4.ip_forward=1 net.ipv6.conf.all.forwarding=1
  inH python3 -I "$T/nslisten.py" "[::]:22" 172.31.254.1:53/udp > /dev/null 2>&1 &
  inR python3 -I "$T/nslisten.py" "[::]:80" "[::]:443" "[::]:445" 192.168.77.1:53/udp 9.9.9.9:53/udp > /dev/null 2>&1 &
  inC python3 -I "$T/nslisten.py" "0.0.0.0:8000" > /dev/null 2>&1 &
  inH bash -c "$(declare -f docker_sim); docker_sim"
  sleep 1
}
inbound() {   # from the LAN: can anything open a connection into the container?
  inR python3 -I -c "import sys; sys.path.insert(0, '$K/image'); from selftest import tcp, REACHED
r = tcp('172.31.254.2', 8000, 1.5); print('inbound LAN -> container:8000', r); sys.exit(0 if r in REACHED else 1)"
}
guard() { inC python3 -I "$K/image/selftest.py" --guard; }
blocked_section() {   # the full selftest, [blocked] rows only: all PASS
  inC python3 -I "$K/image/selftest.py" --no-gpu --timeout 1.5 --host-ip 192.168.77.2 --host-ip 203.0.113.5 \
    --lan-gw 192.168.77.1 --show-ips > /tmp/$P-selftest.out 2>&1
  awk '/^\[blocked\]/{f=1; next} /^\[/{f=0} f && NF' /tmp/$P-selftest.out > /tmp/$P-blocked.out
  cat /tmp/$P-blocked.out
  [ -s /tmp/$P-blocked.out ] && ! grep -qv "^  PASS" /tmp/$P-blocked.out
}
snapshot() {
  { inH nft list table inet donorkit 2> /dev/null | sed 's/counter packets [0-9]* bytes [0-9]*/counter/'
    for t in iptables-save ip6tables-save; do inH $t -t filter 2> /dev/null | grep -E "DONOR" | sed 's/\[[0-9:]*\]//'; done; }
}
ours_gone() { ! inH nft list table inet donorkit > /dev/null 2>&1 && ! inH iptables-save | grep -q DONOR && ! inH ip6tables-save | grep -q DONOR; }
install() { inH env DONOR_BRIDGE=donor0 DONOR_SUBNET=172.31.254.0/24 bash "$FW" install --backend "$1" --no-network --no-persist; }

trap cleanup EXIT
echo "netns-firewall.sh: $(uname -r), $(iptables -V), $(nft -v 2> /dev/null)"
for B in "${BACKENDS[@]}"; do
  echo "== backend $B"
  setup || { echo "  FAIL  setup (root? ip netns?)"; FAILS=$((FAILS + 1)); continue; }
  check "no rules: every target answers (the probe sees holes)" inC python3 -I "$T/fwprobe.py" open
  check "no rules: the LAN can open a connection into the container" inbound
  check "no rules: the entrypoint guard refuses to go online" not guard
  check "install ($B)" install "$B"
  for n in $C $H $R; do ip -n "$n" neigh flush all; done   # no cached neighbours: IPv6 must rediscover through the rules
  check "installed: LAN / host / special ranges blocked, internet open" inC python3 -I "$T/fwprobe.py" fw
  check "installed: nothing opens a connection into the container" not inbound
  check "installed: the entrypoint guard passes" guard
  check "installed: selftest [blocked] section all PASS" blocked_section
  snapshot > /tmp/$P-snap1
  check "install again: identical rules (idempotent)" install "$B"
  snapshot > /tmp/$P-snap2
  check "  ... rule sets equal" diff /tmp/$P-snap1 /tmp/$P-snap2
  check "  ... one jump each into our chains" bash -c "[ \$(ip netns exec $H iptables-save | grep -c -- '-j DONOR-') -eq \
$([ "$B" = iptables ] && echo 3 || echo 0) ]"
  inH bash -c "$(declare -f docker_sim); iptables -w -F FORWARD; ip6tables -w -F FORWARD; docker_sim"
  check "Docker restart (FORWARD rebuilt): still blocked" inC python3 -I "$T/fwprobe.py" fw
  O=iptables; [ "$B" = iptables ] && O=nft
  check "switch to $O: still blocked" bash -c "$(declare -f install inH); H=$H FW=$FW install $O && ip netns exec $C python3 -I $T/fwprobe.py fw"
  check "  ... only the $O rules are left" bash -c "if [ $O = nft ]; then ! ip netns exec $H iptables-save | grep -q DONOR; else ! ip netns exec $H nft list table inet donorkit; fi"
  check "uninstall" inH env DONOR_BRIDGE=donor0 bash "$FW" uninstall
  check "uninstalled: every target answers again" inC python3 -I "$T/fwprobe.py" open
  check "uninstalled: nothing of ours is left" ours_gone
  check "uninstall again (idempotent)" inH env DONOR_BRIDGE=donor0 bash "$FW" uninstall
done
check "render (dry run) prints the nft rules" bash -c "DONOR_BRIDGE=donor0 bash $FW render --backend nft | grep -q 'reject with icmpx admin-prohibited'"
echo "NETNS-FIREWALL $([ $FAILS = 0 ] && echo PASS || echo "FAIL ($FAILS)")"
exit $((FAILS > 0))
