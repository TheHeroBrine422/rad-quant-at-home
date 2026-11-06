#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# donor-firewall.sh install|uninstall|status|apply|render -- keep the research container off the volunteer's LAN.
# install-firewall.sh / uninstall-firewall.sh call this. Run as root on the HOST (not in the container).
#
# What it does (all of it scoped to the research network's bridge, nothing else on the host changes):
#   network  the Docker network donor-research (bridge donor0, subnet 172.31.254.0/24, IPv4 only, no inter-container
#            traffic). The container gets only this network.
#   forward  traffic from donor0 to private / local / special ranges is rejected: RFC1918 (10/8, 172.16/12, 192.168/16),
#            link-local and cloud metadata (169.254/16), CGNAT and Tailscale (100.64/10), 0/8, 127/8, 192.0.0/24,
#            198.18/15, multicast and reserved (224/4, 240/4), the subnets of the host's own public addresses; IPv6 ULA
#            (fc00::/7), link-local (fe80::/10), multicast, NAT64 (64:ff9b::/96), mapped and loopback, the host's own
#            IPv6 prefixes. Everything else (the internet) is allowed. Nothing from outside may open a connection INTO
#            the container (replies to its own connections only).
#   input    the container may not open connections to the host itself (any host address, the bridge gateway included);
#            only IPv6 neighbour discovery (the IPv6 form of ARP) gets through.
#   DNS      the container uses public resolvers (run.sh: --dns 1.1.1.1 --dns 9.9.9.9), never the LAN's.
# Rejects answer with ICMP admin-prohibited, so a probe fails at once (selftest.py tells "rejected" from "no answer").
#
# Backends (--backend, default auto = nft when the nft tool works, else iptables):
#   nft       its own table `inet donorkit` (forward / input base chains at priority filter-10). Docker, firewalld and
#             ufw never touch it, and a drop or reject in any base chain is final, so it holds whatever Docker's own
#             firewall backend is (iptables-legacy, iptables-nft, or Docker's nftables backend).
#   iptables  chains DONOR-FWD / DONOR-IN (and DONOR-FWD6 / DONOR-IN6 in ip6tables), jumped to from DOCKER-USER (the
#             chain Docker evaluates before its own rules), from FORWARD and from INPUT; filled atomically
#             (iptables-restore --noflush). Use this where nft is missing.
# Persistence: a systemd unit donor-firewall.service (Before=docker.service, so the rules are in place before Docker
# starts any container at boot) runs `/usr/local/sbin/donor-firewall apply` with the settings saved in
# /etc/donor-kit/firewall.conf. The same on Fedora, Ubuntu / Debian and Arch (OPERATOR.md "Firewall").
#
# Settings (env, saved at install): DONOR_NET (donor-research), DONOR_BRIDGE (donor0), DONOR_SUBNET (172.31.254.0/24).
# Flags: --backend nft|iptables|auto, --no-persist (rules until the next reboot only), --no-network (do not create the
# Docker network; tests), --remove-network (uninstall: also remove the network), --dry-run (render: print the rules).
set -euo pipefail
CMD=${1:-}; shift || true
BACKEND=auto PERSIST=1 NETWORK=1 RMNET=0
while [ $# -gt 0 ]; do case $1 in
  --backend) BACKEND=$2; shift ;;
  --no-persist) PERSIST=0 ;;
  --no-network) NETWORK=0 ;;
  --remove-network) RMNET=1 ;;
  *) echo "unknown option $1" >&2; exit 2 ;;
esac; shift; done
CONF=/etc/donor-kit/firewall.conf
if [ "$CMD" = apply ] && [ -f "$CONF" ]; then . "$CONF"; fi
NET=${DONOR_NET:-donor-research} BR=${DONOR_BRIDGE:-donor0} SUBNET=${DONOR_SUBNET:-172.31.254.0/24}
[ "$BACKEND" = auto ] && BACKEND=${DONOR_FW_BACKEND:-auto}
SBIN=/usr/local/sbin/donor-firewall UNIT=/etc/systemd/system/donor-firewall.service
TABLE=donorkit
log() { echo "donor-firewall: $*" >&2; }

BLOCK4="0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 172.16.0.0/12 192.0.0.0/24 192.168.0.0/16
198.18.0.0/15 224.0.0.0/4 240.0.0.0/4"
BLOCK6="::/127 ::ffff:0:0/96 64:ff9b::/96 64:ff9b:1::/48 fc00::/7 fe80::/10 ff00::/8"

# the host's own subnets and addresses (on-link routes + global addresses; private ones overlap the static ranges,
# which is harmless): a neighbour on a public LAN segment or a LAN with public IPv6 is still the LAN. The research
# bridge, Docker's bridges and veths are skipped; a suspiciously short on-link prefix is replaced by the address alone.
host_prefixes() {   # $1 = 4|6
  local min=8; [ "$1" = 6 ] && min=32
  { ip -o -"$1" route show table main scope link 2> /dev/null | awk '{d = ""; for (i = 2; i < NF; i++) if ($i == "dev") d = $(i + 1); print $1, d}'
    ip -o -"$1" addr show scope global 2> /dev/null | awk '{print $4, $2}'
    ip -o -"$1" addr show scope global 2> /dev/null | awk -v f="$1" '{split($4, a, "/"); print a[1] (f == 4 ? "/32" : "/128"), $2}'
  } | awk -v br="$BR" -v min="$min" '$2 != br && $2 !~ /^(docker|br-|veth|lo$)/ && $1 ~ /\// && $1 !~ /^(fe80|ff)/ {
        split($1, a, "/"); if (a[2] + 0 < min) next; print $1 }' | sort -u | tr '\n' ' '
}

pick_backend() {
  if [ "$BACKEND" = auto ]; then
    if command -v nft > /dev/null && nft list tables > /dev/null 2>&1; then BACKEND=nft
    elif command -v iptables > /dev/null; then BACKEND=iptables
    else log "neither nft nor iptables found"; exit 1; fi
  fi
  case $BACKEND in nft|iptables) ;; *) log "backend: nft | iptables | auto"; exit 2 ;; esac
}

render_nft() {
  local h4 h6 b4 b6
  h4=$(host_prefixes 4); h6=$(host_prefixes 6)
  b4=$(echo $BLOCK4 $h4 | tr ' ' ','); b6=$(echo $BLOCK6 $h6 | tr ' ' ',')
  cat << EOF
# donor-kit firewall (research-kit/firewall/donor-firewall.sh): bridge $BR, subnet $SUBNET
add table inet $TABLE
delete table inet $TABLE
table inet $TABLE {
  set blocked4 { type ipv4_addr; flags interval; auto-merge; elements = { $b4 } }
  set blocked6 { type ipv6_addr; flags interval; auto-merge; elements = { $b6 } }
  chain forward {
    type filter hook forward priority filter - 10; policy accept;
    iifname "$BR" ip daddr @blocked4 counter reject with icmpx admin-prohibited comment "container -> LAN / private"
    iifname "$BR" ip6 daddr @blocked6 counter reject with icmpx admin-prohibited comment "container -> LAN / private (v6)"
    oifname "$BR" ct state established,related accept
    oifname "$BR" counter drop comment "nothing opens connections into the container"
  }
  chain input {
    type filter hook input priority filter - 10; policy accept;
    iifname "$BR" ct state established,related accept
    iifname "$BR" icmpv6 type { nd-neighbor-solicit, nd-neighbor-advert } accept comment "IPv6 neighbour discovery (ARP's twin)"
    iifname "$BR" counter reject with icmpx admin-prohibited comment "container -> the host itself"
  }
}
EOF
}

# iptables-restore input for one family (4|6), applied with --noflush: only our chains are (re)written
render_ipt() {
  local fam=$1 blk rej
  if [ "$fam" = 4 ]; then blk="$BLOCK4 $(host_prefixes 4)"; rej=icmp-admin-prohibited
  else blk="$BLOCK6 $(host_prefixes 6)"; rej=icmp6-adm-prohibited; fi
  local f=DONOR-FWD i=DONOR-IN
  [ "$fam" = 6 ] && f=DONOR-FWD6 i=DONOR-IN6
  echo "*filter"
  echo ":$f - [0:0]"
  echo ":$i - [0:0]"
  echo "-F $f"
  echo "-F $i"
  for n in $blk; do echo "-A $f -i $BR -d $n -j REJECT --reject-with $rej"; done
  echo "-A $f -o $BR -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN"
  echo "-A $f -o $BR -j DROP"
  echo "-A $i -i $BR -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN"
  if [ "$fam" = 6 ]; then for nd in neighbour-solicitation neighbour-advertisement; do
    echo "-A $i -i $BR -p ipv6-icmp --icmpv6-type $nd -j RETURN"; done; fi
  echo "-A $i -i $BR -j REJECT --reject-with $rej"
  echo "COMMIT"
}

ipt_jump() {   # tool parent target: make sure parent jumps to target, once, at the top
  local t=$1 p=$2 c=$3
  $t -w -C "$p" -j "$c" 2> /dev/null && return 0
  $t -w -I "$p" 1 -j "$c"
}

apply_rules() {
  pick_backend
  if [ "$BACKEND" = nft ]; then
    render_nft | nft -f -
    log "nft table inet $TABLE loaded (bridge $BR)"
    # a leftover iptables install would only double the rules; remove it
    remove_ipt quiet
  else
    for fam in 4 6; do
      t=iptables; [ "$fam" = 6 ] && t=ip6tables
      command -v $t > /dev/null || { [ "$fam" = 6 ] && { log "no ip6tables: IPv6 not filtered (the network has no IPv6)"; continue; }; exit 1; }
      f=DONOR-FWD i=DONOR-IN; [ "$fam" = 6 ] && f=DONOR-FWD6 i=DONOR-IN6
      render_ipt "$fam" | "$t-restore" -w --noflush
      $t -w -N DOCKER-USER 2> /dev/null || true        # Docker reuses an existing DOCKER-USER and never flushes it
      ipt_jump $t DOCKER-USER "$f"
      ipt_jump $t FORWARD "$f"                         # also when Docker's DOCKER-USER jump is missing (iptables off)
      ipt_jump $t INPUT "$i"
    done
    log "iptables chains DONOR-FWD / DONOR-IN loaded (bridge $BR)"
    command -v nft > /dev/null && nft delete table inet $TABLE 2> /dev/null || true
  fi
}

remove_ipt() {
  for t in iptables ip6tables; do command -v $t > /dev/null || continue
    for pair in "DOCKER-USER DONOR-FWD" "FORWARD DONOR-FWD" "INPUT DONOR-IN" "DOCKER-USER DONOR-FWD6" \
                "FORWARD DONOR-FWD6" "INPUT DONOR-IN6"; do
      set -- $pair
      while $t -w -D "$1" -j "$2" 2> /dev/null; do :; done
    done
    for c in DONOR-FWD DONOR-IN DONOR-FWD6 DONOR-IN6; do $t -w -F $c 2> /dev/null && $t -w -X $c 2> /dev/null || true; done
  done
  [ "${1:-}" = quiet ] || log "iptables chains removed"
}

make_network() {
  command -v docker > /dev/null || { log "docker not found: install Docker first (or --no-network)"; exit 1; }
  if docker network inspect "$NET" > /dev/null 2>&1; then
    b=$(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "$NET")
    [ "$b" = "$BR" ] || { log "network $NET exists with bridge '$b', not $BR: remove it (docker network rm $NET)"; exit 1; }
    log "Docker network $NET exists (bridge $BR)"
  else
    docker network create --driver bridge --subnet "$SUBNET" --ipv6=false \
      --opt com.docker.network.bridge.name="$BR" --opt com.docker.network.bridge.enable_icc=false \
      --label org.rad-quant-at-home.research-kit=1 "$NET" > /dev/null
    log "Docker network $NET created (bridge $BR, $SUBNET)"
  fi
}

persist() {
  mkdir -p /etc/donor-kit
  { echo "# written by donor-firewall.sh install; read by donor-firewall.service"
    echo "DONOR_NET=$NET"; echo "DONOR_BRIDGE=$BR"; echo "DONOR_SUBNET=$SUBNET"; echo "DONOR_FW_BACKEND=$BACKEND"; } > "$CONF.part"
  mv "$CONF.part" "$CONF"
  install -m 0755 "$(readlink -f "$0")" "$SBIN.part" && mv "$SBIN.part" "$SBIN"
  command -v systemctl > /dev/null && [ -d /run/systemd/system ] || {
    log "no systemd: the rules last until the next reboot; run '$SBIN apply' at every boot, before Docker starts"; return 0; }
  cat > "$UNIT" << EOF
[Unit]
Description=donor-kit: keep the research container (bridge $BR) off the LAN and the host
Documentation=https://github.com/TheHeroBrine422/rad-quant-at-home/tree/main/research-kit
Wants=network-online.target
After=network-online.target
Before=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$SBIN apply

[Install]
WantedBy=multi-user.target
WantedBy=docker.service
EOF
  systemctl daemon-reload
  systemctl enable donor-firewall.service > /dev/null 2>&1
  systemctl start donor-firewall.service 2> /dev/null || true
  log "persisted: donor-firewall.service (enabled; runs before docker.service at boot)"
}

status() {
  local ok=0
  if command -v nft > /dev/null && nft list table inet $TABLE > /dev/null 2>&1; then
    echo "nft: table inet $TABLE loaded"; nft list table inet $TABLE | grep -E "counter|elements" | sed 's/^/  /'; ok=1; fi
  if command -v iptables > /dev/null && iptables -w -S DONOR-FWD > /dev/null 2>&1; then
    echo "iptables: DONOR-FWD $(iptables -w -S DONOR-FWD | grep -c REJECT) reject rules; jumps:"
    for p in DOCKER-USER FORWARD INPUT; do iptables -w -S $p 2> /dev/null | grep -E -- "-j DONOR-(FWD|IN)$" | sed 's/^/  /'; done; ok=1; fi
  [ -f "$CONF" ] && echo "saved settings: $(grep -v '^#' "$CONF" | tr '\n' ' ')"
  command -v systemctl > /dev/null && echo "donor-firewall.service: $(systemctl is-enabled donor-firewall.service 2> /dev/null || echo absent), $(systemctl is-active donor-firewall.service 2> /dev/null || true)"
  [ $ok = 1 ] && echo "FIREWALL: ACTIVE" || { echo "FIREWALL: NOT ACTIVE"; return 1; }
}

case $CMD in
  render) pick_backend; if [ "$BACKEND" = nft ]; then render_nft; else render_ipt 4; render_ipt 6; fi ;;
  apply) [ "$(id -u)" = 0 ] || { log "run as root"; exit 1; }; apply_rules ;;
  install)
    [ "$(id -u)" = 0 ] || { log "run as root (sudo)"; exit 1; }
    apply_rules                                        # rules first: the network never exists without them
    [ $NETWORK = 1 ] && make_network
    [ $PERSIST = 1 ] && persist
    log "done. Check with: sudo $0 status; then ./run.sh up and ./run.sh selftest" ;;
  uninstall)
    [ "$(id -u)" = 0 ] || { log "run as root (sudo)"; exit 1; }
    if command -v systemctl > /dev/null && [ -f "$UNIT" ]; then
      systemctl disable --now donor-firewall.service > /dev/null 2>&1 || true; rm -f "$UNIT"; systemctl daemon-reload; fi
    rm -f "$SBIN" "$CONF"; rmdir /etc/donor-kit 2> /dev/null || true
    command -v nft > /dev/null && nft delete table inet $TABLE 2> /dev/null && log "nft table removed" || true
    remove_ipt
    if [ $RMNET = 1 ] && command -v docker > /dev/null && docker network inspect "$NET" > /dev/null 2>&1; then
      docker network rm "$NET" > /dev/null && log "Docker network $NET removed"; fi
    log "uninstalled. A container still on $NET now has LAN access: stop it (./run.sh down) or remove it" ;;
  status) status ;;
  *) echo "usage: donor-firewall.sh install|uninstall|status|apply|render [--backend nft|iptables] [--no-persist] [--no-network] [--remove-network]" >&2; exit 2 ;;
esac
