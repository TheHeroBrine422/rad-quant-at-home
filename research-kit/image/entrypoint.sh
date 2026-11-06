#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# entrypoint.sh -- PID 1's child in the research container (docker --init runs it), as the unprivileged user `research`.
#
#   1. LAN guard: refuse to come online when the host firewall (firewall/install-firewall.sh) is missing, i.e. when the
#      host answers on the bridge gateway. DONOR_ALLOW_UNFIREWALLED=1 skips it (our own tests only).
#   2. tailscaled in userspace-networking mode: its own tailnet node, no TUN device, no capabilities, state in
#      /data/tailscale. First start: logs in with the auth key (TS_AUTHKEY or /data/tailscale/authkey, deleted after
#      use), tagged tag:donor, Tailscale SSH on. Later starts reuse the state; no key needed. TS_LOGIN_SERVER: a
#      self-hosted control server (headscale) instead of Tailscale's.
#   3. resume-queues.sh --local in the background: relaunches the research queues registered in /data/queue.d.
#   4. waits on tailscaled; docker stop (SIGTERM) stops it, and the queues are killed with the container (they are
#      written to survive that: atomic outputs, resumable steps).
set -u
umask 027
log() { echo "$(date -u +%FT%TZ) donor-kit: $*"; }
K=/opt/donor-kit
export HOME=/data/home

# /data is the one volume: home, tailscale state, queues, work. Lay it out on the first start.
mkdir -p /data/home /data/tailscale /data/queue.d /data/work /data/tmp /data/torch_ext /data/cache || {
  log "FATAL: /data is not writable by uid $(id -u) (a bind mount must belong to uid $(id -u))"; exit 1; }
chmod 700 /data/tailscale
[ -f /data/home/.bashrc ] || cp -n /etc/skel/.bashrc /etc/skel/.profile /data/home/ 2> /dev/null || true

if [ "${DONOR_ALLOW_UNFIREWALLED:-0}" != 1 ]; then
  if ! python3 -I "$K/selftest.py" --guard; then
    log "FATAL: the host answers on the container's gateway, so the host firewall is not installed."
    log "Run 'sudo ./firewall/install-firewall.sh' on the host, then start the container again. Not going online."
    exit 3
  fi
  log "LAN guard: the host gateway is blocked (firewall present)"
else
  log "WARNING: DONOR_ALLOW_UNFIREWALLED=1, the LAN guard is skipped (tests only)"
fi

SOCK=/run/tailscale/tailscaled.sock
TS=(tailscale --socket="$SOCK")
env -u TS_AUTHKEY tailscaled --tun=userspace-networking --statedir=/data/tailscale --socket="$SOCK" --port="${TS_PORT:-0}" \
  > /data/tailscale/tailscaled.log 2>&1 &
TSD=$!
trap 'log "stopping"; kill -TERM $TSD 2> /dev/null; wait $TSD; exit 0' TERM INT
for _ in $(seq 60); do "${TS[@]}" status --json > /dev/null 2>&1 && break
  [ -S "$SOCK" ] && "${TS[@]}" status 2>&1 | grep -q "Logged out\|NeedsLogin" && break; sleep 0.5; done

HOSTNAME_TS=${TS_HOSTNAME:-donor-$(hostname)}
state() { "${TS[@]}" status --json 2> /dev/null | python3 -Ic 'import json,sys; print(json.load(sys.stdin).get("BackendState",""))'; }
if [ "$(state)" != Running ]; then
  KEY=${TS_AUTHKEY:-}; KF=/data/tailscale/authkey
  [ -z "$KEY" ] && [ -f "$KF" ] && KEY=$(tr -d ' \n' < "$KF")
  if [ -z "$KEY" ]; then
    log "not logged in and no auth key (TS_AUTHKEY or $KF): tailscale stays offline; the container keeps running"
  # --advertise-tags is a safety net: even an untagged key by mistake gives a tag:donor node, never a device of the
  # key's owner (which would get the owner's access). TS_TAGS="" leaves the tags to the key (some self-hosted
  # control servers refuse both at once).
  elif TAGS=${TS_TAGS-tag:donor}; "${TS[@]}" up --auth-key="$KEY" --hostname="$HOSTNAME_TS" --ssh ${TAGS:+--advertise-tags="$TAGS"} \
        --accept-dns=false --accept-routes=false --timeout=120s ${TS_LOGIN_SERVER:+--login-server="$TS_LOGIN_SERVER"}; then
    rm -f "$KF"; log "tailscale: logged in as $HOSTNAME_TS (the auth key is no longer needed: remove TS_AUTHKEY)"
  else
    log "tailscale up failed (expired or used key?); see /data/tailscale/tailscaled.log"
  fi
  unset KEY TS_AUTHKEY
else
  "${TS[@]}" set --ssh --hostname="$HOSTNAME_TS" --accept-routes=false 2> /dev/null || true
fi
log "tailscale: $(state) $("${TS[@]}" ip -4 2> /dev/null | head -1)"

# queues registered in /data/queue.d (resume-queues.sh); a little later, so a login right after the start sees a calm box
( sleep "${RESUME_DELAY_S:-15}"; QUEUE_D=/data/queue.d TORCH_EXTENSIONS_DIR=/data/torch_ext \
  bash "$K/resume-queues.sh" --local >> /data/queue.d/boot.log 2>&1 ) < /dev/null > /dev/null 2>&1 &

wait $TSD
log "tailscaled exited ($?); see /data/tailscale/tailscaled.log"
exit 1
