#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# run.sh up|down|start|pause|unpause|status|logs|selftest|shell|remove|purge -- the research container on the
# volunteer's host. Settings: donor.env next to this script (copy donor.env.example) or the environment.
#
#   up        create and start the container (checks first: Docker, the firewall, the network, the GPU devices)
#   down      stop it (docker stop; the GPU and RAM are free again, /data is kept). start: start it again
#   pause     freeze every process in place (CPU drops to zero; GPU memory and RAM stay allocated). unpause: resume
#   status    container state, limits, firewall, Tailscale
#   selftest  the in-container self test with this host's addresses (PASS / FAIL table to paste back); --speed adds a
#             download test
#   shell     a shell inside, as the research user (to look around)
#   remove    delete the container (the data volume stays). purge: also delete the data volume, i.e. everything
#
# How it runs: the research network only (firewall/install-firewall.sh made it and keeps it off the LAN), public DNS,
# no capabilities, no-new-privileges, Docker's default seccomp profile (ROCm: plus the NUMA calls, seccomp-rocm.json),
# read-only image, not privileged, no Docker socket, the GPU by its device nodes only, CPU / RAM / process limits, one
# data volume at /data. Nothing is published to the host or the LAN; we reach it through its own Tailscale node.
set -euo pipefail
K=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
[ -f "$K/donor.env" ] && { set -a; . "$K/donor.env"; set +a; }
VARIANT=${DONOR_VARIANT:-rocm}
NAME=${DONOR_NAME:-donor-research}
IMAGE=${DONOR_IMAGE:-ghcr.io/theherobrine422/rad-quant-at-home-research:$(cat "$K/../VERSION" 2> /dev/null || echo latest)-$VARIANT}
NET=${DONOR_NET:-donor-research} BR=${DONOR_BRIDGE:-donor0}
DATA=${DONOR_DATA:-donor-research-data}
CPUS=${DONOR_CPUS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}
MEM=${DONOR_MEM:-24g} PIDS=${DONOR_PIDS:-4096} SHM=${DONOR_SHM:-8g}
TSH=${TS_HOSTNAME:-donor-$(hostname -s | tr -c 'a-zA-Z0-9-\n' -)}
UIDR=10010
say() { echo "run.sh: $*" >&2; }
die() { say "$*"; exit 1; }

firewall_ok() {
  if command -v systemctl > /dev/null && [ "$(systemctl is-active donor-firewall.service 2> /dev/null)" = active ]; then return 0; fi
  if [ "$(id -u)" = 0 ]; then
    command -v nft > /dev/null && nft list table inet donorkit > /dev/null 2>&1 && return 0
    command -v iptables > /dev/null && iptables -w -C "DOCKER-USER" -j DONOR-FWD > /dev/null 2>&1 && return 0
  fi
  return 1
}

gpu_flags() {
  G=()
  if [ "$VARIANT" = rocm ]; then
    [ -e /dev/kfd ] || die "/dev/kfd is missing: is the amdgpu driver loaded? (lsmod | grep amdgpu)"
    G+=(--device /dev/kfd)
    if [ "${DONOR_DRI:-all}" = all ]; then G+=(--device /dev/dri)
    else for d in $DONOR_DRI; do G+=(--device "/dev/dri/$d"); done; fi   # e.g. DONOR_DRI="renderD129": one card only
    for grp in video render; do gid=$(getent group $grp | cut -d: -f3); [ -n "$gid" ] && G+=(--group-add "$gid"); done
    case ${DONOR_SECCOMP:-kit} in
      kit) G+=(--security-opt "seccomp=$K/seccomp-rocm.json") ;;
      default) ;;                                          # Docker's built-in profile; try it, ROCm may warn about NUMA
      *) die "DONOR_SECCOMP: kit | default" ;;
    esac
  elif [ "$VARIANT" = cuda ]; then
    G+=(--gpus "${DONOR_GPUS:-all}")                      # all, or "device=0" / "device=0,1"
  elif [ "$VARIANT" != cpu ]; then                         # cpu: no GPU (CPU-only lenders, tests)
    die "DONOR_VARIANT: rocm | cuda | cpu"
  fi
  if command -v getenforce > /dev/null && [ "$(getenforce 2> /dev/null)" = Enforcing ] && [ "${DONOR_SELINUX_LABEL:-}" = disable ]; then
    G+=(--security-opt label=disable)
  fi
}

host_addrs() { ip -o addr show scope global 2> /dev/null | awk -v br="$BR" '$2 != br && $2 !~ /^(docker|br-|veth)/ {split($4, a, "/"); print a[1]}'; }
lan_gws() { { ip -4 route show default; ip -6 route show default; } 2> /dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "via") print $(i + 1)}' | sort -u; }

up() {
  command -v docker > /dev/null || die "Docker is not installed"
  docker info > /dev/null 2>&1 || die "cannot talk to Docker (sudo, or add yourself to the docker group)"
  if docker container inspect "$NAME" > /dev/null 2>&1; then
    say "$NAME exists; starting it (./run.sh remove first to recreate it with new settings)"; docker start "$NAME" > /dev/null; return; fi
  if ! firewall_ok; then
    [ "${DONOR_SKIP_FIREWALL_CHECK:-0}" = 1 ] || die "the firewall is not active: sudo $K/firewall/install-firewall.sh (then run this again)"
    say "WARNING: firewall check skipped (DONOR_SKIP_FIREWALL_CHECK=1)"
  fi
  b=$(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "$NET" 2> /dev/null) \
    || die "no Docker network $NET: sudo $K/firewall/install-firewall.sh creates it"
  [ "$b" = "$BR" ] || die "network $NET uses bridge '$b', the firewall guards $BR"
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { say "pulling $IMAGE"; docker pull "$IMAGE" > /dev/null || die "image $IMAGE not found: docker pull it, or ./build.sh $VARIANT"; }
  gpu_flags
  case $DATA in
    /*) [ -d "$DATA" ] || die "DONOR_DATA=$DATA does not exist"
        [ "$(stat -c %u "$DATA")" = $UIDR ] || die "DONOR_DATA=$DATA must belong to uid $UIDR: sudo chown $UIDR:$UIDR $DATA" ;;
    *) docker volume inspect "$DATA" > /dev/null 2>&1 || docker volume create --label org.rad-quant-at-home.research-kit=1 "$DATA" > /dev/null ;;
  esac
  # the Tailscale auth key goes into the volume (read once, deleted by the entrypoint), never into the container's
  # configuration, where `docker inspect` would show it for good
  if [ -n "${TS_AUTHKEY:-}" ]; then
    printf '%s' "$TS_AUTHKEY" | docker run --rm -i --network none --cap-drop ALL --security-opt no-new-privileges=true \
      -v "$DATA:/data" --entrypoint sh "$IMAGE" -c 'mkdir -p /data/tailscale && chmod 700 /data/tailscale && umask 077 && cat > /data/tailscale/authkey'
    say "Tailscale auth key placed in the data volume (used once at the first start, then deleted); remove TS_AUTHKEY from donor.env"
  fi
  docker run -d --name "$NAME" --hostname "$TSH" --restart unless-stopped --init \
    --network "$NET" --dns 1.1.1.1 --dns 9.9.9.9 \
    --cap-drop ALL --security-opt no-new-privileges=true \
    --read-only --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
    --tmpfs /run/tailscale:rw,nosuid,nodev,size=16m,uid=$UIDR,gid=$UIDR,mode=0700 \
    --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM" --pids-limit "$PIDS" --shm-size "$SHM" \
    --ulimit nofile=65536:65536 --stop-timeout 30 \
    -v "$DATA:/data" -e TS_HOSTNAME="$TSH" ${TS_LOGIN_SERVER:+-e TS_LOGIN_SERVER="$TS_LOGIN_SERVER"} \
    --label org.rad-quant-at-home.research-kit=1 \
    "${G[@]}" ${DONOR_EXTRA_RUN_ARGS:-} "$IMAGE" > /dev/null
  say "started $NAME ($IMAGE): $CPUS CPUs, $MEM RAM, $PIDS processes, data in $DATA. Tailscale name: $TSH"
  say "next: ./run.sh selftest (and send us the table)"
}

status() {
  docker ps -a --filter "name=^/$NAME$" --format 'container {{.Names}}: {{.Status}} ({{.Image}})' | grep . || echo "container $NAME: not created"
  docker stats --no-stream --format 'usage: CPU {{.CPUPerc}}, RAM {{.MemUsage}}, processes {{.PIDs}}' "$NAME" 2> /dev/null || true
  docker inspect -f 'limits: {{.HostConfig.NanoCpus}} nano-CPUs, {{.HostConfig.Memory}} B RAM, {{.HostConfig.PidsLimit}} processes; caps dropped: {{.HostConfig.CapDrop}}; privileged: {{.HostConfig.Privileged}}; read-only: {{.HostConfig.ReadonlyRootfs}}' "$NAME" 2> /dev/null || true
  if firewall_ok; then echo "firewall: active"; else echo "firewall: NOT VERIFIED (run as root, or: sudo $K/firewall/donor-firewall.sh status)"; fi
  docker exec "$NAME" tailscale status --self=true --peers=false 2> /dev/null | sed 's/^/tailscale: /' || true
}

case ${1:-} in
  up) up ;;
  down|stop) docker stop "$NAME" > /dev/null && say "stopped $NAME (./run.sh start to resume)" ;;
  start) firewall_ok || [ "${DONOR_SKIP_FIREWALL_CHECK:-0}" = 1 ] || die "the firewall is not active: sudo $K/firewall/install-firewall.sh"
         docker start "$NAME" > /dev/null && say "started $NAME" ;;
  pause) docker pause "$NAME" > /dev/null && say "paused $NAME (GPU memory and RAM stay allocated; ./run.sh down frees them)" ;;
  unpause|resume) docker unpause "$NAME" > /dev/null && say "resumed $NAME" ;;
  status) status ;;
  logs) docker logs --tail "${2:-50}" "$NAME" ;;
  selftest) shift; A=(); for a in $(host_addrs); do A+=(--host-ip "$a"); done; for g in $(lan_gws); do A+=(--lan-gw "$g"); done
            docker exec "$NAME" selftest "${A[@]}" "$@" ;;
  shell) docker exec -it "$NAME" bash -l ;;
  remove) docker rm -f "$NAME" > /dev/null && say "removed $NAME (data volume $DATA kept; ./run.sh purge deletes it)" ;;
  purge) docker rm -f "$NAME" > /dev/null 2>&1 || true
         case $DATA in /*) say "DONOR_DATA is a directory: delete $DATA yourself" ;; *) docker volume rm "$DATA" > /dev/null 2>&1 && say "deleted the data volume $DATA" || true ;; esac
         say "left: the image (docker image rm $IMAGE), the network and firewall (sudo $K/firewall/uninstall-firewall.sh --remove-network)" ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
