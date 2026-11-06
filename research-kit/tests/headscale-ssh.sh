#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# headscale-ssh.sh -- Tailscale SSH into the research container, with tailscaled as the unprivileged user under the run
# flags (cap-drop ALL, no-new-privileges, read-only root), against a throwaway headscale control server on an INTERNAL
# Docker network (no internet, no DERP, nothing touches a real tailnet). Nodes: two donor containers (tests/
# Dockerfile.lite, the entrypoint as shipped, single-use keys, tagged tag:donor) and a "user" node (untagged, user admin).
# Policy (tests/headscale-policy.hujson): the policy.hujson rules in headscale's dialect (user admin@ -> tag:donor, ssh
# as research). Checks: the user logs in as research (non-interactive command; the research environment is loaded),
# root is refused; a donor can reach neither the user nor the other donor; the auth key file is gone after login.
# Needs docker; ~2 min. Output: "HEADSCALE-SSH PASS". E2E_KEEP=1 keeps everything for a look.
set -u
K=$(cd "$(dirname "$0")/.." && pwd); T=$K/tests
HS_IMAGE=${HS_IMAGE:-headscale/headscale:v0.29.7}; LITE=${LITE_IMAGE:-rqah-research-lite}
N=rk-hs P=rk-hs-   # network and container prefix
FAILS=0
not() { ! "$@"; }
check() { local name=$1; shift; if "$@" > /tmp/$P.out 2>&1; then echo "  ok    $name"
  else echo "  FAIL  $name"; sed 's/^/        /' /tmp/$P.out | tail -20; FAILS=$((FAILS + 1)); fi; }
cleanup() { docker rm -f ${P}server ${P}donor1 ${P}donor2 ${P}user > /dev/null 2>&1
  docker volume rm ${P}d1 ${P}d2 ${P}hs > /dev/null 2>&1; docker network rm $N > /dev/null 2>&1; }
[ "${E2E_KEEP:-0}" = 1 ] || trap cleanup EXIT
cleanup
docker image inspect "$LITE" > /dev/null 2>&1 || docker build -q -f "$T/Dockerfile.lite" -t "$LITE" "$K/image" > /dev/null
docker build -q -t ${P}client - > /dev/null << EOF
FROM $LITE
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client && rm -rf /var/lib/apt/lists/*
EOF
docker network create --internal $N > /dev/null
CFG=$(mktemp -d); cp "$T/headscale-policy.hujson" "$CFG/policy.hujson"
cat > "$CFG/config.yaml" << 'EOF'
server_url: http://rk-hs-server:8080
listen_addr: 0.0.0.0:8080
metrics_listen_addr: 127.0.0.1:9090
grpc_listen_addr: 127.0.0.1:50443
noise: {private_key_path: /var/lib/headscale/noise_private.key}
prefixes: {v4: 100.64.0.0/10, v6: fd7a:115c:a1e0::/48, allocation: sequential}
derp: {server: {enabled: false}, urls: [], paths: [/etc/headscale/derp.yaml], auto_update_enabled: false}
disable_check_updates: true
database: {type: sqlite, sqlite: {path: /var/lib/headscale/db.sqlite}}
policy: {mode: file, path: /etc/headscale/policy.hujson}
dns: {magic_dns: true, base_domain: hs.test, override_local_dns: false, nameservers: {global: []}}
unix_socket: /var/run/headscale/headscale.sock
unix_socket_permission: "0770"
logtail: {enabled: false}
log: {level: info}
EOF
# headscale needs a DERP map; this region is unreachable on purpose (the internal network has no way out), so the
# nodes must find each other directly on the shared network, as they will over the internet with NAT traversal
cat > "$CFG/derp.yaml" << 'EOF'
regions:
  900:
    regionid: 900
    regioncode: none
    regionname: unreachable
    nodes:
      - {name: 900a, regionid: 900, hostname: derp.invalid, stunport: -1, derpport: 443}
EOF
chmod -R a+rX "$CFG"
docker run -d --name ${P}server --network $N -v "$CFG:/etc/headscale:ro,z" -v ${P}hs:/var/lib/headscale "$HS_IMAGE" serve > /dev/null
HS() { docker exec ${P}server headscale "$@"; }
for _ in $(seq 30); do HS users list > /dev/null 2>&1 && break; sleep 1; done
HS users create admin > /dev/null; HS users create donors > /dev/null
uid() { HS users list -o json | python3 -c "import json,sys; print([u['id'] for u in json.load(sys.stdin) if u['name']=='$1'][0])"; }
key() { HS preauthkeys create -u "$(uid "$1")" -e 1h "${@:2}" -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['key'])"; }

donor() {   # name hostname: the run.sh flags, the key seeded into the volume as run.sh does
  # headscale 0.29 refuses the entrypoint's --advertise-tags=tag:donor ("requested tags ... not permitted", both with a
  # tagged key and with a key of the tag owner), so here the tag comes from the key alone (TS_TAGS=""). The
  # advertise-tags path is for Tailscale's control plane: OPERATOR.md "still to test".
  printf '%s' "$(key donors --tags tag:donor)" | docker run --rm -i --network none -v ${P}$1:/data --entrypoint sh "$LITE" \
    -c 'mkdir -p /data/tailscale && chmod 700 /data/tailscale && umask 077 && cat > /data/tailscale/authkey'
  docker run -d --name ${P}donor${1#d} --hostname "$2" --init --network $N --cap-drop ALL \
    --security-opt no-new-privileges=true --read-only --tmpfs /tmp:rw,exec,nosuid,nodev,size=256m \
    --tmpfs /run/tailscale:rw,nosuid,nodev,size=16m,uid=10010,gid=10010,mode=0700 --memory 1g --pids-limit 256 \
    -v ${P}$1:/data -e TS_HOSTNAME="$2" -e TS_LOGIN_SERVER=http://rk-hs-server:8080 -e DONOR_ALLOW_UNFIREWALLED=1 \
    -e TS_TAGS= -e RESUME_DELAY_S=1 "$LITE" > /dev/null
}
donor d1 donor-hs1; donor d2 donor-hs2
UKEY=$(key admin)
docker run -d --name ${P}user --network $N --entrypoint sh ${P}client -c "tailscaled --tun=userspace-networking \
  --statedir=/var/lib/ts --socket=/run/tailscale/tailscaled.sock > /tmp/tsd.log 2>&1 & sleep 2; \
  tailscale up --login-server=http://rk-hs-server:8080 --auth-key=$UKEY --hostname=you-laptop --ssh --timeout=60s; sleep infinity" > /dev/null
U() { docker exec ${P}user "$@"; }
D1() { docker exec ${P}donor1 "$@"; }
running() { docker exec "$1" tailscale status --json 2> /dev/null | grep -q '"BackendState": "Running"'; }
waitfor() { for _ in $(seq "$1"); do "${@:2}" > /dev/null 2>&1 && return 0; sleep 1; done; "${@:2}"; }
ip_of() { docker exec "$1" tailscale ip -4 | head -1; }
banner() { docker exec "$1" sh -c "(sleep 4) | timeout 10 tailscale nc $2 22 | head -c 20" 2> /dev/null | grep -q '^SSH-'; }

echo "headscale-ssh: $HS_IMAGE, $(docker exec ${P}donor1 tailscale version | head -1 2> /dev/null)"
check "donor 1 logged in (tagged key, userspace, uid 10010)" waitfor 60 running ${P}donor1
check "donor 2 logged in" waitfor 60 running ${P}donor2
check "user node logged in" waitfor 60 running ${P}user
check "  donor 1 is tagged tag:donor" bash -c "docker exec ${P}donor1 tailscale status --json | grep -q '\"tag:donor\"'"
check "  the auth key file is gone after login" bash -c "! docker exec ${P}donor1 test -e /data/tailscale/authkey"
IU=$(ip_of ${P}user); I1=$(ip_of ${P}donor1); I2=$(ip_of ${P}donor2)
# first contact: without DERP the peers need a moment to find the direct path; disco pings set it up
for ip in $I1 $I2; do U tailscale ping -c 10 --timeout 3s "$ip" > /dev/null 2>&1; done
D1 tailscale ping -c 3 --timeout 2s "$IU" > /dev/null 2>&1   # path only: disco pings ignore the policy
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes"
sshu() { U ssh $SSHO -o "ProxyCommand=tailscale nc %h %p" "$@"; }
check "user -> donor 1: Tailscale SSH as research (non-interactive command)" \
  bash -c "docker exec ${P}user ssh $SSHO -o 'ProxyCommand=tailscale nc %h %p' research@$I1 'id -un; id -u' | tr '\n' ' ' | grep -q '^research 10010'"
check "  the research environment is loaded in 'ssh host cmd'" \
  bash -c "docker exec ${P}user ssh $SSHO -o 'ProxyCommand=tailscale nc %h %p' research@$I1 'echo \$PATH \$HF_HOME' | grep -q '/opt/venv/bin.*/data/cache/huggingface'"
check "  the session sees the donor's data volume and queues" \
  bash -c "docker exec ${P}user ssh $SSHO -o 'ProxyCommand=tailscale nc %h %p' research@$I1 'test -w /data/work && resume-queues --local --status; echo ok' | grep -q ok"
check "user -> donor 1 as root: refused by the policy" bash -c "docker exec ${P}user ssh $SSHO -o 'ProxyCommand=tailscale nc %h %p' root@$I1 true 2>&1 | grep -i 'policy does not permit'"
check "user -> donor 2: Tailscale SSH as research too" \
  bash -c "docker exec ${P}user ssh $SSHO -o 'ProxyCommand=tailscale nc %h %p' research@$I2 'id -un' | grep -qx research"
check "donor 1 -> user: nothing (Tailscale SSH port of the user node)" not banner ${P}donor1 $IU
check "donor 1 -> donor 2: nothing" not banner ${P}donor1 $I2
check "  (control: the user node's SSH port does answer the user itself)" banner ${P}user $IU
check "donor container still has no capabilities, NoNewPrivs" bash -c "docker exec ${P}donor1 grep -E '^(CapEff|NoNewPrivs)' /proc/1/status | tr '\n' ' ' | grep -q 'CapEff:.0000000000000000 NoNewPrivs:.1'"
rm -rf "$CFG"
echo "HEADSCALE-SSH $([ $FAILS = 0 ] && echo PASS || echo "FAIL ($FAILS)")"
exit $((FAILS > 0))
