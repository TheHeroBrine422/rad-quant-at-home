#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# docker-e2e.sh -- the kit end to end with a real Docker daemon, without touching this host's own firewall: a
# Docker-in-Docker container (its own network namespace and iptables / nftables) plays the volunteer's host. Inside it:
# run.sh refuses without the firewall; the entrypoint's LAN guard keeps a container off Tailscale when the firewall is
# missing; install-firewall.sh; run.sh up (tests/Dockerfile.lite = the research image without the GPU stack); the full
# selftest (the real LAN of this box -- its address and default gateway -- must be blocked, the internet reachable);
# the resume hook across restarts; tailscaled up as the unprivileged user; uninstall opens the holes and the selftest
# sees them; the iptables backend with Docker's real DOCKER-USER chain; purge and uninstall leave nothing.
# Needs: docker, root (privileged container), internet. ~3 min. Output: one line per check, "DOCKER-E2E PASS".
#   E2E_DIND_IMAGE (default docker:29-dind), E2E_KEEP=1 keeps the dind container for a look.
set -u
K=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" != --inner ]; then
  D=rk-e2e-dind
  docker rm -f $D > /dev/null 2>&1
  docker run -d --privileged --name $D -v "$K:/kit:ro" "${E2E_DIND_IMAGE:-docker:29-dind}" > /dev/null || exit 1
  for _ in $(seq 60); do docker exec $D docker info > /dev/null 2>&1 && break; sleep 1; done
  docker exec $D apk add -q bash python3 iproute2 nftables iptables ip6tables > /dev/null || { echo "apk add failed"; exit 1; }
  # the real LAN as seen from this box: its own address and its default gateway (never written to files; masked in output)
  LANIP=$(ip -4 route get 1.1.1.1 2> /dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
  LANGW=$(ip -4 route show default | awk '{print $3; exit}')
  docker exec -e E2E_LANIP="$LANIP" -e E2E_LANGW="$LANGW" $D bash /kit/tests/docker-e2e.sh --inner
  rc=$?
  [ "${E2E_KEEP:-0}" = 1 ] || docker rm -f -v $D > /dev/null
  exit $rc
fi

# ---- inside the dind container (root, its own netns, its own dockerd) ----
FAILS=0 R=/tmp/e2e; mkdir -p $R
check() { local name=$1; shift; if "$@" > $R/check.out 2>&1; then echo "  ok    $name"
  else echo "  FAIL  $name"; sed 's/^/        /' $R/check.out | tail -30; FAILS=$((FAILS + 1)); fi; }
not() { ! "$@"; }
export DONOR_VARIANT=cpu DONOR_IMAGE=rqah-research-lite DONOR_CPUS=2 DONOR_MEM=2g DONOR_PIDS=512 DONOR_SHM=256m \
  TS_HOSTNAME=donor-e2e RESUME_DELAY_S=5
RUN=/kit/run.sh FW=/kit/firewall
EXTRA=(); [ -n "${E2E_LANIP:-}" ] && EXTRA+=(--host-ip "$E2E_LANIP"); [ -n "${E2E_LANGW:-}" ] && EXTRA+=(--lan-gw "$E2E_LANGW")
selftest() { $RUN selftest --no-gpu "${EXTRA[@]}" > $R/selftest.out 2>&1; cat $R/selftest.out; }
section_all_pass() { awk -v s="[$1]" '$0 == s {f=1; next} /^\[/ || /^SELFTEST/ {f=0} f && NF' $R/selftest.out > $R/sec.out
  cat $R/sec.out; [ -s $R/sec.out ] && ! grep -qvE "^  (PASS|INFO)" $R/sec.out; }
section_has_fail() { awk -v s="[$1]" '$0 == s {f=1; next} /^\[/ || /^SELFTEST/ {f=0} f && NF' $R/selftest.out | grep -q "^  FAIL"; }
logs_have() { docker logs donor-research 2>&1 | grep -q "$1"; }
waitfor() { for _ in $(seq "$1"); do eval "$2" > /dev/null 2>&1 && return 0; sleep 1; done; eval "$2"; }
inC() { docker exec donor-research bash -c "$1"; }

echo "docker-e2e (inside dind): $(docker version -f 'Docker {{.Server.Version}}'), $(iptables -V)"
check "build the lite image" docker build -q -f /kit/tests/Dockerfile.lite -t rqah-research-lite /kit/image
check "run.sh up refuses without the firewall" bash -c "! $RUN up && ! docker container inspect donor-research"
docker network create --driver bridge --subnet 172.31.254.0/24 -o com.docker.network.bridge.name=donor0 donor-research > /dev/null
guard_refuses() {   # Docker restarts it (unless-stopped): it must refuse at every start and never start tailscaled
  DONOR_SKIP_FIREWALL_CHECK=1 $RUN up && waitfor 30 "logs_have 'FATAL: the host answers'" && sleep 10 &&
  [ "$(docker logs donor-research 2>&1 | grep -c 'FATAL: the host answers')" -ge 2 ] &&
  ! logs_have 'tailscale:' && [ "$(docker inspect -f '{{.RestartCount}}' donor-research)" -ge 1 ]
}
check "  (forced past run.sh) the entrypoint guard keeps it offline, at every restart" guard_refuses
$RUN remove > /dev/null 2>&1
check "install-firewall.sh (auto backend; existing network reused)" $FW/install-firewall.sh
check "  status: ACTIVE" bash -c "$FW/donor-firewall.sh status"
check "run.sh up" $RUN up
check "  entrypoint: LAN guard passed, tailscaled up, not logged in" waitfor 30 "docker logs donor-research 2>&1 | grep -q 'tailscale: NeedsLogin'"
check "  the guard line" logs_have "LAN guard: the host gateway is blocked"
check "  tailscaled runs as uid 10010" bash -c "docker exec donor-research ps -eo uid,args | grep -E '^ *10010 +tailscaled'"
selftest_runs() { selftest; grep -q '^SELFTEST' $R/selftest.out; }
check "selftest (nft): runs" selftest_runs
cp $R/selftest.out $R/selftest-nft.txt
check "  [sandbox] all PASS" section_all_pass sandbox
check "  [blocked] all PASS (incl. this box's LAN address and gateway)" section_all_pass blocked
check "  [internet] DNS, HF, PyPI, GitHub PASS" bash -c "grep -E '^  PASS  (DNS|TCP|Hugging Face API|Hugging Face file|PyPI|GitHub)' $R/selftest.out | wc -l | grep -qx 6"

# resume hook: a registered queue is relaunched after every start until its DONE holds
inC 'mkdir -p /data/queue.d /data/work && cat > /data/queue.d/10-e2e.sh << "EOF"
CMD="bash -c \"echo started >> /data/work/e2e.out; exec sleep 600\""
DONE="grep -q finished /data/work/e2e.done"
LOG=/data/work/e2e.log
EOF'
docker restart -t 5 donor-research > /dev/null
check "resume hook: queue launched after a restart" waitfor 40 "inC 'grep -c started /data/work/e2e.out' | grep -qx 1"
check "  resume-queues --status: running" bash -c "docker exec donor-research resume-queues --local --status | grep -qx '10-e2e running'"
docker stop -t 5 donor-research > /dev/null; docker start donor-research > /dev/null
check "  relaunched after docker stop / start" waitfor 40 "inC 'grep -c started /data/work/e2e.out' | grep -qx 2"
inC 'echo finished > /data/work/e2e.done'
docker restart -t 5 donor-research > /dev/null; sleep 15
check "  not relaunched once DONE holds" bash -c "docker exec donor-research resume-queues --local --status | grep -qx '10-e2e done' && [ \$(docker exec donor-research grep -c started /data/work/e2e.out) = 2 ]"

check "uninstall-firewall.sh" $FW/uninstall-firewall.sh
selftest_fails() { selftest > /dev/null; grep -q '^SELFTEST FAIL' $R/selftest.out; }
check "  selftest now FAILS [blocked] (it sees the holes)" selftest_fails
cp $R/selftest.out $R/selftest-nofw.txt
check "  ... the gateway answers" section_has_fail blocked
check "install-firewall.sh --backend iptables (Docker's DOCKER-USER)" $FW/install-firewall.sh --backend iptables
check "  DOCKER-USER jumps to DONOR-FWD" bash -c "iptables -S DOCKER-USER | grep -- '-j DONOR-FWD'"
blocked_pass() { selftest > /dev/null; section_all_pass blocked; }
check "  selftest (iptables): [blocked] all PASS" blocked_pass
cp $R/selftest.out $R/selftest-iptables.txt
check "run.sh purge" $RUN purge
check "uninstall-firewall.sh --remove-network" $FW/uninstall-firewall.sh --remove-network
check "  nothing left (container, volume, network, rules, settings)" bash -c "! docker container inspect donor-research && \
  ! docker volume inspect donor-research-data && ! docker network inspect donor-research && ! iptables-save | grep -q DONOR && \
  ! nft list table inet donorkit && [ ! -e /etc/donor-kit ] && [ ! -e /usr/local/sbin/donor-firewall ]"
echo "--- selftest table with the nft firewall (as a volunteer would paste it) ---"; cat $R/selftest-nft.txt
echo "DOCKER-E2E $([ $FAILS = 0 ] && echo PASS || echo "FAIL ($FAILS)")"
exit $((FAILS > 0))
