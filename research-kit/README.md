# research-kit: lend a GPU for research (trusted volunteers only)

A container that a volunteer runs on their own machine and that **we** (the rad-exl3-mix researchers and our agents)
reach over Tailscale to run experiments. It's separate from, and optional to, the rad-quant-at-home job queue:

| | job queue ([../README.md](../README.md)) | research access (this folder) |
|---|---|---|
| what runs | fixed job types built into the worker image | whatever we type in a shell |
| trust needed | none: results are verified by score | yes: we get a shell on your machine |
| who | anyone | people who know us personally |

- Volunteers: read **[VOLUNTEER.md](VOLUNTEER.md)** (one page: what we can and cannot do, how to stop it).
- Us: **[OPERATOR.md](OPERATOR.md)** (onboarding, Tailscale policy and keys, host profile, rules, per-distro notes).

## How it is locked down

```
 volunteer's host                                                          internet
 ┌──────────────────────────────────────────────────────────────┐            ▲
 │  container donor-research (uid 10010, no caps, read-only)    │            │ allowed
 │   tailscaled --tun=userspace-networking ── Tailscale SSH ────┼── bridge donor0 ──► nft / DOCKER-USER rules
 │   /data (one volume)   GPU: /dev/kfd + /dev/dri | --gpus     │            │ rejected: RFC1918, 169.254/16,
 └──────────────────────────────────────────────────────────────┘            │ 100.64/10, host's own subnets,
                                                                              ▼ ULA / link-local, the host itself
```

- **Network:** its own Docker bridge (`donor0`, IPv4 only), DNS through 1.1.1.1 / 9.9.9.9. Host firewall rules
  (`firewall/`) reject everything private, link-local, CGNAT / Tailscale, special and multicast, the host's own subnets
  and the host itself; nothing can open a connection into the container. No published ports.
- **Container:** not privileged, `--cap-drop ALL` (tailscaled in userspace mode needs no capabilities), no-new-privileges,
  Docker's default seccomp profile (ROCm: plus only the NUMA memory-policy calls, `seccomp-rocm.json`), read-only
  image, no Docker socket, GPU through its device nodes only, CPU / RAM / process limits, one data volume.
- **Tailscale:** the container is its own node, tagged `tag:donor`, on *our* tailnet (the host's own Tailscale, if
  any, is untouched). The policy ([tailscale/policy.hujson](tailscale/policy.hujson)) lets only us reach donors, as
  the user `research`; donors reach nothing.
- **Fail closed:** `run.sh up` refuses to start without the firewall, and the entrypoint refuses to bring Tailscale
  up if the host answers on the container's gateway (i.e. the rules are missing).
- **Preemption:** the owner stops it at will; queues registered in `/data/queue.d` are relaunched at every start
  (`image/resume-queues.sh`).

## Files

| path | what |
|---|---|
| `Dockerfile.rocm`, `Dockerfile.cuda`, `build.sh` | the images (`ghcr.io/theherobrine422/rad-quant-at-home-research:<version>-rocm` / `-cuda`); the GPU stack pins come from the worker's `../docker/Dockerfile.*` |
| `image/` | what goes into the image: `entrypoint.sh`, `selftest.py` (+ `selftest.sh`), `gpucheck.py`, `resume-queues.sh`, `research-env.sh` |
| `run.sh`, `donor.env.example` | the volunteer's tool: `up`, `down`, `start`, `pause`, `selftest`, `status`, `remove`, `purge` |
| `compose.yaml` | the same settings for compose users (`--profile rocm` / `cuda`) |
| `seccomp-rocm.json`, `tools/mkseccomp.py` | Docker's default profile (moby seccomp/v0.3.0) + `get_mempolicy`, `mbind`, `set_mempolicy`, `set_mempolicy_home_node` without CAP_SYS_NICE |
| `firewall/` | `install-firewall.sh`, `uninstall-firewall.sh`, `donor-firewall.sh` (nft or iptables; systemd unit for boot) |
| `tailscale/policy.hujson` | the tailnet policy snippet, with tests that fail the save if a donor could reach anything |
| `locks/` | `pip freeze` of each built image |
| `tests/` | `netns-firewall.sh`, `docker-e2e.sh`, `headscale-ssh.sh`, `cuda-smoke-rootfs.sh`; results in [tests/RESULTS.md](tests/RESULTS.md) |

## Quick start

Volunteer (VOLUNTEER.md has the details):

```bash
sudo ./firewall/install-firewall.sh
cp donor.env.example donor.env && $EDITOR donor.env    # variant, name, Tailscale key, limits
./run.sh up && ./run.sh selftest
```

Us: [OPERATOR.md](OPERATOR.md) "Onboarding a volunteer".

## Building and publishing

`./build.sh rocm` / `./build.sh cuda` build `ghcr.io/theherobrine422/rad-quant-at-home-research:<VERSION>-<variant>`
(also `:<variant>` and `:<variant>-<commit>`). ROCm: Python 3.14, torch 2.15.0+rocm7.14 with the TheRock 7.14.1 SDK
(hipcc) and only the gfx1201 device libraries (`ROCM_DEVICE_EXTRA=device-all` for every arch): 15.6 GB on disk,
4.1 GB compressed. CUDA: CUDA 12.8.1 devel (nvcc), Python 3.12, torch 2.11.0+cu128: 27 GB, 9.5 GB compressed.
Pushing: [../README.md](../README.md) "Publishing images (maintainer)", then `docker push
ghcr.io/theherobrine422/rad-quant-at-home-research --all-tags`.
