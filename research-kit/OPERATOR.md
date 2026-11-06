# Research access on a volunteer's machine: operator notes

For us (the user and the agents). The volunteer's side is [VOLUNTEER.md](VOLUNTEER.md); how the pieces fit is in
[README.md](README.md).

## Rules

1. **Trusted people only.** We know them personally, they read VOLUNTEER.md, and they agreed.
2. **No secrets on donor boxes.** No API tokens, ssh keys, git credentials, HF write tokens, Tailscale keys other than
   the box's own single-use auth key. The owner has root on the host and can read everything in the container,
   including what we type. Results are **pulled** from our side (rsync / scp over Tailscale SSH); the donor node can't
   push to us anyway (the policy lets it reach nothing).
3. **Public data only.** Public corpora, public checkpoints, public Hessians and code. Never our private traces,
   session logs, unpublished material others shared with us, or anything derived from them per document (goldens,
   slices). Those stay on our own machines (the project's data rules).
4. **Preemptible by design.** The owner stops the container whenever they want, without warning. Every queue is
   resumable and registered in `/data/queue.d` (below); never fight the owner for the GPU, never ask them to keep it
   running for us.
5. **Stay inside the limits** the owner set, and keep `/data` tidy: delete intermediates once a result is logged.
6. **Don't try to get around the sandbox** (privilege escalation, scanning, the LAN, other tailnet devices), even
   "to test it". `selftest` is the test.
7. No ssh agent forwarding (`ssh -A`) and no port forwarding from our machines into a donor box.

## Onboarding a volunteer

1. **Tailscale policy, once:** merge [tailscale/policy.hujson](tailscale/policy.hujson) into our tailnet policy
   (admin console > Access controls). Read its header first: the default allow-all rule must go, or donors could
   reach our devices. The policy's `tests` block makes a save fail if a donor could reach us or another donor.
2. **One auth key per volunteer** (admin console > Settings > Keys > Generate auth key):
   - **Reusable: off** (single-use: once the box has logged in, the key is worthless, so it doesn't matter who saw it),
   - **Ephemeral: off** (the node's state lives in the box's `/data` volume, so it survives restarts and reboots without
     a key; an ephemeral node is deleted soon after it goes offline, and a stopped container would have to log in again),
   - **Pre-approved: on**, **Tags: `tag:donor`**, expiry: a few days (only matters until it's used).
   - Tagged devices don't expire, so the box stays reachable until we remove it.
   - API alternative (with our own API access token, never stored on a donor box):
     `POST https://api.tailscale.com/api/v2/tailnet/-/keys` with
     `{"capabilities": {"devices": {"create": {"reusable": false, "ephemeral": false, "preauthorized": true, "tags": ["tag:donor"]}}}, "expirySeconds": 259200, "description": "donor-<name>"}`.
   - A reusable + ephemeral key is the other valid choice (one key for many short-lived boxes), but then the key must
     stay on the box, and anyone with it can add `tag:donor` nodes. Not for long-lived volunteer machines.
3. **Send the volunteer** the kit (this folder), the key over a private channel, and the name to use (`TS_HOSTNAME`,
   e.g. `donor-alex`). They follow VOLUNTEER.md "Setup".
4. **They send back the selftest table** (`./run.sh selftest`). Every row PASS (warnings are OK, e.g. Tailscale before
   login). Keep it with their host profile.
5. **Log in:** `ssh research@donor-alex` (or `tailscale ssh research@donor-alex`). Check `selftest` once more from
   inside, `nvidia-smi` / `rocminfo`, `df -h /data`.
6. **Write the host profile** (below) into AGENTS.md's machine table, and add the box to the watchdog as preemptible.

### Revoking a donor

- **Remove the device:** admin console > Machines > `donor-<name>` > Remove (API: `DELETE /api/v2/device/{id}`). Its
  node key is dead at once; it can't log in again without a new auth key.
- If its auth key was never used (or was reusable): Settings > Keys > Revoke.
- Take it out of AGENTS.md and the watchdog; stop pulling from it.
- The volunteer removes everything with VOLUNTEER.md "Stopping for good". Nothing of ours needs cleaning up there
  (rule 2).

## Host profile template (for AGENTS.md)

```markdown
| **donor-<name>** (lent by <first name>, research shell) | <GPU> x<n> (<VRAM> each, <gfx1201 / sm_89>), <driver> |
| <CPU threads allowed> of <host threads> | <RAM limit> | /data: <free GB>, <filesystem> | <down / up Mbps> |

- Access: `ssh research@donor-<name>` (Tailscale SSH, tag:donor). Image: rad-quant-at-home-research:<variant>-<commit>.
- **Preemptible**: the owner stops it at will (gaming, work). Assume a job can die at any moment; queues in
  /data/queue.d, resumed at every start. Watchdog: `donor-<name> preemptible pre 10`.
- Owner rules: <hours it is usually available / "nights only" / "ask before multi-day runs" / temperature or power
  limits / "never more than N GB on /data">.
- Public data only (OPERATOR.md rules); no secrets on the box. Results are pulled from our side.
- Selftest: PASS on <date> (table in local/<...>).
```

## Working on a donor box

- The shell: Python venv `/opt/venv` (torch; ROCm: torch 2.15.0+rocm7.14 with the TheRock 7.14.1 SDK, hipcc included;
  CUDA: torch 2.11.0+cu128 with nvcc 12.8), build tools, git, tmux. `pip install --user` goes to `/data/home/.local`.
  No root, no apt: anything else is a pip / conda-style user install on `/data`, or a rebuild of the image.
- Everything that grows lives on `/data` (the one volume): `/data/work`, `/data/cache` (HF, pip, triton),
  `/data/torch_ext`, `/data/tmp`. The image is read-only; `/tmp` is a 2 GB RAM disk.
- A non-interactive `ssh host cmd` may not read the profile: start scripts with
  `. /etc/profile.d/research-env.sh`.
- **Queues survive stops** the same way as on our own preemptible GPU boxes: write outputs
  as `.part` then rename, skip finished steps, no `.done` marker before the data, and register the driver in
  `/data/queue.d/NAME.sh` (`CMD=`, `DONE=`, optional `LOG=`, `AFTER=`, `CWD=`; format in `image/resume-queues.sh`).
  The container runs `resume-queues --local` ~15 s after every start; `resume-queues --local --status` shows the queue
  states. `touch /data/queue.d/NAME.off` before stopping a queue on purpose, `/data/queue.d/PAUSE` stops all launches.
- Downloads: from the box itself (public data); nothing large through our machines.

## Firewall: how it works, per distro

`firewall/donor-firewall.sh` (header) is the reference. In short: the container sits on its own bridge `donor0`
(network `donor-research`, 172.31.254.0/24, IPv4 only). Traffic from that bridge to private, link-local, CGNAT /
Tailscale, special and multicast ranges, and to the host's own subnets, is rejected; so is any connection from the
container to the host itself, and any new connection into the container. The internet is allowed; DNS goes to 1.1.1.1
/ 9.9.9.9 directly.

- **Backend:** `nft` (default where it works): its own table `inet donorkit`, which Docker, firewalld and ufw never
  touch, and which holds whatever firewall backend Docker uses (a drop in any base chain is final). `iptables`: chains
  `DONOR-FWD` / `DONOR-IN` jumped to from Docker's `DOCKER-USER`, `FORWARD` and `INPUT`. Prefer nft:
  `firewall-cmd --reload` with firewalld's old iptables backend, or a script that flushes iptables, removes the
  iptables variant until the next boot or `install-firewall.sh`.
- **Persistence** is the same on every systemd distro: `donor-firewall.service` (enabled; `Before=docker.service`, so
  the rules are loaded before Docker starts the container at boot) runs `/usr/local/sbin/donor-firewall apply` with
  `/etc/donor-kit/firewall.conf`. It recomputes the host's subnets at every boot.
  - **Fedora / RHEL:** `nft` is installed; firewalld runs alongside and doesn't interfere. Docker from Docker's repo or
    Fedora's `moby-engine` both work. SELinux: if the container can't open the GPU, `sudo setsebool -P
    container_use_devices 1` (`DONOR_SELINUX_LABEL=disable` in donor.env is the blunter fallback).
  - **Ubuntu / Debian:** `sudo apt install nftables` if `nft` is missing (the iptables backend works without it). ufw
    is unaffected (Docker bypasses it anyway). Don't enable the distro's `nftables.service` just for this: its default
    config starts with `flush ruleset`, which would also wipe Docker's rules; our unit doesn't need it.
  - **Arch:** `sudo pacman -S nftables` if missing. Same as above about `nftables.service`.
  - No systemd: the script says so; run `/usr/local/sbin/donor-firewall apply` at boot before Docker.
- **Out of scope:** **Docker Desktop** (Linux, macOS, Windows) runs containers in its own VM; rules on the host don't
  see that traffic, and rules inside the VM don't persist. **Windows / WSL2:** the GPU path differs (`/dev/dxg`, no
  `/dev/kfd`), and WSL's NAT or mirrored networking puts the Windows host and the LAN behind addresses this script
  doesn't know about. **Rootless Docker / Podman:** container traffic leaves through slirp4netns / pasta as the
  user's own traffic, so `DOCKER-USER` and the bridge rules don't apply. For all of these: a native Linux install with
  the system Docker daemon, or no research access.
- **Known limits:** a LAN service the owner exposes to the internet (router port forward, UPnP) is reachable at their
  public IP like from anywhere else (hairpin NAT). A non-private subnet that the host routes to (a VPN with public
  addresses, a Tailscale subnet route with a public range) is reachable; private ones are blocked. If the host's
  addresses change, the boot unit catches up at the next boot (or run `install-firewall.sh` again).

## What still has to be tested on a real donor host (not possible on our boxes)

Our tests: [tests/RESULTS.md](tests/RESULTS.md). Still open, in this order, on the first real host:

1. `sudo ./firewall/install-firewall.sh` on a systemd host (Fedora, Ubuntu or Arch): unit enabled, then **reboot** and
   check `sudo ./firewall/donor-firewall.sh status` shows the rules again and the container came back with them
   (`./run.sh selftest` PASS).
2. `sudo systemctl restart docker`, then `./run.sh selftest` again (rules survive a Docker restart). With firewalld:
   `sudo firewall-cmd --reload`, selftest again.
3. ROCm image on an R9700: `./run.sh up`, `./run.sh selftest` (rocminfo shows gfx1201; the torch matmul passes) with the
   default `DONOR_SECCOMP=kit`. Then once with `DONOR_SECCOMP=default`: if ROCm works under Docker's plain default
   profile too, we can drop `seccomp-rocm.json`. Also a hipcc build (`torch.utils.cpp_extension`), since the image
   carries only the gfx1201 device libraries.
4. CUDA image on an NVIDIA host with the Container Toolkit: `./run.sh up` (`--gpus`), selftest PASS.
5. Tailscale end to end on our real tailnet: the policy saves (its tests pass), a `tag:donor` single-use key logs the
   box in, `ssh research@donor-<name>` works, `ssh` with a command (`ssh research@donor-x nvidia-smi`) gets the
   environment, and from the donor box `tailscale ping` / `nc` to our devices and to another donor fail.
6. A LAN with IPv6: the selftest's IPv6 row, and from inside `curl -6 https://...` fails (the network has no IPv6).
