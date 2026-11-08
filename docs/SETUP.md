# `rqah setup`: the donor's setup wizard

`client/setup.py` (stdlib only, Python 3.10+, no imports from the rest of the repo) looks at the machine, asks what
to donate, and writes the worker's configuration into one directory. It writes nothing outside that directory and
never runs sudo. The only network traffic comes from the two opt-in steps, the image pull and the worker's first
start. It runs on the **host**, not inside the worker container (it has to see the host's GPUs, RAM, disk and Docker).

```bash
python3 client/setup.py              # or `rqah setup` once the CLI hook-up is merged
python3 client/setup.py --check      # validate the current config against this machine; changes nothing
python3 client/setup.py --show       # what it detects and the defaults it would offer
```

## What it writes (default `~/.config/rad-quant-at-home`, or `--config-dir DIR`; the directory is 0700)

| file | mode | contents |
|---|---|---|
| `worker.env` | 0600 | every worker setting as `RQAH_*` plus the invite. Plain `KEY=VALUE` (no quotes, `$` or `#`), so `docker run --env-file` and compose read it the same way |
| `compose.yaml` | 0644 | `docker compose up -d` in that directory: the image, devices, caps, `env_file: worker.env`, the state volume, `container_name: rad-quant-at-home` |
| `docker-run.sh` | 0700 | the same as one `docker run` line, for hosts without the compose plugin |
| `setup.json` | 0600 | the answers (no invite) and the hardware seen; a re-run starts from it |

Files are replaced atomically. When a file changes, the previous version is kept as `NAME.prev` (with the same mode).
The compose file and the run line use the same named volume (`rqah-state`), so switching from one to the other keeps
the token and the cache.

Image: `ghcr.io/theherobrine422/rad-quant-at-home-worker:<VERSION>-rocm7.14-gfx1201` for AMD (gfx1201 only: R9700,
RX 9070 / 9070 XT), `:<VERSION>-cu128` for NVIDIA (compute capability 8.0+, driver 570+); `image` overrides it.
AMD gets `/dev/kfd`, `/dev/dri`, the host's numeric video / render gids and `seccomp=unconfined`. NVIDIA gets
`--gpus all` (compose: a device reservation) and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the worker's indices match
`nvidia-smi`. Both get `no-new-privileges`, `--cap-drop ALL`, a memory limit, a 15 min stop timeout and
`/etc/localtime` (read-only, so that the time windows use local time). Nothing else is mounted.

## Detection (read-only)

- **AMD:** the kfd topology (`/sys/class/kfd/kfd/topology/nodes/*/properties`: `gfx_target_version` → gfx1201,
  `drm_render_minor`), joined with `/sys/class/drm/card*` (VRAM from `mem_info_vram_total`, a connected connector =
  the card drives a display). The ROCm device index is the card's position among the kfd GPU nodes, so an iGPU shifts
  it (shown, not selectable). `rocminfo` is used for marketing names if it is installed; without it a small PCI-id
  table names the gfx1201 cards. amdgpu cards without a kfd node are listed as not usable.
- **NVIDIA:** `nvidia-smi --query-gpu=index,name,memory.total,compute_cap,display_active,driver_version,pci.bus_id`. If
  `nvidia-smi` is missing but there is an NVIDIA PCI display device, it says so.
- **Host:** `/proc/meminfo`, `/proc/cpuinfo`, free disk (statvfs) and file system (`/proc/mounts`) where the cache
  will live (the Docker root's `volumes/` for the named volume, or the chosen path), the time zone, SELinux enforcing.
- **Docker:** `docker --version` (Podman shim?), `docker info --format '{{json .}}'` (daemon reachable, permission,
  root dir, runtimes, rootless), `docker compose version`. NVIDIA Container Toolkit: `nvidia-ctk` /
  `nvidia-container-cli` / the runtime hook, an `nvidia` runtime, or a CDI spec. `/dev/kfd`, `/dev/dri/renderD*` and
  the video / render groups (from `/etc/group`).

## Questions and defaults

| setting | default | notes |
|---|---|---|
| queue URL | (required) | `https://`; plain `http://` only for localhost |
| invite | (none) | optional: without one the worker registers on probation (server-scored quantization jobs only, one at a time until a few results pass; evaluations later); an invite skips that. Read without echo (curses: not drawn; plain: getpass), or `--invite-file F` / `-` / `$RQAH_INVITE`. Never a flag value, never printed, stored only in `worker.env`. Enter keeps the stored one, `-` removes it (after registration the token is in the state volume and the invite is unused) |
| display name, leaderboard | `donor`, off | the server's display-name rules (kept in sync by the test) plus no `#` / `$`. The board credits by name (case-insensitive): use the same name on every machine to be credited together. A warning if you opt in with the default `donor` (you would share its row with everyone who kept it) |
| vendor | the one with the most usable VRAM | asked only when both have usable cards |
| cards | every usable card of that vendor | iGPUs, other gfx targets, cc < 8.0 and drivers < 570 are refused |
| VRAM per card | the card's size in GB, 1 less if it drives a display | the smallest of the selected cards; a warning for mixed models |
| allocator cap | off | advanced; for a card you also use while it runs |
| host RAM for jobs | 75% of RAM, at most RAM − 4 | 16 GB → 12, 32 → 24, 64 → 48 (the LM head needs 48) |
| container memory limit | RAM for jobs + 8, below the host's RAM | advanced; follows the RAM answer |
| state location | the Docker volume `rqah-state` | advanced; a host path must belong to uid 10001 (setup prints the `sudo install -d` line and does not run it); `:z` on SELinux |
| input cache | 200 GB on AMD cards ≥ 16 GB (evaluations), else 50 | bounded by the free disk minus 25 GB |
| speed caps, day / month budgets, reset day | 0 = unlimited, day 1 | day budgets are advanced; the reset day is asked only when there is a monthly budget |
| time windows | always | `HH:MM-HH:MM`, comma-separated, local time |
| on stop, container name, image | finish, `rad-quant-at-home`, default | advanced |
| update automatically | no | the hourly timer (above, "Updating"); the summary says which way updates happen |

The summary lists the jobs the declared caps will get (the README job table and docs/SMALL-CARDS.md: below 6 GB
nothing, 6-7 GB small Hessian groups, 8-11 GB layers up to K6 ("small-card mode"), 12 GB everything trunk, AMD 16 GB
evaluations ≤ 4.5 bpw, 32 GB ≤ 7 bpw, and the LM head with 48 GB of RAM), plus a note when declaring a bigger size
would add jobs (e.g. a 12 GB card that drives the display). **Errors** stop the write (`--force` overrides them):
an unusable or missing card, VRAM above the card, RAM above the host, a cache that doesn't fit the disk, no
`/dev/kfd` for AMD, no Container Toolkit for NVIDIA (when Docker is installed), < 6 GB of VRAM, bad formats.
**Warnings** don't: no Docker, no access to the daemon, Podman, a display card declared at full size, too little
RAM for all cards at once, a cache without reflinks for evaluations, running as root, a small data budget.

## After writing (each step asked; `--smoke`, `--pull`, `--start` with `--yes`)

1. **Local checks** (no network): `docker compose config --quiet`, then `docker image inspect` to see whether the
   image is there. If it is, the image runs once **offline** (`--network none`, no env file, only the GPU devices)
   to print the cards torch sees, and the result is compared with the selection.
2. **Pull** the image (opt-in, several GB).
3. **Start** the worker (opt-in): `docker compose up -d` (or `docker-run.sh`). On its first start the worker registers
   with the invite. Otherwise setup prints the command. A worker container of that name that is running is **drained**
   first (its jobs finish, or `--release` hands them back; below), then removed (never its volume) and created anew.

## Updating (`--update`, alias `rqah update`; 0.5)

```bash
python3 client/setup.py --update [--to X.Y.Z] [--yes] [--release] [--drain-timeout MIN] [--dry-run] [--config-dir DIR]
```

1. **Which release and digest:** `GET <queue>/api/v0/version` (no token; the queue in `setup.json`): the current
   release and, per stack, its image and `sha256` digest. Or `--to X.Y.Z`: the digest then comes from that release's
   `release.json` on GitHub (or `--digest`). The release must be `X.Y.Z` and the image
   `ghcr.io/theherobrine422/rad-quant-at-home-worker:X.Y.Z-<your stack>` (the repository is fixed in setup.py), so a
   queue can only point at releases we published. No digest: refused (`--no-digest-check` for an image you built
   yourself). An advertised release older than the pinned one is refused (a downgrade needs `--to`). A release whose
   manifest says `setup_min` newer than this setup.py asks for a `git pull` first. A config that overrides `image` is
   left alone.
2. **The plan**, then a confirmation (`--yes` skips it; without a terminal it needs `--yes`); `--dry-run` stops here.
3. `docker pull` the new tag, then `docker image inspect` its RepoDigests: the advertised digest must be there, else
   **refused** (the tag was moved or tampered with) and nothing else happens. A failed pull changes nothing either.
4. **Drain** the running container: `docker update --restart no` (its exit must not bring it back), `docker kill
   --signal TERM` (the worker finishes its running jobs and exits; with `on_stop` abandon it hands them back at once),
   wait up to `--drain-timeout` minutes (60), then a second SIGTERM (it hands the jobs back to the queue, not
   counted against them) and at last `docker stop`. `--release` sends both signals at once.
5. Rewrite `compose.yaml`, `docker-run.sh` and `setup.json` with the new pin `REPO:TAG@DIGEST` (`settings.version`,
   `settings.digest`; the previous files as `*.prev`); `worker.env` and the invite are unchanged.
6. `docker rm` the stopped container (the `rqah-state` volume or state directory stays: token, input cache, data
   budget counters, running-job records) and start it again from the new files, then check that it runs the new tag.

`rqah status` inside the container shows when an update is available, this command and the release notes. `rqah
update` inside the container refuses (it has to run on the host, where Docker is).

**Automatically** (the question "Update automatically", `--auto-update on|off`, default off; README "Updates and
versions" has the trade-off): on writes `rqah-update.service` and `rqah-update.timer` into the config directory and,
after you agree (`--yes` agrees), links them into your systemd user manager (`systemctl --user link`, `enable --now`:
the only files setup ever puts outside its directory, and only then). The timer runs `setup.py --update --auto` hourly:
it acts only while `auto_update` is on, only for a newer release with a digest, published at least 24 h ago, and lets
the running jobs finish (up to 6 h) before the restart; a lock keeps two runs apart. Off disables the timer. Without
systemd it prints the cron line instead. The units name this setup.py's path: move the checkout, run setup again.

## Re-running and `--check`

A re-run loads `setup.json` (and the invite from `worker.env`) and shows the current values as the defaults. Flags
and `--answers` override single values, and a changed vendor or card choice re-derives the values that follow from it
unless they are given too. Apply the changes by recreating the container (`docker compose up -d` does it; for the run
line, `docker stop`, `docker rm`, then `sh docker-run.sh`).

`--check` (exit 0 ok, 1 problems, 2 no config; `--json`) re-detects the hardware and validates the stored answers
against it: the same errors and warnings as the summary. It also flags missing files, `worker.env` / `setup.json`
readable by others, env lines that docker and compose would read differently (quotes, `$`, `#`), and files that
differ from what setup would write for `setup.json` (hand edits).

## Non-interactive

```bash
python3 client/setup.py --yes --server https://queue.example.org --invite-file invite.txt --gpus 0 --ram-gb 24 \
   --cache-gb 100 --window 22:00-08:00 --max-down-gb-month 300 --leaderboard on --name "Ada"
python3 client/setup.py --yes --answers answers.json < /dev/null         # keys as in setup.json "settings"
```

The order is: flags, then `--answers`, then the current config, then the machine's defaults. Without a terminal and
without `--yes` it exits 2. `--plain` forces plain prompts, `--curses` forces the form. The form is used when stdin
and stdout are terminals, `TERM` is set and the terminal is at least 80 x 24. Keys: ↑↓ / j k, Enter edit, Space
toggle, `a` advanced, `s` summary, `w` write, `q` quit. Over SSH either one works. Ctrl-C quits without writing.

## Screens (from `tests/setup_test.py`, fake machines)

The form (2 × R9700, 128 GB):

```
 rad-quant-at-home setup 0.5.0 · editing ~/.config/rad-quant-at-home
 CPU     Fake CPU 9000 (16 threads)   RAM 128 GB (100 GB available now)
 GPU AMD    #0  AMD Radeon AI PRO R9700             32 GB  gfx1201          ok
 GPU AMD    #1  AMD Radeon AI PRO R9700             32 GB  gfx1201          ok
 Docker  29.8.2, compose 5.6.0, data in /var/lib/docker

 Account
 › Queue URL                        https://queue.example.org
   Invite code                      (stored, hidden)
   Display name                     Ada
   On the public leaderboard        no
 GPUs
   Cards to donate                  0,1
   VRAM per card (GB)               32
 Memory and disk
   Host RAM for jobs (GB)           96
   Input cache (GB)                 200
 Network
   Download speed cap (Mbit/s)      0
   Upload speed cap (Mbit/s)        0
   Download budget per month (GB)   0
   Upload budget per month (GB)     0
 Schedule
   Run only during                  always

 the https:// URL of the queue (from us)
 ↑↓ move  Enter edit  Space toggle  [a] show advanced  [s] summary & save  [q] quit
```

The summary of the plain prompts (RTX 4070 driving the display, 32 GB):

```
What will be written (in ~/.config/rad-quant-at-home):
  worker.env      the settings below + your invite  (mode 0600)
  compose.yaml    image ghcr.io/theherobrine422/rad-quant-at-home-worker:0.5.0-cu128
  docker-run.sh   the same as one docker run line
  setup.json      these answers (no invite), for the next run

What the worker may use:
  queue           https://queue.example.org
  cards           0: NVIDIA GeForce RTX 4070
  VRAM            11 GB per card; reported as rtx4070
  host RAM        24 GB for all jobs; container limit 31 GB
  disk            input cache ≤ 50 GB in the Docker volume rqah-state
  network         down unlimited, up unlimited; budget down no cap, up no cap
  when            22:00-08:00 (America/Chicago); on stop: finish the running jobs
  leaderboard     listed as 'Grace' (with every opted-in machine of that name)

Jobs it will get (small-card mode, docs/SMALL-CARDS.md):
  + 27B layers up to K6 (quant_layer), and every Hessian group but down_proj above K6 (quant_tensor)
  - no mix evaluations (the CUDA image has quantization jobs only)
  (declaring 12 GB would add more jobs: only if nothing else, e.g. your desktop, uses the card)

It runs only the job types built into the image, makes only outgoing HTTPS requests to the queue,
opens no port, and reads and writes only its state volume. Your token stays in that volume.

Write these files? [Y/n]
```

## Tests

`tests/setup_test.py [WORKDIR]` (no GPU, no network, ~20 s; writes the screens to `WORKDIR/screens/`): fake roots
for a 4 / 8 / 12 (display and headless) / 16 / 32 GB matrix, 2 × R9700 plus an iGPU, gfx1100, cc 6.1 plus an old
driver, amdgpu without KFD, no Docker, Docker denied / down / Podman, NVIDIA without the toolkit; the files' contents
and modes, `docker compose config` on them (if the plugin is installed), re-runs, `--check`, the validation rules,
the plain prompts (scripted and on a pseudo-terminal), the curses form (rendered, and driven on a pseudo-terminal
with a hidden invite), the smoke step's commands, nothing written outside the config dir, Python 3.10 grammar,
stdlib only.

`tests/update_test.py` runs `--update` and `--start` against a simulated Docker (the order of the steps, the drain and
its timeout, `--release`, `--dry-run`, refusals); `tests/update_docker_test.py` runs the drain against a real container.

Not tested: real AMD hardware (the kfd / drm layout is the documented one), the smoke step against a real image (none
is published yet), `--update` / `--start` with a real worker image and GPU (the drain is tested with a real container
running the worker on noop jobs).
