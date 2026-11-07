# rad-quant-at-home: donate GPU time to the rad-exl3 quantizations (v0.4)

> **Status: pre-release.** There is no public queue server yet and the job types still change; please read, comment and
> open issues, but don't run a worker against anything until a release is announced.

**rad-quant-at-home** (short: `rqah`) is a Folding@home-style job system that lets people **donate compute to the
[rad-exl3-mix](https://github.com/TheHeroBrine422/rad-exl3-mix) project**: the EXL3 quantizations of Qwen3.8-27B
(the EXL3-V3 size ladder) and the search over mixes of it. It is **not a quantizer**. The quantizer, its settings and
the research live in rad-exl3-mix; this repository is the plumbing that ships that quantizer's jobs to donors' GPUs,
brings the results back, and checks them before they count.

Donors run a worker (a container); it pulls jobs from our queue, runs them and uploads the results. **All code is here
and open (Apache-2.0); every job type is built into the worker.** The server only names a job type and sends
parameters that must match that type's schema; it never sends code.

Names: the project and its images are `rad-quant-at-home` (`rad-quant-at-home-server`, `rad-quant-at-home-worker`),
the command line is `rad-quant-at-home` with the alias `rqah`, the environment is `RQAH_*`, and points are "lumens".
0.3 still reads the 0.2 variables `RADQUANT_*` (with a deprecation note); the v0 names `RXMQ_*` / `RXMW_*` are gone.

Status: built and tested on CPU with fakes standing in for the GPU, and with real quantizer runs on an RTX 4070
(TESTS.md). **Not deployed**: nothing listens beyond localhost, no image is published. Open: publishing the images on
GHCR (public packages; pulling needs no login), the server host and its volumes, the worker image builds (CUDA: its
steps were built and run on the 4070 without Docker, `tests/cuda_rootfs.sh`, but the Docker build needs a host with
≥ 8 GB of free RAM per compile job; ROCm: with radiance inside), `docker run --gpus` on an NVIDIA host, and one real
R9700 job of each type.

What is new in 0.4:
- **Bandwidth limits for donors** on data caps or slow links: rate caps (`--max-down-mbps`, `--max-up-mbps`) and day /
  month data budgets (`--max-down-gb-day` ...). The server offers a worker only jobs whose downloads and uploads fit
  what is left ("Data caps" below).
- **Scheduling** (docs/SCHED.md): ladder levels get a default priority by K's granularity (integer K first, then
  halves, quarters, eighths), and `admin.py reprioritize` changes it by level pattern. Inside a priority class, whole
  levels finish first, and a worker gets the jobs whose inputs it already caches. A higher-priority job never starves
  behind lower ones for longer than `--class-wait-s`.

What is new in 0.3:
- **Verification by score** (docs/VERIFY.md): a quant result is accepted when its codes, decoded on the server with
  the exact integer decode, have the Hessian-weighted error expected for that tensor at that K. No re-runs (except one
  anchor per tensor), any GPU vendor or driver, best of N.
- **Small cards**: a 27B layer job needs 8 GB of VRAM (through K6, measured), and `quant_tensor` jobs (one Hessian group
  of a layer) let 6 GB cards take everything but down_proj (docs/SMALL-CARDS.md, TESTS.md section 11).
- **NVIDIA donors**: a CUDA worker image (quantization jobs).

## For donors

**What it is.** Your GPU helps quantize Qwen3.8-27B into the EXL3-V3 size ladder of rad-exl3-mix and score candidate
mixes of it against reference outputs. The results become openly published models; donors get points ("lumens") and,
opt-in, credit in the model cards.

**What you need.** Linux, Docker, and one of:
- **AMD** (all job types): the amdgpu kernel driver (no ROCm install), a Radeon AI PRO R9700 (32 GB) or another gfx1201
  card. 16 GB cards also get the smaller evaluation jobs.
- **NVIDIA** (quantization jobs): an RTX 30xx / 40xx / 50xx or newer data-center card (compute capability 8.0+), the
  NVIDIA driver (570 or newer) and the NVIDIA Container Toolkit. 8 GB is enough for most jobs, 12 GB for all of them.

Host RAM: 16 GB or more (a 27B layer job peaks at ~10 GB on an 8 GB card, less on bigger ones; the LM-head job needs
48 GB and goes only to workers that declare that much). Disk: the cache you allow (`--cache-gb`; 50 GB is plenty
for quantization, 200 GB recommended for evaluation jobs, where XFS or btrfs lets mixes be assembled with reflinks)
plus ~20 GB.

**Set it up with the wizard** (optional): `python3 client/setup.py` (alias `rqah setup`; one file, Python 3.10+, no
dependencies) looks at your GPUs, RAM, disk and Docker, asks what to donate (cards, VRAM / RAM / cache caps,
bandwidth limits and data budgets, time windows, leaderboard, the invite without echoing it), shows what the worker
will be allowed to do, and writes `worker.env` (mode 0600), `compose.yaml` and `docker-run.sh` for the right image into
`~/.config/rad-quant-at-home`. Re-running it edits that config, `--check` validates it, and `--yes` with flags or
`--answers FILE` runs it without questions. It needs no sudo, and it uses the network only if you ask it to pull the
image or start the worker. Details: [docs/SETUP.md](docs/SETUP.md). Or write the line yourself:

**Run it** (one line; the invite code and the queue URL come from us). AMD:

```bash
docker run -d --name rad-quant-at-home --restart unless-stopped --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined --security-opt no-new-privileges --cap-drop ALL --memory 40g --stop-timeout 900 \
  -v rqah-state:/var/lib/rqah -v /etc/localtime:/etc/localtime:ro \
  -e RQAH_SERVER=https://<queue-url> -e RQAH_INVITE=<invite-code> -e RQAH_NAME=<name-for-the-leaderboard> \
  ghcr.io/theherobrine422/rad-quant-at-home-worker:0.4.0-rocm7.14-gfx1201 --gpu-name r9700 --gpus 0 --vram-gb 32 --ram-gb 32 --cache-gb 200
```

NVIDIA (`--gpus all` hands the cards to the container; the worker's own `--gpus 0` picks among them):

```bash
docker run -d --name rad-quant-at-home --restart unless-stopped --gpus all \
  --security-opt no-new-privileges --cap-drop ALL --memory 24g --stop-timeout 900 \
  -v rqah-state:/var/lib/rqah -v /etc/localtime:/etc/localtime:ro \
  -e RQAH_SERVER=https://<queue-url> -e RQAH_INVITE=<invite-code> -e RQAH_NAME=<name-for-the-leaderboard> \
  ghcr.io/theherobrine422/rad-quant-at-home-worker:0.4.0-cu128 --gpu-name rtx4070 --gpus 0 --vram-gb 12 --ram-gb 16 --cache-gb 50
```

Declare the VRAM the worker may really use: a card that also drives your desktop has ~1 GB less (e.g. `--vram-gb 11`
on a 12 GB card), and `--vram-cap-gb 7` caps the worker's allocator below that if you game or work on the same card.

Several cards in one box: `--gpus 0,1,2,3` (one job per card at once; jobs that need two cards get them together)
and `--ram-gb` for the RAM all of them may use together. Only at night: add `--window 22:00-08:00` (repeatable, your
local time). `docker/worker/compose.yaml` holds the same settings for compose users (AMD).

**Data caps.** If your internet plan has a monthly allowance, or the link is slow, tell the worker:
`--max-down-gb-month 300 --max-up-gb-month 100` (also `-day`; 1 GB = 10^9 bytes) counts the data its jobs move and
stops taking new jobs when the allowance is used up. The month starts again on `--budget-reset-day` (1-28, e.g. your
ISP's billing day), the day at local midnight. The counters live in the state volume, so restarts and new containers
keep them. `--max-down-mbps 50 --max-up-mbps 10` caps the rates (megabits per second, all jobs together), and it
applies to resumed transfers too. With a budget set, the server sends only jobs whose downloads (inputs you don't
already have) and expected uploads fit what is left, and it prefers jobs whose inputs your worker already holds. When
the allowance runs out, the running jobs finish and nothing new starts until the reset. `rqah status` shows what is
left and when it resets. The worker counts payload bytes (inputs and results), not TCP/TLS overhead (a few percent),
so leave your ISP's cap some margin. Typical sizes are in "Data per job": a 27B layer job downloads ~1.5 GB the
first time and uploads 50-300 MB.

**Stop it.** `docker stop rad-quant-at-home` finishes the running jobs (up to 15 min) and exits; `docker kill -s TERM
rad-quant-at-home` while that waits hands them back at once. `docker start rad-quant-at-home` resumes. To leave for
good: `docker rm -f rad-quant-at-home && docker volume rm rqah-state` (the volume holds your token and the cache).

**What runs on your machine.** Only the job types built into the image, all in this repository: quantizing one layer,
one part of a layer or the LM head (`quant_layer`, `quant_tensor`, `quant_head`: rad-exl3-mix's `quantize/ourq.py`,
copied into the image at a pinned commit), scoring a candidate mix (`eval_mix`, AMD image: the mix is assembled from
cached pieces, then radiance runs it over a public reference), and for donors we know personally and flag "trusted",
recording reference outputs of the bf16 model (`golden_record`). The worker only makes outgoing HTTPS requests to the
queue; it opens no port, runs no shell command and no code from the server, reads no file outside its volume and
uploads only each job's declared outputs.

**Data.** You download public data only: the published Qwen3.8-27B weights (cut per layer), the published calibration
Hessians, our quantized ladder levels, and reference outputs ("goldens") recorded on public text (the challenge's
published calib corpus and the public xcorp corpora). **No private data ever**: the server refuses any golden or
corpus not marked public, and anything derived from our own (private) agent traces or session logs is refused by name
on top of that (`jobs/golden_common.py`). What you upload: the job's results (quantized shards, per-position scores,
reference rows) and a small `job.json` (image, GPU model, timings). Nothing about your machine beyond the caps you
declare and the GPU name.

**Points.** Lumens are R9700 card-minutes of verified work (+10% for a result back within twice the reference time).
Quantization results are verified by their score: the server decodes your codes and checks their error (docs/VERIFY.md),
so any GPU and any driver is fine. Other results: a random 3-5% of every donor's results (and your first of each kind)
are re-run on our own hardware (evaluations: ~8%, a few of their documents). An evaluation far off what we predicted for
that mix is refused, or re-run in full before it earns. A result that fails its check puts the worker in quarantine
and voids its unconfirmed points.

**Check on it.** `docker exec rad-quant-at-home rqah status` shows your workers, their running jobs, your lumens and
the queue; `rqah jobs` lists recent jobs and whether each is verified yet, `rqah points` the ledger, `rqah logs -f`
the server's log of your workers (section "Donor CLI"). Add `--json` to any of them for scripts.

**Leaderboard.** The public status page lists only donors who opt in, by display name: start the worker with `-e
RQAH_LEADERBOARD=1` (the name is `RQAH_NAME`), or later `docker exec rad-quant-at-home rqah config set
--leaderboard on --display-name "Your Name"`. Off again with `--leaderboard off`. The board credits by display
name: **use the same name on every machine** and they share one row (case does not matter). A name is a label, not
an account: nothing about verification depends on it, and anyone can show any name.

## Research access (trusted volunteers only)

Separate from the job queue, and optional. **The queue needs no trust**: it runs only the job types built into the
worker, and every result is verified by its score before it counts. **Research access is different: we get a shell
on your machine** (in a locked-down container we reach over Tailscale) and run our own experiments there. It is only
for people who know us personally. What we can and cannot do, the protections (a firewall that keeps the container
off your LAN, no root, limits) and how to stop it: [research-kit/VOLUNTEER.md](research-kit/VOLUNTEER.md). The kit
itself: [research-kit/](research-kit/README.md).

## Architecture

```
            admin.py (server host: datapack, ladder-add, golden-add, enqueue-*, batch-*, check, trust, points)
                 |
                 v
   data root ---------- server.py (stdlib HTTP + SQLite WAL) -------- results/<job>/   -> export -> ourq layout
   models/<m>@<rev>/     lease / heartbeat / upload / complete / fail     spool/<job>/<lease>/ (uploads in flight)
   goldens/ corpora/     GET /data/<file> (HTTP ranges)                   server/verify.py: spot checks, lumens
   manifest.json         /api/v0/search/batch (the search's token)         server/batch.py: eval_mix batches
                 ^                     ^
                 |  sha256-checked     |  bearer token per worker
                 |  downloads          |
            worker.py  (donor: lease -> fetch into a capped cache -> run built-in job on its cards -> upload -> complete)
```

- **Server** (`server/server.py`, Python stdlib only): workers register with an invite code and get a token (only
  its sha256 is stored). The job table holds type, params, inputs, requirements, state, lease owner / expiry,
  attempts, result, spot-check state. A reaper thread puts jobs whose leases missed their heartbeat back in the queue
  (0.5: also leases past their deadline, `--lease-cap-x` 3 × the job type's reference runtime and at least
  `--lease-cap-min-s` 2 h after the lease was issued, however often they heartbeat; abandoning and leasing the job again
  does not restart that clock) and deletes the spooled uploads of leases that can no longer complete (an expired lease's upload is kept while its
  job is still open: the late-result rule); a job fails for good after `max_attempts` (default 5). Completion is
  idempotent: once a job is accepted, any later completion or upload for it is acknowledged and ignored. A late
  result from an expired lease is still accepted if nobody has finished the job yet, and the newer lease is then
  told to stop. Files are served with HTTP Range support, only to a worker holding an active lease on a job that
  lists the file, and goldens / corpora only while their privacy marker says public. No HTTP endpoint changes
  anything as admin; the search's batch endpoint (its own token) can only enqueue eval_mix jobs on public goldens.
- **Scheduler** (0.2, 0.4: docs/SCHED.md): every job type declares what one job needs (`requires`: GPU, cards, VRAM per card, host RAM;
  by model size or bpw), every worker what it has (`--gpus` / `--cards`, `--vram-gb`, `--ram-gb`, `--cache-gb`). A job
  is leased only to a worker whose declaration covers it and whose cards and RAM are free now (the server counts its
  active leases). Gang jobs (more than one card) on a busy multi-card worker: the server reserves the worker for the
  job (it gets nothing else while its cards drain), then leases it the gang job; a worker that can run it at once
  still takes it. Trusted-only types go to workers the operator flagged `trusted`, spot-check re-runs to `verifier`
  workers (ours), never to the original donor; quarantined workers get nothing. 0.4: priority classes strictly in
  order, whole levels first inside a class, then the jobs whose inputs the worker caches (it reports its cache when
  leasing). It filters by the worker's declared data budget (each job type declares its expected upload,
  `upload_bytes`), and a higher-class job that a busy worker keeps passing over drains that worker after
  `--class-wait-s`. A worker that failed, rejected or let expire a lease on a job is not offered that job again for
  `--refail-s` (default 1 h) after that lease ended, so a donor who switches a machine off and on, or hits a transient
  error, is not benched from it for good (`--refail-s 0`: only after the operator requeues it; `admin.py requeue` also
  gives a still-queued job this fresh chance; verifier workers are never excluded).
- **Admin** (`server/admin.py`): data (`datapack`, `ladder-add`, `golden-add`, `corpus-add`, `manifest`), jobs
  (`enqueue-level`, `enqueue-golden`, `batch-submit` / `-status` / `-collect`, `list`, `queue`, `reprioritize`, `events`,
  `requeue`, `cancel`),
  verification (`check`, `spot`, `quarantine`, `unquarantine`, `exclude-donor`, `batch-adopt`), workers (`invite`, `add-worker`, `workers`, `trust`,
  `verifier`, `disable-worker`), donors (`invite --personal`, `donors`, `donor ID --hide`), `points`,
  `status-page`, `export` (the accepted shards of a level in `quantize/ourq.py`'s layout, so knap4 and the search read
  them unchanged). RUNBOOK.md walks through them.
- **Worker** (`worker/worker.py`): a loop of lease → fetch → run → upload → complete per job, with a heartbeat thread
  per job. One process serves all the box's cards: each job runs in its own thread, its processes see only its cards
  (`ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`, `--gpu-env`), all jobs share one input cache and one RAM budget.
  Every step can be resumed: each running job and its phase are in `STATE/current/<job>-<lease>.json`, downloads
  resume with a Range request and are renamed only once the sha256 matches, uploads resume from the server's offset,
  ourq skips a finished shard. A killed worker restarted on the same state dir picks up its jobs if the leases are
  still alive and drops them otherwise. The input cache is content-addressed (sha256) with a byte cap (`--cache-gb`):
  least-recently-used blobs not pinned by a running job are evicted, a blob two jobs need is downloaded once, and the
  server never leases a job whose inputs exceed the cap. Configured by flags; every flag has a `RQAH_*` variable.
  - Stopping: SIGTERM/SIGINT or `--stop-file` finishes the current jobs, then exits. A second signal, or
    `--on-stop abandon`, hands them back immediately (the abandoned attempts are not counted).
  - `--window HH:MM-HH:MM` (repeatable, local time): no new leases outside the windows.
  - `--gpu-lock FILE` (`{card}` = one lock per card, e.g. `/var/lock/gpu-card{card}.lock` on a shared box): flock around GPU jobs.
  - `--vram-cap-gb`: caps torch's allocator in job processes (a shared card).
  - Bandwidth (0.4, `worker/bandwidth.py`): token-bucket rate caps shared by all jobs; data budgets per calendar day
    and budget month in `STATE/bandwidth.json`. Running jobs reserve what they still have to move.
- **Donor API and CLI** (`server/donor.py`, `client/rqah.py`): a donor reads their own workers, jobs, lumens and
  log with any one of their worker tokens, and sets their public profile (section "Donor CLI").
- **Status page** (`server/statuspage.py`): one static HTML file, rebuilt every minute from the database and served
  at `/` without a login (section "Public status page").
- **Checker** (`checker/`): verifies returned quant shards without trusting the worker (below); its exact decode
  (`checker/exl3dec.py`) is also what the score verifier (`server/score.py`, docs/VERIFY.md) runs.
- **Server container** (`docker/server/`): a slim Python image (stdlib server + numpy for the checker), non-root
  (uid 10001), read-only root filesystem, configured by `RQAH_*` variables, `/healthz` healthcheck, compose file.
- **Worker containers** (`docker/`, built by `docker/build.sh` from one commit of this repo plus the quantizer paths of
  one rad-exl3-mix commit): `Dockerfile.rocm` (gfx1201: exllamav3 `a3bfbfd0` + rad-exl3-mix's patch series, the side
  extensions precompiled, radiance from our fork's image, transformers for the staged golden recorder; not built yet)
  and `Dockerfile.cuda` (CUDA 12.8, torch 2.11.0+cu128, the same quantizer; quantization jobs only; its steps built
  and run on an RTX 4070 without Docker, TESTS.md section 12).

## Job types (`jobs/`)

| type | does | inputs | outputs | requires (one job) |
|---|---|---|---|---|
| `quant_layer` | `ourq.py --layers L-L` (replicate mode: `--save-wq` + per-row error claim) | `weights/LNN.safetensors`, the layer's 4 input-Hessian files, `config.json`, optional H_out files | `q-LNN.safetensors`, `proxy.jsonl`, `stamp.json`, `job.json`, `ourq.log` (replicate mode: + `rowerr.npz`) | 27B: 1 card, 8 GB VRAM up to K6 (12 GB above), 12 GB RAM; small models 4 / 8 |
| `quant_tensor` (0.3) | one Hessian group of a layer: `ourq.py --layers L-L --only <group>` | `weights/LNN-gG.safetensors` (`datapack --groups`), the group's Hessian, `config.json` | `q-LNN-gG.safetensors` + the same small files | 27B: down_proj group 8 GB VRAM up to K6 (12 GB above), 12 GB RAM; the others 6 GB, 4 GB RAM |
| `quant_head` | `ourq.py --only-head` (replicate: `--save-wq`) | `weights/head.safetensors`, `lm_head.input` Hessian, `config.json` | `q-head.safetensors` + the same small files | 27B: 1 card, 32 GB, **48 GB RAM** (peak ~22 GB); small 8 / 16 |
| `eval_mix` | assemble a candidate mix from cached ladder pieces, `radiance --kld-ref` | the base level's skeleton, one piece per weight, the golden (public) | `rows.f32` (`--kld-out` FILE.rows), `kld.json`, `job.json` | ≤ 4.5 bpw: 1 card 16 GB, 16 GB RAM; ≤ 7 bpw: 32 GB, 24 GB RAM; above: 2 cards (TP2) |
| `golden_record` | bf16 reference rows of a public corpus chunk: `radiance --kld-record --kld-top N` (TP2) or `eval/gold27/mkgold27.py` (staged, one card) | `bf16.rad` or the HF checkpoint, the chunk (public) | `kld.json`, `tokens.i32`, `top.bin`, `rows.f32`, `gen.u8`, `job.json` | **trusted donors only**; tp2: 2 cards × 32 GB, 64 GB RAM; staged: 1 card, 48 GB RAM |
| `noop` | sleeps, reads its inputs, writes a deterministic file | `test/*` | `out.bin` | what its parameters say (tests) |

Model size comes from `models/<m>@<rev>/model.json` (`n_params`, written by `datapack` or `model-info`); 8 G
parameters and up is the 27B class. bpw = the assembled container's bytes × 8 / `n_params` (the real-bpw rule).

Parameters (`quant_layer`): `model`, `revision`, `arch` (`qwen3_5`), `layer`, `K` (multiple of 1/8, 0.125–8),
`codebook` (`mul1|i8|i4|i8c`), `level` (a name), `settings` (`sigma_reg`, `sigma_reg_in`, `sigma_reg_out`,
`scaled_hessian`, `qtf2`, `hout` snapshot id, `seed_offset`), `stamp`, the settings stamp: a sha256 prefix of
everything except the layer, and (0.3) `verify`: `replicate` (default; the 0.2 checker and spot checks) or `score`
(docs/VERIFY.md: the server scores the codes; the job skips `--save-wq` and the per-row claim). `quant_tensor` adds
`group` (the index of the Hessian group in the layer plan). The worker recomputes the stamp and refuses a mismatch, so every job of a level
provably carries the same settings. `job.json` records the settings stamp, ourq's code stamp, the image id, the GPU
and the timings.

The per-row claim exists because ourq's own `wq/<key>.rowerr.npz` is computed on the weight `quantize_exl3` has
regularized in place (the fix `proxy2b/exp/w04-blockgain/rowerr.patch` is not merged), so the job computes it itself
(`jobs/rowerr_tool.py`) from ourq's dequantized weights. Those stay local and are deleted before upload.

**Versions.** The server and the worker each derive a job's inputs and outputs from their own copy of `jobs/`, so
every type has a `VERSION`, bumped on any change of parameters, inputs, outputs or meaning. A worker declares the
versions it runs, the server leases only matching jobs, and dedupe keys include the version. Deploy order: server
first, then the new image. 0.2: `noop` 2 (requirement parameters), `eval_mix` 1 (implemented), `golden_record` 1 (new);
the quant types stay at 2 (only their declared requirements changed, which the server computes). 0.3: `quant_layer` and
`quant_head` 3 (the `verify` parameter; score mode uploads no `rowerr.npz`; ourq runs with the CPU refit fallback),
`quant_tensor` 1 (new). 0.5: `eval_mix` 2 (`docs` / `sub_rows`: a spot check of a few documents).

### eval_mix: the search's candidates, assembled on the donor

- **Ladder pieces** (`common/radfile.py`, `admin.py ladder-add MODEL REV LEVEL FILE.rad`): each ladder level's `.rad`
  container is cut once into `models/<m>@<rev>/rad/<level>/skeleton.radh` (the level's bytes before its data: header,
  strings, metadata, tables, vocab) and `t/<weight>.radw`, one piece per weight (its directory entry, encoding,
  planes and data; 4 KiB-aligned). A piece depends only on the weight, so weights that are the same in two levels
  (norms, embeddings) are one blob.
- **A candidate** is a base level plus a sorted list of `[weight, level]` pairs (and optional `[weight, act mode]`
  pairs: `f16`, `a8t`, `a8g<N>`). Its inputs are the base's skeleton, the named pieces, the base's pieces for every
  other weight and the golden. The donor's content-addressed cache keeps all of them across candidates, so a new
  candidate downloads only the pieces it has not seen (TESTS.md: the second candidate of a batch fetched exactly its
  two new pieces), up to the cache cap (LRU).
- **Assembly** is a port of `mix/radmix.cpp` from pieces: the same layout, string interning, encoding deduplication
  and `exl3.act.*` metadata, byte for byte what radmix writes from the whole files (`tests/radmix_port.py`, against
  the real radmix built from our radiance headers, 17/17). On XFS / btrfs the data is reflinked from the cache (no
  copy, no extra disk); elsewhere copied. The container is deleted after the run. The donor reports its sha256; the
  server checks the size, and a spot-check re-run must reproduce the hash.
- **Scoring**: `radiance --model MIX --tp <cards> --kv-cache-dtype bf16 --max-model-len L --max-num-seqs 1
  --host-headroom-mib 8192 --kld-ref GOLDEN --kld-out kld.json`; the upload is radiance's `kld.json.rows` as
  `rows.f32` (`[rows, 4]` f32 {kl, nll_q, nll_p, top1}, reference row order; `eval/cmp115.py` and `mix/paired.py` read
  column 0).
- **The server's check at completion** (stdlib, < 1 s per 1M rows): column `nll_p` equals the golden's own
  `rows.f32[:, 0]` bit for bit (radiance copies it), so the right golden was scored, every row present and in order;
  every KL finite and ≥ 0 (to rounding), top1 ∈ {0, 1}, the report's mean and the result's `kld_mean` /
  `container_sha256` equal to the rows' / `job.json`'s.
- **Plausibility** (0.5, #4): a donor can write rows that pass those checks without running the model, so they do
  not make a result true. The search sends its knapsack's predicted mean KL with each candidate (`pred_kld`; it stays
  on the server, workers never see it). A result above 3× the prediction (+0.001) is refused at completion (after two
  donors' refusals the job goes to our verifiers: the prediction was off, not both donors); one below 0.5× is
  accepted as **suspect**, earns nothing yet and is re-run in full on a verifier. The search hunts mixes that beat
  the prediction by a few to ~20%, not by half. A batch may set `"band": {"worse", "better"}` within 1.5–10× /
  0.1–0.9×.
- **Spot checks of a few documents**: `--eval-spot-rate` (8%) of each donor's evaluations, and its first one, are
  re-scored on a verifier on `--eval-spot-docs` (3) random documents of the golden: the verifier cuts a golden of
  just those documents (`golden_common.sub_golden`; a document scores the same alone) and its rows must match the
  donor's rows of those documents (per row as below, and each document's mean KL within 0.5%). A faker who did not
  run the model is caught on whichever documents are drawn.
- **Adoption** (0.5, #4): anything the search acts on gets a full trusted re-run first: `POST .../batch/ID/adopt
  {"candidates": [...]}` (`searchq.py adopt ID C... [--wait]`, `admin.py batch-adopt`). Status reports each
  candidate's `trust`: `confirmed` (a verifier ran or fully re-ran it and it matched), `rerun-pending`, `spot-pending`
  / `spot-pass`, `unverified`; act on an adopted candidate only once it is `confirmed`.
- **Batches** (`server/batch.py`): the search posts ~100 candidates at once (`POST /api/v0/search/batch` with the
  search token, `client/searchq.py`, or `admin.py batch-submit SPEC.json`), polls `GET .../batch/ID` and fetches each
  candidate's rows (`GET .../batch/ID/rows/CAND`, `searchq.py collect`). The server checks the golden is public,
  every piece exists, computes each candidate's exact assembled size (so its card class) and enqueues one job per
  candidate; a candidate identical to an earlier one (same base, mix, act, golden, settings) reuses that job's result.
  A result invalidated after a quarantine shows as `invalidated` in the status, and `collect` deletes its stale rows.

### golden_record: the big-box tier (trusted donors only)

The bf16 27B reference rows of one chunk of a public corpus (`admin.py corpus-add NAME FILE.jsonl --public
--chunk-docs N`, `enqueue-golden --method tp2|staged`). TP2 runs our radiance on a 2-card worker (a gang job), staged
runs `eval/gold27/mkgold27.py` on one card from the HF checkpoint (`datapack --hf`; `--bf16-rad` for TP2). Only for
workers the operator flagged `trusted` (`admin.py trust ID`): long jobs, not cheaply re-checked. The server's check
at completion: the layout's sizes, the documents and their sources and radiance's corpus hash (`corpus_fnv1a64`,
radiance's FNV-1a with its own offset basis) against our chunk, and on ~256 random rows: the head sorted and finite,
head + tail = 1 within 1e-4, the true next token's log p = −nll. Spot checks re-record the chunk on our hardware:
tokens identical, per-row nll mean |Δ| ≤ 0.02, the same argmax on ≥ 98% of sampled rows (another kernel path moves
nll by ~1e-2: `eval/gold27/README.md`).

## Security model

**What the server can make a worker do:**
- Run one of the job types built into the worker's image, with parameters that pass that type's strict schema
  (unknown keys, wrong types and out-of-range values are rejected) and the type's extra checks (settings stamp,
  sorted mixes, act modes).
- Download files from the configured server URL only. File names must be relative manifest paths of a fixed form
  (no `..`), and they must be exactly the files the job type requires for those parameters (plus the optional
  files it allows). Each file is checked against its sha256 before use.
- Upload only the output files the job type defines, from the job's own output directory, each under a size cap.
- Use the cards, RAM and disk the donor declared: the worker checks a leased job's requirements against what it has
  free and hands back what does not fit.

**What it cannot do:**
- Send code, shell commands or argv. ourq's and radiance's command lines are built by `jobs/` from validated values
  and run without a shell.
- Make the worker read or upload any other local file, point it at another host, run it outside its time windows,
  or make it open a port: the worker only makes outgoing requests (the job and lease ids the server sends are checked
  before they name any path).

Tested against a fake malicious server (TESTS.md): unknown types, extra parameters, path traversal, inputs the
type doesn't allow, a malformed eval_mix, a bad settings stamp, a job id that is a path, an empty lease id, a job id carrying a newline and a terminal escape. All were refused, with no download attempted.

**What a worker can do to the server:** lease and finish jobs (as many at once as its slots, at most
`--max-leases`), read only the files its own active leases list, upload only the declared outputs of its own leases
within the caps (chunks ≤ 64 MB, JSON bodies ≤ 1 MB), declared by a non-negative `Content-Length`; a short body is
truncated back to the acknowledged offset. A result is accepted only if its sha256 and size match and the
type's structural check passes. Tokens are per worker, stored hashed, and can be disabled
(`admin.py disable-worker ID`); a quarantined worker can no longer complete anything. With its token a donor reads
its own donor's workers, jobs, lumens and events (`/api/v0/me...`), never another donor's, and writes only its
display name and its leaderboard opt-in.

**What anyone can read without a token:** `/healthz` and the status page at `/`. The page is a prebuilt file: no
request reaches the database, and nothing on it can be written. It holds only aggregate counts, level and public
corpus names, and opted-in display names (section "Public status page").

**Data rules (AGENTS.md):** donors only ever get public data. Goldens and corpora carry a privacy marker
(`golden.json` / `corpus.json`, written by `golden-add` / `corpus-add`, which refuse to add anything not `--public`).
A golden or corpus without a marker, marked otherwise, or with any document whose source is one of ours
(`agent-trace…`, `pi-session…`, `traces/`, `private/`, `user/`) is refused at four points: `golden-add` /
`corpus-add`, `manifest` (its files are never listed, so never served), job submission (`batch-submit`,
`enqueue-golden`), and every data request (the marker is re-read per request). The worker re-checks the marker before
it runs. The user's traces stay on our machines; trace-based evals run on our cards.

**Not covered yet:** TLS (the server speaks plain HTTP: put it behind a TLS terminator such as Caddy, or a tunnel),
rate limiting, sandboxing beyond the container (the worker image runs as uid 10001 with `--cap-drop ALL` and
`no-new-privileges`, but needs `seccomp=unconfined` for ROCm's NUMA calls and no `--read-only` because torch's
extension loader takes a lock file), signed images.

## Verification, quarantine, points (`server/verify.py`)

- **Every result:** the server's structural check at completion (sha256, sizes, the type's `accept`: exact tensor
  names / dtypes / shapes for quant shards; in replicate mode `rowerr.npz`, a stored, uncompressed npz with one float
  `err`/`ref` pair per tensor of the right length; the golden's NLL column for eval rows; the row-level consistency of a
  recorded golden).
- **Quant results, score mode** (0.3, `"verify": "score"`, docs/VERIFY.md): the scorer (`server.py --score-threads 1`
  or `admin.py score --loop S`) decodes every tensor of the returned codes with the exact integer decode, computes
  its Hessian-weighted error against our own weights and Hessians, and accepts the result when every tensor lies in
  the band expected for it at that K (from earlier results of the same tensor: same K, neighbouring K, or the
  rate-distortion slope; one-sided). No re-runs, except one anchor re-run on a verifier for the first result of a
  tensor in a series; other GPUs, vendors, drivers and seeds are all fine; copies of a job (best of N) are all
  accepted and the export keeps the best (0.5: a donor is leased at most one copy of a job). A tensor out of its band
  fails the job (quarantine); codes equal to a published model's fail (replay); codes equal to another donor's pass
  but earn nothing. "Equal" includes near-copies (0.5): sharing at least half of 256 trellis tiles sampled with the
  queue's secret, so the same quant with a few words changed is the same quant. `admin.py scores` lists every scored
  tensor.
- **Quant results, replicate mode** (the 0.2 path): the CPU checker (`admin.py check`, below) recomputes every
  tensor's proxy error from the returned codes against our own Hessians and bf16 weights and compares it with the
  worker's log (`|ratio − 1| ≤ 2e-3`). A FAIL quarantines the donor (unless `--no-quarantine`).
- **Spot checks** (all types except score-mode quant jobs): when a donor's result is accepted, it is drawn for a re-run with probability `--spot-rate`
  (default 4%, the 3-5% band; TESTS.md: 4.28% of 5,000), and always for a donor's first result of each type
  (`--spot-first 1`). The re-run is the same job (type, parameters, inputs) leased only to a `verifier` worker (our own
  hardware, `admin.py verifier ID`), never the donor. When it comes back, the type's `compare` decides:
  - quant: each tensor's proxy error within 1.5% of the re-run's (seed spread ~0.5%: TESTS.md v0 §4); the checker
    has already tied each log to its codes, so this compares the codes' quality;
  - eval_mix: the same container sha256; per position |ΔKL| ≤ 1e-3 + 5% on ≥ 99.5% of rows, the mean KL within 0.2%,
    top-1 flips ≤ 1% (cross-GPU spread: to calibrate on the first R9700 pairs). 0.5: the draw re-scores 3 random
    documents (`--eval-spot-rate` 8%), each document's mean within 0.5%; a full re-run (`trust_state`) for suspect and
    adopted candidates (above);
  - golden_record: identical tokens, per-row nll and argmax as above;
  - noop: identical bytes.
  The original's `verify_state` becomes `pass` or `fail`; `admin.py spot JOB` draws one by hand.
- **Quarantine** (a failed spot check, a failed checker run, or `admin.py quarantine ID --reason ...`): the worker
  gets no more leases (it is told so) and cannot complete anything; its active leases are cancelled; every accepted
  result of it that no re-run (or, score mode, no score) confirmed is invalidated: its lumens revoked, its files moved to
  `results/<job>.invalid-<time>`, the job queued again for anyone but that worker (batches see it as pending again).
  An evaluation counts as confirmed only after a full trusted re-run (a spot check of a few documents does not).
  `admin.py unquarantine ID` lifts it (invalidated results stay requeued). Quarantine is per worker; a donor caught
  faking is dropped as a whole by the operator: `admin.py exclude-donor DONOR --reason ...` quarantines all its workers
  and invalidates all their unconfirmed results.
- **Lumens** (`admin.py points [--json]`, the leaderboard): a job's reference cost in R9700 card-minutes
  (`lumens(p)` per type: 27B layer 3, head 15, eval ~2.5 per 100k positions × cards, golden ~20 per 1M positions ×
  cards; `quant_tensor` 0.5, its down_proj group 1.5), +10% if the result came back within twice that. Granted to the
  worker whose result was accepted once it is verified: quant results need the checker's or the scorer's pass, drawn
  results their spot check's pass, others their structural check. Revoked on quarantine. Verifier workers earn none; timed-out or superseded leases earn none.

### The quant checker (`checker/`)

For each returned shard:
1. **Structure:** exactly `{key}.trellis` (int16 `[in/16, out/16, 16K]`), `.suh` (f16 `[in]`), `.svh`
   (f16 `[out]`) and `.mul1` (int32 marker) for every linear of the layer plan, with the shapes our bf16 weights
   imply. `proxy.jsonl` covers each tensor once with the job's K, codebook and scaled-Hessian setting. `job.json`
   carries the job's settings stamp. `suh` / `svh` finite, non-zero, |x| ≤ 1e3, and `.mul1` is the MUL1 marker
   (docs/VERIFY.md "Format"; the same rule the score verifier applies); in replicate mode `rowerr.npz` is a valid npz
   with one float `err`/`ref` pair per tensor of the right length.
2. **Proxy recompute:** `exl3dec.py` decodes each tensor from its codes on the CPU. It is a numpy transcription of
   exllamav3's decode: integer and fractional K, qtf2's sub-1 salt, the mul1/i8/i8c/i4 codebooks. The checker
   then computes rel = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) against our own Hessian and bf16 weight.
   - **Full mode:** compares rel with the logged `proxy_err`. exllamav3 logs its post-refit error against the
     undamped Hessian, which is the same quantity.
   - **Sampled mode (`--rows-frac 0.25`):** scores random 128-wide output blocks. The checker picks the seed, so
     the worker can't know which blocks. It compares them with the worker's per-row claim on exactly those
     blocks, and also checks the claim's total against the log.

   Tolerance: `|ratio − 1| ≤ 2e-3`. Measured spread for honest results is ≤ 1.7e-4, which comes from fp16
   rounding of the stored scales. Results are not bit-identical across GPUs, but the check compares the codes a
   worker returned with that worker's own numbers, so cross-GPU nondeterminism doesn't enter it. A comparison that is
   not a finite number fails.
3. Valid-but-bad codes with an honest log pass this check; the spot checks below catch them (a trusted re-run's
   proxy error per tensor, within the seed spread).

Cost on the queue host (8 threads, niced), per 27B layer:

| mode | wall | peak RSS |
|---|---|---|
| full | 50–75 s | 2.6 GB |
| sampled 25% | 13–18 s | — |

Batch checks should go layer by layer: the Hessian load and tr(W H Wᵀ) are the same for every level.

## Donor CLI (`client/rqah.py`)

What a donor (or an agent working for one) runs to see their own part of the queue. Stdlib only; installed as
`rad-quant-at-home` (alias `rqah`) in the worker images, where it reads the container's settings (`RQAH_SERVER`, the token file in the
state volume), so no flags are needed:

```bash
docker exec rad-quant-at-home rqah status            # profile, workers, running jobs, lumens, pending, queue depth,
                                                     # data budget left (0.4: DATA LEFT, this box's day / month counters)
docker exec rad-quant-at-home rqah jobs --limit 20   # recent jobs (--active: running only; --worker ID)
docker exec rad-quant-at-home rqah points            # lumens per worker / type, the grant / revoke ledger, rank
docker exec rad-quant-at-home rqah logs -f           # the server's events for your workers (--local: this box's job records)
docker exec rad-quant-at-home rqah config            # server, token fingerprint, caps from the environment, profile
docker exec rad-quant-at-home rqah config set --leaderboard on --display-name "Ada"
# outside a container:
python3 client/rqah.py --server https://<queue-url> --token-file ~/.rqah-token status
```

- **Donors and workers.** Every worker belongs to one donor; status, jobs, points and logs cover all workers of the
  donor of the token used, and nothing else. Workers registered with one personal invite (`admin.py invite --uses 3
  --personal`) are one donor; every other registration makes a new donor. That grouping is for your own view (and
  the operator's `exclude-donor`); the **leaderboard credits by display name**, so machines registered separately
  are credited together by giving them the same name. `rqah link` (and `POST /api/v0/me/link`, `admin.py link`) was
  removed: the endpoint answers 410 with a pointer to display names.
- **Verification state per job** (`jobs`): `running`; `verified` (lumens granted); `accepted` (no further check
  due); `awaiting check` (quant results until the checker runs); `spot-check pending` (re-running on our hardware);
  `failed check` / `failed spot check`; `voided` (invalidated after a quarantine); `superseded` (another worker
  finished first); and the lease states `expired`, `abandoned`, `failed`, `rejected`, `cancelled`. Pending jobs show
  the lumens they are expected to earn.
- **`--json`** before or after any command: one JSON object on stdout with `"ok": true` and the server's fields
  (`logs -f --json`: one event per line). Errors: `{"ok": false, "error": ..., "http": code}`. Exit codes: 0 ok, 1
  server error (including a refused setting: HTTP 400), 2 usage or missing configuration, 3 token refused, 4 server
  unreachable.
- **Privacy.** The token is never printed (`config` shows the first 12 hex digits of its sha256). Everything the API
  returns is about the donor's own workers. Free text in it (errors, quarantine reasons, ledger notes) is scrubbed of
  paths, addresses, URLs, e-mails and tokens. Display names: 1–32 printable characters; addresses, e-mails, URLs,
  token-like strings, long hex and `<>&"'` are refused.

API (any worker token; `server/donor.py`): `GET /api/v0/me`, `/me/jobs?limit=&active=1&worker=`,
`/me/points?limit=`, `/me/events?after=ID&limit=`; `POST /api/v0/me/profile {display_name, leaderboard}` (a donor
the operator hid cannot change its name). `/me` reports the donor's `rank`, `board_lumens` (the whole name's total)
and `name_hidden`. Registration takes `"leaderboard": true` (the worker's `--leaderboard` / `RQAH_LEADERBOARD=1`).

## Public status page (`server/statuspage.py`)

One static HTML file anyone can read, with no login. It has no JavaScript, no forms and no external resources; a
`<meta refresh>` reloads it every minute. It shows:
- **campaign progress:** "ladder 41 / 73 levels (… checked)", the levels in progress with their jobs done, "search 1,240
  evals", golden chunks, and spot checks passed / failed / pending;
- **jobs:** running, queued, done in the last 24 h and done in total per job type, plus cards busy and donors active;
- **the leaderboard:** one row per display name (case-insensitive; the most common spelling is shown) with the
  lumens and verified jobs of every opted-in donor showing that name, and last active; one line for everyone else
  ("N donors not listed: X lumens");
- **recent work:** the last 15 accepted jobs (level and layer, "LM head", "search · mix on <level>", "golden · <corpus>
  <chunk>"), each with who did it (a listed name, "a donor" or "our hardware") and whether it is verified;
- **the last update time** (UTC).

- **Serving.** `server.py` rebuilds it every `--status-interval` seconds (`RQAH_STATUS_INTERVAL`, default 60;
  0 turns it off) and serves the last good copy at `GET /` and `/index.html`. Requests never touch the database. The
  headers are a strict CSP (`default-src 'none'`), `nosniff` and `no-referrer`. With `--status-dir DIR`
  (`RQAH_STATUS_DIR`), it also writes `DIR/index.html` atomically (0644) for any static host. In the container,
  use a directory inside the out volume, e.g. `/srv/rqah/out/status`. `admin.py status-page OUT` and
  `statuspage.py --db DB --out FILE` (a read-only connection, e.g. from cron) build it once.
- **Campaign totals** (optional): `campaign.json` next to the database, or `--campaign` / `RQAH_CAMPAIGN`. It is
  re-read at every build:
  `{"title": "EXL3-V3 ladder for Qwen3.8-27B", "ladder": {"levels": 73}, "search": {"evals": 5000}, "goldens":
  {"chunks": 120}, "note": "..."}`. `ladder.levels` may also be a list of level names. A level counts as done when
  every one of its quant jobs is accepted, and as checked when every job also passed the checker.
- **What it never shows:** worker names, tokens, invites, addresses, hostnames, GPU names, job parameters other than
  level / layer / public corpus names, errors, quarantine reasons, file paths. Names come from the opt-in display name
  only. A stored name that fails the display-name rules is left off. A quarantined worker's lumens count nowhere, but
  nothing else happens to its name: the row stays with everyone else's lumens (otherwise anyone could take a name
  off the board by cheating under it). The operator's `admin.py donor ID --hide` on any donor hides that **name**
  for every donor showing it (the operator's call; `--leaderboard off` drops just one donor, who may opt in again).
  Every string is HTML-escaped. Identifiers (levels, corpora)
  must also match `[a-z0-9][a-z0-9._-]*` to be shown at all.
- **Example** (the test fixture, rendered as text):

```
rad-quant-at-home
EXL3-V3 ladder for Qwen3.8-27B · updated 2026-11-05 17:33 UTC
Campaign
ladder 1 / 73 levels (1 checked)   search 5 evals of 5,000 planned, 3 queued   goldens 1 / 2 chunks
spot checks 1 passed 1 failed, 1 pending
 level in progress | jobs done
 k2.5-i8           | 1 / 3
 k4-i8             | 0 / 1
Jobs
2 cards busy on 1 workers · 7 donors active in the last 24 h
 job type                 | running | queued | done (24 h) | done (all)
 quantize a layer         | 1       | 3      | 1           | 4
 quantize the LM head     | 0       | 0      | 0           | 1
 score a candidate mix    | 0       | 3      | 3           | 5
 record reference outputs | 0       | 1      | 1           | 1
 test job                 | 0       | 0      | 2           | 2
plus 1 verification re-runs on our own hardware
Donors
95.5 lumens of verified work so far. ...
 # | name         | lumens | verified jobs | last active
 1 | Ada Lovelace | 69.0   | 5             | now
 2 | Zoë 🚀 Team  | 20.0   | 1             | now
3 donors not listed: 6.5 lumens
Recent work
 2026-11-05 17:33 UTC | search · mix on k3.25-i8 | Ada Lovelace | verified
 2026-11-05 17:31 UTC | k2.5-i8 · layer 0        | Zoë 🚀 Team  | awaiting check
 2026-11-04 13:45 UTC | search · mix on k3.25-i8 | a donor      | spot check pending
 ...
```

## Data per job

Measured numbers are in TESTS.md.

| | down | up |
|---|---|---|
| 27B layer | ~0.8 GB weights + ~0.7 GB Hessians (≈ 1.5 GB) | 50–300 MB shard + ~1 MB small files |
| 27B Hessian group (`quant_tensor`) | its weights + its one Hessian: 0.1–0.4 GB (down_proj's 17408² Hessian is 0.6 GB of it) | its share of the shard |
| 0.8B layer (test) | 43 MB weights + 38 MB Hessians | see TESTS.md |
| 0.8B Hessian group (test, measured) | 18.9 MB | 3.2 MB |

The Hessians and weights of a layer are the same for every level, so a donor working down the ladder layer by
layer hits its cache: the scheduler (0.4) gives a worker, inside the level being finished, the layers whose inputs it
holds (docs/SCHED.md). A worker with a data budget is charged only for the inputs it does not hold, and for the
type's expected upload (`upload_bytes` in `jobs/*.py`).

v0.2 job types (from the file layouts; the fake-model sizes in TESTS.md §8):

| job | down | up |
|---|---|---|
| eval_mix, 27B | the first candidate of a base: its skeleton + one piece per weight ≈ the container (8-17 GB) + the golden (top-512: 4.1 GB per 1M positions, ~0.4 GB for a 100k slice); each later candidate only the pieces it has not cached (tens to hundreds of MB for a typical search move: a 27B `down_proj` at 4 bpw is ~45 MB) | rows 16 B per position (1.6 MB per 100k) + the report |
| golden_record, 27B | once: `bf16.rad` (54 GB, TP2) or the HF checkpoint (54 GB, staged); per chunk a few MB of corpus | per 1M positions 4.1 GB (`top.bin`) + 12 MB rows + 4 MB tokens |


## How to run

RUNBOOK.md is the operator's walk-through (start the server, load a job batch, watch, verify, pay points).

### Server without Docker (tests, our own boxes)

```bash
A="python3 -I server/admin.py --db Q.sqlite --data DATA --results RESULTS"
$A datapack qwen38-27b <rev> <bf16 ckpt dir> <calib hessians dir> [--hout <snap> <dir>] [--hf] [--bf16-rad FILE]
$A ladder-add qwen38-27b <rev> k3-i8 /srv/rqah/sources/ladder/k3-i8.rad       # each level the search mixes from
$A golden-add xcorp-s07 <golden dir> --public --corpus xcorp --pool search
$A manifest --jobs 8
$A enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8 \
   --sigma-reg 0.025 --sigma-reg-in 0.005 --scaled-hessian --qtf2 --verify score --head [--copies 2] [--unit group]
python3 -I server/server.py --db Q.sqlite --data DATA --spool SPOOL --results RESULTS --bind 127.0.0.1 --port 8740 \
   --score-threads 1 [--search-token-file F] [--admin-token-file F]   # expose only through a TLS terminator / tunnel
$A invite --uses 1                       # one per donor
$A list --level k3-i8; $A scores --level k3-i8; $A export --level k3-i8 OUT --checked; $A points
# replicate mode (0.2) instead: no --verify score, then $A check --level k3-i8 --rows-frac 0.25
```

Every `server.py` flag can also be set as `RQAH_<FLAG>` in the environment (`RQAH_DB`, `RQAH_PORT`,
`RQAH_SPOT_RATE`, `RQAH_SCORE_THREADS`, ...; the 0.2 `RADQUANT_<FLAG>` still works in 0.3); a flag on the command line wins. `admin.py` reads
`RQAH_DB`, `RQAH_DATA` and `RQAH_RESULTS` the same way.

### Server container (a plain Linux host)

Files: `docker/server/Dockerfile`, `entrypoint.sh` (`rad-quant-at-home serve | admin ... | healthcheck | version`, alias `rqah`),
`compose.yaml`, `.env.example`. Every `RQAH_*` setting also takes its v0 name `RXMQ_*` in 0.2. Needs Docker
with the compose plugin (tested: Docker 29.8, compose 5.6; Podman untested).

```bash
# 1. build (from a commit; --worktree for uncommitted code). Tags rad-quant-at-home-server:0.4.0, :0.4.0-<commit>, :latest
docker/server/build.sh
# 2. configure: copy .env.example to .env next to compose.yaml and set at least the directories
cd docker/server && cp .env.example .env
#    RQAH_DB_DIR=/srv/queue/db  RQAH_DATA_DIR=/srv/queue/data  RQAH_OUT_DIR=/srv/queue/out  RQAH_SOURCES_DIR=/srv/models
sudo mkdir -p /srv/queue/{db,data,out} && sudo chown 10001:10001 /srv/queue/{db,data,out}   # or RQAH_UID/GID = yours
#    SELinux enforcing (getenforce)? also RQAH_SELINUX_RELABEL=1 (bind mounts get :z)
# 3. start; it restarts by itself (unless-stopped) and is healthy once /healthz answers
docker compose up -d && docker compose ps
# 4. data, levels, invites (admin = admin.py inside the image, same volumes)
docker compose run --rm admin datapack qwen38-27b <rev> /srv/rqah/sources/qwen38-27b-bf16 /srv/rqah/sources/hessians/calib
docker compose run --rm admin manifest --jobs 8
docker compose run --rm admin enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8 ...
docker compose exec server rqah admin invite --uses 1
docker compose exec server rqah admin list --level k3-i8
docker compose run --rm admin check --level k3-i8 --rows-frac 0.25          # CPU-heavy: 8 CPUs / 8 GB by default
docker compose logs -f server; docker compose restart server; docker compose down   # down keeps the volumes
```

What lives where (each `*_DIR` is a named volume by default, or a host path):

| volume | in the container | contents | server access |
|---|---|---|---|
| `RQAH_DB_DIR` | `/srv/rqah/db` | `queue.sqlite` (+ `-wal`, `-shm`): jobs, leases, workers, events | read-write; local disk only, not NFS |
| `RQAH_DATA_DIR` | `/srv/rqah/data` | per-file inputs + `manifest.json` | read-only (`run --rm admin` writes it) |
| `RQAH_OUT_DIR` | `/srv/rqah/out` | `results/<job>/` and `spool/` (uploads in flight) | read-write; one filesystem, because completion renames spool to results |
| `RQAH_SOURCES_DIR` | `/srv/rqah/sources` | bf16 checkpoint and Hessians that `datapack` symlinks from | read-only |
| `RQAH_SECRETS_DIR` | `/run/secrets/rqah` | `admin-token` (optional) | read-only |

`datapack` writes symlinks to the Hessians. Run it through `docker compose run --rm admin` with source paths under
`/srv/rqah/sources`, so that the links resolve inside the server container too. Running admin.py on the host instead
also works, against the same bind-mounted files: the WAL database handles two processes, and the tests do exactly
this. SQLite gives its journal files the database owner when run as root. Files the host side writes must be
readable by uid 10001 (admin.py writes them 0644).

Settings (`.env`; defaults in brackets):

| variable | meaning |
|---|---|
| `RQAH_PUBLISH` [`127.0.0.1:8740`] | host address:port to publish on; keep it on loopback (or LAN / tunnel), with TLS in front |
| `RQAH_LEASE_S` [120], `RQAH_HEARTBEAT_S` [30], `RQAH_RETRY_S` [30], `RQAH_MAX_LEASES` [16] | leases and polling |
| `RQAH_CHUNK_MAX_MB` [64] | largest upload PUT |
| `RQAH_UPLOAD_MAX_MB` [0 = off] | ceiling per uploaded file, on top of each job type's own caps |
| `RQAH_SPOOL_MAX_GB` [0 = off] | uploads get 507 (the worker backs off and resumes) while the spool holds this much |
| `RQAH_SEARCH_TOKEN_FILE` [off] | enables the search's batch API (`/api/v0/search/...`: eval_mix on public goldens only) for that bearer token (≥ 32 characters, a file in `RQAH_SECRETS_DIR`) |
| `RQAH_SPOT_RATE` [0.04], `RQAH_SPOT_FIRST` [1] | spot checks: the fraction of each donor's accepted jobs re-run on a verifier, and every donor's first N per type |
| `RQAH_EVAL_SPOT_RATE` [0.08], `RQAH_EVAL_SPOT_DOCS` [3] | eval_mix spot checks: the fraction drawn, and how many random documents of the golden the verifier re-scores (0: all) |
| `RQAH_LEASE_CAP_X` [3], `RQAH_LEASE_CAP_MIN_S` [7200] | a lease's deadline: this many times its job type's reference runtime, at least this many seconds (0.5, #7) |
| `RQAH_GANG_WAIT_S` [0], `RQAH_RESERVE_MAX_S` [21600] | gang jobs: how long one waits before a busy multi-card worker drains for it; when a drain is given up |
| `RQAH_CLASS_WAIT_S` [900], `RQAH_CLASS_SCAN` [5000] | 0.4 (docs/SCHED.md): how long a job a busy worker could run waits before that worker is drained for it instead of getting lower-priority work; queued jobs looked at per priority class and lease |
| `RQAH_ADMIN_TOKEN_FILE` [off] | enables the read-only `GET /api/v0/admin/summary` (counts per level, workers, last events) for that bearer token (≥ 32 characters, a file in `RQAH_SECRETS_DIR`, readable by uid 10001) |
| `RQAH_UID` / `RQAH_GID` [10001] | user the container runs as |
| `RQAH_MEM_LIMIT` [2g], `RQAH_ADMIN_MEM_LIMIT` [8g], `RQAH_ADMIN_CPUS` [8] | resource caps |
| `RQAH_SELINUX_RELABEL` [off] | `1`: bind mounts get `:z` (needed on SELinux hosts) |

Nothing that changes the queue goes over HTTP. Admin commands run on the host, via `docker compose exec` or
`run --rm admin`; the admin token gives a read-only view only.

### Donor worker (Docker: ROCm or CUDA)

The run lines are in "For donors" above (`docker run --gpus all` for the CUDA image needs the NVIDIA Container
Toolkit on the host; the worker's own `--gpus` then picks among the cards the container sees). The token is given at run time only, by one of:
- `RQAH_INVITE`: a one-time invite. The worker registers on its first start and keeps the token in the state
  volume (`/var/lib/rqah/token`, mode 0600). The invite stays visible in `docker inspect`, which is harmless
  because it is used up.
- `RQAH_TOKEN_FILE`: a file mounted read-only.
- `RQAH_TOKEN`: the token in an env variable. This shows in `docker inspect`, so prefer the other two.

Worker options (each flag also `RQAH_<FLAG>`, e.g. `--max-up-gb-month` = `RQAH_MAX_UP_GB_MONTH`):

| option [default] | meaning |
|---|---|
| `--server`, `--token-file` / `--invite` + `--name` | the queue and how to authenticate (see above) |
| `--state` [`/var/lib/rqah` in the image] | token, running jobs, input cache, `worker.jsonl`, `bandwidth.json` |
| `--gpus 0,1,...` / `--cards N`, `--gpu-name`, `--vram-gb` (per card) | the cards jobs may use |
| `--slots` [one per card] | jobs at once |
| `--ram-gb` | host RAM all running jobs together may use |
| `--cache-gb` [200] | disk cap of the input cache |
| `--window HH:MM-HH:MM` (repeatable; `RQAH_WINDOWS=a-b,c-d`) [always] | local-time windows for new jobs |
| `--vram-cap-gb`, `--gpu-lock FILE` | shared cards: torch allocator cap, flock around GPU jobs |
| `--job-types a,b` [all the image has] | job types to accept |
| `--on-stop finish\|abandon` [finish], `--stop-file F` | what a stop does to running jobs |
| `--max-down-mbps`, `--max-up-mbps` [0 = unlimited] | **0.4** rate caps in Mbit/s, all jobs together, resumed transfers included |
| `--max-down-gb-day`, `--max-down-gb-month` [0 = unlimited] | **0.4** download budget (1 GB = 10^9 bytes) per calendar day / budget month, local time |
| `--max-up-gb-day`, `--max-up-gb-month` [0 = unlimited] | **0.4** upload budget, the same |
| `--budget-reset-day N` [1] | **0.4** day of the month (1-28) the monthly budgets start again |
| `--max-jobs N`, `--idle-exit N`, `--poll S` | tests and one-off runs |

More on each:
- **Cards:** `--gpus 0,1,...` (or `--cards N`), `--vram-gb` (per card), `--gpu-name`; `--slots` (jobs at once,
  default one per card). Without cards the worker is CPU-only (test jobs only).
- **RAM:** `--ram-gb`: host RAM all running jobs together may use; the server leases only what fits beside the jobs
  already running (a 27B head job alone needs 48).
- **Disk:** `--cache-gb` caps the input cache in the state volume. Allow that plus one job's staging: ~2 GB for a 27B
  layer, one assembled container (up to ~25 GB) for an eval on a filesystem without reflinks.
- **Time windows:** `--window` (repeatable) or `RQAH_WINDOWS=a-b,c-d`, in the container's local time (the host's,
  via the `/etc/localtime` mount, or `-e TZ=...`). Outside them the worker takes no new jobs.
- `--vram-cap-gb` caps torch's allocator on a shared card; `--gpu-lock FILE` (with `{card}`: per card) for shared boxes.
- **Bandwidth (0.4):** "Data caps" above; the counters are in `STATE/bandwidth.json` (`rqah status` reads them).
  The month runs from `--budget-reset-day` to the day before it in the next month. A spent download budget still
  allows jobs whose inputs are all cached; a spent upload budget allows nothing new.
- **Job types:** the CUDA image declares `noop,quant_layer,quant_head,quant_tensor` (no radiance in it); the ROCm image
  every type. A 27B layer job needs 8 GB of VRAM up to K6 and 12 GB above, its down_proj group the same, every other
  group 6 GB (docs/SMALL-CARDS.md): declare `--vram-gb` honestly and the server sends only what fits.

`docker/worker/compose.yaml` (with `.env.example`) holds the same settings plus `restart: unless-stopped` (AMD):

```bash
cd docker/worker && cp .env.example .env    # set RQAH_SERVER, RQAH_INVITE, VIDEO_GID, RENDER_GID, caps
docker compose up -d && docker compose logs -f
```

### Worker on our own boxes (no Docker)

```bash
RXM=/path/to/rad-exl3-mix STATE=... EXL3_SRC=... EXL3_PATH=... TORCH_EXT=... GPU_LOCK='/var/lock/gpu-card{card}.lock' \
  RADIANCE=... RADIANCE_HOME=... docker/run-local.sh --server ... --token-file ... --gpus 0 --gpu-env ROCR_VISIBLE_DEVICES
```

A verifier (spot-check re-runs) is such a worker registered with `admin.py add-worker` and flagged
`admin.py verifier ID`.

### Publishing images (maintainer; a host with Docker and ~40 GB free)

The images live on GHCR under `ghcr.io/theherobrine422/` as public packages, so donors pull them without a login.
Pushing needs a classic GitHub token with `write:packages` (never stored in this repo or on a donor box):

```bash
echo $GHCR_TOKEN | docker login ghcr.io -u TheHeroBrine422 --password-stdin
```

Then build and push (below). After the first push of each image, in GitHub's package settings
(github.com/TheHeroBrine422 > Packages > the image > Package settings): set the visibility to **public**, and link the
package to this repository ("Connect repository"), so its page shows the source and the README.

#### Worker images

The build context is this repo at a commit plus the quantizer paths of a rad-exl3-mix checkout at a commit
(`RXM_REPO`, default `../rad-exl3-mix`; `RXM_COMMIT`, default its HEAD): both are recorded in the image (`/opt/rqah/IMAGE`,
labels).

```bash
IMAGE_REPO=ghcr.io/theherobrine422/rad-quant-at-home-worker RXM_COMMIT=<rxm-commit> RADIANCE_IMAGE=radiance:<fork-commit> \
  docker/build.sh rocm <commit>                      # never from --worktree
IMAGE_REPO=ghcr.io/theherobrine422/rad-quant-at-home-worker RXM_COMMIT=<rxm-commit> docker/build.sh cuda <commit>
docker image inspect ghcr.io/theherobrine422/rad-quant-at-home-worker:0.4.0-cu128 --format '{{json .Config.Labels}}'
docker push ghcr.io/theherobrine422/rad-quant-at-home-worker --all-tags
```

The CUDA build compiles exllamav3 and the side extensions for `TORCH_CUDA_ARCH` (default `8.0;8.6;8.9;12.0`) with
`MAX_JOBS` parallel nvcc processes (default 8; each needs ~2-3 GB of RAM: on a small builder use
`BUILD_ARGS="--build-arg MAX_JOBS=1"`, and `--build-arg TORCH_CUDA_ARCH=8.9` for a quick single-arch test build).

Tags:
- `0.4.0-rocm7.14-gfx1201`, `0.4.0-cu128`: what donors pin.
- `0.4.0-rocm` / `0.4.0-cuda` and `rocm` / `cuda`: these move.
- `rocm-<commit>-rxm<commit>`, `cuda-...`: the exact source commits.

#### Research images (research-kit/)

```bash
research-kit/build.sh rocm && research-kit/build.sh cuda   # ghcr.io/theherobrine422/rad-quant-at-home-research:<VERSION>-rocm / -cuda
docker push ghcr.io/theherobrine422/rad-quant-at-home-research --all-tags
```

Bump `VERSION` with every release, and a job type's `VERSION` with any change of its meaning (above).
Deploy the server first, then the new image. Run one real job with the built image on
our own R9700 first (`--max-jobs 1`). That checks the GPU flags and the read-only code paths, which no build host has
exercised yet.

### Tests

Every push to main and every pull request runs the no-GPU suites below on GitHub Actions
(`.github/workflows/tests.yml`, Python 3.12 and the latest 3.x); the GPU tests stay manual.

- `tests/e2e_noop.py`: no GPU, ~5 min, 134 checks (141 with the container). Runs the server as a subprocess, or
  `E2E_SERVER=docker` for the container; the workers always run on the host. The GPU side is faked on the CPU (`tests/fakes.py`,
  `fake_radiance.py`, `fake_mkgold27.py`): requirements, multi-card and gang scheduling, eval_mix batches end to end,
  privacy refusals, spot checks with cheating donors, quarantine, lumens, golden_record, the eval plausibility band,
  spot checks of a few documents, adoption and `exclude-donor` (#4); w25: the donor CLI (every
  command, `--json`, exit codes, isolation, personal invites, credit by display name, `link` gone) and the served
  status page.
- `tests/sched_test.py` (0.4): the scheduling rules in process, a few seconds (docs/SCHED.md): K-granularity
  priorities, `reprioritize`, class order, level-first, cache affinity, the budget filter (incl. a randomized
  check), the cache report protocol, the class-wait drain.
- `tests/bandwidth_e2e.py` (0.4): the bandwidth limits end to end with noop jobs, ~3 min: the budget counters (day,
  month, reset day, year wrap), rate caps on a 30 s fetch / upload / resumed fetch (±10%), budgets across a SIGKILL
  and a restart, `rqah status`, the status page.
- `tests/setup_test.py`: the setup wizard (docs/SETUP.md) on fake machines (fake sysfs and command output), ~20 s, no
  GPU, no network: a 4 / 8 / 12 / 16 / 32 GB card matrix, multi-card with an iGPU, unsupported cards, no Docker,
  the files and their modes, re-runs, `--check`, the plain prompts and the curses form on a pseudo-terminal.
- `tests/statuspage_test.py`: the status page from a fixture database, a few seconds: well-formed XHTML, element
  allow-list, no active content, no token / IP / path / hostname patterns, escaping, the numbers, the display-name
  rules.
- `tests/radmix_port.py`: `common/radfile.py` against the real `mix/radmix.cpp`, byte for byte (needs g++ and
  radiance's headers: `RADIANCE_ABI`, `EXL3_ACT_INC`, or a built `RADMIX`).
- `tests/score_test.py` (0.3): verification by score on a synthetic layer, no GPU, ~1 min (docs/VERIFY.md).
- `tests/score_real.py` (0.3): the same on real quantizer output (honest ladders, seed repeats, real attacks), CPU.
- `tests/cuda_smoke.sh` (0.3): a local server with its scorer + one `quant_tensor` job and its anchor re-run from the
  CUDA image (`MODE=docker`), from its extracted filesystem (`MODE=rootfs`) or from the box's own stack
  (`MODE=native`).
- `tests/quant_e2e.sh`: one real layer vs a direct ourq run (GPU). `tests/checker_tamper.py`, `tests/checker_nan_test.py`
  (the replicate checker on the synthetic layer: NaN scale / NaN claim / 0-0 claim, and a quant_tensor group result
  through `check_job`; a `rowerr.npz` that is not a zip, misses a `.ref`, has the wrong length, is
  compressed, cut short or has a bad CRC, each refused by `accept` and failed on structure, CPU), `tests/dec_rates.sh`.
- Measurement tools (`tools/`): `hfrange.py` (single tensors of a HF checkpoint by HTTP range), `memwrap.py` (peak VRAM
  / RSS of a quantizer run), `curve_loo.py` and `seed_spread.py` (the data behind the score band).

## Next (v1)

- **Before opening:** the namespace, the ROCm image build with radiance inside (`docker/radiance-wrap.sh` is
  untested), one real job of each type on an R9700 (`--max-jobs 1`), the CUDA image under `docker run --gpus` on an
  NVIDIA host, then calibrate the eval tolerance on the first donor / verifier pairs (cross-GPU spread of per-position
  KL).
- **Web UI on the donor side:** start / stop button, time windows, storage and bandwidth limits (the worker already
  has the flags, bandwidth limits included since 0.4; the UI writes them and the stop file). A preference for
  small-upload jobs on slow uplinks beyond the budget filter.
- **Leaderboard, beyond the status page:** teams, per-donor public pages ("your card quantized K2.375 of
  EXL3-V3"), credits in the model cards (`admin.py donors --json` / `points --json` have the numbers). A share of points for bytes served if seeding is ever
  peer-to-peer.
- **Verification by score**, after the first campaign: re-tune `TOL` / `NOISE` from the honest best-of-N pairs; a GPU
  scorer job for our verifiers; binding codes to their seed (the sign vectors) against copying between copies.
- **Bootstrap seeding** of the ~105 GB (cache-aware scheduling is in 0.4).
- **Healing jobs** (hours each, checkpointed state of tens of MB); **layer-streamed statistics** for new models
  (Σdy² partial sums, input-Hessian partials by token chunk and layer group); **big-box** backward passes (Σdy², REAL-Q
  Fisher), fuller H_out accumulators, long-context checks: trusted donors, with split-half / per-layer comparisons
  against our snapshots.
- TLS / a reverse proxy in front, rate limits, token rotation, signed images (cosign), reproducible builds.
