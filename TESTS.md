# rad-quant-at-home: tests

Times are US Central (CST). Sections 1-7: v0 (w13 / w13b, 2026-11-04; they use the v0 names `rxm-queue`, `RXMQ_*`,
`RXMW_*`, `rxmq`). Section 8: v0.2 (w24, 2026-11-05). Section 9: the donor CLI and the public status page (w25,
2026-11-05). Sections 10-12: 0.3 (w47, 2026-11-06): verification by score, small cards, the CUDA image, the rename.
Section 13: 0.4 (w57, 2026-11-06): bandwidth limits and the scheduling rules.
Older sections describe the code and names of their time (the project lived in rad-exl3-mix's `volunteer/` and was
called `radquant` until 0.3; the environment names in them were updated to `RQAH_*`). Machines: the VM (noop
end-to-end, the server container, image builds), an RTX 4070 test box (the real quantizer runs, CPU checks), the
queue host (server throughput, checker timing on the 27B, the v0.2 suite). Nothing was exposed beyond localhost.

## 1. End-to-end with noop jobs (`tests/e2e_noop.py`, VM, localhost)

A server with `--lease-s 6 --heartbeat-s 1.5`, workers as subprocesses. Each run takes ~50 s. Three runs
(3:44–4:10 PM CST): **39/39 checks passed every time**. w13b grew the suite to 54 checks (local) / 61 (container):
see section 7.

| scenario | what is checked | result |
|---|---|---|
| tokens | no token / bad token → 401; bad invite → 403; an invite with 3 uses registers 3 workers, the 4th → 403 | pass |
| data rules | a file without an active lease listing it → 403; not in the manifest → 404; `../` → 404 | pass |
| 3 workers, one killed (SIGKILL) mid-job | 9 jobs × 3 s; the victim's lease expires, the job is re-leased and accepted **once**, by another worker (events: `lease, expire, lease, accept`); every result byte-exact | pass |
| killed worker restarted after its lease died | finds `current.json`, heartbeat says the lease is gone, drops the job, no leftovers | pass |
| killed worker restarted within its lease | resumes the same lease and completes it (events: `lease, accept`, no expiry) | pass |
| corrupted upload | bytes with one bit flipped, true sha256 declared → 422 `bad: [out.bin]`, staged file deleted; correct re-upload accepted | pass |
| chunked / resumable upload | wrong offset → 409 with the resume point; two chunks at offsets 0 / 3000; a short chunk (`Content-Length: 3000`, 1000 bytes sent) → 400 with the old offset, spool truncated back | pass |
| negative `Content-Length` | `-1` on a JSON endpoint (`register`) and on an upload → 400 `bad content-length`, body not read (the client keeps its side open and still gets the reply: the server closes), spool unchanged | pass |
| duplicate completion | a second `complete` (even with a different sha) → 200 `duplicate`, ignored; upload to an accepted job → 409; stored result unchanged, one `accept` event | pass |
| caps | upload of a name that is not an output → 403; over the per-file cap → 413 | pass |
| another worker's lease | heartbeat on someone else's lease → 403 | pass |
| abandon | `fail` with `abandon` → job back to `queued`, attempt not counted | pass |
| disk cap eviction | cache cap 700 kB, four 300 kB inputs a, b, c, a → 2 evictions, cache ≤ cap; a job needing 900 kB is not leased to this worker | pass |
| Range resume | 120 kB of a 300 kB input already in `<sha>.part` → 180 kB downloaded, sha256 verified | pass |
| malformed jobs from our server | `noop` with `seconds: "3"` and with an extra key `cmd` (injected into the DB) → every worker rejects, the job fails after `max_attempts` (2) with 2 `reject` events; workers keep running | pass |
| fake malicious server | 9 crafted leases (unknown type `shell`; extra key `argv`; input `test/../../etc/passwd`; an input the type does not allow; `eval_mix` (unimplemented); `quant_layer` with a wrong settings stamp; a job id that is a path; an empty lease id; a job id carrying a newline and a terminal escape) → all 9 rejected via `fail` with `reject`, **no download attempted**, no file created at the path the job id names, the hostile id escaped in the worker log | pass |
| graceful stop | SIGTERM → finishes the current job, exits 0; `--on-stop abandon` → job handed back within 5 s, attempt not counted | pass |
| time windows | window arithmetic incl. across midnight; a worker outside its window leases nothing | pass |

Security fix round (issue #5, one attempt per worker per job; verifiers exempt): a worker that failed a job is not
offered it again until `admin requeue`, and a verifier that failed one is offered it again. macOS dev box, 2026-11-06,
**110/110** (plain process).

Request overhead: a 1 s noop job takes 1.01 s lease-to-complete on localhost.

Security fix round (issue #1, the three Content-Length cases above): macOS dev box, 2026-11-06, **111/111** (plain process).
Security fix round (issue #2, the three lease-id cases above): macOS dev box, 2026-11-06, **110/110** (plain process).

## 2. Server throughput (the queue host, loopback, niced)

One noop job with a 1.5 GB input and a 64 MB output: download + sha256 check **4.7 s** (~320 MB/s, limited by
hashing), upload of 64 MB **0.48 s** (both sides hash it). The stdlib server (sendfile for data) is far from the
bottleneck for home connections; the queue host's 10 Gbps link is the limit for the bootstrap.

## 3. Checker (CPU decode + proxy recompute)

**Decoder validation** against ourq's own dequantized weights (`--save-wq`) on 27B layer 10 at K2.5 i8 (fractional,
an earlier convergence run's `q-none` arm): every tensor matches to a relative Frobenius error of 4.0e-4 (fp16 rounding of the
stored scales and of `wq`). The recomputed proxy matches the logged `proxy_err` to ≤ 1.6e-4 (in_proj_qkv 1.3e-5,
in_proj_z 3.0e-5, out_proj 8.7e-6, gate 4.9e-6, up 4.3e-6, down 1.6e-4).

**ourq's `rowerr.npz` is wrong** (known; `proxy2b/exp/w04-blockgain/rowerr.patch` not merged): err/ref ≈ 1.0008
instead of 0.0032 for in_proj_z. The jobs therefore compute the per-row claim themselves (`jobs/rowerr_tool.py`);
its totals equal the logged proxy to ≤ 1e-5.

**Sampled mode** (25% of output blocks, checker's seed) needs the per-row claim: without it, a block sample can't
be compared with a whole-tensor number (out_proj's sample came out 1.66× its mean). With the claim, two seeds give
≤ 1.7e-4 on the sampled blocks.

**Timing per 27B layer on the queue host** (8 OpenBLAS threads, `nice -n 19`, other jobs running):

| mode | wall | notes |
|---|---|---|
| full | 50.6–75.3 s | peak RSS 2.6 GB; down_proj is ~40% (17408² Hessian: 2.4 s load, 2 GEMMs) |
| sampled 25% | 13.4 s | |

At 25%: ~14 min per level (64 layers) in one process; several in parallel. Cheapest correct decode path: this
numpy decoder on the CPU (0.8–5.4 s per tensor). exllamav3's reconstruct needs a GPU. `mix/hproxy.cpp` decodes
radiance `.rad` containers (HIP), not raw ourq shards, so it would need a conversion first.

**Tolerance:** `|ratio − 1| ≤ 2e-3`, ~12× the largest honest deviation measured (1.7e-4 on the 27B, 2–3e-5 on the
0.8B). Cross-GPU nondeterminism does not enter: the check compares the codes a worker returned with that worker's
own logged numbers.

**Tamper test** (`tests/checker_tamper.py`, on the 0.8B layer-5 result, 4070 CPU): **8/8 as expected**.

| case | checker |
|---|---|
| untouched, full / sampled | pass (2.0e-5 / 3.0e-5) |
| 1% of one tensor's trellis words changed, full / sampled | fail (ratio off by 4.0 / 3.9) |
| per-row claim 10% low | fail (0.11) |
| logged proxy 5% low | fail (0.053) |
| a tensor missing from the shard | fail (structure) |
| wrong settings stamp in job.json | fail (structure) |

Added after this run (issue #3) and **NOT RUN YET** on the real result: a NaN scale, a NaN per-row claim, a 0/0 per-row
claim (each must fail; `tests/checker_nan_test.py` covers the same three on the synthetic layer, a quant_tensor
group result through `check_job`, and a `rowerr.npz` that is not a zip, misses a `.ref`, has the wrong length,
is compressed, has a member cut short or a bad CRC, each refused by `accept` and failed on structure by `check_job`
(issue #13), 12/12 on CPU).

## 4. One real `quant_layer` on the RTX 4070 (`tests/quant_e2e.sh`)

0.8B, layer 5, K3 i8, settings as `proxy2b/q2b.sh`: qtf2, scaled Hessian, σ 0.025, input group 0.005. Everything
ran on the box: server on localhost, worker with `--vram-cap-gb 7.2`, and every GPU step inside one `flock` on the
shared GPU lock.

**Session 1** (lock held 4:25:57–4:26:41 PM CST, **43.5 s of GPU**):

| step | result |
|---|---|
| direct `ourq.py` (as q2b.sh, `--layers 5-5 --save-wq`) | 4.9 s wall, of which the layer 3 s: **~1.9 s per-process overhead** (imports, CUDA init, cached extension loads); peak VRAM 976 MiB |
| worker | lease → stage → ourq → rowerr_tool → upload took ~7 s per attempt; ourq and the rowerr step ran fine every time, but **every upload was refused**: the server had been started before `rowerr.npz` became an output (version skew). 5 attempts, then the job failed as designed. Fix: VERSION 2 for the quant types, version in the dedupe key, README rule |
| direct, `--seed-offset 1` | 4.4 s; per-tensor proxy vs seed 0: −0.37% to +0.21% (in_proj_z −0.37, qkv −0.18, out −0.13, up −0.03, gate +0.21, down +0.20): the seed spread a quality bound has to allow |

**Server-side path on the real output (CPU, no GPU):** the direct run's shard + proxy log + a job.json + the
per-row claim from `rowerr_tool.py` (CPU fallback) → `quant_layer.accept()` OK (exact keys / dtypes / shapes
against the datapack's bf16 weights, log, stamp) → checker PASS, full 2.0e-5, sampled 2.2e-5 → tamper test 8/8
(section 3).

**Session 2** (byte-for-byte comparison, plus `tests/dec_rates.sh`: the decoder at K0.5 i8 (sub-1, salt),
K2.125 i8 (1/8 step), K3.5 mul1 and K2.25 i8c): **NOT RUN YET.** It is queued on the GPU lock since 4:27 PM CST.
At 5:27 PM CST the lock was still held by a deadlock between two other scripts on the box (`hconv.sh` holds the lock
and waits for no `quantize/ourq.py` process to exist; `hest.sh`'s `flock … python3 …/quantize/ourq.py` matches that
pattern while it waits for the lock). The session needs ~3 min of GPU. It writes `quant_e2e.out` /
`dec_rates.out` in its work dir and leaves the GPU lock as soon as it finishes.

## 5. Data per job

| job | down | up |
|---|---|---|
| 0.8B layer (measured) | 81.3 MB: weights 43.0 MB + 4 Hessians 38.3 MB + config 3 kB | shard 8.1 MB + rowerr ~0.3 MB + logs / stamps ~35 kB |
| 27B layer (from the files) | **1.55 GB**: weights 0.74–0.77 GB + Hessians 0.79 GB (once per layer: the same for every level) | shard ≈ params × K / 8: 116–120 MB at K2.5, ~190 MB at K4, ~380 MB at K8; rowerr ~1 MB |
| 27B head | 2.54 GB weights + 52 MB Hessian | ~0.4 GB at K2.5 (1.27 G params) + rowerr 4 MB |
| bootstrap (whole trunk + head) | 48.7 GB weights + 50.3 GB Hessians + 2.6 GB head ≈ 102 GB | — |

Worker overhead per quant job, measured on the 4070: ~1.9 s for ourq's process start, plus a second short process
for the per-row claim. Staging, hashing and HTTP are well under a second for the 0.8B; the 27B's 1.55 GB of inputs
hash in ~5 s. Compared with ~2–3 min of quantization per 27B layer on an R9700, the overhead is a few percent.

## 6. Not tested in v0

- The worker images: not built. Both Dockerfiles pass `docker build --check`. A ROCm build (several GB of wheels
  plus an exllamav3 compile) is too heavy for our build hosts; it is the first step of publishing (README).
  Untested, therefore:
  - the GPU run flags (`/dev/kfd`, `/dev/dri`, the video / render groups, `seccomp=unconfined`);
  - whether torch's extension loader runs from the image's precompiled `torch_ext` without writing into the
    read-only source tree.
- The server container on Podman, and on the queue host itself.
- `quant_head` end to end.
- Any 27B job through the queue.
- Cross-GPU (R9700 vs 4070) differences.
- TLS.
- More than one machine.

## 7. The server container (`docker/server/`, VM, 2026-11-04, 5:36–6:14 PM CST)

Docker 29.8.2 and compose 5.6.0 on the VM (Fedora, SELinux enforcing).

**Image:** `build.sh --worktree` builds `rxm-queue:0.1.0` in 19 s, 284 MB, on `python:3.14-slim-trixie`
(digest `sha256:930e1abba7c4…`) + numpy 2.5.4. Contents checked: only LICENSE, NOTICE and
`volunteer/{common,jobs,server,checker}` + the entrypoint and healthcheck. User 10001, OCI labels (source, revision,
version, created, licences), `HEALTHCHECK` on `/healthz`.

**The noop suite against the container** (`E2E_SERVER=docker tests/e2e_noop.py`). The server runs from
`compose.yaml` (project `rxmq-e2e`), with the test dir's `db/`, `data/`, `out/` and `secrets/` bind-mounted. The
workers, admin.py and the test's own SQLite reads all run on the host against the same files. Final code: **61/61
twice** (6:10 and 6:14 PM CST). The plain-process path (`E2E_SERVER=local`) on the same code: **54/54**. Earlier
iterations passed 59/59 and 60/60 as checks were added. Every check from section 1 passes in both modes, plus:

| check | result |
|---|---|
| `/healthz` without a token → 200; the container turns `healthy` within ~4 s (`start_interval` 2 s) | pass |
| server process (PID 1) runs as uid 10001; read-only rootfs, `cap_drop: ALL`, restart policy `unless-stopped` | pass |
| `docker compose exec server rxmq admin ...` and `docker compose run --rm admin ...` beside the running server | pass |
| port published on `127.0.0.1` only | pass |
| server-wide upload ceiling (`RXMQ_UPLOAD_MAX_MB=48`) under a job type's 64 MiB cap → 413 at byte 48 MiB + 1 | pass |
| read-only admin summary: no token / a worker's token / a wrong token → 401; admin token → 200 (counts per level, workers without tokens) | pass |
| worker configured by environment only: `RXMW_INVITE` registers once, token file written 0600, the next start uses it; `RXMW_IMAGE` reported in caps | pass |
| **restart mid-run** (2 workers, 6 jobs × 3 s, 200 kB outputs; at the time 2 accepted, 2 leased, 2 queued): `docker compose restart` (0.7 s) | back in 0.9 s; all 6 accepted once, byte-exact, `PRAGMA integrity_check` ok, no lease expired; workers exit 0 |
| **crash mid-run:** SIGKILL of the container's PID 1 from the host | the restart policy brings it back (RestartCount 1), healthy in 0.5 s; same result |
| **recreate mid-run:** `docker compose down` + `up` (a new container on the same volumes) | back in 1.2 s; same result |
| **crash mid-upload:** a 40 MiB output in 1 MiB chunks, SIGKILL when 8–9 MiB are in the spool | the partial upload survives in the spool (once with a half-written chunk: 8.5 → 8.8 MiB); the worker resumes from the server's offset (sent 39.2–40.0 MiB in total, not 48+), accepted once, byte-exact |

The local mode runs the same restart scenarios with SIGTERM (graceful exit, code 0), SIGKILL and a respawn.

**Defaults as a user would run them:** named volumes, no `.env`. `docker compose up -d` gives a healthy server in
4 s, with or without `RXMQ_SELINUX_RELABEL=1`. Admin works via `run --rm admin invite` and `exec server rxmq admin
workers`. `down -v` leaves nothing behind.

**Found and fixed on the way:**
- **SELinux:** a bind-mounted host dir is not writable by the container (`unable to open database file`), so
  `RXMQ_SELINUX_RELABEL=1` adds `:z` (opt-in, because it relabels the host dir).
- **manifest.json was written 0600** (`mkstemp`). When admin.py ran as another user (root on the host), the
  container's server could not read it: every data request got 404 and nothing was leased. `write_json_atomic` now
  gives the mode `open()` would (0644 under umask 022). The server also logs an unreadable manifest and keeps the
  last good one.
- The spool and the results must share a filesystem (completion renames). The compose file puts both under one
  volume, and the server falls back to copy + delete on `EXDEV`.
- Docker's port proxy resets connections while a container starts: the test client now treats that as "not up".

**Worker image (not built):** `docker build --check` passes for `Dockerfile.rocm` and `Dockerfile.cuda` (after
re-declaring the base-image ARG after `FROM`). The donor `docker run` flags parse (`docker create` with stand-in
devices: `seccomp=unconfined`, `no-new-privileges`, numeric `--group-add`, `--memory`, `--stop-timeout`).
`docker/worker/compose.yaml` renders with `docker compose config` and refuses to start without `RXMW_SERVER`,
`VIDEO_GID` or `RENDER_GID`.

## 8. v0.2 (w24, 2026-11-05): rename, requirements, multi-card, eval_mix, verification, golden_record

No GPU (none was free): every GPU step is faked on the CPU. `tests/fakes.py` writes valid RAD2 containers and
radiance-layout goldens; `tests/fake_radiance.py` takes radiance's command line (`--kld-ref` / `--kld-out` rows that
are a function of the container's sha256, `nll_p` copied from the golden as radiance does; `--kld-record`), and
`tests/fake_mkgold27.py` takes mkgold27's. `FAKE_RAD_BIAS` makes a cheating donor, `FAKE_RAD_NOISE` a verifier on
"another GPU" (1e-4 relative jitter per row). Workers declare fake cards (`--gpus 0,1,2,3 --gpu-env
CUDA_VISIBLE_DEVICES`); job processes report the device variable they were given.

**Results** (final code):

| suite | where | result |
|---|---|---|
| `tests/radmix_port.py` | VM (radmix built from radiance-hero `008e8ff` headers), the queue host (built there) | **17/17** both |
| `tests/e2e_noop.py`, server as a process | VM, 2:53-2:57 AM CST (~4 min) | **90/90** (earlier code: 89/89 at 2:37 and 2:50) |
| same | the queue host (queue host, Python 3.14.4, `nice -n 10`, other jobs running), 2:54-2:57 AM CST | **90/90** (earlier code: 89/89 at 2:47) |
| `E2E_SERVER=docker`: the rebuilt `rad-quant-at-home-server:0.2.0` image, `compose.yaml` | VM, 3:01 AM CST | **97/97** (earlier code: 96/96 at 2:47) |

Sections 1 and 7's 61 checks all still pass (same scenarios, now with RQAH_* names, one legacy name kept on purpose
in each mode: the worker's second start uses only `RXMW_*`, the server reads `RXMQ_UPLOAD_MAX_MB`, the container
answers to `rxmq` too). New checks:

| scenario | what is checked | result |
|---|---|---|
| radmix port | `common/radfile.py`'s cut + assemble vs the real `mix/radmix.cpp` on a fake 4-level ladder (multi-plane entries, 3 encodings, metadata with an existing `exl3.act` key, a string that only occurs inside another): empty mix, 4 weights from 3 levels, every linear from another level, act lines (one replacing BASE's key, two new), act only, a weight named from the base itself | byte-identical in all 6 (radmix's copy path; sha256 computed in the same pass matches) |
| | bf16 weights identical across levels are one blob; skeleton = the level's bytes before data_off; a piece of another weight / a missing piece / `a8g100` refused | pass |
| requirements | a 16 GB card / 32 GB RAM worker gets only the job that fits; a 32 GB / 64 GB worker gets the 32 GB-VRAM, 40 GB-RAM and head-sized (48 GB RAM) jobs; a 64 GB-VRAM job stays queued, never leased | pass |
| multi-card | one worker with 4 cards: 8 single-card jobs, at most 4 at once (measured 4), each process sees only its own `CUDA_VISIBLE_DEVICES`, no two concurrent jobs share a card | pass |
| gang | a 4-card job queued behind 6 singles and before 2 more: the server reserved the worker (+2.7 s), leased it nothing else while its cards drained, then the gang job (+5.2 s) with `CUDA_VISIBLE_DEVICES=0,1,2,3`; the 2 later singles only after it | pass (0 leases during the drain) |
| RAM budget | 3 jobs of 40 GB RAM on a 64 GB worker with 4 free cards: one at a time | pass |
| stop file | a 2-card worker with 2 jobs running (both heartbeat threads and the main loop see the file): both finish, none abandoned, the third stays queued, exit 0 | pass |
| privacy | `golden-add` refuses a golden with an `agent-trace/` document and anything without `--public`; `corpus-add` refuses `pi-session/` documents; `manifest` never lists a golden marked private; the search API refuses a golden whose marker lies (says public, documents from traces) and one marked private; a leased golden file is refused (403) while its marker says private | pass |
| search API | wrong / worker token -> 401; 6 candidates -> 4 jobs + 2 reused (a duplicate, a mix naming only the base); a missing piece refused; status, rows per candidate, `batch-collect` and `client/searchq.py collect` give the stored bytes | pass |
| eval_mix sizes | bpw from the exact assembled size: base-like mixes 4.500 bpw -> 16 GB cards, mixes with bigger pieces (4.55) -> 32 GB only (never the 16 GB worker), the ~8 bpw model -> a 2-card gang job (`radiance --tp 2`, cards 0,1) | pass |
| eval_mix correctness | every container a donor assembled = the one assembled here from the server's pieces (sha256); its rows = the fake radiance's on that container, byte for byte | pass (8 candidates) |
| cache | the 2nd candidate on a worker downloaded exactly its 2 new pieces (16,384 B of 2.55 MB inputs); a cache capped at one candidate's inputs + 4 KiB evicted 2 LRU pieces and stayed under the cap; results still exact | pass |
| spot checks | a donor's first job of each type is re-run on a verifier; honest donors' eval rows pass against the verifier's jittered re-run (3 donors), noop / quant-like results pass (8 passes in all); a donor writing salted noop bytes and one with KL rows biased by +0.01 fail | pass |
| quarantine | both cheaters quarantined; their unconfirmed results (3 noop, 3 eval) invalidated (files moved to `.invalid-*`, lumens revoked, requeued excluding them) and redone right by others; a quarantined worker's lease gets `{"quarantined": true}` | pass |
| lumens | cheaters back to 0 (4 revoke rows), verifier 0, honest donors > 0, quick-return bonus granted; `admin.py points --json`; the admin summary shows lumens, spot counts, quarantine flags | pass |
| draw rate | in-process, 5,000 accepted jobs of one donor: 211 drawn (4.22%; 4.28% in another run) incl. its first; 1,000 of a verifier: 0 | pass |
| eval accept | the server refuses rows with a changed reference NLL, a negative KL, rows that disagree with the report's mean, a short file; takes the real one | pass |
| golden_record | an untrusted 2-card worker is never leased one; after `trust`: TP2 (gang, cards 0,1) and staged (1 card) recorded and accepted (sizes, documents, sources, radiance's corpus hash, row consistency); the two recorders agree on a chunk; the trusted donor's first golden passed its spot check on a verifier | pass |
| golden accept | a wrong nll column, an unsorted head, a truncated `top.bin`, the golden of another chunk: all refused | pass |
| reflinks | the queue host's XFS (`/data`): cut two fake levels, assemble a mix with and without cloning: whole 4 KiB blocks were cloned (`FICLONERANGE`), the rest copied; both outputs byte-identical, the streamed sha256 = the file's | pass (the VM's tmpfs only exercises the copy fallback) |

**Found and fixed on the way:**
- **Tokens and invite codes could start with `-`** (`secrets.token_urlsafe`), which a command line reads as a flag
  (`--token -x…`: argparse error, the worker never starts). Hit by chance in one run. Now `rq…` / `ri…` (only the
  hashes are stored, so existing tokens stay valid).
- The test's SQLite helper opened a connection per call; under Python 3.14 they are freed only by the cyclic GC and
  reached the queue host's fd limit (1024): `unable to open database file`. Latent since v0 (the VM's limit is 524,288). One
  shared connection now.
- A multi-card worker briefly saw the cards of a job the server had just accepted as busy (its thread was still
  writing stats) and handed the next job back once; a job now frees its cards the moment the server accepts it, and
  `start()` waits up to 10 s for ending threads.
- A malformed job on a worker without a GPU was handed back as "does not fit" instead of rejected: validation now
  comes before the resource check.
- With several jobs, two heartbeat threads seeing the stop file at once would have counted as a second stop and
  abandoned the jobs; only the first sighting counts now.

**Not tested (needs a GPU, an image build or the user):**
- The worker image with radiance inside (`Dockerfile.rocm` copies the radiance image's rootfs and runs it through
  `docker/radiance-wrap.sh`: untested), and any real eval_mix / golden_record: radiance's `--kld-ref` / `--kld-record`
  were faked. Whether radiance accepts a container assembled from pieces is implied by the byte identity with radmix
  (which our GPU box's evals use), not run.
- The eval spot-check tolerance (per-row 1e-3 + 5%, mean 0.2%, top-1 flips 1%) against real cross-GPU spread: calibrate
  on the first donor / verifier pairs. The quant tolerance (1.5% per tensor) rests on the 0.8B seed spread (section 4).
- The quant path through the 0.2 worker on a GPU (the code path is unchanged except the device variable:
  `ourq.py --device 0` inside `CUDA_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`); `tests/quant_e2e.sh` on the 4070 again.

## 9. w25 (2026-11-05): donor CLI, donor API, public status page

Runs (2026-11-05, CST):

| suite | where | result |
|---|---|---|
| `tests/statuspage_test.py` (fixture database, a few seconds) | VM | **19/19** |
| same | the queue host (no xmllint there: that check is skipped) | **18/18** |
| `tests/e2e_noop.py`, server as a plain process | VM, 3:13–3:20 AM | **108/108** (90 earlier + 18 new) |
| same | the queue host (queue host, `nice -n 19`), ~3:23–3:27 AM | **108/108** |
| same, server as the container (`rad-quant-at-home-server-w25:0.2.0`, built `--worktree`; the page written to `/srv/rqah/out/status` by the read-only uid-10001 container) | VM, 3:24–3:29 AM | **115/115** |

**Status page against a fixture** (`tests/statuspage_test.py`). The fixture database holds what must stay private:
- worker names of our machines (`gpuhost-verifier`, `home-4070`) and their tokens;
- an IP in a job error, in a quarantine reason and in a revoke note;
- paths in job parameters, plus a level named `/srv/secret/challenge/secret-level`;
- donors who must not be listed: one who did not opt in, one quarantined who did, and two who opted in with an IP as
  display name or with markup written straight into the database;
- a GPU name with a hostname (`home.lan`);
- markup in campaign.json's title and note.

| check | result |
|---|---|
| well-formed: `xml.etree` parses the page; `xmllint --noout`; `xmllint --html` has no errors other than HTML5 element names | pass |
| shape: element allow-list, balanced tags; no script / form / input / iframe / img / link; no `on*`, `href`, `src`, `style` attributes; meta refresh, charset, title | pass |
| no token, hex-id, IPv4 / IPv6, e-mail, URL, absolute-path or hostname patterns; none of the fixture's secrets (tokens, worker names, the error, the quarantine reason, params, the opted-out and quarantined donors) | pass |
| escaping: operator title and note with `<b>` / `</p><script>` escaped; `render()` given a hostile name, label and state escapes them all (no element appears); names that are markup or an address are left off (counted as "not listed") | pass |
| numbers: ladder 1 / 73 (1 checked) with campaign.json; search 5 evals of 5,000, 3 queued; goldens 1 / 2; spot checks 1 passed / 1 failed / 1 pending (from the events: a quarantine resets the job's `verify_state`); levels in progress; per type running / queued / 24 h / all with re-runs apart; 2 cards busy | pass |
| leaderboard: opted-in donors by lumens (Ada 69.0 / 5 jobs, Zoë 20.0); not the opted-out, the quarantined, the address / markup names or the verifier; "3 donors not listed: 6.5 lumens" | pass |
| recent work: newest first; labels without paths; "by" is a listed name, "a donor" or "our hardware"; states verified / awaiting check / spot check pending | pass |
| `statuspage.py --db --out`: builds from a read-only connection with campaign.json found next to the db; file 0644; the database files unchanged | pass |
| display-name rules: accepts Latin / CJK / emoji names; refuses empty, > 32 chars, IPs, e-mails, URLs, `rq…` tokens, long hex, `<>&`, control / format characters. `scrub()` takes paths, IPs, tokens, e-mails and URLs out of donor-facing text | pass |

**e2e, run_w25** (after the v0 / 0.2 scenarios, on their database; status page every 1 s, `--status-dir` on):

| check | result |
|---|---|
| `status --json` (`--json` before or after the command): only the donor's own worker; lumens and verified jobs equal `admin.py points`; queue depth; could-run count; local running jobs; human form | pass |
| `jobs --json`: only its own leases, states from the documented set, the lumens of `verified` rows sum to its total; `--worker` of another donor's worker returns nothing (isolation); human table | pass |
| quarantined donor: flagged, reason scrubbed ("spot check of job 84 failed (re-run 87)"), results `voided`, a `revoke` in the ledger, 0 lumens | pass |
| `points --json`: total, per worker and type, grants in the ledger; human form | pass |
| `logs --json`: only its own workers' events (lease, accept, points, spot_draw, spot_pass, invalidate), no token / IP / path / URL pattern; `logs --local`: the box's job records | pass |
| `config --json`: token fingerprint, never the token; env caps; profile; the worker's declared caps | pass |
| `config set --display-name`: an IP, `<b>x</b>`, an e-mail and an empty name are refused (HTTP 400, exit 1) | pass |
| opt-in: not on the page before; `config set --display-name "Honest Donor" --leaderboard on` puts it on the page within the rebuild interval; `status` then shows rank 1 of 1 | pass |
| `/`: `text/html`, CSP `default-src 'none'`, nosniff, no-referrer; HEAD 200 with an empty body; POST / PUT → 404 | pass |
| `/` content: no token / IPv4 / path / URL pattern, none of the run's tokens (admin, search, every worker), no worker name, port or work directory; well-formed; search evals = the accepted eval_mix jobs | pass |
| `--status-dir`: the same bytes as served, 0644, no temp files left; `admin.py status-page` and `donors` | pass |
| `link`: e16 + e32 → `status` from either shows both workers with summed lumens; linking to itself moves nothing; a token not of this server → 403 | pass |
| personal invite (`--uses 2 --personal`): both registrations are one donor; registering with `"leaderboard": true` opts in under the worker name | pass |
| exit codes: bad token 3, unreachable server 4, no token 2 (JSON error objects); `/api/v0/me` without a token → 401 | pass |

Refactor checked by the 90 earlier checks: the lease filter moved to `server/sched.py` (shared with the donor API's
"could run" count), and workers from before donors are each given a donor on database open.

**Not tested:** the CLI inside a built worker image (`/usr/local/bin/radquant` is added to `Dockerfile.rocm` /
`.cuda`; neither image has been built yet). A static host serving `--status-dir` (it is one plain file). The page
in a browser: it was read as text and parsed, not rendered visually.

## 10. Verification by score (w47, 2026-11-06; docs/VERIFY.md)

**Synthetic, no GPU** (`tests/score_test.py`, VM, ~1 min): a tiny qwen3_5 GDN layer (every linear 256 × 256), random
PSD Hessians, EXL3 codes made in numpy, weights = decode(codes) + noise. **31/31** (1:28 PM, final code; the shard
merge / subset round trip of section 12 included).

| check | result |
|---|---|
| band logic: repeat = minimum at that K; interp = linear in log between neighbours; extrap along 1.37 / bit; one-sided; k/v's own wider curve table; the noise term 4 · NOISE / √n | pass |
| cold start: first result provisional (no points), one anchor re-run queued for verifiers, `verify_state` untouched; the verifier's re-run scored as trusted rows; the original then passes on the repeat basis and earns; the verifier earns nothing | pass |
| honest repeat by another donor (other codes, same quality): pass | pass |
| perturbed codes (1 % of one tensor's words), random codes, K2 codes padded to K3's shape, a K2 shape declared K3, a wrong `.mul1` marker, a NaN scale, a published model's codes replayed: all fail | pass |
| a failed score quarantines the donor; its score-verified result stands, its unconfirmed (replicate-mode) one is requeued | pass |
| a published import is a replay guard only (state `guard`), never a band reference | pass |
| a duplicate of another donor's codes passes but earns nothing (`duplicate_of`) | pass |
| a result 1/8 bit worse than its interpolated neighbours fails every tensor; one 2 % off the curve passes | pass |
| re-check: 6 % above the curve passes on a reference 1 bit away (extrap, tol 7 %); when neighbours on both sides arrive it is re-decided on interp (tol 4 %): fail, quarantine, lumens revoked, the job requeued for anyone else | pass |
| best of N: the export takes the passed copy with the lowest total error | pass |

**Real quantizer output** (`tests/score_real.py`, scoring on the 4070 box's CPU, 4 threads, niced; the runs from the
4070 under its GPU lock, 12:59–1:07 PM): Qwen3.5-0.8B layers 6 (GDN) and 7 (full attention), ladder settings (i8, qtf2,
scaled Hessian, σ 0.025 / 0.005): K 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4 (seed 0), K3 seeds 1-4, K2.5 seeds 1-2, and
three real attacks. A campaign through the real scorer in a fresh queue: **14/14** (1:21 PM, final tolerances).

| step | result |
|---|---|
| K3, first result per layer: provisional, anchor queued; the anchor (a verifier, seed 1) → both layers pass (repeat) | pass |
| honest repeats, seeds 2-4 by three other donors | 6/6 pass |
| the other levels in ladder order (2.5, 3.5, 2, 4, 2.25, 2.75, 3.25), each on the curve of the earlier ones (interp, or extrap for K2 / K4) | 14/14 pass |
| honest repeats at K2.5, seeds 1-2 | 4/4 pass |
| the scorer's rel = ourq's own logged proxy_err on every honest tensor | worst \|ratio − 1\| 1.2e-4 |
| attack: K3 codes of the K2 dequantized weights (K2 quality in K3 codes; real ourq run) | fail (total rel 0.0163 vs honest 0.0033, ×5) |
| attack: K3 with the Hessian damped away (σ 1.0: no LDLQ to speak of; real ourq run) | fail (+8 % / +9 % total rel) |
| attack: 1 % of one tensor's trellis words changed | fail |
| attack: random codes | fail (rel 1.4) |
| attack: K2 codes padded to K3's shape | fail (rel 7-9) |
| attack: an honest K2.75 shard declared K3 | fail (structure) |
| info: K3 codes made with the mul1 codebook, declared i8 | pass (+0.9 % / +1.0 %): i8 is int8-rounded mul1, so these are good i8 codes; the score judges codes as the kernel decodes them |
| every honest result and the verifier's still accepted after the attacks | pass |
| scoring time per 0.8B layer | 0.7-2.2 s |

A first run (3 % repeat tolerance, no noise term) showed the honest **0.8B seed spread** is larger than that: 5 seeds of
K3, per tensor range of log rel up to 4.1 % (in_proj_z), 3.4 % (q_proj), 1.8 % (in_proj_qkv) and ≤ 1.3 % for the rest
(`tools/seed_spread.py`). Not a size effect alone: SD · √n (n = in × out) is up to 26 for the layer-input group
(q/k/v_proj, in_proj_qkv, in_proj_z, which use the lighter input damping) and ≤ 6.3 for the rest. Hence the noise
term `4 · NOISE / √n` with NOISE 30 / 8 (docs/VERIFY.md).

**27B, honest spread and cost** (layer 10 from the HF checkpoint by range, its calib Hessians; K2 seeds 0-2 on the 4070,
1:02–1:07 PM; `score_real.py --only-spread`, **2/2**): per-tensor range of rel over the 3 seeds ≤ 0.10 % (out_proj),
≤ 0.02 % for the rest (SD · √n ≤ 2.9). Scorer vs ourq's log: worst 3.0e-4. **Scoring a 27B layer: 30 s the first time
(including the one-time tr(W H Wᵀ)), ~18 s after (mean 22 s of 3), 4 threads, niced**, while the quant of that layer took
96-107 s on the 4070. On an existing K2.5 shard of the same layer (an earlier convergence run): 32 s, scorer vs log 1.6e-4.

**Curve data** (`tools/curve_loo.py` on rad-exl3-mix `data/v3/levels`, 400 tensors of the 27B at K 2-4.5, i8): slope
1.358-1.370 per bit; neighbour-interpolation residual SD 0.15-0.26 % (max 0.4-1.5 %) except k/v_proj (SD 1.0-1.3 %,
max 3.5-6.4 %); extrapolation SD ~0.3-1 % per bit (k/v up to ~6 %). That data mixed damping settings for k/v.

## 11. Small cards: VRAM and host RAM per 27B tensor (w47, 2026-11-06; docs/SMALL-CARDS.md)

RTX 4070 (12 GB), torch's allocator capped at 7.2 GB (the shared box's limit, ≈ an 8 GB card), `tools/memwrap.py` around
`quantize/ourq.py` with the ladder settings, one process per tensor (`--only`), `OURQ_REFIT_CPU_FALLBACK=1`, no
`--save-wq` (score mode). Layer 10 (GDN) and layer 39's attention from the HF checkpoint by range (`tools/hfrange.py`,
975 MB). Units under the box's GPU lock: a (K2 + K4) 12:39–12:44 PM, b (K6) ~12:52–12:55 PM; unit d below.
VRAM = torch peak allocated / reserved in GB (+ ~0.3-0.5 GB CUDA context); RAM = peak RSS; time = wall incl. ~1.5 s start.

| tensor | K2 | K4 | K6 | RAM (GB) | time K2 / K4 / K6 (s) |
|---|---|---|---|---|---|
| L39 k_proj | 1.62 / 1.76 | 0.86 / 1.00 | 0.68 / 0.82 | 1.75 | 2.7 / 2.4 / 4.5 |
| L39 v_proj | 1.62 / 1.76 | 0.86 / 1.00 | 0.68 / 0.82 | 1.76 | 2.0 / 2.0 / 3.1 |
| L39 q_proj | 3.31 / 3.48 | 2.55 / 2.74 | 2.36 / 2.54 | 1.86 | 8.1 / 7.6 / 15.7 |
| L39 o_proj | 2.44 / 2.62 | 1.69 / 1.86 | 1.50 / 1.66 | 1.92 | 5.1 / 4.8 / 9.0 |
| L10 in_proj_qkv | 2.98 / 3.15 | 2.22 / 2.39 | 2.03 / 2.21 | 1.85 | 7.1 / 6.7 / 13.1 |
| L10 in_proj_z | 2.31 / 2.50 | 1.56 / 1.74 | 1.37 / 1.55 | 1.81 | 5.0 / 4.4 / 8.1 |
| L10 out_proj | 2.44 / 2.62 | 1.69 / 1.86 | 1.50 / 1.66 | 1.92 | 5.3 / 4.8 / 8.4 |
| L10 gate_proj | 3.77 / 4.14 | 3.01 / 3.38 | 2.82 / 3.19 | 1.92 | 11.1 / 10.4 / 20.6 |
| L10 up_proj | 3.77 / 4.14 | 3.01 / 3.38 | 2.82 / 3.19 | 1.91 | 11.0 / 10.5 / 20.8 |
| L10 down_proj | 6.24 / 6.59, refit on the CPU | 6.61 / 6.86, refit on the CPU | **fails**: cuBLAS EXECUTION_FAILED at 5.65 / 7.17 | 9.6 / 9.8 (CPU refit) | 63.9 / 61.1 / – |
| L10 down_proj K2, `--save-wq` (replicate mode) | 6.24 / 6.90, CPU refit | | | 10.24 | 67.7 |
| L10 down_proj K2, no CPU fallback | OOM: "tried to allocate 1.13 GiB" at 6.24 allocated | | | 4.49 | – |

Unit d (1:19–1:24 PM, after waiting ~23 min for the lock): down_proj **K3** 6.86 / 7.11 (CPU refit, 10.39 GB RAM, 127 s
wall, 64 s of it the quant), **K5** 6.38 / 7.14 (CPU refit, 10.39 GB, 65 s), **K6 with `CUBLAS_WORKSPACE_CONFIG=:4096:8`**
6.47 / 7.04 (CPU refit, 10.39 GB, 73 s): the K6 failure was cuBLAS's workspace; the jobs set that variable on cards
below 12 GB since 0.3. q_proj **K8**: 2.30 / 2.49, 9 s. Total GPU time of my units: a ~6 min, b ~3 min, c ~8 min,
d ~5 min, each under the box's 15-min rule.

So on an 8 GB card every 27B trunk tensor quantizes at K2-K6 with the CPU refit for down_proj; without that fallback
down_proj needs ~7.4 GB allocated (~8 GB with the context: a 10-12 GB card). Split proposal and overhead:
docs/SMALL-CARDS.md.

## 12. The rename, the suites after it, and the CUDA image (w47, 2026-11-06)

**Rename** `radquant` → `rad-quant-at-home` (the repo and its directory, images `rad-quant-at-home-server` /
`-worker`, command `rad-quant-at-home` + alias `rqah`, `client/rqah.py`, `RQAH_*`; the 0.2 `RADQUANT_*` still read,
the v0 `RXMQ_*` / `RXMW_*` / `rxmq` gone; volumes `rqah-*`; image paths `/opt/rqah`, `/var/lib/rqah`, `/srv/rqah`).
The repo had been cut out of rad-exl3-mix's `volunteer/` without fixing its build: `docker/build.sh` (worker),
`docker/server/build.sh`, `run-local.sh`, `quant_e2e.sh`, `dec_rates.sh` still looked for `volunteer/` and
`quantize/ourq.py` inside the same tree. Now the worker image is built from two trees (`app/` = this repo, `rxm/` = the
quantizer paths of a rad-exl3-mix commit, `RXM_REPO` / `RXM_COMMIT`), and the scripts take `RXM` for the checkout.

| suite | where | result |
|---|---|---|
| `tests/e2e_noop.py`, server as a process, right after the rename | VM, 12:39-12:43 PM | **108/108** |
| same, final 0.3 code | VM, 1:18-1:22 PM | **108/108** |
| same, server as the rebuilt container (`rad-quant-at-home-server:0.3.0`, `--worktree`, 7 s build) | VM, 1:29-1:33 PM | **115/115** |
| `tests/statuspage_test.py` (fixture paths and machine names neutralized) | VM | **19/19** |
| `tests/score_test.py` | VM | **31/31** (+ the shard merge / subset round trip) |
| `tests/score_real.py` (0.8B campaign; 27B spread) | 4070 box, CPU | **14/14**; **2/2** |
| `tests/cuda_smoke.sh MODE=native` (the box's own torch / exllamav3; the new job, worker and scorer code on a GPU) | 4070, under its lock 1:27:22-1:27:32 PM | **PASS** (below) |
| `tests/radmix_port.py` | not re-run (`common/radfile.py` unchanged) | – |

**Native smoke** (`MODE=native`, 0.8B layer 6, K3, `--verify score --unit group`): datapack `--groups`, 4 `quant_tensor`
jobs; a donor worker took one (group 0: in_proj_qkv + in_proj_z; 18.9 MB down, 3.2 MB up, 3.0 s of quantization, 1.1
GB VRAM), the scorer (`--score-threads 1`) made it provisional and queued its anchor; a verifier worker took the
anchor; both scored pass (rel 3.75e-3 / 1.98e-3, the donor on the repeat basis), the donor got 0.55 lumens (0.5 + the
quick-return bonus), the verifier 0. The two runs (same card, same software) returned **bit-identical codes**: the
quantizer is deterministic on one stack.

**The CUDA worker image.** `docker/build.sh cuda --worktree` on the VM, through a buildx builder capped at 2 CPUs /
4 GB (the VM is shared): the base image, the apt step, torch 2.11.0+cu128 and the pinned packages, and the COPY of
the quantizer tree all built; then the exllamav3 extension compile **did not fit**. One `cc1plus` for one exllamav3
unit takes ~5.8 GB (4.1 GB resident + 1.7 GB swapped), so even one compile job thrashed the builder (100+ GB of swap
reads) while the VM went into memory pressure. I stopped it and removed the builder. The Docker build needs a host
with ~6 GB of free RAM per `MAX_JOBS` (the default 8 wants ~48 GB; `BUILD_ARGS="--build-arg MAX_JOBS=2"` ~12 GB).

The 4070 box has no Docker (and no root, no user namespaces), so its steps were replayed there without Docker:
`tests/cuda_rootfs.sh` reads the pins from `Dockerfile.cuda` and builds the same tree under a prefix (`opt/venv`:
python 3.12 venv, torch 2.11.0+cu128 and the pinned packages from the same indexes; `opt/rqah`: app/ + rxm/ as in the
image, exllamav3 a3bfbfd0 + the 0001-0007 patch series built for sm_89, the six side extensions via `docker/warm.py`).
CPU only, niced, 2 compile jobs: 1:27-1:52 PM (venv 4.5 min, exllamav3 14 min, side extensions 6 min), 7.0 GB.
`import exllamav3...quantize` OK. Then **`tests/cuda_smoke.sh MODE=rootfs`: PASS** (GPU lock 2:01:20-2:01:30 PM): the
side extensions were taken from the tree (5 s, no recompile); a donor worker from the tree took one `quant_tensor`
job (4.5 s), its anchor ran on a verifier worker from the tree, both passed by score, the donor got 0.55 lumens.
Peak 1.1-1.2 GB VRAM per process. **The codes are bit-identical to the native run's** (same sha256 per tensor): the
image's stack quantizes exactly like the box's own.

**Not tested** (needs a Docker host with an NVIDIA GPU, i.e. the user): `docker build` of `Dockerfile.cuda` to the end;
`docker run --gpus all` of the image (device flags, `--cap-drop ALL`, the uid-10001 user, the read-only code paths,
`/usr/local/bin/rqah` inside it). The commands:

```bash
# on a host with Docker, the NVIDIA Container Toolkit and >= 16 GB free RAM; rad-exl3-mix checked out next to this repo
cd rad-quant-at-home
BUILD_ARGS="--build-arg TORCH_CUDA_ARCH=8.9 --build-arg MAX_JOBS=2" docker/build.sh cuda          # ~30-60 min
docker run --rm --gpus all --entrypoint python rad-quant-at-home-worker:cuda -c "import torch; print(torch.cuda.get_device_name(0))"
MODE=docker WORK=/tmp/rqah-smoke CKPT=<Qwen3.5-0.8B dir> HESS=<its calib Hessians> IMAGE=rad-quant-at-home-worker:cuda \
  tests/cuda_smoke.sh                                 # expects "SMOKE PASS"; the host needs python3 + numpy for the server
```

## 13. Bandwidth limits and scheduling (0.4, w57, 2026-11-06)

All on the VM, no GPU (noop jobs); the server and workers ran as processes on localhost.

| suite | when (CST) | result |
|---|---|---|
| `tests/e2e_noop.py`, 0.3 code (baseline before the changes) | 4:16-4:20 PM | **108/108** |
| `tests/score_test.py`, `tests/statuspage_test.py`, 0.3 code | 4:16 PM | **31/31**, **19/19** |
| `tests/sched_test.py` (new; in process, ~5 s) | 4:48 PM | **26/26** |
| `tests/bandwidth_e2e.py` (new; ~2 min) | 4:41-4:43 PM | **22/22** |
| `tests/e2e_noop.py`, server as a process, 0.4 code | 4:38-4:41 PM | **108/108** (one test fixed, below) |
| same, server as the container (`rad-quant-at-home-server:0.4.0`, `--worktree`, 6 s build) | 4:44-4:48 PM | **115/115** |
| `tests/score_test.py`, `tests/statuspage_test.py`, 0.4 code | 4:48 PM | **31/31**, **19/19** |

**Rate caps** (`bandwidth_e2e.py`, `--max-down-mbps 2 --max-up-mbps 2`, i.e. 250,000 B/s). A noop job with a 7.5 MB
input and a 7.5 MB output: fetch **29.77 s**, upload **29.77 s** (30 s expected, ±10% allowed; two runs: 29.76 /
29.77 s). A Range-resumed fetch with half of the blob already in `<sha>.part`: 3,750,000 B in **14.76 s** (15 s). The
bucket alone: 500 kB in 1.75 s (2 s minus the quarter-second burst). The worker paces its reads from the socket and
the bytes it hands to it, so the link sees the cap after the first TCP window.

**Budgets.** The counters, with a fake clock (TZ America/Chicago, reset day 15): the tighter of day and month; kept
across instances; the next reset is local midnight when the day is spent, the 15th when the month is; a new day and a
new budget month zero their counters; January 14 still counts in the month that began December 15. Running jobs'
reservations come off the declared budget. End to end: with `--max-down-gb-day 0.0025` and four noop jobs with
distinct 1 MB inputs, the worker was SIGKILLed after its first job. Its `bandwidth.json` held 1,000,000 B. Restarted,
it took exactly **one** more job (2,000,000 B counted; 0.5 MB left is below the 1 MiB "spent" floor), then got
`over_budget` from the server for the other two, logged "data budget spent (download) ... until <local midnight>"
and idled out. `rqah status` shows the DATA LEFT column, the spent note and this machine's counters (`--json`:
`workers[].budget`, `local_budget`). The status page shows "1 workers at their data cap until it resets". A job whose
input was already cached still ran on the spent download budget (0 B down). `--max-up-gb-month 0.0025
--budget-reset-day 5` with three 1 MB outputs: two uploaded (2,002,000 B counted), month key `...-05`, the third
stayed queued.

**Scheduler** (`sched_test.py`, `Queue.lease` in process):
- `level_priority`: 3 → 40, 2.5 → 30, 2.25 → 20, 2.125 → 10, 2.0625 → 5, finer → 0, plus the boost.
- `enqueue-level` on a fake datapack gives the same, with heads +1 and `--priority-boost`.
- `reprioritize 'q-k2.*' --by-granularity --priority-boost 1` changes only the matching levels, and `--dry-run`
  changes nothing. `--priority 7` leaves an accepted job alone, and a pattern that matches nothing is an error.
- Classes strictly in order, and inside a class whole levels first: with ids interleaved A1 B1 A2 B2 A3, the
  workers got A1 A2 A3 B1 B2.
- Cache affinity inside a level: a cached A3 came before A1 and A2, B came after A, and a cached lower-class job came
  last. A budgeted worker got the least download first.
- Budget filter: only the cached job fit a 5000 B download budget. Otherwise the answer was `over_budget` with a
  ≥ 60 s retry. An upload budget 1 byte short of a job's expected upload refused that job.
- Randomized: 250 lease requests against 300 random jobs with random budgets and cache sets. 231 granted, 19 refused,
  and **no grant over budget**. Every `expect` equalled the true uncached download and the upload.
- Cache report protocol: an unknown rev → `cache_unknown`, the list is stored, the stored caps keep only the rev and
  count, and after a restart the rev is unknown again.
- Class-wait drain: a fresh higher-class job that does not fit now lets a lower-class job through. Aged past
  `--class-wait-s`, it drains the worker (reservation), the worker gets nothing while draining, and it gets the job
  once the room is free. An aged job does not drain a worker for work of its own class, nor for a job over the
  worker's budget.
- Lease cost: 29 ms per request with 5000 queued jobs in one class, 36 ms with a 1000-entry cache report and a
  budget.

**Changed test.** `e2e_noop.py`'s disk-cap scenario queued inputs a, b, c, a in one level and expected two evictions
(FIFO). With cache affinity, the worker takes the second `a` job right after the first, because it holds that input,
so only one eviction happened. The four jobs now get one level each, which keeps FIFO order and the two evictions.
`bandwidth_e2e.py` checks the affinity itself with a real worker: jobs m1, m2, m1 of one level ran as m1, m1, m2, with
1,000,000 / 0 / 1,000,000 B downloaded.

