# rad-quant-at-home: operator runbook

Start the server, load a job batch, watch it, verify the results, pay points. Everything here runs on the queue host
(the server container) or on our own boxes; nothing goes over HTTP that changes the queue except the search's batch
endpoint (its own token, eval_mix on public goldens only). Commands use `A` for admin.py:

```bash
# in the container (data and results read-write):  A="docker compose run --rm admin"
# beside it, inside the running server:            A="docker compose exec server rqah admin"
# plain process:                                   A="python3 -I server/admin.py --db Q.sqlite --data DATA --results RESULTS"
```

## 1. Start the server

1. Build the image from a commit: `docker/server/build.sh <commit>` (tags `rad-quant-at-home-server:0.2.0`, `:0.2.0-<commit>`).
2. `cd docker/server && cp .env.example .env`; set the volume directories (`RQAH_DB_DIR`, `_DATA_DIR`,
   `_OUT_DIR`, `_SOURCES_DIR`: local disk; the DB not on NFS; spool and results share `OUT`), `RQAH_PUBLISH`
   (loopback or the tunnel's address) and, on SELinux hosts, `RQAH_SELINUX_RELABEL=1`. `chown 10001:10001` the
   DB and OUT directories.
3. Secrets in `RQAH_SECRETS_DIR` (≥ 32 random characters each, `chmod 0400`, owned by 10001):
   `admin-token` (read-only summary) and `search-token` (the search's batch API), and set
   `RQAH_ADMIN_TOKEN_FILE=/run/secrets/rqah/admin-token`, `RQAH_SEARCH_TOKEN_FILE=/run/secrets/rqah/search-token`.
   `python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/admin-token`.
4. `docker compose up -d && docker compose ps` (healthy within seconds).
5. TLS: a terminator in front (Caddy: `queue.<domain> { reverse_proxy 127.0.0.1:8740 }`) or a tunnel. Nothing is
   public until this step; it needs the user's go-ahead (outward-facing).
6. Our own workers: a verifier for spot checks on one of our cards (`run-local.sh` on the GPU box under its card lock,
   `--gpu-lock '/var/lock/gpu-card{card}.lock'`), registered with `$A add-worker verifier-1` and flagged
   `$A verifier ID`. Without a verifier running, drawn results wait (their points stay pending).
7. Donors: `$A invite --uses 1 --note "<who>"` per donor; send the code and the queue URL privately. One person with
   several boxes: `$A invite --uses 3 --personal --note "<who>"` (every registration with it is one donor). Big-box
   donors we know personally: `$A trust ID` after they registered.
8. Status page (public, read-only; on by default every 60 s at `/`): write `campaign.json` into the DB directory
   (`{"title": ..., "ladder": {"levels": 73}, "search": {"evals": N}}`; README "Public status page"). For a static
   host as well: `RQAH_STATUS_DIR=/srv/rqah/out/status` and publish that directory (e.g. `rsync` from cron).
   Look at it before anything is public: `curl -s http://127.0.0.1:8740/ | less`.

## 2. Load data (once per model / level / golden)

```bash
$A datapack qwen38-27b <rev> /srv/rqah/sources/qwen38-27b-bf16 /srv/rqah/sources/hessians/calib \
   [--hout <snap> <dir>] [--hf] [--bf16-rad /srv/rqah/sources/qwen38-27b-bf16.rad]   # also writes model.json
$A ladder-add qwen38-27b <rev> k3-i8 /srv/rqah/sources/ladder/k3-i8.rad           # each level the search mixes from
$A golden-add xcorp-s07 /srv/rqah/sources/goldens/xcorp-s07 --public --corpus xcorp --pool search
$A corpus-add xcorp-m2 /srv/rqah/sources/corpora/xcorp-m2.jsonl --public --chunk-docs 64  # for golden_record
$A manifest --jobs 8                                                                         # after every change
```

- Data rules (AGENTS.md): **only public data.** `golden-add` / `corpus-add` refuse anything without `--public` and any
  document whose source is one of ours (agent traces, session logs); `manifest` never lists a golden or corpus whose
  marker is not public; the server re-checks the marker on every file it serves. Never put a trace-derived golden in
  the data root at all. A golden built from calib must be in the document-use ledger (`eval/pools/ledger.jsonl`) and
  follow the pools: search accept / reject only on search-pool or xcorp goldens, never verify.
- Disk: `ladder-add` writes each level once more as pieces (reflinked from the `.rad` on XFS / btrfs, copied
  otherwise: ~10-17 GB a 27B level). `df -h` first.

## 3. Load a job batch

- **A ladder level** (quant jobs): `$A enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8
  <frozen settings> --verify score --head` (64 layer jobs + the head, verified by score: docs/VERIFY.md; requirements
  from model.json: 27B layers 8 GB VRAM up to K6 (12 GB above) / 12 GB RAM, the head 32 GB / 48 GB RAM).
  - Small cards: `--unit group` makes one `quant_tensor` job per Hessian group (4 per layer; needs `datapack --groups`
    once): 6 GB cards take everything but down_proj; 8 GB cards already take whole layers up to K6 (docs/SMALL-CARDS.md).
  - Best of N: `--copies 2` queues each job twice with seed_offset +0 / +1; every copy in its band is accepted (and
    gives the band a free honest repeat), the export keeps the better one. A donor (all its tokens) is leased at most
    one copy of a job, and a copy whose codes nearly equal another result's earns nothing (#11). With fewer donors
    than copies, the spare copies wait for another donor or a verifier.
  - Anchors: the first result for each tensor of a new series waits (provisional) for one re-run on a verifier. To
    skip that, import a level we quantized ourselves with the same settings first: `$A score-import --level k3-i8
    /path/to/ourq/output` (q-LNN.safetensors files). Released models' shards: `$A score-import --level L DIR --published
    "v2 k3"` (replay guard only).
- **Search candidates** (eval_mix): from the search box, `client/searchq.py --server https://<queue> --token-file
  search-token submit SPEC.json`, then `wait ID` and `collect ID OUT` (rows per candidate in radiance's FILE.rows
  layout). Or on the queue host: `$A batch-submit SPEC.json`, `batch-status ID`, `batch-collect ID OUT`. The spec is in
  `server/batch.py`. Repeated candidates reuse their earlier result. Give every candidate the knapsack's prediction
  (`"pred_kld"`): far-worse results are refused, far-better ones re-run in full before they count (#4). Before the
  search adopts a candidate: `searchq.py adopt ID CAND... --wait` (or `$A batch-adopt ID CAND...`) queues its full
  trusted re-run; use it only once its `trust` is `confirmed`.
- **Reference goldens** (trusted big boxes): `$A enqueue-golden --model qwen38-27b --rev <rev> --corpus xcorp-m2
  --method tp2|staged --top 512 --max-model-len 32768`.
- Priorities (docs/SCHED.md): higher first, strictly by class. Ladder levels default by K's granularity: integer K 40,
  halves 30, quarters 20, eighths 10, sixteenths 5 (heads +1); `enqueue-level --priority-boost B` / `--priority P`.
  Later: `$A reprioritize 'k2.*' --by-granularity --priority-boost 5 --dry-run` (then without `--dry-run`), `$A queue`
  shows the order. Everything else defaults to 0 (`--priority` on `enqueue-noop` / `enqueue-golden`, the batch
  spec's `priority`), so search batches and goldens queue behind every ladder level unless given a priority.

## 4. Watch

- `$A list [--level L] [--state S]`, `$A events [JOB]`, `$A workers` (flags, declared caps, last seen).
- `curl -H "Authorization: Bearer $(cat secrets/admin-token)" http://127.0.0.1:8740/api/v0/admin/summary`: counts per
  level and state, workers, lumens, spot-check counts, gang reservations, spool bytes, the last events.
- `docker compose logs -f server`.
- Donors: `$A donors` (display name, opt-in, workers, lumens, rank of the name's row). The board credits by display
  name (case-insensitive): every opted-in donor showing a name adds to its row, so one person's machines are
  credited together by using one name; there is no linking. A name that should not be on the page: `$A donor ID
  --hide` hides **that name for every donor showing it** (`--unhide`; a hidden donor cannot rename itself). To
  take one donor off a row without hiding the name: `--leaderboard off` (the donor may opt in again) or
  `exclude-donor` if it cheated. A quarantined worker's lumens drop out of its name's row by themselves; nothing else
  happens to the name. Donors see their own side with `radquant status | jobs | points | logs`.
- A job that keeps failing: `$A events JOB` (worker errors are in the `fail` events), fix, `$A requeue JOB`. A queued
  job no worker takes (every capable worker failed it within the last `--refail-s`, default 1 h): `$A requeue JOB`
  resets its attempts and the exclusions at once. A lease ends at its deadline even while it heartbeats
  (`RQAH_LEASE_CAP_X` 3 × the type's reference runtime, at least `RQAH_LEASE_CAP_MIN_S` 2 h; `expire` events with
  "past its deadline"): the job is reissued, and the old holder's result still counts if it comes first.
  A level to redo: `$A requeue --level L --force` (accepted results are moved aside). Stop work: `$A cancel --level L`.
- A gang job waits for a multi-card worker to drain (`reserve` events); a drain that has not finished after
  `RQAH_RESERVE_MAX_S` is given up and may be retried by another worker (not by the same one for another
  `RQAH_RESERVE_MAX_S`).

## 5. Verify

- **Quant levels, score mode:** keep a scorer running: `server.py --score-threads 1` (`RQAH_SCORE_THREADS=1`; ~2.6 GB
  RAM at the 27B, so raise `RQAH_MEM_LIMIT` to 4g) or, with the container's 2 GB limit, `docker compose run --rm admin
  score --loop 30` beside it (8 CPUs / 8 GB by default). ~20-30 s per 27B layer at 4 threads. Pass: points; fail: the
  donor is quarantined and its unconfirmed results requeued (its score-verified ones stand); provisional: an anchor
  re-run is queued for a verifier. `$A scores --level L` shows every tensor's error, basis, band and state; `$A events
  JOB` the `score_*` events. A `score_trusted_fail` event means one of our own verifiers returned broken codes: look at
  that box.
- **Quant levels, replicate mode:** `$A check --level L --rows-frac 0.25` (CPU checker: ~15 s per 27B layer sampled,
  ~1 min full; `RQAH_ADMIN_CPUS` / `_MEM_LIMIT` bound it). A FAIL quarantines the donor and requeues its unconfirmed
  results; `--no-quarantine` to look first. An `ERROR` (the checker's own failure: a missing Hessian, an I/O error) is
  recorded as `check_state='error'`, quarantines nobody, and is retried by a later `$A check --level L --recheck`.
- **Spot checks** run by themselves: 4% of every donor's accepted results plus its first of each type are re-run on a
  verifier and compared (`spot_draw`, `spot_pass`, `spot_fail` events; `verify_state` in `$A list`). `$A spot JOB`
  draws one by hand. eval_mix: 8% (`RQAH_EVAL_SPOT_RATE`), each re-scoring 3 random documents; suspect (far better
  than predicted) and adopted candidates get a full re-run (`trust_draw`, `trust_pass`, `trust_fail` events). Keep a
  verifier running while the search works. `suspect` events list results far better than predicted;
  `implausible_to_verifier` a job two donors returned far worse than predicted (check the prediction).
- **On a `spot_fail`:** the donor is already quarantined and its unconfirmed results requeued. Read the comparison
  (`verify_detail` of the job: worst tensor / rows out of tolerance / container hashes). A hardware fault or a
  tolerance that is too tight (eval rows: calibrate on the first R9700 pairs) is lifted with `$A unquarantine ID`; a
  cheat stays, and `$A disable-worker ID` revokes the token. A donor caught faking (any of its workers): `$A
  exclude-donor DONOR --reason ...` quarantines all its workers and invalidates every result of theirs that nothing
  confirmed (the search's batches show those candidates as invalidated). Nothing does this donor-wide by itself.
- **Export** only verified work: `$A export --level L OUT --checked` (ourq layout: knap4 and the search read it; of
  best-of-N copies the passed one with the lowest error, `quant_tensor` groups merged into one shard per layer).

## 6. Pay points

Lumens are granted by the server as results are verified and revoked on quarantine; nothing to do per job.
`$A points` (table) / `$A points --json` (per worker: id, name, lumens, verified jobs, flags); `$A donors --json`
per donor (the leaderboard sums the opted-in donors of one display name, without quarantined workers). Worker names are what donors registered with and are never published;
the status page shows only opted-in display names. Credits in the model cards: the donors with lumens
on that release's levels (`$A list --level L` has the `by` worker of each job).

## 7. Upgrades and backups

- A job type's meaning changes → bump its `VERSION`; a release → bump `VERSION`. Deploy the server first,
  then the new worker image; old workers stop getting the changed type.
- 0.3 renamed the project to rad-quant-at-home: `RQAH_*` variables (the 0.2 `RADQUANT_*` are read in 0.3 with a
  deprecation note and removed in 0.4), the `rad-quant-at-home` / `rqah` command, images `rad-quant-at-home-server` /
  `-worker`, volumes `rqah-*`. The v0 names (`RXMQ_*`, `RXMW_*`, `rxmq`) are gone.
- Backups: `sqlite3 queue.sqlite ".backup q.bak"` (consistent under WAL) and the results directory; the data root can
  be rebuilt from the sources (`datapack`, `ladder-add`, `manifest`).
