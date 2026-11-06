# Scheduling: which job a worker gets (0.4)

The server answers every lease request with at most one job. This file describes how it picks that job. The code is
`Queue.lease` in `server/server.py` and the helpers in `server/sched.py`; `tests/sched_test.py` checks every rule
below.

## 1. Hard filters (a job the worker could never run now is skipped)

A queued job is a candidate for a worker only if:

1. **Static fit.** The worker runs the job's type and version. Its declared cards, VRAM per card, host RAM and disk
   (`--cache-gb`) cover the job's `requires`. Trusted-only types go to trusted workers, spot-check re-runs to
   verifiers (never to the original donor), and quarantined workers get nothing (all as in 0.2).
2. **Data budget (0.4).** The worker may declare the bytes left in its data budget (`budget.down`, `budget.up`; see
   §4). Two numbers are compared with it:
   - **Download:** the sizes of the job's inputs whose sha256 the worker's cache report does *not* list. Inputs the
     worker already holds cost nothing.
   - **Upload:** the job type's expected upload, `upload_bytes(params, input sizes)` in `jobs/<type>.py`. For a
     quant job that is the bf16 weight file's parameter count × K / 8 plus the small files. For `eval_mix` it is
     16 B per scored position. For `golden_record` it is the top-N rows. For `noop` it is `size`.

   A job is offered only if both numbers fit. There is no exception for priority: an over-budget job is never
   offered.
3. **Room now.** The job's cards and RAM must fit beside the worker's active leases. A job that fits statically but
   not now can trigger a drain (§3).

When jobs were skipped only because of the budget, the answer is `{"job": null, "over_budget": N, "why": ...}` with
a retry of at least 60 s. The worker logs it once.

## 2. Order of the candidates

**Priority classes are strict.** `jobs.priority` is an integer, and every distinct value is a class. Classes are
scanned from the highest down. A job of a lower class is leased only when no candidate of a higher class fits now.
Nothing below reorders jobs across classes.

**Ladder levels get their class from K's granularity** (user, 2026-11-06), so the coarse steps come first:

| K | example | default priority |
|---|---|---|
| integer | 2, 3, 4 | 40 |
| halves | 2.5 | 30 |
| quarters | 2.25, 2.75 | 20 |
| eighths | 2.125 | 10 |
| sixteenths (head budgets, if ever) | 2.0625 | 5 |
| finer | | 0 |

- `admin.py enqueue-level` sets this priority by default. `--priority-boost B` adds B, and `--priority P` replaces
  the default.
- A level's `quant_head` job gets one more than its layers: it is the longest job of the level, so it should start
  first.
- `admin.py reprioritize PATTERN --by-granularity [--priority-boost B]` (or `--priority P`) re-sets the queued and
  leased jobs of every level matching a shell pattern (e.g. `'q27-k2.*'`). Heads again get +1. Add `--dry-run` to see
  the changes first.
- `admin.py queue` lists the queue in this order.
- Spot-check re-runs keep the original's priority + 10 (they only go to verifiers).

**Inside a class:**
- **Whole levels first.** Each level is ranked by the id of its first job, whatever that job's state, so a level keeps
  its place when some of its jobs are requeued. The earliest-ranked level's queued jobs come before any job of the
  next level. A level therefore finishes and becomes usable before the next one starts, even when the job ids of two
  levels interleave (e.g. levels enqueued in parts, or requeues). Jobs without a level rank by their own id.
- **Then cache affinity, inside the level.** Among one level's jobs, those whose inputs the worker already caches
  (most cached bytes first) come before the rest, and ties go by id. A worker that just quantized layer 12 at K3 and
  still holds its weights and Hessians gets layer 12 of the next level when that level's turn comes.
- **Budgeted workers** (a finite `budget.down`) are ordered by least download first, then level, then id. A donor on
  a data cap gets the fully cached jobs first, then the smallest downloads. This can interleave levels, but only
  within the class.

Cache affinity and the budget never move a job across classes.

## 3. No starvation of a higher class

Strict class order can still starve a job when the worker that could run it is always busy. Say a 48 GB head job
waits on a box whose RAM is half taken by smaller jobs. Each time a small job ends, the free room goes to the next
small job, of a lower class. Two drains prevent this:

- **Gang jobs** (cards > 1), as in 0.2. Once one has waited `--gang-wait-s` (default 0), the first busy worker that
  could run it is reserved for it. That worker gets nothing else until enough of its cards are free.
- **Class wait** (0.4). A queued job has waited `--class-wait-s` (default 900 s, counted since it last became queued)
  and is a candidate for this worker except for room now. If the worker would otherwise get a job of a *lower* class,
  it is reserved for the waiting job instead and drains. This never happens for a job of the same class, and never
  for a job over the worker's budget (the worker could not run it anyway).

In both cases another worker that can run the job at once still takes it, and the reservation goes away. A drain
that has not freed the room within `--reserve-max-s` (default 6 h) is given up, after which the job may reserve
again. A worker's soft preferences (level order, cache affinity, budget order) thus delay a higher class by at most
one lease cycle. Hard constraints delay it by at most `--class-wait-s` plus the time the running jobs need to end.

Cost: a lease request now looks at every candidate of the highest class that has one, instead of stopping at the
first fit. On the VM, with 5000 queued jobs in one class, a request takes ~29 ms (~36 ms with a 1000-entry cache
report and a budget), all inside the database transaction. `--class-scan` (default 5000) caps how many jobs of one
class are looked at.

## 4. What the worker declares

Each lease request carries the 0.2 caps plus two optional objects:

- `cache: {rev, shas}`. `shas` holds the first 16 hex digits of the sha256 of every complete blob in the input cache
  (the most recently used 32768). It is sent only when its revision `rev` changes; the server keeps the last list per
  worker in memory. After a server restart, a known `rev` sent without its list gets `cache_unknown`, and the worker
  resends the list at once. The stored caps keep only `rev` and the count. The cache report is a scheduling hint: a
  blob evicted between the lease and the fetch is downloaded again (and counted). The worker re-checks the budget
  before fetching (below).
- `budget: {down, up, spent, reset}` (only if a budget is configured). `down` and `up` are the bytes left, the tighter
  of the day and month limits, *minus what the worker's running jobs still have to move*: inputs not fetched yet, and
  their expected upload, which the server sends with each job as `expect: {down, up}`. `null` means unlimited.
  `spent` lists the directions with less than 1 MiB left, and `reset` is when the binding limit resets (unix time).
  This is what `rqah status` and the status page show (§5).

The worker checks every fresh lease against its budget itself before fetching anything. A job that does not fit,
because of a stale cache report or a 0.3 server that ignores budgets, is handed back as abandoned (the attempt is not
counted), and the worker leases nothing for 60 s. A running job always finishes, even if its real upload turns out
larger than expected. The overrun is counted.

Compatibility: a 0.4 worker against a 0.3 server gets `400 unknown key` for the new caps. It then leases with the 0.3
caps and enforces its budget only by handing jobs back. Deploy the server first.

## 5. Where it shows

- `rqah status`: a DATA LEFT column per worker (declared down / up left, or SPENT) and a note per spent worker ("only
  jobs whose inputs it already caches" when only the download budget is spent, otherwise "no new jobs"), with the
  reset time. It also shows this machine's counters from `STATE/bandwidth.json` (used and limit per day and month).
  `--json` carries `workers[].budget` and `local_budget`.
- Status page: "N workers at their data cap until it resets" (workers seen in the last 15 min that declare a spent
  direction). It is a count only, with no names or amounts.
- `admin.py workers` prints the stored caps, including `budget`.
