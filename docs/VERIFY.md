# Verifying donated quantization by score

Status: implemented in 0.3 (`server/score.py`, behind the job parameter `"verify": "score"`; `admin.py enqueue-level
--verify score`). The 0.2 path (checker + replication spot checks, `"verify": "replicate"`) stays the default until the
first real campaign has run on it. Tests: `tests/score_test.py` (synthetic, no GPU) and `tests/score_real.py` (real
quantizer output); numbers in TESTS.md section 10.

## The idea

A quantization job's output is a set of EXL3 trellis codes. Whether they are good is a fact about the codes, not about
who made them or how: decode them exactly and measure their error against the weights they stand for. So the server
does not ask "did the donor run the quantizer the way we would have?" (replication: a second GPU run per checked job,
and two runs never match bit for bit across vendors, drivers or even runs) but "are these codes as good as a quant
of this tensor at this bitrate should be?".

For each tensor of a returned shard the server computes, with its own copies of the bf16 weight `W` and the layer's
input Hessian `H` (the published calibration Hessians):

    rel = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ),   ΔW = decode(codes) − W

`decode` is `checker/exl3dec.py`: a numpy transcription of exllamav3's integer trellis decode (integer and fractional
K, qtf2's salt, the mul1 / i8 / i8c / i4 codebooks) followed by the stored sign/scale vectors and the 128-wide
Hadamard transforms. It reproduces the quantizer's own dequantized weights to fp16 rounding, and its `rel` equals the
quantizer's logged `proxy_err` to ≤ 2e-4 (27B, every tensor of a layer). This is the same proxy the whole quantizer
minimizes, so it is the quantity worth checking.

The result is **accepted when `rel` lies in the band expected for that tensor at that K** (below), and the codes are
valid for the declared format and K. The band is one-sided: a lower error is never a cheat. The server computes it
from the codes, and finding codes with low error is exactly the work being donated. Consequences:

- GPU nondeterminism, other vendors (CUDA donors), other drivers, other seeds: all fine. Only the quality is judged.
- The worker's own log (`proxy.jsonl`) is not trusted and not needed. The per-row claim (`rowerr.npz`) and the
  `--save-wq` pass that produced it are dropped in score mode: less GPU time, VRAM and upload.
- The server can keep the **best of N**: copies of a job (different `seed_offset`) can go to several donors, every
  copy in its band is accepted, and the export takes the one with the lowest error (`--copies N`).

## What is checked, per job

1. **Structure** (at upload, `accept()`): exactly the tensors of the job's layer plan (`.trellis` int16
   `[in/16, out/16, 16K]`, `.suh` f16 `[in]`, `.svh` f16 `[out]`, `.mul1` int32), shapes from our bf16 weights.
   The trellis word count fixes K: codes of another K do not fit.
2. **Format**: `suh` / `svh` finite, non-zero, |x| ≤ 1e3; `.mul1` is exllamav3's MUL1 constant (0x83DCD12D, also
   for i8 / i8c, whose decode is the mul1 hash plus rounding). Any int16 pattern is a valid tail-biting trellis, so
   the codes themselves need no further format check.
3. **Error**: the exact decode and `rel` per tensor (above). `tr(W H Wᵀ)` is the same for every result of a tensor
   and is computed once (`score_den`).
4. **Band**: `log rel ≤ log center + tol` for every tensor (next section).
5. **Replay / duplicates**: the sha256 of each tensor's code material (trellis + suh + svh) is stored, and (0.5, #11)
   fingerprints of 256 of its 16 × 16 trellis tiles (4 bytes each), sampled at positions drawn from a secret of the
   queue database. Codes equal to a released model's (imported with `score-import --published`), or sharing at least
   half of the sampled tiles with them, fail. The same against another donor's earlier result for the same tensor
   and K (or the donor's own other copy): they pass (they are a valid result) but earn nothing: the first submission
   of given codes earns. Honest copies use another `seed_offset`, so other sign vectors and other codes in every
   tile; dodging the check means changing words in more than half the tiles, and 1 % of the words already costs
   ×1.1–4 in rel (out of the band). The scheduler also leases each donor at most one copy of a job (`jobs.grp`).

Verdicts: every tensor in its band → **pass** (points granted, a reference for later jobs). A tensor in the trusted
band but out of the all-rows band, with no trusted row at its K → **suspect** (next section; an anchor decides). Any
tensor out of its band or a structure / format error → **fail**: the donor is quarantined as before, its accepted results that nothing
confirmed are requeued, but **its score-verified results stand** (their quality is proven by their own codes). No
reference yet for some tensor → **provisional** (next sections).

## The band

Results are grouped into **series**: model, revision, architecture, codebook and quantizer settings except
`seed_offset`. The levels of one ladder differ only in K, so a series is one rate-distortion curve per tensor.
The center comes from earlier **passed** rows of the same tensor in the same series, best basis first:

| basis | when | center | tol (log), + the noise term |
|---|---|---|---|
| repeat | rows at the same K (honest repeats, best-of-N copies, the anchor) | trusted rows: their minimum; all passed rows: their upper median | 0.02 |
| interp | rows below and above K | linear in log rel between the nearest K on each side | 0.04 (k/v: 0.08) |
| extrap | rows on one side only | the nearest, moved along the codebook's slope: −d log rel / dK = 1.37 | 0.04 + 0.03 · \|ΔK\| (k/v: 0.08 + 0.12 · \|ΔK\|) |

plus a **noise term** `4 · NOISE / √n` for a tensor of n = in × out weights: the honest seed noise of the quantizer
(its error is a sum over many independent tiles, so it falls like 1/√n). NOISE = 30 for the layer-input group
(q/k/v_proj, in_proj_qkv, in_proj_z: lighter damping, noisier) and 8 for the rest. At the 27B that adds 0.3 % (mlp)
to 5 % (k_proj, v_proj: 5 M weights);
on a 0.8B proxy up to 9 %. When trusted rows exist (our verifiers' re-runs, or our own results imported with
`score-import`), a second bound is computed from them alone. That way a chain of donor results, each a little worse
than the previous, cannot drag the curve.

**The trusted bound is the authority (0.5, #12).** A result that fails it fails. A result that passes it but fails the
all-rows bound is not failed: with trusted rows at its own K (repeat basis) it passes, otherwise it is **suspect**
(accepted, no points yet, not a reference) and the job is re-run once on a verifier; that anchor at the same K then
decides it. Without trusted rows the all-rows bound decides alone. The all-rows bound is centred on the **upper median**
of the passed rows at a K, not their minimum: one result a few percent better than the honest quantizer (more search
effort, another quantizer) is allowed, and it must not turn every honest repeat after it into a fail and quarantine
their donors. A result more than the repeat tolerance below our own trusted center passes and writes a
`score_outlier` event for the operator (better than our quantizer: worth a look, not a punishment).

**Why these numbers** (`tools/curve_loo.py`, `tools/seed_spread.py`; TESTS.md section 10):

- The curve is very regular. Our own 27B levels (rad-exl3-mix `data/v3/levels`: 400 tensors × K 2–4.5 in quarter
  steps, i8): log rel falls by 1.358–1.370 per bit (the Gaussian rate-distortion slope is 2 ln 2 = 1.386).
- Leave-one-K-out prediction of a tensor's log rel from its other K: neighbour interpolation residual SD 0.15–0.26 %,
  max 0.4–1.5 % for every family except k_proj / v_proj (SD 1.0–1.3 %, max 3.5–6.4 %). Extrapolating with the family
  shape from one K: SD grows ~0.3–1 % per bit (k/v: up to ~6 % per bit). Those levels mixed damping settings for k/v
  within one "tensor", so a clean series should be tighter.
- Honest repeats (`--seed-offset` 0–2) of a 27B layer at K2: per-tensor range ≤ 0.10 % (out_proj), ≤ 0.02 % for the
  rest (SD · √n ≤ 2.9). On the 0.8B proxy (5 seeds, tensors 25–100× smaller): range up to 4.1 % (in_proj_z), SD · √n
  up to 26 for the layer-input group and ≤ 6.3 for the rest, which is where NOISE comes from.

The cheats the band must catch are much larger: one ladder step (1/8 bit) is +17 % in rel (e^(1.37/8)), a whole bit
×3.9. At the 27B the bands are 2–7 % (repeat) and 4–13 % (interp, k/v the widest) wide. Every honest result we have passes, and a
quantization that ignores the Hessian (+8 % on the 0.8B) fails. Extrapolating more than ~3 bits is looser than
1/8 bit; the re-checks below close that gap as the ladder fills in.

## Cold start: anchors

The first result for a tensor in a series has no reference. It is accepted **provisionally** (no points yet) and the
job is re-run once on a **verifier** (our own hardware, `admin.py verifier ID`). The verifier's result is scored as
**trusted** rows (a reference whatever its value), and the provisional job is then decided against it on the repeat
basis. This is the only replication left: one job per tensor and series (for a ladder: the first level of each layer
and the head, ~1/73 of the work), or none at all if we import references we made ourselves:

    admin.py score-import --level k3-i8 /path/to/our/ourq/output     # q-LNN.safetensors of the same settings

## Re-checks

Every new passed row re-evaluates the passed rows of its tensor that were decided on a wider basis (extrap, interp)
or are still provisional. A row that is now out of its band fails retroactively: quarantine, points revoked, the job
requeued for anyone else. So a result that slipped through an extrapolated band (e.g. 1/8 bit worse, judged from a
reference 2 bits away) fails as soon as the ladder fills in around it. Score-verified rows decided on the repeat basis
are not re-checked.

## Threat model

| threat | what the donor does | what catches it | test |
|---|---|---|---|
| garbage | random or junk codes (no work) | rel ≈ 1–3 instead of 1e-3–1e-2 | random codes (synthetic and real 0.8B) |
| corrupted / partly done | some tensors or tiles not quantized, bit flips | that tensor's rel above its band (1 % of one tensor's words changed: rel ×1.1–4) | perturbed codes |
| lazy, cheaper quant | fewer bits of search (e.g. a lower-K trellis inside the K container, a constrained search), no Hessian / LDLQ | rel above the band: a lower K costs +17 % per 1/8 bit, ignoring the Hessian costs more | real 0.8B: K2 quality re-encoded as K3 codes; K3 with the Hessian damped away |
| wrong K | codes of a lower K, declared as K | the trellis shape (16K words per tile): structure error; padded to the right shape: garbage decode | K2 shard as K3; K2 codes padded to K3 |
| wrong format | another codebook (mul1 instead of i8), wrong marker, broken scales | decoded with the declared codebook: rel far out; marker / NaN / zero / huge scales: format error | real mul1 codes declared i8; marker; NaN scale |
| replay | a published model's codes for the same tensor, as is or with a few words changed | sha256 or near-duplicate (≥ half the sampled tiles) match with a `--published` import | replay tests |
| duplicate | the same codes submitted twice (two copies of a best-of-N job, or colluding donors), as is or with a few words changed per copy | sha256 or near-duplicate match with another donor's row or the donor's own other copy: accepted, no points; one copy of a job per donor in the scheduler | duplicate and perturbed-copy tests |
| collusion on quality | several donors return the same slightly-worse quant | the trusted bound (anchors / imports) and re-checks against neighbouring K | 1/8-bit-worse test, re-check test |
| griefing honest donors | one result a few % better than honest, so that honest repeats fall out of a band centred on it | the all-rows center is the upper median; an all-only fail with the trusted bound passing is suspect, decided by an anchor, never a quarantine (#12) | better-than-honest-then-honest test |
| wrong data | quantizing another tensor, another layer, other weights | rel against our weights: garbage-level | (follows from the decode) |

**What it does not catch** (and why that is acceptable):

- A donor who delivers a *good* quant made some other way (another quantizer, another machine, or our own
  unpublished result if it leaked) passes. That is fine for the model. For points it is a fraud only if the result
  was copied. The guards are hashes (published releases, other donors) and the fact that nobody but the server and
  the donor sees a result.
- Determinism: the quantizer is deterministic on one GPU and software stack (the smoke test's donor and verifier,
  same card, returned bit-identical codes), so two honest donors with identical setups may too; the second earns
  nothing for that tensor. Codes equal (or nearly equal) to a *trusted* row (our anchor or import) are not counted as
  a duplicate. Best-of-N copies avoid it by construction (each copy has its own `seed_offset`).
- A cheat that stays inside the tolerance (≤ 3–5 % worse than honest) passes. That is the price of not replicating,
  and it is below the spread between neighbouring ladder steps.
- Binding codes to the seed (checking that `sign(suh)` / `sign(svh)` follow the job's seed) would stop copying across
  best-of-N copies. It needs the quantizer's device RNG on the server, so it is left out.

### Identity: trust is per worker, not per donor (#9, won't fix)

Quarantine, the spot-check draw (a worker's first result of each type, then a random share) and the exclusion of a
worker from a job it was caught on all key on the worker id. A donor can hold several ids (a personal invite with
`--uses N`, `rqah link`, more invites), so a caught donor's other machines keep working until the operator acts.
Keying these on the donor instead would not stop a cheater: registering again with a fresh invite makes a new donor
as easily as a new worker, so donor-wide automation would cost complexity and protect nothing. What it would do is
punish honest donors for one broken machine (a bad card or driver fails a re-run; its siblings are fine). So:

- **Every worker is checked on its own.** A new token's first result of each type is always re-run (or scored), so a
  new identity buys no unchecked work; counting the first N per donor would be weaker, not stronger.
- **Nothing a worker delivers counts without a check that does not trust it.** Quant codes are scored by the server
  itself; for evaluations (#4) a result far off the knapsack's prediction is refused or re-run in full, a share is
  spot-checked on random documents, and anything the search adopts is re-run in full on our hardware. A cheat that
  re-registers starts from zero unchecked work.
- **Donor-wide action is the operator's call:** `admin.py exclude-donor DONOR --reason ...` quarantines every worker
  of the donor and invalidates every result of theirs that nothing confirmed (lumens revoked, jobs requeued for
  others, the search's batches see them as invalidated). `disable-worker` revokes a token.
- **Later, if re-registration becomes a real problem:** limits on invites (they are the only way in, and the operator
  hands them out), and blocks by IP range at the reverse proxy.

## Cost

Decoding is cheap. The cost is one GEMM `H·ΔW` per tensor (`tr(W H Wᵀ)` is the same for every result of a tensor
and is computed once). Measured on a 27B layer (CPU, numpy/OpenBLAS, 4 threads, niced): **30 s for the first result
of a layer** (with the cached trace), **~18 s after**. A 0.8B layer takes 0.7–2 s. Quantizing that 27B layer takes
~100 s on an RTX 4070 capped at 7.2 GB (~2.5 min on an R9700), so one 4-thread scorer keeps up with ~5 such cards; run
several (`admin.py score --loop` per CPU group, or `--score-threads 1` in the server) for a bigger campaign. For
scale: the whole ladder (73 levels × 64 layers) is ~4,700 layer scores, ~24 CPU-hours at 4 threads. A GPU verifier
would do the same GEMMs in well under a second; that is a possible later job type, not needed for the first campaign.

## Operating it

    admin.py datapack ... [--groups]                         # once; --groups for quant_tensor jobs (small cards)
    admin.py score-import --level <first level> <our ourq output dir>   # optional: anchors without re-runs
    admin.py enqueue-level ... --verify score [--unit group] [--copies 2]
    server.py ... --score-threads 1                          # or: admin.py score --loop 30 beside it
    admin.py scores --level L                                # every scored tensor: rel, basis, band, state
    admin.py export --level L OUT --checked                  # best passed copy per job; groups merged per layer

Tolerances, family multipliers and the slope are the module constants `TOL`, `FAMILY` and `SLOPE` in
`server/score.py`. Re-tune them from the first campaign's honest repeats (best-of-N copies give them for free).
