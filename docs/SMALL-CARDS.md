# Quantization jobs on 8 GB and 12 GB cards

Status: measured on an RTX 4070 (12 GB, CUDA) with torch's allocator capped at 7.2 GB (the box is shared, so the cap
stands in for an 8 GB card: ~7.5 GB usable once the CUDA context and a desktop are counted). Implemented in 0.3: the
27B requirements of `quant_layer` and the `quant_tensor` job type (one Hessian group). Raw rows: TESTS.md section 10.

## What one 27B tensor needs (`quantize/ourq.py`, one tensor per process)

Ladder settings (i8 codebook, qtf2, scaled Hessian, σ 0.025 / 0.005), score mode (no `--save-wq`), one process per
tensor (`--only`), weights fetched from the HF checkpoint by range (tools/hfrange.py): layer 10 (gated delta-net) and
the attention tensors of layer 39. VRAM = torch's peak allocated / reserved (the CUDA context, ~0.3–0.5 GB, comes on
top); host RAM = the process's peak RSS; time = wall per process including ~1.5 s of start-up.

| tensor (out × in) | K2 VRAM alloc / res (GB) | K4 | K6 | host RAM | time K2 / K4 / K6 (s) |
|---|---|---|---|---|---|
| k_proj, v_proj (1024 × 5120) | 1.62 / 1.76 | 0.86 / 1.00 | 0.68 / 0.82 | 1.8 GB | 2–3 / 2 / 3–5 |
| q_proj (12288 × 5120) | 3.31 / 3.48 | 2.55 / 2.74 | 2.36 / 2.54 | 1.9 GB | 8 / 8 / 16 |
| o_proj, out_proj (5120 × 6144) | 2.44 / 2.62 | 1.69 / 1.86 | 1.50 / 1.66 | 1.9 GB | 5 / 5 / 8–9 |
| in_proj_qkv (10240 × 5120) | 2.98 / 3.15 | 2.22 / 2.39 | 2.03 / 2.21 | 1.8 GB | 7 / 7 / 13 |
| in_proj_z (6144 × 5120) | 2.31 / 2.50 | 1.56 / 1.74 | 1.37 / 1.55 | 1.8 GB | 5 / 4 / 8 |
| gate_proj, up_proj (17408 × 5120) | 3.77 / 4.14 | 3.01 / 3.38 | 2.82 / 3.19 | 1.9 GB | 11 / 10 / 21 |
| **down_proj (5120 × 17408)** | **6.24 / 6.59** | **6.61 / 6.86** | **6.47 / 7.04** with a small cuBLAS workspace (without: fails at 5.65 / 7.17) | **9.6–10.4 GB** | **64 / 61 / 73** |

down_proj also at K3: 6.86 / 7.11, and K5: 6.38 / 7.14 (both with the CPU refit, 10.4 GB RAM). q_proj at K8: 2.30 / 2.49.

down_proj is the only problem, and for two reasons:
1. **The scale refit** after the trellis search needs a second copy of its 17408² fp32 Hessian (1.13 GiB). Under the
   cap it runs out (`OOM ... tried to allocate 1.13 GiB` at 6.24 GiB allocated), and ourq's CPU fallback
   (`OURQ_REFIT_CPU_FALLBACK=1`, always set by the jobs since 0.3) does the refit on the CPU: +~50 s and ~10 GB of
   host RAM. With the GPU refit it needs ~7.4 GB allocated, ~8 GB on the card: fine on 10–12 GB cards.
2. **At K6 cuBLAS's workspace** no longer fits beside it: `CUBLAS_STATUS_EXECUTION_FAILED` at 7.17 GB reserved (5.65
   GB allocated). With `CUBLAS_WORKSPACE_CONFIG=:4096:8` (a small workspace; the jobs set it on cards below 12 GB
   since 0.3) it runs: 6.47 / 7.04 GB, 73 s. K3 and K5 fit without it.

Everything else peaks at ≤ 4.2 GB, less at higher K (smaller per-tile buffers), so **6 GB cards could take every
tensor but down_proj**, and host RAM is ~2 GB per job.

`--save-wq` (the 0.2 replicate mode's per-row claim) adds a second Hessian load per tensor: down_proj at K2 peaks at
6.90 GB reserved and 10.2 GB RSS with it, so score mode is also the small-card mode.

## The job split

| card | gets | how |
|---|---|---|
| ≥ 12 GB (RTX 3060 12 GB, 4070, 4070 Ti, 3080 Ti, 4080, 3090, 4090, 5070 Ti, ...) | every trunk quant job at every K | `quant_layer`: 8 GB VRAM up to K6, 12 GB above; 12 GB RAM |
| 8 GB (RTX 3060 Ti, 3070, 4060, 4060 Ti 8 GB, ...) | whole layers up to K6 (measured K2-K6); above K6 every group but down_proj | `quant_layer` up to K6; above K6 `quant_tensor` (`--unit group`): down_proj groups require 12 GB, the others 6 GB |
| 6 GB | every group but down_proj | `quant_tensor` with `--unit group` |

The LM head (`quant_head`, 248k × 5120) is not covered here: it stays a 32 GB / 48 GB-RAM job for the big cards.

- **Per layer** (`quant_layer`, the default): the requirement is the largest tensor's, i.e. down_proj's: 8 GB VRAM and
  12 GB RAM up to K6 (CPU refit, small cuBLAS workspace), 12 GB VRAM above K6 (not measured there; to be lowered once
  K7 / K8 are measured). One job per layer, so no overhead; a layer takes ~1.7 min on
  the 4070 at K2 (gate + up + down dominate: ~10 + 10 + 60 s).
- **Per Hessian group** (`quant_tensor`, `enqueue-level --unit group`, `datapack --groups`): the unit is the set of
  linears that share one input Hessian (GDN: qkv + z | out | gate + up | down; attention: q / k / v | o | gate + up |
  down), because ourq's seeds and the in-place preparation of a shared Hessian make a whole group give exactly the
  codes of the full-layer run, while a lone tensor of a group would not. Each group job downloads only its own
  weights (`weights/LNN-gG.safetensors`) and its one Hessian, so the total download is the same as for a layer job.
  Requirements: down_proj group 8 GB (12 GB above K6) and 12 GB RAM; the others 6 GB and 4 GB RAM.
- **Per row block** (not implemented): splitting down_proj's 5120 output channels would not help. The 17408² Hessian
  (1.13 GiB, twice during the refit) dominates its memory, and every block needs all of it. The per-tensor scales and
  the global scale search also span all rows, so blocks would need a two-phase protocol (fix the scales first) and
  would no longer reproduce the full run. The CPU refit is the cheaper answer for 8 GB.

**Overhead of the group split**, per 27B layer: 4 jobs instead of 1, so 3 more leases, uploads and process starts
(~2 s each for ourq's imports and CUDA init, plus ~1 s of HTTP), ~10 s per layer, ~6 % of a K2 layer on the 4070 and
less at higher K. The server holds 4× the job rows (73 levels × 64 layers × 4 ≈ 19k, nothing for SQLite). Download:
the same bytes. Scoring: the same work, in four pieces. Use groups only where they buy something: 6 GB cards,
preemptible cards (shorter units: the longest group is down_proj's ~1 min), or 8 GB cards above K6. Big cards are
better served by whole layers.

**Recommendation for the main ladder run**: whole layers (`quant_layer`) in score mode for everyone with ≥ 8 GB, which
covers the trunk through K6 on 8 GB cards; add a group-split copy of the levels above K6 (and of any level, for 6 GB
donors) only if such donors show up.

## Host RAM

The README's old 24 GB per 27B layer was an R9700-side estimate. Measured per process here: ~2 GB, and ~10 GB for
down_proj when its refit falls back to the CPU. Declared requirements in 0.3: 12 GB per layer or down_proj group,
4 GB per other group. A 16 GB desktop can run one job; give the container `--memory` a few GB above `--ram-gb`.
