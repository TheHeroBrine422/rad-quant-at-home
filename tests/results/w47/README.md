# w47 raw results (2026-11-06; TESTS.md sections 10-12)

| file | what |
|---|---|
| `vram-mem.jsonl`, `vram-units.log` | peak VRAM / RSS per 27B tensor (tools/memwrap.py; units a, b, d), RTX 4070 capped at 7.2 GB |
| `real-units.log` | the real quantizer runs for the score tests (0.8B ladder, seeds, attacks; 27B layer 10 K2 seeds 0-2) |
| `sr08.log` | tests/score_real.py on the 0.8B runs: 14/14 |
| `sr27.log`, `score-27b-k2.5-existing.log` | the 27B layer: seed spread, scorer vs log, scoring time |
| `seed-spread-and-smoke.txt` | tools/seed_spread.py (0.8B K3 x5 seeds; 27B K2 x3) and the rootfs smoke test of the CUDA stack |
| `curve-loo-27b.txt` | tools/curve_loo.py on our 27B levels (rad-exl3-mix data/v3/levels) |
