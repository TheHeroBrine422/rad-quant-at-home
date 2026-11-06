#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fake_mkgold27.py MODEL CORPUS OUT --scratch DIR [...] -- eval/gold27/mkgold27.py's command line on the CPU (tests
only): the same golden as fake_radiance.py --kld-record (tests/fakes.py record_golden, top 512 unless
FAKE_GOLD_TOP), so the staged and the TP2 paths agree. Checks that MODEL holds config.json and the index's shards."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes

model, corpus, out = sys.argv[1:4]
idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))
missing = [f for f in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(model, f))]
if missing or not os.path.exists(os.path.join(model, "config.json")): sys.exit(f"fake mkgold27: checkpoint incomplete {missing}")
assert "--scratch" in sys.argv
n = fakes.record_golden(corpus, out, int(os.environ.get("FAKE_GOLD_TOP", "512")), model=os.path.basename(model.rstrip("/")),
                        salt=os.environ.get("FAKE_RAD_SALT", ""))
print(f"fake mkgold27: {n} rows -> {out}")
