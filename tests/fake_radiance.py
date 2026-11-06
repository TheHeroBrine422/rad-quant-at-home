#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""fake_radiance.py -- radiance's command line for the two modes the worker runs, on the CPU (tests only):
  --model M.rad ... --kld-ref DIR --kld-out FILE        FILE + FILE.rows (tests/fakes.py score)
  --model M.rad ... --kld-record DIR --kld-corpus C [--kld-top N]   a golden (tests/fakes.py record_golden)
Other flags are accepted and recorded. FAKE_RAD_BIAS adds to every KL (a cheating donor), FAKE_RAD_NOISE is a relative
per-row jitter (cross-GPU nondeterminism), FAKE_RAD_SLEEP seconds of "GPU work", FAKE_RAD_FAIL=1 exits 3."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fakes

a = sys.argv[1:]
opt = {a[i]: a[i + 1] for i in range(len(a) - 1) if a[i].startswith("--") and not a[i + 1].startswith("--")}
print("fake radiance", " ".join(a), flush=True)
if os.environ.get("FAKE_RAD_FAIL") == "1": sys.exit(3)
time.sleep(float(os.environ.get("FAKE_RAD_SLEEP", "0")))
if "--kld-ref" in opt:
    rep = fakes.score(opt["--model"], opt["--kld-ref"], opt["--kld-out"], float(os.environ.get("FAKE_RAD_BIAS", "0")),
                      float(os.environ.get("FAKE_RAD_NOISE", "0")))
    rep["tp"] = int(opt.get("--tp", 1)); json.dump(rep, open(opt["--kld-out"], "w"), indent=1)
    print(f"kld: mean {rep['all']['kld']['mean']:.6f} over {rep['all']['positions']} positions (tp {opt.get('--tp')})")
elif "--kld-record" in opt:
    n = fakes.record_golden(opt["--kld-corpus"], opt["--kld-record"], int(opt.get("--kld-top", 16)), model=opt["--model"],
                            salt=os.environ.get("FAKE_RAD_SALT", ""))
    print(f"kld: reference recorded: {n} positions")
else:
    sys.exit("fake radiance: nothing to do")
