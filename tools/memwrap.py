#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""memwrap.py CAP_GB OUT.jsonl TAG SCRIPT [ARGS...] -- run SCRIPT (as __main__, e.g. quantize/ourq.py) with torch's
allocator capped at CAP_GB per visible GPU (as worker/capwrap.py), then append one JSON line to OUT.jsonl: wall time,
torch's peak allocated / reserved bytes on device 0, the process's peak RSS (host RAM), the exit status. The cap only
bounds the caching allocator: the CUDA context (~0.3-0.5 GB) comes on top (sample nvidia-smi for the card's view)."""
import json, os, resource, runpy, sys, time, traceback

cap, out, tag, script = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
import torch
if torch.cuda.is_available() and cap > 0:
    for d in range(torch.cuda.device_count()):
        tot = torch.cuda.get_device_properties(d).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot), d)
sys.argv = [script] + sys.argv[5:]
t0, status = time.time(), "ok"
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    status = "ok" if e.code in (None, 0) else f"exit {e.code}"
except BaseException as e:
    status = f"error {e!r}"[:300]; traceback.print_exc()
finally:
    r = {"tag": tag, "status": status, "wall_s": round(time.time() - t0, 1), "cap_gb": cap,
         "peak_alloc_gb": round(torch.cuda.max_memory_allocated(0) / 2 ** 30, 3) if torch.cuda.is_available() else None,
         "peak_reserved_gb": round(torch.cuda.max_memory_reserved(0) / 2 ** 30, 3) if torch.cuda.is_available() else None,
         "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)}
    with open(out, "a") as f: f.write(json.dumps(r) + "\n")
    print("MEM", json.dumps(r), flush=True)
sys.exit(0 if status == "ok" else 1)
