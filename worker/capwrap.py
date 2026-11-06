# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""capwrap.py CAP_GB SCRIPT [ARGS...] -- run SCRIPT (as __main__) with torch's caching allocator capped at CAP_GB on
every visible GPU (worker --vram-cap-gb: a shared card). Results are unchanged unless the cap is hit (then OOM)."""
import runpy, sys

cap = float(sys.argv[1]); script = sys.argv[2]
import torch
if torch.cuda.is_available():
    for d in range(torch.cuda.device_count()):
        tot = torch.cuda.get_device_properties(d).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / tot), d)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name="__main__")
