#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# gpucheck.py -- torch sees the GPU(s) and computes correctly: per device a 4096^2 bf16 matmul against an fp32
# reference, and its rate. Prints one JSON line {"ok": bool, "detail": str}. GPUCHECK_N sets the size,
# GPUCHECK_VRAM_CAP_GB caps the allocator on a shared card.
import json
import os
import time


def main():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"import torch: {e}"}
    if not torch.cuda.is_available():
        return {"ok": False, "detail": f"torch {torch.__version__}: no GPU visible"}
    n = int(os.environ.get("GPUCHECK_N", "4096"))
    cap = float(os.environ.get("GPUCHECK_VRAM_CAP_GB", "0"))   # shared cards: keep the allocator under this
    hip = getattr(torch.version, "hip", None)
    parts, ok = [], True
    for d in range(torch.cuda.device_count()):
        dev = torch.device("cuda", d)
        if cap:
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.mem_get_info(dev)[1]), dev)
        g = torch.Generator(device=dev).manual_seed(0)
        a = torch.randn(n, n, device=dev, generator=g)
        b = torch.randn(n, n, device=dev, generator=g)
        ref = a @ b
        ab, bb = a.bfloat16(), b.bfloat16()
        c = ab @ bb
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        reps = 10
        for _ in range(reps):
            c = ab @ bb
        torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t) / reps
        err = ((c.float() - ref).norm() / ref.norm()).item()
        good = err < 2e-2
        ok &= good
        free, total = torch.cuda.mem_get_info(dev)
        parts.append(f"cuda:{d} {torch.cuda.get_device_name(dev)} {total / 2**30:.0f} GiB ({free / 2**30:.0f} free), "
                     f"bf16 {2 * n**3 / dt / 1e12:.0f} TFLOP/s, rel err {err:.1e}{'' if good else ' BAD'}")
        del a, b, ref, ab, bb, c
    stack = f"HIP {hip}" if hip else f"CUDA {torch.version.cuda}"
    return {"ok": ok, "detail": f"torch {torch.__version__} ({stack}); " + "; ".join(parts)}


if __name__ == "__main__":
    print(json.dumps(main()))
