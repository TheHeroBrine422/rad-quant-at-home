#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# cuda-smoke-rootfs.sh ROOTFS -- the CUDA research image's GPU stack on a CUDA box WITHOUT Docker (e.g. a GPU box that is
# itself a container): ROOTFS holds the image's /opt (docker create IMG; docker cp CID:/opt - | tar -C ROOTFS -xf -).
# The image's venv python is a symlink to /usr/bin/python3, so the box needs the same Python (Ubuntu 24.04: 3.12).
# Runs: the image's gpucheck.py (torch sees the card, bf16 matmul vs fp32), then a CUDA extension built with
# torch.utils.cpp_extension (nvcc from the box, CUDA 12.8 like the image) and run on the card.
# Env: GPU_LOCK (flock it for the whole GPU part), LOCK_WAIT_S (default 3600), VRAM_CAP_GB (default 7.2),
# TORCH_CUDA_ARCH_LIST (default 8.9), MAX_JOBS (default 4). Output: "CUDA-SMOKE PASS|FAIL".
set -u
ROOTFS=$(cd "${1:?usage: cuda-smoke-rootfs.sh ROOTFS}" && pwd); PY=$ROOTFS/opt/venv/bin/python
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
echo "image: $(cat "$ROOTFS/opt/donor-kit/IMAGE")"
echo "python: $($PY -c 'import sys; print(sys.version.split()[0])'), box: $(. /etc/os-release; echo "$PRETTY_NAME"), nvcc: $(nvcc --version | tail -2 | head -1)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader
cat > "$W/ext.py" << 'PYEOF'
import os, time, torch
from torch.utils.cpp_extension import load_inline
src = r"""
__global__ void axpb(const float* x, float* y, float a, float b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a * x[i] + b; }
torch::Tensor run(torch::Tensor x, double a, double b) {
  auto y = torch::empty_like(x); int n = x.numel();
  axpb<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), (float)a, (float)b, n); return y; }
"""
t = time.time()
m = load_inline("donorkit_smoke", cpp_sources="torch::Tensor run(torch::Tensor x, double a, double b);",
                cuda_sources=src, functions=["run"], build_directory=os.environ["EXT_DIR"], verbose=False)
x = torch.randn(1 << 20, device="cuda")
y = m.run(x, 3.0, 0.5)
err = (y - (3 * x + 0.5)).abs().max().item()
print(f"extension: built in {time.time() - t:.0f} s, max err {err:.1e}", "OK" if err < 1e-5 else "BAD")
raise SystemExit(0 if err < 1e-5 else 1)
PYEOF
mkdir -p "$W/ext"
run() {
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} GPUCHECK_VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} "$PY" -I "$ROOTFS/opt/donor-kit/gpucheck.py" | tee "$W/gpu.json" &&
  grep -q '"ok": true' "$W/gpu.json" &&
  EXT_DIR=$W/ext TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} MAX_JOBS=${MAX_JOBS:-4} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
    PATH=${CUDA_HOME:-/usr/local/cuda}/bin:$PATH nice -n 19 "$PY" -I "$W/ext.py"
}
t0=$(date +%s)
if [ -n "${GPU_LOCK:-}" ]; then
  exec 9> "$GPU_LOCK"; echo "waiting for $GPU_LOCK"; flock -w "${LOCK_WAIT_S:-3600}" 9 || { echo "CUDA-SMOKE FAIL (lock)"; exit 1; }
  echo "lock held after $(( $(date +%s) - t0 )) s"
fi
t1=$(date +%s)
if timeout 840 bash -c "$(declare -f run); W=$W ROOTFS=$ROOTFS PY=$PY run"; then r=PASS; else r=FAIL; fi
echo "GPU part $(( $(date +%s) - t1 )) s"
echo "CUDA-SMOKE $r"
[ $r = PASS ]
