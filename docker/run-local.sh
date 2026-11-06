#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# run-local.sh [worker flags...] -- the rad-quant-at-home worker without docker, on a box that already has the quantizer stack (our own
# GPU machines). Same flags as the image's entrypoint; paths from the environment:
#   STATE       state + cache dir                        EXL3_SRC   the fork's exllamav3_ext sources (--frac-ext)
#   EXL3_PATH   PYTHONPATH of the built exllamav3        TORCH_EXT  TORCH_EXTENSIONS_DIR for the side extensions
#   PYTHON      the python with torch (default python3)  GPU_LOCK   optional: flock this file around GPU jobs
#                                                        ({card} = one lock per card, e.g. /var/lock/gpu-card{card}.lock)
#   RADIANCE    optional: the radiance binary (eval_mix / golden_record); RADIANCE_HOME is passed to job processes
# e.g. STATE=... EXL3_SRC=... EXL3_PATH=... TORCH_EXT=... run-local.sh --server https://queue.example --token-file
#      ~/.rqah-token --gpu-name r9700 --gpus 0 --vram-gb 32 --ram-gb 30 --cache-gb 200 --window 22:00-08:00
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
RXM=${RXM:-$(cd "$R/.." && pwd)/rad-exl3-mix}          # the rad-exl3-mix checkout whose quantize/ourq.py the jobs run
: "${STATE:?}" "${EXL3_SRC:?}" "${EXL3_PATH:?}" "${TORCH_EXT:?}"
PY=${PYTHON:-python3}
exec "$PY" -I "$R/worker/worker.py" --state "$STATE" --python "$PY" --ourq "$RXM/quantize/ourq.py" \
  --exl3-src "$EXL3_SRC" --env "PYTHONPATH=$EXL3_PATH" --env "TORCH_EXTENSIONS_DIR=$TORCH_EXT" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${GPU_LOCK:+--gpu-lock "$GPU_LOCK"} \
  ${RADIANCE:+--radiance "$RADIANCE"} ${RADIANCE_HOME:+--env "RADIANCE_HOME=$RADIANCE_HOME"} \
  --image "local-$(git -C "$R" rev-parse --short HEAD 2>/dev/null || echo unknown)" "$@"
