#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# cuda_rootfs.sh ROOT -- the steps of docker/Dockerfile.cuda without Docker, into ROOT/opt/venv and ROOT/opt/rqah (for a
# box that has the CUDA toolkit and Python 3.12 but no Docker, e.g. a GPU sandbox; then tests/cuda_smoke.sh MODE=rootfs
# ROOTFS=ROOT). The pins (torch, its index, the Python packages, exllamav3's repo and commit) are read from the
# Dockerfile, so the two cannot drift. Same layout as the image: app/ = this repo's working tree (the build.sh paths),
# rxm/ = rad-exl3-mix at RXM_COMMIT (default HEAD) of RXM_REPO (default ../rad-exl3-mix). Resumable: finished steps
# are skipped. Env: TORCH_CUDA_ARCH (default 8.9), MAX_JOBS (default 2: one exllamav3 unit needs up to ~6 GB of RAM).
set -euo pipefail
ROOT=$(mkdir -p "$1" && cd "$1" && pwd)
APP=$(cd "$(dirname "$0")/.." && pwd); DF=$APP/docker/Dockerfile.cuda
RXM=${RXM_REPO:-$(cd "$APP/.." && pwd)/rad-exl3-mix}
arg() { sed -n "s/^ARG $1=//p" "$DF" | head -1 | sed 's/^"//; s/"$//'; }
# PY_DEPS spans two lines in the Dockerfile (a trailing backslash)
PY_DEPS=$(awk '/^ARG PY_DEPS=/{f=1} f{print} f&&!/\\$/{exit}' "$DF" | sed 's/^ARG PY_DEPS=//; s/\\$//' | tr -d '"\n')
TORCH_SPEC=$(arg TORCH_SPEC); TORCH_INDEX=$(arg TORCH_INDEX); EXL3_REPO=$(arg EXL3_REPO); EXL3_COMMIT=$(arg EXL3_COMMIT)
ARCH=${TORCH_CUDA_ARCH:-8.9}; JOBS=${MAX_JOBS:-2}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/build.log"; }
log "pins: $TORCH_SPEC @ $TORCH_INDEX | $PY_DEPS | exllamav3 $EXL3_COMMIT | arch $ARCH, $JOBS jobs"
V=$ROOT/opt/venv R=$ROOT/opt/rqah
if [ ! -f "$V/.done" ]; then
  python3 -m venv "$V" && "$V/bin/pip" install --no-cache-dir --index-url "$TORCH_INDEX" "$TORCH_SPEC" \
    && "$V/bin/pip" install --no-cache-dir $PY_DEPS && touch "$V/.done"; log "venv done"
fi
mkdir -p "$R/app" "$R/rxm"
tar -C "$APP" --exclude=__pycache__ --exclude='*.pyc' -cf - LICENSE NOTICE LICENSES VERSION common jobs worker client docker checker \
  | tar -C "$R/app" -xf -
RC=$(git -C "$RXM" rev-parse --short "${RXM_COMMIT:-HEAD}")
git -C "$RXM" archive --format=tar "$RC" -- LICENSE NOTICE LICENSES quantize/ourq.py quantize/exl3-patches v3/e3/exl3frac \
  v3/e3/exl3frac2 v3/e3/exl3cb eval/gold27/mkgold27.py proxy2b/common.py ':(exclude)v3/e3/**/*.sh' | tar -C "$R/rxm" -xf -
if [ ! -f "$R/exllamav3/.done" ]; then
  rm -rf "$R/exllamav3"; git clone -q "$EXL3_REPO" "$R/exllamav3"
  ( cd "$R/exllamav3" && git checkout -q "$EXL3_COMMIT" \
    && git -c user.name=build -c user.email=build@localhost am -q "$R"/rxm/quantize/exl3-patches/000*.patch \
    && TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=$JOBS "$V/bin/python" setup.py build_ext --inplace > "$ROOT/exl3-build.log" 2>&1 \
    && rm -rf build .git && touch .done )
  log "exllamav3 built"
fi
export PYTHONPATH=$R/exllamav3 TORCH_EXTENSIONS_DIR=$R/torch_ext TORCH_CUDA_ARCH_LIST=$ARCH MAX_JOBS=$JOBS
if [ ! -f "$R/torch_ext/.done" ]; then
  CUDA_VISIBLE_DEVICES= "$V/bin/python" "$R/app/docker/warm.py" "$R/rxm" "$R/exllamav3/exllamav3/exllamav3_ext" > "$ROOT/warm.log" 2>&1
  touch "$R/torch_ext/.done"; log "side extensions built"
fi
CUDA_VISIBLE_DEVICES= "$V/bin/python" -c "import exllamav3.modules.quant.exl3_lib.quantize, torch; print('import ok, torch', torch.__version__)" | tee -a "$ROOT/build.log"
echo "rad-quant-at-home-worker $(cat "$APP/VERSION") cuda-rootfs $ARCH repo $(git -C "$APP" rev-parse --short HEAD)-dirty rad-exl3-mix $RC exllamav3 $EXL3_COMMIT" > "$R/IMAGE"
log "done: $(cat "$R/IMAGE")"
