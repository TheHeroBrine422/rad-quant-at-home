#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# cuda_smoke.sh -- the CUDA worker image end to end: a local server (this checkout, host python3 + numpy) with its
# scorer on, one layer of a small qwen3_5 model as quant_tensor jobs verified by score (docs/VERIFY.md), one donor worker
# that takes ONE job (a Hessian group) and one verifier worker that takes its anchor re-run, both from the image; then
# the scorer must pass both and grant the donor its lumens. TESTS.md section 11.
#
#   MODE=docker  (default) `docker run --gpus all --network host IMAGE`: a host with Docker and the NVIDIA Container Toolkit
#   MODE=rootfs  ROOTFS/opt/rqah and ROOTFS/opt/venv: the image's tree extracted, or built in place without Docker by
#                tests/cuda_rootfs.sh. Uses the tree's own prebuilt side extensions (ROOTFS/opt/rqah/torch_ext); an
#                extracted tree at another path than it was built for gets them compiled again first (torch's JIT keys
#                them by absolute path), with the box's nvcc
#   MODE=native  no image: the box's own python / exllamav3 build (RXM = a rad-exl3-mix checkout, EXL3_SRC, EXL3_PATH,
#                TORCH_EXT as for tests/quant_e2e.sh): the job, worker and scorer code on a real GPU
# Env: WORK (scratch, wiped), CKPT (e.g. a Qwen3.5-0.8B checkpoint), HESS (its calib Hessians, challenge layout),
#      IMAGE (docker; default rad-quant-at-home-worker:cuda), ROOTFS (rootfs), LAYER (default 6), K (default 3),
#      GPU_LOCK (optional: flock it around the GPU part), VRAM_CAP (GB, default 7.2), PORT (default 18790)
set -u
APP=$(cd "$(dirname "$0")/.." && pwd)
: "${WORK:?}" "${CKPT:?}" "${HESS:?}"
MODE=${MODE:-docker} IMAGE=${IMAGE:-rad-quant-at-home-worker:cuda} LAYER=${LAYER:-6} K=${K:-3} VRAM_CAP=${VRAM_CAP:-7.2}
PORT=${PORT:-18790}
rm -rf "$WORK"; mkdir -p "$WORK"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$WORK/smoke.log"; }
A=(python3 -I "$APP/server/admin.py" --db "$WORK/q.sqlite" --data "$WORK/data" --results "$WORK/results")
"${A[@]}" datapack smoke r0 "$CKPT" "$HESS" --layers "$LAYER-$LAYER" --no-head --groups > /dev/null || exit 1
"${A[@]}" manifest > /dev/null
"${A[@]}" enqueue-level --model smoke --rev r0 --K "$K" --codebook i8 --level smoke-k$K --layers "$LAYER-$LAYER" \
  --sigma-reg 0.025 --sigma-reg-in 0.005 --scaled-hessian --qtf2 --verify score --unit group | tee -a "$WORK/smoke.log"
TD=$("${A[@]}" add-worker donor | awk '{print $4}'); TV=$("${A[@]}" add-worker verifier | awk '{print $4}')
"${A[@]}" verifier "$("${A[@]}" workers | awk '$2=="verifier"{print $1}')" > /dev/null
nice -n 10 python3 -I "$APP/server/server.py" --db "$WORK/q.sqlite" --data "$WORK/data" --spool "$WORK/spool" \
  --results "$WORK/results" --bind 127.0.0.1 --port "$PORT" --score-threads 1 --score-poll-s 2 --spot-rate 0 \
  --status-interval 0 > "$WORK/server.log" 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
sleep 2
if [ "$MODE" = rootfs ]; then
  : "${ROOTFS:?}"; R=$ROOTFS/opt/rqah PY=$ROOTFS/opt/venv/bin/python
  export TORCH_EXTENSIONS_DIR=$R/torch_ext TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} MAX_JOBS=${MAX_JOBS:-4}
  log "rootfs: checking the side extensions (compiled only if the tree moved)"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$R/exllamav3 nice -n 19 "$PY" "$R/app/docker/warm.py" "$R/rxm" "$R/exllamav3/exllamav3/exllamav3_ext" \
    >> "$WORK/warm.log" 2>&1 || { log "warm failed (warm.log)"; exit 1; }
  log "rootfs: extensions ready in $(( $(date +%s) - t0 )) s"
fi
worker() {  # name token state
  if [ "$MODE" = native ]; then
    python3 -I "$APP/worker/worker.py" --state "$3" --python python3 --ourq "$RXM/quantize/ourq.py" --exl3-src "$EXL3_SRC" \
      --env "PYTHONPATH=$EXL3_PATH" --env "TORCH_EXTENSIONS_DIR=$TORCH_EXT" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --gpu-env CUDA_VISIBLE_DEVICES --image native-smoke --server "http://127.0.0.1:$PORT" --token "$2" --name "$1" \
      --gpu-name smoke --gpus 0 --vram-gb 12 --ram-gb 16 --cache-gb 5 --vram-cap-gb "$VRAM_CAP" --max-jobs 1 --idle-exit 3 --poll 1
  elif [ "$MODE" = docker ]; then
    docker run --rm --gpus all --network host --security-opt no-new-privileges --cap-drop ALL -v "$3:/var/lib/rqah" \
      "$IMAGE" --server "http://127.0.0.1:$PORT" --token "$2" --name "$1" --gpu-name smoke --gpus 0 --vram-gb 12 \
      --ram-gb 16 --cache-gb 5 --vram-cap-gb "$VRAM_CAP" --max-jobs 1 --idle-exit 3 --poll 1
  else
    "$PY" -I "$R/app/worker/worker.py" --state "$3" --python "$PY" --ourq "$R/rxm/quantize/ourq.py" \
      --exl3-src "$R/exllamav3/exllamav3/exllamav3_ext" --env "PYTHONPATH=$R/exllamav3" --env "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --job-types noop,quant_layer,quant_head,quant_tensor \
      --gpu-env CUDA_VISIBLE_DEVICES --image rootfs-smoke \
      --server "http://127.0.0.1:$PORT" --token "$2" --name "$1" --gpu-name smoke --gpus 0 --vram-gb 12 --ram-gb 16 --cache-gb 5 \
      --vram-cap-gb "$VRAM_CAP" --max-jobs 1 --idle-exit 3 --poll 1
  fi
}
mkdir -p "$WORK/wd" "$WORK/wv"; chmod 0777 "$WORK/wd" "$WORK/wv"
gpu() {
  ( while sleep 0.5; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$WORK/vram.log" 2>/dev/null; done ) &
  MON=$!
  t0=$(date +%s); worker donor "$TD" "$WORK/wd" > "$WORK/donor.log" 2>&1; log "donor worker: rc $? in $(( $(date +%s) - t0 )) s"
  for i in $(seq 1 60); do                                 # the scorer marks it provisional and queues the anchor re-run
    "${A[@]}" list --type quant_tensor | grep -q "chk provisional" && break; sleep 1
  done
  t0=$(date +%s); worker verifier "$TV" "$WORK/wv" > "$WORK/verifier.log" 2>&1; log "verifier worker: rc $? in $(( $(date +%s) - t0 )) s"
  kill $MON
}
if [ -n "${GPU_LOCK:-}" ]; then
  exec 9>>"$GPU_LOCK"; log "waiting for $GPU_LOCK"; flock 9; log "lock held"; gpu; exec 9>&-; log "lock released"
else gpu; fi
for i in $(seq 1 60); do                                   # the scorer: provisional -> anchor -> pass
  st=$("${A[@]}" list --type quant_tensor | awk '$4=="accepted"' | grep -c "chk pass")
  [ "$st" -ge 2 ] && break; sleep 2
done
"${A[@]}" list --type quant_tensor | tee -a "$WORK/smoke.log"
"${A[@]}" scores | tee -a "$WORK/smoke.log"
"${A[@]}" points | tee -a "$WORK/smoke.log"
ok=$("${A[@]}" list --type quant_tensor | awk '$4=="accepted"' | grep -c "chk pass")
log "peak VRAM per process (MiB): $(awk -F', ' '{gsub(" MiB","",$2); if ($2+0>m[$1]) m[$1]=$2+0} END {for (p in m) printf "%s:%s ", p, m[p]}' "$WORK/vram.log" 2>/dev/null)"
if [ "$ok" -ge 2 ]; then log "SMOKE PASS: the donor's job and its anchor re-run accepted and passed by score"; exit 0; fi
log "SMOKE FAIL (see $WORK/*.log)"; exit 1
