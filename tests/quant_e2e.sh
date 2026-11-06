#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# quant_e2e.sh -- one real quant_layer through server + worker vs a direct ourq.py run of the same layer on the same GPU
# (TESTS.md). Needs a datapack + an enqueued level in $WORK (admin.py datapack / manifest / enqueue-level, see TESTS.md).
# Env (no defaults, nothing machine-specific in the repo): WORK (db, data, results), MODEL_DIR (bf16 checkpoint),
# HESS_DIR, EXL3_SRC (fork's exllamav3_ext), EXL3_PATH (PYTHONPATH of the built exllamav3), TORCH_EXT, GPU_LOCK,
# VRAM_CAP (GB), LAYER, K, CB, SIGIN. Every GPU step runs inside one flock on $GPU_LOCK (fd 9).
set -u
V=$(cd "$(dirname "$0")/.." && pwd); R=${RXM:-$(cd "$V/.." && pwd)/rad-exl3-mix}   # R: the rad-exl3-mix checkout (quantize/ourq.py)
: "${WORK:?}" "${MODEL_DIR:?}" "${HESS_DIR:?}" "${EXL3_SRC:?}" "${EXL3_PATH:?}" "${TORCH_EXT:?}" "${GPU_LOCK:?}" "${VRAM_CAP:?}"
LAYER=${LAYER:-5} K=${K:-3} CB=${CB:-i8} SIGIN=${SIGIN:-0.005} PORT=${PORT:-18760}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $WORK/quant_e2e.log; }
P=$WORK/p; mkdir -p $P/challenge/hessians; ln -sfn $HESS_DIR $P/challenge/hessians/calib
python3 -c "import json; c=json.load(open('$MODEL_DIR/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$P/template.json', 'w'))"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nice -n 19 python3 -I $V/server/server.py --db $WORK/q.sqlite --data $WORK/data --spool $WORK/spool --results $WORK/results \
  --port $PORT > $WORK/server.log 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
TOK=$(python3 -I $V/server/admin.py --db $WORK/q.sqlite add-worker gpu4070 | awk '{print $4}')
direct() {   # direct ourq.py with proxy2b/q2b.sh's settings (one layer); $1 = out dir, rest = extra args
  out=$1; shift; mkdir -p $WORK/cwd; ( cd $WORK/cwd && PYTHONPATH=$EXL3_PATH RXM_WORK=$P RXM_CKPT=$MODEL_DIR RXM_TEMPLATE=$P/template.json \
    TORCH_EXTENSIONS_DIR=$TORCH_EXT nice -n 19 python3 $V/worker/capwrap.py $VRAM_CAP $R/quantize/ourq.py -K $K -o $out --device 0 \
    --layers $LAYER-$LAYER --codebook $CB --qtf2 --frac-ext $EXL3_SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in $SIGIN "$@" ) \
    > $out.log 2>&1
}
log "waiting for $GPU_LOCK"
exec 9>>$GPU_LOCK; flock 9 || { log "flock failed"; exit 1; }                       # held (by fd 9) until exec 9>&- below
( exec 9>&-; while sleep 1; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $WORK/vram.log 2>/dev/null; done ) &
MON=$!
log "lock held"
t0=$(date +%s.%N)
direct $WORK/direct --save-wq; rc=$?; t1=$(date +%s.%N); log "direct run: rc $rc $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s"
nice -n 19 python3 -I $V/worker/worker.py --server http://127.0.0.1:$PORT --token $TOK --state $WORK/worker \
  --gpu-name rtx4070 --vram-gb 12 --ram-gb 32 --cache-gb 20 --vram-cap-gb $VRAM_CAP --ourq $R/quantize/ourq.py \
  --exl3-src $EXL3_SRC --env PYTHONPATH=$EXL3_PATH --env TORCH_EXTENSIONS_DIR=$TORCH_EXT --image local-4070 \
  --max-jobs 1 --idle-exit 2 --poll 1 > $WORK/worker.log 2>&1
rc=$?; t2=$(date +%s.%N); log "worker run: rc $rc $(awk "BEGIN{printf \"%.1f\", $t2 - $t1}") s"
direct $WORK/seed1 --seed-offset 1; rc=$?; t3=$(date +%s.%N); log "seed-offset 1 run: rc $rc $(awk "BEGIN{printf \"%.1f\", $t3 - $t2}") s"
kill $MON; exec 9>&-; log "GPU total $(awk "BEGIN{printf \"%.1f\", $t3 - $t0}") s; lock released"
S=q-L$(printf %02d $LAYER).safetensors
J=$(python3 -I $V/server/admin.py --db $WORK/q.sqlite list | awk '$2=="quant_layer"{print $1; exit}')
if cmp $WORK/direct/$S $WORK/results/$J/$S; then log "shard: BYTE-IDENTICAL (direct vs worker)"; else log "shard: DIFFERENT"; fi
cmp <(cut -d, -f1-4 $WORK/direct/proxy.jsonl) <(cut -d, -f1-4 $WORK/results/$J/proxy.jsonl) && log "proxy_err: identical"
log "peak VRAM per process (MiB): $(awk -F', ' '{gsub(" MiB","",$2); if ($2>m[$1]) m[$1]=$2} END {for (p in m) printf "%s:%s ", p, m[p]}' $WORK/vram.log 2>/dev/null)"
log "DONE"
