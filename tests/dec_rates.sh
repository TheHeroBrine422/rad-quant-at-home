#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# dec_rates.sh -- validate checker/exl3dec.py on more rates / codebooks: direct ourq.py runs of one layer (--save-wq)
# under one flock on $GPU_LOCK, then the CPU checker against ourq's own dequantized weights (check.py ourq --wq).
# Env as quant_e2e.sh (WORK, MODEL_DIR, HESS_DIR, EXL3_SRC, EXL3_PATH, TORCH_EXT, GPU_LOCK, VRAM_CAP, LAYER);
# RATES = "K:codebook ..." (default: sub-1, a 1/8 step, plain mul1 at a half step, the cubic int8 codebook).
set -u
V=$(cd "$(dirname "$0")/.." && pwd); R=${RXM:-$(cd "$V/.." && pwd)/rad-exl3-mix}   # R: the rad-exl3-mix checkout (quantize/ourq.py)
: "${WORK:?}" "${MODEL_DIR:?}" "${HESS_DIR:?}" "${EXL3_SRC:?}" "${EXL3_PATH:?}" "${TORCH_EXT:?}" "${GPU_LOCK:?}" "${VRAM_CAP:?}"
LAYER=${LAYER:-5} RATES=${RATES:-"0.5:i8 2.125:i8 3.5:mul1 2.25:i8c"}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $WORK/dec_rates.log; }
P=$WORK/p; mkdir -p $P/challenge/hessians $WORK/rates $WORK/cwd; ln -sfn $HESS_DIR $P/challenge/hessians/calib
python3 -c "import json; c=json.load(open('$MODEL_DIR/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$P/template.json', 'w'))"
log "waiting for $GPU_LOCK"
exec 9>>$GPU_LOCK; flock 9 || { log "flock failed"; exit 1; }
t0=$(date +%s.%N); log "lock held"
for kc in $RATES; do K=${kc%%:*} CB=${kc##*:}; O=$WORK/rates/K$K-$CB
  SIGIN=$(python3 -c "print(0.05 if $K < 1 else 0.005)")
  ( cd $WORK/cwd && PYTHONPATH=$EXL3_PATH RXM_WORK=$P RXM_CKPT=$MODEL_DIR RXM_TEMPLATE=$P/template.json TORCH_EXTENSIONS_DIR=$TORCH_EXT \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 19 python3 $V/worker/capwrap.py $VRAM_CAP $R/quantize/ourq.py -K $K -o $O \
    --device 0 --layers $LAYER-$LAYER --codebook $CB --qtf2 --frac-ext $EXL3_SRC --scaled-hessian --sigma-reg 0.025 \
    --sigma-reg-in $SIGIN --save-wq ) > $O.log 2>&1; log "K$K $CB: rc $?"
done
exec 9>&-; log "GPU $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t0}") s; lock released"
for kc in $RATES; do K=${kc%%:*} CB=${kc##*:}
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 python3 -I $V/checker/check.py ourq \
    --dir $WORK/rates/K$K-$CB --layer $LAYER --K $K --codebook $CB --ckpt $MODEL_DIR --hess $HESS_DIR --wq > $WORK/rates/K$K-$CB.check 2>&1
  log "K$K $CB: $(tail -1 $WORK/rates/K$K-$CB.check) | max wq_rel_fro $(python3 -c "
import json,sys; r=[json.loads(l) for l in open('$WORK/rates/K$K-$CB.check') if l.startswith('{')]; print('%.1e' % max(x['wq_rel_fro'] for x in r))")"
done
log "DONE"
