#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# build.sh rocm|cuda -- build the research image ghcr.io/theherobrine422/rad-quant-at-home-research:<version>-<variant>.
#
# The GPU stack pins (base image, wheel index, torch, ROCm) are read from the worker's ../docker/Dockerfile.<variant>,
# so the research shell and the job-queue worker run the same stack and cannot drift; everything else (tools,
# Tailscale, the start hook) is this folder's Dockerfile.<variant>. The build context is image/ only.
# Tags: <version>-<variant> (what volunteers pull and run.sh starts; version = ../VERSION), <variant> (latest), and
# <variant>-<commit> (the exact source). Env: IMAGE_REPO (default ghcr.io/theherobrine422/rad-quant-at-home-research),
# VERSION, DOCKER (default docker), BUILD_ARGS (extra --build-arg ..., e.g. "--build-arg TORCH_CUDA_ARCH=8.9"),
# DOCKER_BUILD_FLAGS. Never pushes (publishing: ../README.md "Publishing images (maintainer)").
set -euo pipefail
VAR=${1:?usage: build.sh rocm|cuda}; K=$(cd "$(dirname "$0")" && pwd); W=$K/../docker/Dockerfile.$VAR
DOCKER=${DOCKER:-docker}; REPO=${IMAGE_REPO:-ghcr.io/theherobrine422/rad-quant-at-home-research}
VERSION=${VERSION:-$(cat "$K/../VERSION")}
[ -f "$K/Dockerfile.$VAR" ] && [ -f "$W" ] || { echo "variant: rocm | cuda (needs ../docker/Dockerfile.$VAR)"; exit 2; }
arg() { sed -n "s/^ARG $1=//p" "$W" | head -1 | sed 's/^"//; s/"$//'; }
case $VAR in
  rocm) PINS=(PY_IMAGE TORCH_SPEC ROCM_SPEC GPU_ARCH) ;;   # ROCM_INDEX: ours (the worker's is wrong: patches/)
  cuda) PINS=(CUDA_IMAGE TORCH_SPEC TORCH_INDEX TORCH_CUDA_ARCH) ;;
esac
BA=()
for p in "${PINS[@]}"; do v=$(arg "$p"); [ -n "$v" ] || { echo "no ARG $p in $W"; exit 1; }; BA+=(--build-arg "$p=$v"); done
C=$(git -C "$K" rev-parse --short HEAD 2> /dev/null || echo unknown)
[ -z "$(git -C "$K" status --porcelain -- . 2> /dev/null)" ] || C=$C-dirty
echo "pins from $W: ${BA[*]}"
# shellcheck disable=SC2086
"$DOCKER" build -f "$K/Dockerfile.$VAR" "${BA[@]}" --build-arg KIT_COMMIT="$C" \
  --build-arg CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" ${BUILD_ARGS:-} \
  --build-arg VERSION="$VERSION" -t "$REPO:$VERSION-$VAR" -t "$REPO:$VAR" -t "$REPO:$VAR-$C" ${DOCKER_BUILD_FLAGS:-} "$K/image"
echo "built $REPO:$VERSION-$VAR (also $REPO:$VAR, $REPO:$VAR-$C)"
