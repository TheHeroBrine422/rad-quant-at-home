#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# build.sh [rocm|cuda] [COMMIT | --worktree] -- build the rad-quant-at-home worker image (rad-quant-at-home-worker).
#
# The build context has two parts, both from committed trees (git archive: never local/, data or untracked files):
#   app/  this repository at COMMIT (default HEAD)
#   rxm/  the quantizer code the jobs run, from a rad-exl3-mix checkout at RXM_COMMIT (default its HEAD): only the
#         paths listed in RXM_PATHS (ourq.py, the exllamav3 patch series, the side extensions, the staged golden
#         recorder, its licence files)
# --worktree takes this repo's paths from its working tree (tests only; tags end in -dirty); rad-exl3-mix still comes
# from RXM_COMMIT unless RXM_WORKTREE=1.
#
# Env: RXM_REPO (default ../rad-exl3-mix next to this repo), RXM_COMMIT, IMAGE_REPO (default rad-quant-at-home-worker,
# e.g. ghcr.io/theherobrine422/rad-quant-at-home-worker), VERSION (default ./VERSION), DOCKER (default docker),
# RADIANCE_IMAGE (rocm: the radiance image of our fork's qc-exl3-v3 branch), BUILD_ARGS (extra --build-arg ..., e.g.
# "--build-arg TORCH_CUDA_ARCH=8.9 --build-arg MAX_JOBS=2" for a quick test build), DOCKER_BUILD_FLAGS.
# Tags:
#   $IMAGE_REPO:$VERSION-<stack>    e.g. 0.3.0-rocm7.14-gfx1201, 0.3.0-cu128   immutable: what donors pin
#   $IMAGE_REPO:$VERSION-<variant>  e.g. 0.3.0-rocm                           moves with rebuilds of the same version
#   $IMAGE_REPO:<variant>           e.g. rocm, cuda                           latest release of that variant
#   $IMAGE_REPO:<variant>-<commit>[-rxm<commit>]                              the exact source commits
# Publishing is a separate, manual step (README "Publishing images (maintainer)"); this script never pushes.
set -euo pipefail
VAR=${1:-rocm}; R=$(cd "$(dirname "$0")/.." && pwd)
RXM=${RXM_REPO:-$(cd "$R/.." && pwd)/rad-exl3-mix}
DOCKER=${DOCKER:-docker}; REPO=${IMAGE_REPO:-rad-quant-at-home-worker}; VERSION=${VERSION:-$(cat "$R/VERSION")}
case $VAR in                                   # keep in step with the ARGs of Dockerfile.$VAR
  rocm) STACK=rocm7.14-gfx1201 ;;
  cuda) STACK=cu128 ;;
  *) echo "variant: rocm | cuda"; exit 2 ;;
esac
[ -d "$RXM/quantize" ] || { echo "no rad-exl3-mix checkout at $RXM (set RXM_REPO)"; exit 2; }
APP_PATHS="LICENSE NOTICE LICENSES VERSION common jobs worker client docker checker"
RXM_PATHS="LICENSE NOTICE LICENSES quantize/ourq.py quantize/exl3-patches v3/e3/exl3frac v3/e3/exl3frac2 v3/e3/exl3cb
           eval/gold27/mkgold27.py proxy2b/common.py"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/ctx/app" "$T/ctx/rxm"
if [ "${2:-}" = "--worktree" ]; then
  C=$(git -C "$R" rev-parse --short HEAD)-dirty
  tar -C "$R" --exclude=__pycache__ --exclude='*.pyc' --exclude=.env --exclude=secrets -cf - $APP_PATHS | tar -C "$T/ctx/app" -xf -
else
  C=$(git -C "$R" rev-parse --short "${2:-HEAD}")
  git -C "$R" archive --format=tar "$C" -- $APP_PATHS | tar -C "$T/ctx/app" -xf -
fi
if [ "${RXM_WORKTREE:-0}" = 1 ]; then
  RC=$(git -C "$RXM" rev-parse --short HEAD)-dirty
  tar -C "$RXM" --exclude=__pycache__ --exclude='*.pyc' --exclude='v3/e3/*/*.sh' --exclude='v3/e3/*/build' -cf - $RXM_PATHS \
    | tar -C "$T/ctx/rxm" -xf -
else
  RC=$(git -C "$RXM" rev-parse --short "${RXM_COMMIT:-HEAD}")
  git -C "$RXM" archive --format=tar "$RC" -- $RXM_PATHS ':(exclude)v3/e3/**/*.sh' | tar -C "$T/ctx/rxm" -xf -
fi
[ -f "$T/ctx/app/docker/Dockerfile.$VAR" ] || { echo "docker/Dockerfile.$VAR is not in $C"; exit 1; }
TAG="$REPO:$VERSION-$STACK"
# shellcheck disable=SC2086
"$DOCKER" build -f "$T/ctx/app/docker/Dockerfile.$VAR" --build-arg VERSION="$VERSION" --build-arg REPO_COMMIT="$C" \
  --build-arg RXM_COMMIT="$RC" --build-arg IMAGE_TAG="$TAG" --build-arg CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  ${RADIANCE_IMAGE:+--build-arg RADIANCE_IMAGE="$RADIANCE_IMAGE"} ${BUILD_ARGS:-} \
  -t "$TAG" -t "$REPO:$VERSION-$VAR" -t "$REPO:$VAR" -t "$REPO:$VAR-$C-rxm$RC" ${DOCKER_BUILD_FLAGS:-} "$T/ctx"
echo "built $TAG (also $REPO:$VERSION-$VAR, $REPO:$VAR, $REPO:$VAR-$C-rxm$RC)"
