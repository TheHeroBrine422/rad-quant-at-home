#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# build.sh [COMMIT | --worktree] -- build the rad-quant-at-home queue server image (rad-quant-at-home-server) from a minimal context (LICENSE, NOTICE and the
# code it runs; never local/, data or untracked files except with --worktree, which copies the same paths
# from the working tree for testing uncommitted code).
# Tags: $IMAGE_REPO:$VERSION, $IMAGE_REPO:$VERSION-<commit> (or -dirty), $IMAGE_REPO:latest.
#   IMAGE_REPO (default rad-quant-at-home-server), VERSION (default VERSION), DOCKER (default docker; podman works)
set -euo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
DOCKER=${DOCKER:-docker}; REPO=${IMAGE_REPO:-rad-quant-at-home-server}; VERSION=${VERSION:-$(cat "$R/VERSION")}
PATHS="LICENSE NOTICE LICENSES VERSION common jobs server checker docker/server"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
if [ "${1:-}" = "--worktree" ]; then
  C=$(git -C "$R" rev-parse --short HEAD)-dirty
  tar -C "$R" --exclude=__pycache__ --exclude='*.pyc' --exclude=.env --exclude=secrets -cf "$T/ctx.tar" $PATHS
else
  C=$(git -C "$R" rev-parse --short "${1:-HEAD}")
  git -C "$R" archive --format=tar "$C" $PATHS > "$T/ctx.tar" || { echo "the paths are not committed at $C (--worktree to test)"; exit 1; }
fi
mkdir "$T/ctx" && tar -C "$T/ctx" -xf "$T/ctx.tar"
"$DOCKER" build -f "$T/ctx/docker/server/Dockerfile" \
  --build-arg VERSION="$VERSION" --build-arg REPO_COMMIT="$C" --build-arg CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t "$REPO:$VERSION" -t "$REPO:$VERSION-$C" -t "$REPO:latest" "$T/ctx"
echo "built $REPO:$VERSION ($REPO:$VERSION-$C)"
