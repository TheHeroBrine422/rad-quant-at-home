#!/bin/sh
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# radiance-wrap.sh -- run radiance from the copied rootfs of its own image (Dockerfile.rocm: /opt/rqah/radiance-root)
# through that rootfs's dynamic loader and libraries, so its ROCm / C++ runtime never mixes with the worker image's.
# UNTESTED until the first image build: check the loader and library paths of the radiance image.
set -eu
. /opt/rqah/radiance.env
R=$RADIANCE_ROOT
LD=$(ls "$R"/lib64/ld-linux-x86-64.so.2 "$R"/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 2>/dev/null | head -1)
export RADIANCE_HOME="${RADIANCE_HOME:-$R/opt/radiance/share/radiance}"
exec "$LD" --library-path "$R/opt/radiance/lib:$R/opt/rocm/lib:$R/lib/x86_64-linux-gnu:$R/usr/lib/x86_64-linux-gnu:$R/lib64" \
  "$R$RADIANCE_BIN" "$@"
