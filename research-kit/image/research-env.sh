# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# /etc/profile.d/research-env.sh -- the research shell's environment. Tailscale SSH sessions do not inherit the image's
# ENV, so login shells read this file, and ~/.bashrc and /etc/bash.bashrc source it on their FIRST line (Debian's
# .bashrc returns early for non-interactive shells such as `ssh host cmd`). Everything that grows lives on /data.
[ -n "${RESEARCH_ENV:-}" ] && return 0 2> /dev/null
export RESEARCH_ENV=1
[ -f /opt/donor-kit/stack.env ] && . /opt/donor-kit/stack.env   # written at build time: PATH / ROCM / CUDA bits
export PATH="$HOME/.local/bin:/opt/venv/bin:${STACK_PATH:+$STACK_PATH:}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR=/data/tmp HF_HOME=/data/cache/huggingface PIP_CACHE_DIR=/data/cache/pip XDG_CACHE_HOME=/data/cache \
  TORCH_EXTENSIONS_DIR=/data/torch_ext TRITON_CACHE_DIR=/data/cache/triton HF_HUB_DISABLE_TELEMETRY=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
