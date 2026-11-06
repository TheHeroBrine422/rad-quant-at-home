# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# The rad-quant-at-home queue server image (rad-quant-at-home-server): server.py + admin.py (+ the CPU checker), Python stdlib +
# numpy, no GPU, non-root.
# Build with docker/server/build.sh (a minimal context of the needed paths: never local/ or data).
# Run with compose.yaml next to this file (README "Server container"); config by RQAH_* environment variables
# (the 0.2 names RADQUANT_* are still read in 0.3).
ARG PY_IMAGE=python:3.14-slim-trixie
FROM ${PY_IMAGE}

ARG VERSION=0.0.0-dev
ARG REPO_COMMIT=unknown
ARG SOURCE=https://github.com/TheHeroBrine422/rad-quant-at-home
ARG CREATED=unknown
ARG NUMPY_SPEC="numpy==2.5.4"

# numpy only for the quant checker and the score verifier (`rqah admin check` / `score`); the server itself is stdlib
RUN pip install --no-cache-dir --root-user-action=ignore "${NUMPY_SPEC}" \
 && groupadd --system --gid 10001 rqah \
 && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent --no-create-home --shell /usr/sbin/nologin rqah

COPY LICENSE NOTICE VERSION /opt/rqah/
COPY LICENSES /opt/rqah/LICENSES
COPY common /opt/rqah/common
COPY jobs /opt/rqah/jobs
COPY server /opt/rqah/server
COPY checker /opt/rqah/checker
COPY docker/server/entrypoint.sh /usr/local/bin/rad-quant-at-home
COPY docker/server/healthcheck.py /opt/rqah/healthcheck.py

# db: the SQLite file (+ -wal/-shm); out: results/ and spool/ on ONE filesystem (completion renames spool -> results);
# data: the per-file inputs (read-only for the server); sources: what datapack symlinks into data (bf16, Hessians)
RUN python -m compileall -q /opt/rqah \
 && chmod 0755 /usr/local/bin/rad-quant-at-home && ln -s rad-quant-at-home /usr/local/bin/rqah && chmod -R a-w /opt/rqah \
 && mkdir -p /srv/rqah/db /srv/rqah/data /srv/rqah/out/results /srv/rqah/out/spool /srv/rqah/sources \
 && chown -R 10001:10001 /srv/rqah/db /srv/rqah/data /srv/rqah/out \
 && echo "rad-quant-at-home-server ${VERSION} repo ${REPO_COMMIT}" > /opt/rqah/IMAGE

ENV RQAH_DB=/srv/rqah/db/queue.sqlite RQAH_DATA=/srv/rqah/data RQAH_RESULTS=/srv/rqah/out/results \
    RQAH_SPOOL=/srv/rqah/out/spool RQAH_BIND=0.0.0.0 RQAH_PORT=8740 PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.title="rad-quant-at-home-server" \
      org.opencontainers.image.description="rad-quant-at-home: job queue server for donated-compute EXL3 quantization and evals " \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.documentation="${SOURCE}/blob/main/README.md" \
      org.opencontainers.image.revision="${REPO_COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.licenses="Apache-2.0"
USER 10001:10001
WORKDIR /srv/rqah
EXPOSE 8740
VOLUME ["/srv/rqah/db", "/srv/rqah/out"]
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 CMD ["python", "-I", "/opt/rqah/healthcheck.py"]
ENTRYPOINT ["/usr/local/bin/rad-quant-at-home"]
CMD ["serve"]
