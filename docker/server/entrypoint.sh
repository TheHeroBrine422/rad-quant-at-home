#!/bin/sh
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# rad-quant-at-home (alias rqah) -- the server image's entrypoint (on PATH, so `docker compose exec server rqah admin
# list` works too).
#   rqah [serve] [server.py flags]   the queue (config from RQAH_* env; flags win)
#   rqah admin COMMAND ...           admin.py on the same db / data / results (RQAH_DB, _DATA, _RESULTS)
#   rqah healthcheck                 exit 0 if /healthz answers
#   rqah version                     image version and repo commit
set -eu
V=/opt/rqah
cmd=${1:-serve}
[ $# -gt 0 ] && shift
case "$cmd" in
  serve)       exec python -I "$V/server/server.py" "$@" ;;
  admin)       exec python -I "$V/server/admin.py" "$@" ;;
  healthcheck) exec python -I /opt/rqah/healthcheck.py ;;
  version)     cat /opt/rqah/IMAGE ;;
  -*)          exec python -I "$V/server/server.py" "$cmd" "$@" ;;
  *)           echo "usage: rad-quant-at-home (rqah) [serve [flags] | admin COMMAND ... | healthcheck | version]" >&2; exit 2 ;;
esac
