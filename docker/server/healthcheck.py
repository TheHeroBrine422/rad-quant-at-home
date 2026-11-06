# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""container healthcheck: GET http://127.0.0.1:$RQAH_PORT/healthz (db answers, spool / results writable)"""
import os, sys, urllib.request

try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('RQAH_PORT') or os.environ.get('RADQUANT_PORT') or '8740'}/healthz", timeout=4)
    sys.exit(0 if r.status == 200 else 1)
except Exception as e:
    print(f"unhealthy: {e!r}"); sys.exit(1)
