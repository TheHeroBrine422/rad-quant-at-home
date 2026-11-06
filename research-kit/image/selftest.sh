#!/bin/sh
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# selftest -- inside the research container: docker exec donor-research selftest [--host-ip IP] [--lan-gw IP] [...]
# (run.sh selftest on the host passes the host's own addresses and gateways for you). Options: selftest.py.
exec python3 -I /opt/donor-kit/selftest.py "$@"
