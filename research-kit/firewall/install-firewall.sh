#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# install-firewall.sh [--backend nft|iptables] [--no-persist] -- run once on the host, as root, BEFORE the first
# ./run.sh up. Creates the research network and the rules that keep it off the LAN and the host, and makes them survive
# reboots. Idempotent: run it again any time (e.g. after the host's addresses change). Details: donor-firewall.sh.
exec "$(dirname "$(readlink -f "$0")")/donor-firewall.sh" install "$@"
