#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# uninstall-firewall.sh [--remove-network] -- remove the rules, the boot unit and the saved settings (as root). Stop
# the research container first (./run.sh down): without the rules it would reach the LAN. Idempotent.
exec "$(dirname "$(readlink -f "$0")")/donor-firewall.sh" uninstall "$@"
