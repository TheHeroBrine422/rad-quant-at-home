#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# mkseccomp.py MOBY_DEFAULT_JSON OUT_JSON -- Docker's default seccomp profile plus the NUMA memory-policy calls.
#
# ROCm's runtime calls get_mempolicy / mbind / set_mempolicy. Docker's default profile allows them only with
# CAP_SYS_NICE, which the research container does not have (cap-drop ALL), so AMD documents seccomp=unconfined.
# This keeps every other rule of the default profile and only drops the CAP_SYS_NICE condition on that one group. The
# kernel still checks CAP_SYS_NICE itself where it matters (MPOL_MF_MOVE_ALL), so nothing gets past the capability.
# Source: moby/profiles tag seccomp/v0.3.0, seccomp/default.json, sha256 6416b47770785a41ac59073cdc77d9fe98517df2799dc83ef207e622de3053f6
#   curl -sO https://raw.githubusercontent.com/moby/profiles/seccomp/v0.3.0/seccomp/default.json
#   python3 tools/mkseccomp.py default.json seccomp-rocm.json
import hashlib
import json
import sys

PINNED = "6416b47770785a41ac59073cdc77d9fe98517df2799dc83ef207e622de3053f6"
GROUP = {"get_mempolicy", "mbind", "set_mempolicy", "set_mempolicy_home_node"}

src, out = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
h = hashlib.sha256(raw).hexdigest()
if h != PINNED:
    sys.exit(f"{src}: sha256 {h}, expected the pinned moby/profiles seccomp/v0.3.0 file ({PINNED})")
prof = json.loads(raw)
hits = [s for s in prof["syscalls"] if set(s["names"]) == GROUP]
if len(hits) != 1 or hits[0].get("includes") != {"caps": ["CAP_SYS_NICE"]}:
    sys.exit("the mempolicy group is not where it was expected: check the source profile by hand")
del hits[0]["includes"]
hits[0]["comment"] = "rad-quant-at-home research-kit: allowed without CAP_SYS_NICE (ROCm NUMA calls)"
with open(out, "w") as f:
    json.dump(prof, f, indent="\t")
    f.write("\n")
print(f"wrote {out}: moby default ({h[:12]}) + {sorted(GROUP)} unconditional")
