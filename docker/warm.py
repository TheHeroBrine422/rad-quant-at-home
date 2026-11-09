# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""warm.py RXM SRC -- compile the side extensions ourq.py and rxq build at run time (exl3frac, exl3cb i8 / i8c [/ i4],
exl3frac2 mul1 / i8 / i8c [/ i4]; their sources under RXM/v3/e3/, RXM = the rad-exl3-mix tree in the image) into $TORCH_EXTENSIONS_DIR at image
build time. SRC: the patched exllamav3_ext sources. Compiling needs no GPU; nothing is run on a device."""
import os, sys

rxm, src = sys.argv[1], sys.argv[2]
for d in ("exl3frac", "exl3cb", "exl3frac2"): sys.path.insert(0, os.path.join(rxm, "v3", "e3", d))
import exl3frac, exl3cb, exl3frac2
exl3frac.load(src)
for v in ("i8", "i8c"): exl3cb.load(src, v)
for v in ("mul1", "i8", "i8c"): exl3frac2.load(src, v)
for m in (exl3cb, exl3frac2):           # i4 (int4 grid: rxq ladder levels whose codebook is i4), where the pinned code has it
    try: m.load(src, "i4")
    except AssertionError as e:         # the variant check comes before any compile; its message is the variant
        if str(e) != "i4": raise
print("extensions built in", os.environ.get("TORCH_EXTENSIONS_DIR"))
