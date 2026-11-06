# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""warm.py RXM SRC -- compile the side extensions ourq.py builds at run time (exl3frac, exl3cb i8 / i8c, exl3frac2 mul1 /
i8 / i8c; their sources under RXM/v3/e3/, RXM = the rad-exl3-mix tree in the image) into $TORCH_EXTENSIONS_DIR at image
build time. SRC: the patched exllamav3_ext sources. Compiling needs no GPU; nothing is run on a device."""
import os, sys

rxm, src = sys.argv[1], sys.argv[2]
for d in ("exl3frac", "exl3cb", "exl3frac2"): sys.path.insert(0, os.path.join(rxm, "v3", "e3", d))
import exl3frac, exl3cb, exl3frac2
exl3frac.load(src)
for v in ("i8", "i8c"): exl3cb.load(src, v)
for v in ("mul1", "i8", "i8c"): exl3frac2.load(src, v)
print("extensions built in", os.environ.get("TORCH_EXTENSIONS_DIR"))
