#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rxq_anchor.py restamp -- our own rxq levels as trusted anchors under another settings stamp (0.6).

The scorer's anchors for a ladder level are our own runs of its jobs (`admin.py score-import --level L DIR`): DIR holds
t-<key>.safetensors files whose metadata name the level's settings stamp and level. Levels we quantized before the
freeze carry the stamp of the settings file they were made with; if the freeze only changed settings that do not reach
their jobs (e.g. #126: the codebook below K1, for a K2.5 level), their codes are what the frozen settings make, and
only the metadata must name the new stamp. This writes such copies:

  rxq_anchor.py restamp --rxq PIPELINE --old OLD.toml --new NEW.toml --out STAGE [--level NAME] [--code-stamp HEX
                        [--accept-code-change REASON]] [--allow-standin] SRC...

SRC: rxq job output dirs or flat dirs of t-files (w92's out/<tag>/<job>/, an assembled job set, rqah result dirs).
Refuses (exit 1, STAGE untouched) unless, for every t-file:
  - its metadata names OLD's settings stamp, a level of OLD at its K / codebook, and the GPU kernels (--allow-standin:
    tests only);
  - every per-job setting rxq derives for its key at its K / codebook is the same under OLD and NEW: rxq.config
    quant_settings (codebook, qtf2, damping incl. sub-1 / lowest level, drift, refit, gain, H_out dir / sigma / ldlq,
    tile CD sweeps), tile_cd module, the block order, the in-job C9 alpha and e_clip, the H_in source / blend, the H_out
    estimator / snapshot. A difference in any of them means other codes: no re-stamp, re-run the anchors;
  - with --code-stamp HEX (the freeze's rxq code stamp): its rxq_code equals it, or --accept-code-change REASON names
    why the code between them cannot change codes (e.g. "only rxq/ladder.py differs"; check with
    `git diff --stat ANCHOR_COMMIT FREEZE_COMMIT -- pipeline/rxq v3/e3/exl3frac v3/e3/exl3frac2 v3/e3/exl3cb` and the
    fork's commits). Without --code-stamp the code stamps are listed, not checked.
Writes STAGE/t-<key>.safetensors: the same header and tensor bytes, metadata rxq_settings / level replaced, plus
rxq_restamp_from (the old stamp and level) and rxq_restamp_note (REASON). With OLD = NEW it links the files unchanged.
Then: `admin.py score-import --level <new level> STAGE --note "..."` (the level's jobs must be enqueued first)."""
import argparse, glob, json, os, shutil, struct, sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return n, json.loads(f.read(n))


def rewrite(src, dst, meta):
    """src -> dst with __metadata__ = meta; the tensors' bytes copied unchanged (offsets are relative to the data start)"""
    n, h = header(src)
    h["__metadata__"] = {str(k): str(v) for k, v in meta.items()}
    hb = json.dumps(h, separators=(",", ":")).encode(); hb += b" " * (-len(hb) % 8)
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb); fi.seek(8 + n)
        while True:
            b = fi.read(1 << 24)
            if not b: break
            fo.write(b)
    os.replace(dst + ".part", dst)


def fingerprint(C, cfg, K, cb, key):
    """everything an rxq job derives from the settings file for one key at (K, codebook)"""
    from rxq import qwen35
    q = dict(C.quant_settings(cfg, K, cb, key))
    fam = qwen35.family(key)
    cd = C.tile_cd(cfg)
    c9 = C.c9_injob(cfg, key)
    ho = cfg["inputs"]["hout"]
    return {"quant": q, "tile_cd": cd if cd and fam in cd[1] else None,
            "block_order": C.block_order(cfg) if q.get("block_order") else None,
            "c9": (c9, cfg["post"]["c9"]["e_clip"]) if c9 is not None else None,
            "hin": {k: cfg["inputs"]["hin"][k] for k in sorted(cfg["inputs"]["hin"])},
            "hout": (ho["estimator"], ho["snapshot"]) if C.hout_applies(cfg, K) else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("restamp")
    x.add_argument("--rxq", required=True, help="rad-exl3-mix pipeline/ (rxq.config, rxq.ladder)")
    x.add_argument("--old", required=True); x.add_argument("--new", required=True); x.add_argument("--out", required=True)
    x.add_argument("--level", help="the new level name (default: rxq.ladder's level of NEW at the files' K / codebook)")
    x.add_argument("--code-stamp"); x.add_argument("--accept-code-change", metavar="REASON")
    x.add_argument("--allow-standin", action="store_true", help="tests only: CPU stand-in codes")
    x.add_argument("src", nargs="+")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(a.rxq))
    from rxq import config as C, ladder as LD
    old, new = C.load(a.old), C.load(a.new)
    so, sn = C.settings_stamp(old), C.settings_stamp(new)
    files = {}
    for s in a.src:
        for f in sorted(glob.glob(os.path.join(s, "t-*.safetensors")) + glob.glob(os.path.join(s, "*", "t-*.safetensors"))):
            b = os.path.basename(f)
            if b in files and files[b] != f: sys.exit(f"{b} twice: {files[b]} and {f} (one level, one seed per stage)")
            files[b] = f
    if not files: sys.exit(f"no t-*.safetensors in {a.src}")
    bad, kc, codes, lv_old = [], set(), {}, set()
    for b, f in files.items():
        m = header(f)[1].get("__metadata__") or {}
        key, K, cb = m.get("key"), float(m.get("K", "nan")), m.get("codebook")
        if b != f"t-{key}.safetensors": bad.append(f"{b}: metadata key {key}"); continue
        if m.get("rxq_settings") != so: bad.append(f"{b}: settings {m.get('rxq_settings')} != OLD's {so}"); continue
        if m.get("rxq_kernels", "gpu") != "gpu" and not a.allow_standin: bad.append(f"{b}: {m['rxq_kernels']} codes"); continue
        kc.add((K, cb)); lv_old.add(m.get("level")); codes[m.get("rxq_code")] = codes.get(m.get("rxq_code"), 0) + 1
        if fingerprint(C, old, K, cb, key) != fingerprint(C, new, K, cb, key):
            fo, fn = fingerprint(C, old, K, cb, key), fingerprint(C, new, K, cb, key)
            bad.append(f"{key} K{K:g} {cb}: the job settings differ under NEW ({[k for k in fo if fo[k] != fn[k]]}): other codes, re-run")
    if len(kc) > 1: bad.append(f"several K / codebooks in one stage: {sorted(kc)}")
    K, cb = next(iter(kc)) if kc else (None, None)
    names_old = {lv["name"] for lv in LD.levels(old) if float(lv["K"]) == K and lv["codebook"] == cb}
    if lv_old - names_old: bad.append(f"level {sorted(lv_old)} is not OLD's level at K{K:g} {cb} ({sorted(names_old)})")
    if a.level: lv_new = a.level
    else:
        cand = [lv["name"] for lv in LD.levels(new) if K is not None and float(lv["K"]) == K and lv["codebook"] == cb]
        lv_new = cand[0] if len(cand) == 1 else None
        if not lv_new: bad.append(f"NEW has no single ladder level at K{K} {cb} ({cand}); give --level")
    if a.code_stamp and set(codes) != {a.code_stamp} and not a.accept_code_change:
        bad.append(f"code stamps {codes} != the freeze's {a.code_stamp}: re-run, or --accept-code-change REASON")
    print(f"{len(files)} t-files, K{K:g} {cb}: settings {so} -> {sn}, level {sorted(lv_old)} -> {lv_new}; code stamps {codes}"
          + (f" (freeze {a.code_stamp})" if a.code_stamp else ""))
    if bad:
        for x in bad[:20]: print("refused:", x)
        print(f"RESTAMP REFUSED: {len(bad)} problem(s), nothing written"); return 1
    os.makedirs(a.out, exist_ok=True)
    for b, f in files.items():
        d = os.path.join(a.out, b)
        if os.path.lexists(d): os.remove(d)
        if so == sn and lv_new in lv_old: os.symlink(os.path.abspath(f), d); continue
        m = dict(header(f)[1].get("__metadata__") or {})
        m.update(rxq_settings=sn, level=lv_new, rxq_restamp_from=f"{so} {m['level']}",
                 rxq_restamp_note=a.accept_code_change or "settings-equivalent at this K / codebook")
        rewrite(f, d, m)
    print(f"RESTAMP OK: {len(files)} files -> {a.out} (level {lv_new}); next: admin.py score-import --level {lv_new} {a.out} --note ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
