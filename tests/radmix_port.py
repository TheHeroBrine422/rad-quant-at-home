#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""radmix_port.py [WORKDIR] -- common/radfile.py (cut into pieces + assemble) against the real mix/radmix.cpp, byte for
byte, on a fake ladder of RAD2 containers (tests/fakes.py). CPU only, a few seconds.

RADMIX: a built radmix binary; otherwise it is compiled here with g++ from mix/radmix.cpp against RADIANCE_ABI
(radiance's abi/ directory: rad_format.h, rad_types.h, rad_encoding.h) and EXL3_ACT_INC (the exl3import plugin's
include/ directory: exl3_act.h). WORKDIR must not be a reflink filesystem (radmix would clone, and its 4 KiB tail
padding then comes from the donor file), e.g. tmpfs."""
import hashlib, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tests"))
from common import radfile as R
import fakes

W = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/tmp/radmix-port")
RES = []


def ok(name, cond, detail=""):
    RES.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def radmix_bin():
    if os.environ.get("RADMIX"): return os.environ["RADMIX"]
    abi, inc = os.environ.get("RADIANCE_ABI"), os.environ.get("EXL3_ACT_INC")
    if not (abi and inc): sys.exit("set RADMIX, or RADIANCE_ABI and EXL3_ACT_INC to build it")
    out = f"{W}/radmix"
    subprocess.run(["g++", "-O2", "-std=c++17", os.path.join(HERE, "..", "mix", "radmix.cpp"), f"-I{abi}", f"-I{inc}",
                    "-o", out], check=True)
    return out


def main():
    if os.path.exists(W): shutil.rmtree(W)
    os.makedirs(W)
    rm = radmix_bin()
    lv = fakes.ladder(f"{W}/levels", Ks=(2, 3, 4, 2.5))
    pieces = {k: R.cut(p, f"{W}/pieces/{k}") for k, p in lv.items()}
    base = "k2"
    names = fakes.weight_names()
    n_all = sum(len(p["pieces"]) for p in pieces.values())
    # identical weights in two levels are one blob
    shas = {}
    for k, p in pieces.items():
        for n, f in p["pieces"].items(): shas.setdefault(n, set()).add(sha(f))
    bf = [n for n in names if "norm" in n or n == "token_embd.weight"]
    ok("cut: the bf16 weights are byte-identical pieces in every level", all(len(shas[n]) == 1 for n in bf),
       f"{len(bf)} weights x {len(lv)} levels; {n_all} pieces")
    ok("cut: skeleton = the level's bytes before data_off", open(pieces[base]["skeleton"], "rb").read() ==
       open(lv[base], "rb").read()[:R.Rad(lv[base]).h["data_off"]])
    cases = {
        "empty mix (base re-laid out)": ({}, {}),
        "4 weights from 3 levels": ({"blk.0.attn_q.weight": "k3", "blk.1.ffn_down.weight": "k4", "blk.2.ffn_up.weight": "k2p5",
                                     "output.weight": "k4"}, {}),
        "every linear from k4": ({n: "k4" for n in names if any(f".{x}." in n for x in fakes.LINEARS)}, {}),
        "mix + act lines (one replaces BASE's key, two new)": (
            {"blk.0.attn_k.weight": "k3", "blk.1.ffn_down.weight": "k4"},
            {"blk.1.ffn_down.weight": "a8g128", "blk.0.attn_k.weight": "a8t", "blk.2.ffn_up.weight": "a8g256"}),
        "act only": ({}, {"blk.2.attn_q.weight": "f16"}),
        "a weight named from the base level itself": ({"blk.0.attn_q.weight": "k2", "blk.0.ffn_up.weight": "k3"}, {}),
    }
    for name, (mix, act) in cases.items():
        mf = f"{W}/{abs(hash(name))}.mix"
        with open(mf, "w") as f:
            f.write("# test\n")
            for t, l in sorted(mix.items()): f.write(f"{t} {lv[l]}\n")
            for t, m in sorted(act.items()): f.write(f"act {t} {m}\n")
        o1, o2 = f"{W}/a.rad", f"{W}/b.rad"
        r = subprocess.run([rm, o1, lv[base], mf], capture_output=True, text=True)
        pp = {n: pieces[mix.get(n, base)]["pieces"][n] for n in names}
        info = R.assemble(o2, pieces[base]["skeleton"], pp, act)
        size = R.mix_size(pieces[base]["skeleton"], pp, act)
        same = r.returncode == 0 and open(o1, "rb").read() == open(o2, "rb").read()
        ok(f"radmix vs radfile: {name}", same and info["sha256"] == sha(o1) and size == info["bytes"] == os.path.getsize(o1),
           f"{info['bytes']} B, radmix: {r.stderr.strip()[-90:]}")
        if same:                                                     # the result parses, every weight from its level
            m = R.Rad(o2); good = True
            for e in m.entries:
                src = R.Rad(lv[mix.get(e["name_s"], base)]); se = src.entries[src.idx[e["name_s"]]]
                good &= open(o2, "rb").read()[e["offset"]:e["offset"] + e["bytes"]] == open(lv[mix.get(e["name_s"], base)], "rb").read()[se["offset"]:se["offset"] + se["bytes"]]
            ok(f"  output parses; every weight's bytes from its level", good)
    # what radmix refuses, radfile refuses
    pp = {n: pieces[base]["pieces"][n] for n in names}
    bad = dict(pp); bad["blk.0.attn_q.weight"] = pieces["k3"]["pieces"]["blk.0.attn_norm.weight"]
    try: R.assemble(f"{W}/c.rad", pieces[base]["skeleton"], bad); r1 = False
    except R.RadError as e: r1 = "holds" in str(e)
    ok("refused: a piece of another weight", r1)
    try: R.assemble(f"{W}/c.rad", pieces[base]["skeleton"], {k: v for k, v in pp.items() if k != "output.weight"}); r2 = False
    except R.RadError as e: r2 = "no piece" in str(e)
    ok("refused: a weight without a piece", r2)
    try: R.assemble(f"{W}/c.rad", pieces[base]["skeleton"], pp, {"blk.0.attn_q.weight": "a8g100"}); r3 = False
    except R.RadError as e: r3 = "activation mode" in str(e)
    ok("refused: a bad act mode (a8g100)", r3)
    print(f"\n{sum(RES)}/{len(RES)} checks passed")
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
