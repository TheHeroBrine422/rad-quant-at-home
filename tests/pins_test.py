#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pins_test.py -- tools/pins.py fill on synthetic repositories (git only, no network, a few seconds): an "upstream"
exllamav3, a fork with two commits, a rad-exl3-mix with the fork's commits as quantize/exl3-patches/000N and the paths
docker/build.sh's RXM_PATHS names. Checks that it writes the pins for a clean, reproducible state and refuses a dirty
rad-exl3-mix tree, a fork the series does not reproduce (an extra commit, uncommitted changes), a gap in the series,
too short a series, and a radiance commit change without a new image digest."""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools import pins as P   # noqa: E402

T = tempfile.mkdtemp(prefix="rqah-pins-test-")
fails = []


def sh(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def g(repo, *a): return sh("git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@localhost", *a)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: f.write(text)


def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
    if not cond: fails.append(name)


def run(*extra, file=None):
    """pins.py fill against the synthetic trees -> (exit code, output)"""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P.main(["fill", "--rxm", RXM, "--fork", FORK, "--exl3-base", BASE, "--min-patches", "2", "--file", file or PF, *extra])
    return rc, buf.getvalue()


try:
    UP = f"{T}/upstream"; os.makedirs(UP); g(UP, "init", "-q")
    write(f"{UP}/exllamav3/quantize.py", "x = 1\n"); g(UP, "add", "-A"); g(UP, "commit", "-qm", "upstream")
    BASE = g(UP, "rev-parse", "HEAD")
    FORK = f"{T}/fork"; sh("git", "clone", "-q", UP, FORK)
    write(f"{FORK}/exllamav3/quantize.py", "x = 2\n"); g(FORK, "commit", "-qam", "p1")
    write(f"{FORK}/exllamav3/ext.cu", "// k\n"); g(FORK, "add", "-A"); g(FORK, "commit", "-qm", "p2")
    RXM = f"{T}/rxm"; os.makedirs(RXM); g(RXM, "init", "-q")
    for p in P.rxm_paths() + ["pipeline/config", "pipeline/tests"]:
        if p.endswith((".py", ".md")) or p in ("LICENSE", "NOTICE"): write(f"{RXM}/{p}", "#\n")
        elif p != "quantize/exl3-patches": write(f"{RXM}/{p}/x.py", "#\n")
    write(f"{RXM}/pipeline/config/v3.toml", "[meta]\nname = 't'\n[inputs]\n[quant]\n[post]\n")
    pd = f"{RXM}/quantize/exl3-patches"; os.makedirs(pd)
    g(FORK, "format-patch", "-q", f"{BASE}..HEAD", "-o", pd)
    write(f"{RXM}/README.md", "rxm\n"); g(RXM, "add", "-A"); g(RXM, "commit", "-qm", "rxm")
    C1 = g(RXM, "rev-parse", "HEAD")
    PF = f"{T}/pins.env"; shutil.copyfile(os.path.join(HERE, "docker", "pins.env"), PF)
    old = open(PF).read()

    rc, out = run("--dry-run")
    ok("clean trees, dry run: PINS OK, nothing written", rc == 0 and "PINS OK (dry run)" in out and open(PF).read() == old, out)
    ok("the fork is reported as base + 0001-0002", f"= exllamav3 {BASE} + 0001-0002" in out, out)
    rc, out = run()
    txt = open(PF).read()
    ok("write: RXM_COMMIT = the commit, comments kept", rc == 0 and f"RXM_COMMIT={C1}\n" in txt
       and txt.count("#") == old.count("#") and len(txt.splitlines()) == len(old.splitlines()), out)

    write(f"{RXM}/pipeline/rxq/new.py", "#\n")
    rc, out = run("--dry-run")
    ok("untracked file in pipeline/rxq: refused", rc == 1 and "working tree differs" in out, out)
    rc, out = run("--dry-run", "--allow-worktree-diff")
    ok("... unless --allow-worktree-diff", rc == 0, out)
    os.remove(f"{RXM}/pipeline/rxq/new.py")
    write(f"{RXM}/quantize/ourq.py", "# changed\n")
    rc, out = run("--dry-run")
    ok("modified ourq.py: refused", rc == 1 and "quantize/ourq.py" in out, out)
    g(RXM, "checkout", "-q", "--", "quantize/ourq.py")
    write(f"{RXM}/docs/notes.md", "elsewhere\n")
    rc, out = run("--dry-run")
    ok("a change outside the image's paths: accepted", rc == 0, out)

    write(f"{FORK}/exllamav3/quantize.py", "x = 3\n")
    rc, out = run("--dry-run")
    ok("fork with uncommitted changes: refused", rc == 1 and "uncommitted changes" in out, out)
    g(FORK, "commit", "-qam", "p3 (not in the series)")
    rc, out = run("--dry-run")
    ok("fork commit missing from the series: refused (tree differs)", rc == 1 and "the fork has commits the series lacks" in out, out)
    rc, out = run("--dry-run", "--fork-rev", "HEAD~1")
    ok("--fork-rev at the series' state: accepted", rc == 0, out)
    g(FORK, "format-patch", "-q", "--start-number", "4", "HEAD~1..HEAD", "-o", pd)
    g(RXM, "add", "-A"); g(RXM, "commit", "-qm", "gap")
    rc, out = run("--dry-run")
    ok("series 0001, 0002, 0004: refused (gap)", rc == 1 and "without gaps" in out, out)
    os.rename(f"{pd}/" + [n for n in os.listdir(pd) if n.startswith("0004")][0], f"{pd}/0003-p3.patch")
    g(RXM, "add", "-A"); g(RXM, "commit", "-qm", "0003")
    rc, out = run("--dry-run")
    ok("series 0001-0003 = the fork: accepted", rc == 0 and "+ 0001-0003" in out, out)
    rc, out = run("--dry-run", "--min-patches", "9")
    ok("--min-patches 9 with 3: refused", rc == 1 and "at least 9" in out, out)
    rc, out = run("--dry-run", "--commit", C1)
    ok("an older rad-exl3-mix commit than the checkout: refused (tree differs on the patches)", rc == 1 and "working tree differs" in out, out)

    rc, out = run("--dry-run", "--radiance-commit", "1" * 40)
    cur = P.REL.pins(PF)
    ok("radiance commit change: needs a new image digest when one is pinned" if cur["RADIANCE_IMAGE"] else
       "radiance commit change with no image pinned: accepted, empty image noted", (rc == 1) if cur["RADIANCE_IMAGE"] else
       (rc == 0 and "RADIANCE_IMAGE is empty" in out), out)
    rc, out = run("--dry-run", "--radiance-commit", "1" * 40, "--radiance-image", "ghcr.io/x/y@sha256:" + "a" * 64)
    ok("radiance commit + digest: accepted", rc == 0, out)
    rc, out = run("--dry-run", "--radiance-image", "docker.io/x:latest")
    ok("a radiance image that is not REPO@sha256: refused", rc == 1 and "format" in out, out)
finally:
    shutil.rmtree(T, ignore_errors=True)
print(f"pins_test: {'ALL PASS' if not fails else f'{len(fails)} FAIL'}")
sys.exit(1 if fails else 0)
