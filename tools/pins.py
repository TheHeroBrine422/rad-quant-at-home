#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pins.py fill -- write docker/pins.env for a release from the trees it is built from (0.6; stdlib + git only).

  pins.py fill --rxm RXM [--commit REV] --fork FORK [--fork-rev REV] [--radiance RAD [--radiance-rev qc-exl3-v3] |
               --radiance-commit SHA] [--radiance-image ghcr.io/...@sha256:<64 hex>] [--min-patches 9] [--dry-run]

Refuses (exit 1, nothing written) unless
  - rad-exl3-mix: REV (default HEAD) is a commit of RXM, and RXM's working tree equals it on every path the worker image
    takes (docker/build.sh RXM_PATHS) plus pipeline/config and pipeline/tests: no modified, staged or untracked file
    there (--allow-worktree-diff: pin a commit other than the checkout's state on purpose);
  - the exllamav3 fork is pinned: its working tree is clean (tracked files) and its tree at --fork-rev (default HEAD) is
    exactly upstream EXL3_COMMIT (the Dockerfiles' ARG, the base the image builds) + `git am` of quantize/exl3-patches/
    000*.patch as they are at REV, numbered 0001..N without gaps, N >= --min-patches. The image builds that series, so
    this is the check that the code we ran (the fork) is the code the donors run;
  - this repository's docker/pins.env has no uncommitted changes (the file is committed after this writes it);
  - the result passes tools/release.py's pin format (RADIANCE_IMAGE empty only warns: the ROCm release build stops on it).
Reported, not refused: whether REV / the radiance commit are on a remote-tracking branch (the release workflow fetches
them anonymously, so the user pushes them before the tag), the settings stamp of pipeline/config/v3.toml at REV (what the
ladder's level names carry). Last line: "PINS OK ..." or "PINS REFUSED: ...". --dry-run checks and prints the diff."""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, tomllib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common.util import stamp                       # noqa: E402
from tools import release as REL                    # noqa: E402

PINS = os.path.join(HERE, "docker", "pins.env")


def git(repo, *a, check=True, inp=None):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, input=inp)
    if check and r.returncode: raise RuntimeError(f"git -C {repo} {' '.join(a)}: {r.stderr.decode().strip()[:300]}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def rxm_paths():
    """docker/build.sh's RXM_PATHS: the one list of rad-exl3-mix paths the worker image carries"""
    s = open(os.path.join(HERE, "docker", "build.sh")).read()
    m = re.search(r'^RXM_PATHS="([^"]*)"', s, re.M)
    if not m: raise RuntimeError("docker/build.sh: no RXM_PATHS")
    return m.group(1).split()


def exl3_base():
    """the upstream exllamav3 commit both Dockerfiles build (ARG EXL3_COMMIT)"""
    got = set()
    for v in ("cuda", "rocm"):
        m = re.search(r"^ARG EXL3_COMMIT=([0-9a-f]{7,40})\s*$", open(os.path.join(HERE, "docker", f"Dockerfile.{v}")).read(), re.M)
        if not m: raise RuntimeError(f"docker/Dockerfile.{v}: no ARG EXL3_COMMIT")
        got.add(m.group(1))
    if len(got) != 1: raise RuntimeError(f"the Dockerfiles build different exllamav3 commits: {sorted(got)}")
    return got.pop()


def check_rxm(rxm, rev, allow_diff, bad, notes):
    commit = git(rxm, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}", check=False)
    if not commit: bad.append(f"rad-exl3-mix: {rev} is not a commit of {rxm}"); return None
    paths = rxm_paths() + ["pipeline/config", "pipeline/tests"]
    paths = [p for p in paths if git(rxm, "ls-tree", "--name-only", commit, "--", p)]
    if not allow_diff:
        diff = git(rxm, "diff", "--name-only", commit, "--", *paths).splitlines()
        untracked = git(rxm, "ls-files", "--others", "--exclude-standard", "--", *paths).splitlines()
        if diff or untracked:
            bad.append(f"rad-exl3-mix: the working tree differs from {commit[:12]} on the image's paths: "
                       f"{(diff + untracked)[:4]}{' ...' if len(diff + untracked) > 4 else ''} (commit them, or --allow-worktree-diff)")
    on = git(rxm, "branch", "-r", "--contains", commit, check=False)
    notes.append(f"rad-exl3-mix {commit[:12]}: " + (f"on {on.split()[0]}" if on else
                 "NOT on any remote-tracking branch yet (the user pushes it before the tag: the release workflow fetches it)"))
    try:
        cfg = tomllib.loads(git(rxm, "show", f"{commit}:pipeline/config/v3.toml"))
        notes.append(f"settings stamp of pipeline/config/v3.toml at {commit[:12]}: {stamp({k: cfg[k] for k in ('meta', 'inputs', 'quant', 'post')})}")
    except Exception as e:                              # noqa: BLE001 (a report line only)
        notes.append(f"settings stamp: not computed ({e})")
    return commit


def check_fork(fork, fork_rev, rxm, commit, min_patches, bad, notes, base=None):
    base = base or exl3_base()
    fc = git(fork, "rev-parse", "--verify", "-q", f"{fork_rev}^{{commit}}", check=False)
    if not fc: bad.append(f"fork: {fork_rev} is not a commit of {fork}"); return
    if fork_rev in ("HEAD", git(fork, "rev-parse", "--abbrev-ref", "HEAD", check=False)):
        dirty = git(fork, "status", "--porcelain", "--untracked-files=no")
        if dirty: bad.append(f"fork: uncommitted changes in {fork}: {dirty.splitlines()[:3]}")
    names = sorted(n for n in git(rxm, "ls-tree", "--name-only", f"{commit}:quantize/exl3-patches").splitlines()
                   if re.fullmatch(r"\d{4}-.*\.patch", n))
    nums = [int(n[:4]) for n in names]
    if nums != list(range(1, len(nums) + 1)): bad.append(f"fork: the patch series at {commit[:12]} is not 0001..N without gaps: {nums}"); return
    if len(nums) < min_patches: bad.append(f"fork: the series has {len(nums)} patches, at least {min_patches} expected"); return
    tmp = tempfile.mkdtemp(prefix="rqah-pins-")
    try:
        repo = os.path.join(tmp, "x")                   # a shared clone: nothing is written into the fork's repository
        subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", fork, repo], check=True, capture_output=True)
        if not git(repo, "rev-parse", "--verify", "-q", f"{base}^{{commit}}", check=False):
            bad.append(f"fork: upstream base {base} (Dockerfile EXL3_COMMIT) is not in {fork}"); return
        git(repo, "checkout", "-q", "--detach", base)
        pd = os.path.join(tmp, "patches"); os.makedirs(pd)
        for n in names:
            with open(os.path.join(pd, n), "wb") as f:
                f.write(subprocess.run(["git", "-C", rxm, "show", f"{commit}:quantize/exl3-patches/{n}"], check=True,
                                       capture_output=True).stdout)
        r = subprocess.run(["git", "-C", repo, "-c", "user.name=pins", "-c", "user.email=pins@localhost", "am", "-q",
                            *[os.path.join(pd, n) for n in names]], capture_output=True)
        if r.returncode: bad.append(f"fork: git am of the series on {base} failed: {r.stderr.decode().strip()[:200]}"); return
        t_series, t_fork = git(repo, "rev-parse", "HEAD^{tree}"), git(fork, "rev-parse", f"{fc}^{{tree}}")
        if t_series != t_fork:
            bad.append(f"fork: {fc[:12]}'s tree {t_fork[:12]} != {base} + 0001-{len(names):04d} at {commit[:12]} ({t_series[:12]}): the "
                       "fork has commits the series lacks (format-patch them into rad-exl3-mix), or the reverse")
        else:
            notes.append(f"fork {fc[:12]} = exllamav3 {base} + 0001-{len(names):04d} (tree {t_fork[:12]})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def render(path, new):
    """pins.env with the KEY=VALUE lines of new replaced (comments and order kept)"""
    out = []
    for l in open(path).read().splitlines(keepends=True):
        k, sep, _ = l.strip().partition("=")
        out.append(f"{k}={new[k]}\n" if sep and not l.lstrip().startswith("#") and k in new else l)
    return "".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("fill")
    x.add_argument("--rxm", required=True); x.add_argument("--commit", default="HEAD")
    x.add_argument("--fork", required=True); x.add_argument("--fork-rev", default="HEAD")
    x.add_argument("--exl3-base", help="upstream exllamav3 commit (default: the Dockerfiles' ARG EXL3_COMMIT; tests only)")
    x.add_argument("--radiance"); x.add_argument("--radiance-rev", default="qc-exl3-v3"); x.add_argument("--radiance-commit")
    x.add_argument("--radiance-image"); x.add_argument("--min-patches", type=int, default=9)
    x.add_argument("--allow-worktree-diff", action="store_true"); x.add_argument("--dry-run", action="store_true")
    x.add_argument("--file", default=PINS)
    a = ap.parse_args(argv)
    bad, notes = [], []
    cur = REL.pins(a.file)
    inside = not os.path.relpath(os.path.abspath(a.file), HERE).startswith("..")
    if inside and not a.dry_run and git(HERE, "status", "--porcelain", "--", os.path.abspath(a.file)):
        bad.append(f"{os.path.relpath(a.file, HERE)} has uncommitted changes")
    commit = check_rxm(a.rxm, a.commit, a.allow_worktree_diff, bad, notes)
    if commit: check_fork(a.fork, a.fork_rev, a.rxm, commit, a.min_patches, bad, notes, a.exl3_base)
    new = dict(cur)
    if commit: new["RXM_COMMIT"] = commit
    if a.radiance_commit: new["RADIANCE_COMMIT"] = a.radiance_commit
    elif a.radiance:
        rc = git(a.radiance, "rev-parse", "--verify", "-q", f"{a.radiance_rev}^{{commit}}", check=False)
        if not rc: bad.append(f"radiance: {a.radiance_rev} is not a commit of {a.radiance}")
        else:
            new["RADIANCE_COMMIT"] = rc
            on = git(a.radiance, "branch", "-r", "--contains", rc, check=False)
            notes.append(f"radiance {rc[:12]}: " + (f"on {on.split()[0]}" if on else "NOT on any remote-tracking branch yet"))
    if a.radiance_image is not None: new["RADIANCE_IMAGE"] = a.radiance_image
    if new["RADIANCE_COMMIT"] != cur["RADIANCE_COMMIT"] and a.radiance_image is None and cur["RADIANCE_IMAGE"]:
        bad.append("RADIANCE_COMMIT changes but RADIANCE_IMAGE does not: give the digest of the image built from the new commit")
    if not new["RADIANCE_IMAGE"]: notes.append("RADIANCE_IMAGE is empty: the ROCm release build stops until it is pinned (radiance-image.yml)")
    text = render(a.file, new)
    tmp = a.file + ".part"
    with open(tmp, "w") as f: f.write(text)
    try:
        try: REL.pins(tmp)                              # exits with a message on a format error
        except SystemExit as e: bad.append(f"format: {e}")
        for n in notes: print("note:", n)
        ch = [f"{k}: {cur[k] or '(empty)'} -> {new[k] or '(empty)'}" for k in cur if cur[k] != new[k]]
        print("changes: " + ("; ".join(ch) if ch else "none (the pins already say this)"))
        if bad:
            for b in bad: print("refused:", b)
            print(f"PINS REFUSED: {len(bad)} problem(s), {os.path.relpath(a.file, HERE)} unchanged"); return 1
        if a.dry_run: print(f"PINS OK (dry run): RXM_COMMIT {new['RXM_COMMIT'][:12]}, nothing written"); return 0
        os.replace(tmp, a.file)
        print(f"PINS OK: wrote {os.path.relpath(a.file, HERE)} (RXM_COMMIT {new['RXM_COMMIT'][:12]}); commit it before the tag")
        return 0
    finally:
        if os.path.exists(tmp): os.remove(tmp)


if __name__ == "__main__":
    sys.exit(main())
