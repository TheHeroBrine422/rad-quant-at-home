#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""release.py COMMAND -- the release workflow's steps that are not docker (.github/workflows/release.yml; stdlib only,
run from the repository root; tests/update_test.py covers them).

  check-tag TAG                  exit 1 unless TAG is v<VERSION> (the workflow publishes only that)
  pins [--need-radiance]         docker/pins.env as KEY=VALUE lines for $GITHUB_ENV, checked (commits are 40 hex,
                                 RADIANCE_IMAGE is REPO@sha256:...; --need-radiance: it must be set)
  manifest --digest STACK=sha256:... [--digest ...] [--commit SHA] [--setup-min X.Y.Z]
                                 release.json on stdout (common/release.py): VERSION, the commit, the digests the
                                 pushes reported, the pinned sources, the changes since the previous tag
  notes --manifest release.json  the GitHub Release text: version, changes since the previous tag, image tags and
                                 digests, how donors update"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from common import release as RL

PINS = os.path.join(HERE, "docker", "pins.env")
PIN_RULES = {"RXM_REPO_URL": r"https://[A-Za-z0-9./_-]+", "RXM_COMMIT": r"[0-9a-f]{40}",
             "RADIANCE_REPO_URL": r"https://[A-Za-z0-9./_-]+", "RADIANCE_COMMIT": r"[0-9a-f]{40}",
             "RADIANCE_IMAGE": r"(ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64})?"}


def version():
    v = open(os.path.join(HERE, "VERSION")).read().strip()
    if not re.fullmatch(RL.VERSION_PAT, v): sys.exit(f"VERSION {v!r} is not X.Y.Z")
    return v


def pins(path=PINS, need_radiance=False):
    out = {}
    for n, l in enumerate(open(path), 1):
        l = l.strip()
        if not l or l.startswith("#"): continue
        k, sep, v = l.partition("=")
        if not sep or k not in PIN_RULES: sys.exit(f"{path}:{n}: unknown line {l[:60]!r}")
        if not re.fullmatch(PIN_RULES[k], v): sys.exit(f"{path}:{n}: {k}={v[:80]!r} does not match {PIN_RULES[k]}")
        out[k] = v
    miss = sorted(set(PIN_RULES) - set(out))
    if miss: sys.exit(f"{path}: missing {miss}")
    if need_radiance and not out["RADIANCE_IMAGE"]:
        sys.exit(f"{path}: RADIANCE_IMAGE is empty: publish the radiance image first (.github/workflows/radiance-image.yml) "
                 "and pin its digest here")
    return out


def git(*a):
    r = subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def changes(tag):
    """the subjects of the commits since the previous v* tag (or the last 50), merges left out"""
    prev = git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", f"{tag}^") if git("rev-parse", "-q", "--verify", tag) else \
        git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*", "HEAD")
    rng = f"{prev}..{tag if git('rev-parse', '-q', '--verify', tag) else 'HEAD'}" if prev else "HEAD"
    log = git("log", "--no-merges", "--format=- %s (%h)", "-n", "60", rng) or ""
    head = f"Changes since {prev}:" if prev else "Changes (first tagged release; the last 60 commits):"
    return head + "\n" + "\n".join(l[:200] for l in log.splitlines())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("check-tag"); x.add_argument("tag")
    x = sp.add_parser("pins"); x.add_argument("--need-radiance", action="store_true"); x.add_argument("--file", default=PINS)
    x = sp.add_parser("manifest"); x.add_argument("--digest", action="append", default=[], metavar="STACK=sha256:...")
    x.add_argument("--commit"); x.add_argument("--setup-min"); x.add_argument("--pins", default=PINS); x.add_argument("--created")
    x = sp.add_parser("notes"); x.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)
    v = version()
    if a.cmd == "check-tag":
        if a.tag != f"v{v}": sys.exit(f"tag {a.tag!r} is not v{v} (VERSION): bump VERSION in the tagged commit, or tag v{v}")
        print(f"tag {a.tag} = VERSION {v}"); return 0
    if a.cmd == "pins":
        for k, val in pins(a.file, a.need_radiance).items(): print(f"{k}={val}")
        return 0
    if a.cmd == "manifest":
        dg = {}
        for d in a.digest:
            s, _, h = d.partition("=")
            if s in dg: sys.exit(f"--digest {s} twice")
            dg[s] = h
        p = pins(a.pins)
        src = {"rad_exl3_mix": f"{p['RXM_REPO_URL']}@{p['RXM_COMMIT']}", "radiance": f"{p['RADIANCE_REPO_URL']}@{p['RADIANCE_COMMIT']}"}
        if p["RADIANCE_IMAGE"]: src["radiance_image"] = p["RADIANCE_IMAGE"]
        commit = a.commit or git("rev-parse", "HEAD")
        try: m = RL.make(v, commit, dg, src, a.created, changes(f"v{v}"), setup_min=a.setup_min)
        except ValueError as e: sys.exit(f"manifest: {e}")
        print(json.dumps(m, indent=1, sort_keys=True)); return 0
    if a.cmd == "notes":
        try: m = RL.loads(open(a.manifest, "rb").read())
        except ValueError as e: sys.exit(f"{a.manifest}: {e}")
        L = [f"rad-quant-at-home {m['version']} (commit {m['commit'][:12]}).", "",
             "Donors: the queue advertises this release once the operator installs it; `rqah status` in your worker says so. "
             "Update on the host with `rqah update` (pulls the tag below, checks its digest, lets the running jobs finish, "
             "recreates the container on the same state volume). Docker is the only supported way to run a donor worker.", "",
             "| image | digest |", "|---|---|"]
        L += [f"| `{im['repo']}:{im['tag']}` | `{im['digest']}` |" for _, im in sorted(m["images"].items())]
        L += ["", "Pull by digest: `docker pull REPO:TAG@DIGEST`. `release.json` (attached) is the machine-readable form: the "
              "operator installs it with `admin.py release-fetch " + m["version"] + "`.", ""]
        if m.get("sources"): L += ["Built from: " + "; ".join(f"{k} {v}" for k, v in sorted(m["sources"].items())), ""]
        if m.get("setup_min"): L += [f"Needs `client/setup.py` {m['setup_min']} or newer for `rqah update`.", ""]
        L += [m.get("changes") or ""]
        print("\n".join(L).rstrip() + "\n"); return 0


if __name__ == "__main__":
    sys.exit(main())
