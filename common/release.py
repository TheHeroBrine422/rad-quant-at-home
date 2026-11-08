# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The release manifest (release.json, stdlib only): what one tagged release published, machine-readable.

Written by tools/release.py in the release workflow (.github/workflows/release.yml) after the worker images are
pushed, attached to the GitHub Release as an asset, fetched onto the queue host with `admin.py release-fetch`, and read
by the server (--release-manifest, default release.json next to the database), which advertises the release and its
image digests to every worker and to `rqah update` (README "Updates and versions"). Releases are not a deploy channel:
nothing installs from them; they record what was built.

  {"format": 1, "name": "rad-quant-at-home", "version": "0.5.0", "tag": "v0.5.0", "commit": "<40 hex>",
   "created": "2026-11-08T14:00:00Z", "notes_url": "https://github.com/TheHeroBrine422/rad-quant-at-home/releases/tag/v0.5.0",
   "images": {"rocm7.14-gfx1201": {"repo": "ghcr.io/theherobrine422/rad-quant-at-home-worker",
                                   "tag": "0.5.0-rocm7.14-gfx1201", "digest": "sha256:<64 hex>"},
              "cu128": {...}},
   "sources": {"rad_exl3_mix": "<commit>", "radiance_image": "<ref@digest>", ...},
   "setup_min": "0.5.0"}                 (optional: the oldest client/setup.py that may install this release)

`images` is keyed by the stack, the part of the tag after the version (docker/build.sh): a worker image's own
RQAH_IMAGE_ID (REPO:VERSION-STACK+COMMIT) names its stack. A digest is what `docker push` reported for that tag (the
image's RepoDigest), so `docker pull REPO:TAG` followed by `docker image inspect` shows it unless the tag was moved."""
import datetime, json, re
from common.schema import validate, SchemaError

FORMAT = 1
NAME = "rad-quant-at-home"
REPO_URL = "https://github.com/TheHeroBrine422/rad-quant-at-home"
IMAGE_REPO = "ghcr.io/theherobrine422/rad-quant-at-home-worker"
STACKS = ("rocm7.14-gfx1201", "cu128")                   # docker/build.sh; keep in step with client/setup.py
VERSION_PAT = r"\d{1,4}\.\d{1,4}\.\d{1,4}"
DIGEST_PAT = r"sha256:[0-9a-f]{64}"
STACK_PAT = r"[a-z0-9][a-z0-9.]{0,31}(-[a-z0-9][a-z0-9.]{0,31}){0,3}"

SCHEMA = {"type": "object", "required": ["format", "name", "version", "tag", "commit", "created", "notes_url", "images"],
          "properties": {
              "format": {"const": FORMAT}, "name": {"const": NAME},
              "version": {"type": "string", "pattern": VERSION_PAT}, "tag": {"type": "string", "pattern": "v" + VERSION_PAT},
              "commit": {"type": "string", "pattern": "[0-9a-f]{40}"},
              "created": {"type": "string", "pattern": r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"},
              "notes_url": {"type": "string", "pattern": re.escape(REPO_URL) + r"/releases/tag/v" + VERSION_PAT},
              "images": {"type": "object", "additionalProperties": {
                  "type": "object", "required": ["repo", "tag", "digest"], "properties": {
                      "repo": {"type": "string", "pattern": r"[a-z0-9][a-z0-9./-]{0,127}"},
                      "tag": {"type": "string", "pattern": VERSION_PAT + "-" + STACK_PAT},
                      "digest": {"type": "string", "pattern": DIGEST_PAT}}}},
              "sources": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 300}},
              "setup_min": {"type": "string", "pattern": VERSION_PAT},
              "changes": {"type": "string", "maxLength": 20000}}}


def check(m, image_repo=IMAGE_REPO):
    """m if it is a well-formed manifest of our images, else ValueError: the schema, tag = "v" + version, every image
    in image_repo, keyed by its stack, tagged VERSION-STACK"""
    try: validate(m, SCHEMA, "release")
    except SchemaError as e: raise ValueError(str(e))
    if m["tag"] != "v" + m["version"]: raise ValueError(f"release: tag {m['tag']} is not v{m['version']}")
    if m["notes_url"] != notes_url(m["version"]): raise ValueError("release: notes_url is not this release's page")
    if not m["images"]: raise ValueError("release: no images")
    for stack, im in m["images"].items():
        if not re.fullmatch(STACK_PAT, stack): raise ValueError(f"release: bad stack {stack!r}")
        if im["repo"] != image_repo: raise ValueError(f"release: {stack}: repo {im['repo']} is not {image_repo}")
        if im["tag"] != f"{m['version']}-{stack}": raise ValueError(f"release: {stack}: tag {im['tag']} is not {m['version']}-{stack}")
    return m


def loads(b, image_repo=IMAGE_REPO):
    try: m = json.loads(b)
    except ValueError as e: raise ValueError(f"release: not JSON ({e})")
    return check(m, image_repo)


def notes_url(version): return f"{REPO_URL}/releases/tag/v{version}"


def asset_url(version=None):
    """the manifest asset of release `version` (None: the latest release)"""
    return f"{REPO_URL}/releases/latest/download/release.json" if not version else f"{REPO_URL}/releases/download/v{version}/release.json"


def make(version, commit, digests, sources=None, created=None, changes=None, image_repo=IMAGE_REPO, setup_min=None):
    """digests: {stack: "sha256:..."} -> a checked manifest"""
    m = {"format": FORMAT, "name": NAME, "version": version, "tag": "v" + version, "commit": commit,
         "created": created or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "notes_url": notes_url(version),
         "images": {s: {"repo": image_repo, "tag": f"{version}-{s}", "digest": d} for s, d in sorted(digests.items())},
         "sources": dict(sources or {})}
    if changes: m["changes"] = changes[:20000]
    if setup_min: m["setup_min"] = setup_min
    return check(m, image_repo)


def advertised(m):
    """what the server sends for a release: {stack: {image: REPO:TAG, digest}}, the notes page, when it was published"""
    a = {"images": {s: {"image": f"{im['repo']}:{im['tag']}", "digest": im["digest"]} for s, im in m["images"].items()},
         "notes": m["notes_url"], "published": m["created"]}
    if m.get("setup_min"): a["setup_min"] = m["setup_min"]
    return a
