# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The .rad container (radiance abi/rad_format.h, format version 2) as far as mixing needs it, and mix/radmix.cpp's
assembly ported to Python (stdlib only), working from per-weight pieces instead of whole donor files.

A ladder level is cut once on the server (`cut`) into
    skeleton.radh       the level's bytes [0, data_off): header, strings, metadata, the old tables, vocab, profile
    t/<weight>.radw     one piece per directory entry: its entry record, encoding, planes and data
so a donor downloads a weight once per (weight, level) and keeps it in its content-addressed cache; a candidate mix
is assembled locally (`assemble`) from the base level's skeleton plus one piece per entry: the mix's level for the
weights it names, the base level's for the rest. The output is byte for byte what `radmix OUT BASE.rad MIXFILE` writes
from the original files when radmix copies (tests/radmix_port.py; when radmix reflinks, the 4 KiB padding after each
weight comes from the donor file, zeros in rad-convert's output). Strings are interned, encodings deduplicated and
`act` keys set exactly as radmix does, in the same order.

Piece (.radw): magic "RADQW\\0\\1\\0", u64 n, n bytes of JSON {name, quantizer, options, n_planes, bytes}, the
RadFileEntry (136 B) with name / quantizer / options / enc / plane_first / offset zeroed, the RadEncoding (568 B),
n_planes RadFilePlane (offsets relative to the entry's first byte), zeros to 4096, the data, zeros to 4096. A piece
depends only on the weight, so identical weights in two levels are one blob. The data sits 4 KiB-aligned, so on a
reflink filesystem (XFS, btrfs) both cutting and assembly clone it instead of copying."""
import errno, fcntl, hashlib, json, os, re, struct

MAGIC, VERSION = 0x31444152, 2
ALIGN_POOL, ALIGN_UNIT, ALIGN_TAB = 2 << 20, 4096, 256
MAX_RANK = 6
HDR_FMT = "<IIQ5Q16Q8Q"                  # RadFileHeader, 248 B
HDR_FIELDS = ["magic", "version", "file_bytes", "arch_id", "model_name", "quant", "recipe", "created_by",
              "str_off", "str_bytes", "meta_off", "meta_count", "dir_off", "dir_count", "plane_off", "plane_count",
              "enc_off", "enc_count", "vocab_off", "vocab_bytes", "prof_off", "prof_count", "data_off", "data_bytes"] + \
             [f"reserved{i}" for i in range(8)]
ENT_FMT = "<QQQII6qiiiIQQQ2Q"           # RadFileEntry, 136 B
ENT_FIELDS = ["name", "quantizer", "options", "enc", "rank"] + [f"shape{i}" for i in range(MAX_RANK)] + \
             ["layer", "expert", "slot", "n_planes", "plane_first", "offset", "bytes", "res0", "res1"]
HDR, ENT, PLANE, ENC, KV = 248, 136, 16, 568, 24
RAD_P_STR = 2
ACT_PREFIX = b"exl3.act."
PIECE_MAGIC = b"RADQW\x00\x01\x00"
WEIGHT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,150}")          # names a piece file can carry
ACT_RE = re.compile(r"f16|a8t|a8g[1-9][0-9]{0,6}")
FICLONERANGE = 0x4020940D
assert struct.calcsize(HDR_FMT) == HDR and struct.calcsize(ENT_FMT) == ENT


class RadError(ValueError):
    pass


def al(x, a): return (x + a - 1) // a * a


def act_ok(mode):
    """exl3_act_parse: f16 | a8t | a8g<N>, N a multiple of 128 in [128, 2^20]"""
    if not isinstance(mode, str) or not ACT_RE.fullmatch(mode): return False
    if mode.startswith("a8g"):
        g = int(mode[3:]); return 128 <= g <= 1 << 20 and g % 128 == 0
    return True


def cstr(blob, off):
    """the NUL-terminated string at off in a string blob"""
    end = blob.find(b"\0", off)
    return bytes(blob[off:end if end >= 0 else len(blob)])


class Rad:
    """a .rad file, or a skeleton (its bytes before data_off): header, strings, directory, planes, encodings, metadata"""
    def __init__(self, path, skeleton=False):
        self.path = path
        with open(path, "rb") as f: self.pre = f.read(HDR)
        if len(self.pre) < HDR: raise RadError(f"{path}: short file")
        self.h = dict(zip(HDR_FIELDS, struct.unpack(HDR_FMT, self.pre)))
        h = self.h
        if h["magic"] != MAGIC or h["version"] != VERSION: raise RadError(f"{path}: not RAD2")
        size = os.path.getsize(path)
        if h["data_off"] % ALIGN_POOL or (not skeleton and h["file_bytes"] != size) or (skeleton and size != h["data_off"]):
            raise RadError(f"{path}: size {size} does not fit the header")
        for k, n in (("str", 1), ("meta", KV), ("dir", ENT), ("plane", PLANE), ("enc", ENC)):
            off, cnt = h[f"{k}_off"], h[f"{k}_bytes" if k == "str" else f"{k}_count"]
            if off + cnt * n > h["data_off"] and cnt: raise RadError(f"{path}: {k} table beyond the pre-data region")
        with open(path, "rb") as f: self.pre = f.read(h["data_off"])
        p = self.pre
        self.strs = p[h["str_off"]:h["str_off"] + h["str_bytes"]]
        self.entries = []
        for i in range(h["dir_count"]):
            raw = p[h["dir_off"] + i * ENT:h["dir_off"] + (i + 1) * ENT]
            e = dict(zip(ENT_FIELDS, struct.unpack(ENT_FMT, raw)))
            e["raw"] = raw; e["name_s"] = cstr(self.strs, e["name"]).decode()
            self.entries.append(e)
        self.planes = [struct.unpack("<QQ", p[h["plane_off"] + i * PLANE:h["plane_off"] + (i + 1) * PLANE])
                       for i in range(h["plane_count"])]
        self.encs = [p[h["enc_off"] + i * ENC:h["enc_off"] + (i + 1) * ENC] for i in range(h["enc_count"])]
        self.meta = [list(struct.unpack("<QIIQ", p[h["meta_off"] + i * KV:h["meta_off"] + (i + 1) * KV]))
                     for i in range(h["meta_count"])]
        self.idx = {e["name_s"]: i for i, e in enumerate(self.entries)}

    def shape(self, e): return (e["rank"], tuple(e[f"shape{i}"] for i in range(MAX_RANK)))


# ---------------------------------------------------------------- copying with reflinks where the filesystem allows
class Mover:
    """pieces of one file into another: FICLONERANGE for whole 4 KiB blocks at aligned offsets (falls back to a copy
    once the filesystem refuses), a plain copy for the rest; optionally hashes what lands in the output"""
    def __init__(self, dst_fd, clone=True):
        self.fd, self.clone, self.cloned, self.copied = dst_fd, clone, 0, 0

    def copy(self, src_fd, soff, doff, n, hasher=None):
        if n <= 0: return
        full = n // ALIGN_UNIT * ALIGN_UNIT
        if self.clone and full and soff % ALIGN_UNIT == 0 and doff % ALIGN_UNIT == 0:
            try:
                fcntl.ioctl(self.fd, FICLONERANGE, struct.pack("<qQQQ", src_fd, soff, full, doff))
                self.cloned += full
                if hasher: self._read(src_fd, soff, full, None, hasher)
                soff += full; doff += full; n -= full
            except OSError as e:
                if e.errno not in (errno.EOPNOTSUPP, errno.EXDEV, errno.EINVAL, errno.ENOTTY, errno.EPERM, errno.EBADF, errno.ENOSYS):
                    raise
                self.clone = False
        self._read(src_fd, soff, n, doff, hasher); self.copied += n

    def _read(self, src_fd, soff, n, doff, hasher):
        while n:
            b = os.pread(src_fd, min(n, 16 << 20), soff)
            if not b: raise RadError("short read")
            if doff is not None: self.write(doff, b); doff += len(b)
            if hasher: hasher.update(b)
            soff += len(b); n -= len(b)

    def write(self, off, b):
        mv = memoryview(b)
        while mv:
            w = os.pwrite(self.fd, mv, off); mv = mv[w:]; off += w


def _zeros(hasher, n):
    z = bytes(min(n, 1 << 20))
    while n:
        k = min(n, len(z)); hasher.update(z[:k]); n -= k


# ---------------------------------------------------------------- pieces
def piece_bytes_header(rad, e):
    """a piece's bytes before its data (padded to 4096)"""
    q, o = cstr(rad.strs, e["quantizer"]).decode(), cstr(rad.strs, e["options"]).decode()
    f = dict(e); f.update(name=0, quantizer=0, options=0, enc=0, plane_first=0, offset=0)
    ent = struct.pack(ENT_FMT, *[f[k] for k in ENT_FIELDS])
    pl = b"".join(struct.pack("<QQ", off - e["offset"], n) for off, n in rad.planes[e["plane_first"]:e["plane_first"] + e["n_planes"]])
    js = json.dumps({"name": e["name_s"], "quantizer": q, "options": o, "n_planes": e["n_planes"], "bytes": e["bytes"]},
                    sort_keys=True, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    head = PIECE_MAGIC + struct.pack("<Q", len(js)) + js + ent + rad.encs[e["enc"]] + pl
    return head + bytes(-len(head) % ALIGN_UNIT)


def cut(rad_path, out_dir, clone=True):
    """a .rad -> out_dir/skeleton.radh + out_dir/t/<weight>.radw (each written .part, then renamed; existing ones of
    the same size are kept). Returns {"skeleton": path, "pieces": {weight: path}}"""
    rad = Rad(rad_path)
    os.makedirs(os.path.join(out_dir, "t"), exist_ok=True)
    sk = os.path.join(out_dir, "skeleton.radh")
    if not (os.path.exists(sk) and os.path.getsize(sk) == len(rad.pre)):
        with open(sk + ".part", "wb") as f: f.write(rad.pre)
        os.replace(sk + ".part", sk)
    out = {"skeleton": sk, "pieces": {}}
    sfd = os.open(rad_path, os.O_RDONLY)
    try:
        for e in rad.entries:
            if not WEIGHT_RE.fullmatch(e["name_s"]): raise RadError(f"weight name not allowed in a piece file: {e['name_s']!r}")
            for off, n in rad.planes[e["plane_first"]:e["plane_first"] + e["n_planes"]]:
                if not (e["offset"] <= off and off + n <= e["offset"] + e["bytes"]): raise RadError(f"{e['name_s']}: plane outside its entry")
            head = piece_bytes_header(rad, e)
            p = os.path.join(out_dir, "t", e["name_s"] + ".radw"); out["pieces"][e["name_s"]] = p
            size = len(head) + al(e["bytes"], ALIGN_UNIT)
            if os.path.exists(p) and os.path.getsize(p) == size: continue
            fd = os.open(p + ".part", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, size)
                m = Mover(fd, clone); m.write(0, head); m.copy(sfd, e["offset"], len(head), e["bytes"]); clone = m.clone
            finally: os.close(fd)
            os.replace(p + ".part", p)
    finally: os.close(sfd)
    return out


def read_piece(path):
    """a piece's header: {name, quantizer, options, entry (dict), entry_raw, enc_raw, planes [(rel, n)], data_off, bytes}"""
    with open(path, "rb") as f:
        b = f.read(16)
        if len(b) < 16 or b[:8] != PIECE_MAGIC: raise RadError(f"{path}: not a piece")
        n = struct.unpack("<Q", b[8:16])[0]
        if n > 1 << 16: raise RadError(f"{path}: bad piece header")
        js = json.loads(f.read(n)); ent = f.read(ENT); enc = f.read(ENC)
        e = dict(zip(ENT_FIELDS, struct.unpack(ENT_FMT, ent)))
        if e["n_planes"] != js["n_planes"] or e["n_planes"] > 64 or e["bytes"] != js["bytes"]: raise RadError(f"{path}: inconsistent piece")
        pl = [struct.unpack("<QQ", f.read(PLANE)) for _ in range(e["n_planes"])]
    data_off = al(16 + n + ENT + ENC + PLANE * e["n_planes"], ALIGN_UNIT)
    if os.path.getsize(path) != data_off + al(e["bytes"], ALIGN_UNIT): raise RadError(f"{path}: size does not fit its header")
    for off, k in pl:
        if off + k > e["bytes"]: raise RadError(f"{path}: plane outside the entry")
    if not WEIGHT_RE.fullmatch(js["name"]): raise RadError(f"{path}: bad weight name")
    return {"name": js["name"], "quantizer": js["quantizer"].encode(), "options": js["options"].encode(), "entry": e,
            "entry_raw": ent, "enc_raw": enc, "planes": pl, "data_off": data_off, "bytes": e["bytes"], "path": path}


# ---------------------------------------------------------------- assembly (radmix.cpp, from pieces)
def plan(skel, pieces, act=None):
    """the mix's layout. skel: Rad of the base's skeleton; pieces: {weight: read_piece()} covering every directory
    entry; act: {weight: mode}. Returns a dict with the new header, the tables' bytes and each weight's offset."""
    act = act or {}
    missing = [e["name_s"] for e in skel.entries if e["name_s"] not in pieces]
    if missing: raise RadError(f"no piece for {len(missing)} weights, e.g. {missing[:3]}")
    extra = set(pieces) - set(skel.idx)
    if extra: raise RadError(f"base has no weight {sorted(extra)[0]}")
    for k, v in act.items():
        if k not in skel.idx: raise RadError(f"act: base has no weight {k}")
        if not act_ok(v): raise RadError(f"act {k} {v}: not an activation mode (f16, a8t, a8g<N>)")
    bh = skel.h
    strs = bytearray(skel.strs)

    def intern(s):
        if not s: return 0
        p = strs.find(s + b"\0")
        if p != -1 and (p == 0 or strs[p - 1] == 0): return p
        p = len(strs); strs.extend(s + b"\0"); return p
    encs = list(skel.encs)

    def enc_of(raw):
        for i, x in enumerate(encs):
            if x == raw: return i
        encs.append(raw); return len(encs) - 1
    dirs, planes, picks = [], [], []
    for be in skel.entries:
        pc = pieces[be["name_s"]]; e = pc["entry"]
        if (e["rank"], tuple(e[f"shape{i}"] for i in range(MAX_RANK))) != skel.shape(be):
            raise RadError(f"{be['name_s']}: shape differs in {os.path.basename(pc['path'])}")
        d = dict(e)
        d["name"] = be["name"]; d["quantizer"] = intern(pc["quantizer"]); d["options"] = intern(pc["options"])
        d["enc"] = enc_of(pc["enc_raw"]); d["plane_first"] = len(planes)
        planes += [list(x) for x in pc["planes"]]
        dirs.append(d); picks.append(pc)
    meta = [list(m) for m in skel.meta]
    for name in sorted(act):                                         # std::map order
        key = ACT_PREFIX + name.encode()
        v = [intern(key), RAD_P_STR, 0, intern(act[name].encode())]
        hit = False
        for i, m in enumerate(meta):
            if cstr(strs, m[0]) == key: meta[i] = list(v); hit = True
        if not hit: meta.append(v)
    h = dict(bh)
    pre = bh["data_off"]
    o = al(pre, ALIGN_TAB)
    h["str_off"], h["str_bytes"] = o, len(strs);                     o = al(o + len(strs), ALIGN_TAB)
    h["dir_off"], h["dir_count"] = o, len(dirs);                     o = al(o + len(dirs) * ENT, ALIGN_TAB)
    h["plane_off"], h["plane_count"] = o, len(planes);               o = al(o + len(planes) * PLANE, ALIGN_TAB)
    h["enc_off"], h["enc_count"] = o, len(encs);                     o = al(o + len(encs) * ENC, ALIGN_TAB)
    if act: h["meta_off"], h["meta_count"] = o, len(meta);           o = al(o + len(meta) * KV, ALIGN_TAB)
    h["data_off"] = al(o, ALIGN_POOL)
    cur = h["data_off"]
    for d, pc in zip(dirs, picks):
        cur = al(cur, ALIGN_UNIT)
        for p in range(d["n_planes"]): planes[d["plane_first"] + p][0] += cur          # relative -> absolute
        d["offset"] = cur; cur += d["bytes"]
    h["data_bytes"] = al(cur, ALIGN_UNIT) - h["data_off"]
    h["file_bytes"] = h["data_off"] + h["data_bytes"]
    tables = [(h["str_off"], bytes(strs)),
              (h["dir_off"], b"".join(struct.pack(ENT_FMT, *[d[k] for k in ENT_FIELDS]) for d in dirs)),
              (h["plane_off"], b"".join(struct.pack("<QQ", *p) for p in planes)),
              (h["enc_off"], b"".join(encs))]
    if act: tables.append((h["meta_off"], b"".join(struct.pack("<QIIQ", *m) for m in meta)))
    return {"header": struct.pack(HDR_FMT, *[h[k] for k in HDR_FIELDS]), "h": h, "pre": pre, "tables": tables,
            "dirs": dirs, "picks": picks, "file_bytes": h["file_bytes"]}


def assemble(out, skeleton_path, piece_paths, act=None, clone=True):
    """write the mix to out (out.part, then renamed): skeleton + one piece per weight ({weight: path}); returns
    {"bytes", "sha256", "cloned", "copied"}. The sha256 is of the output's bytes, computed in the same pass."""
    skel = Rad(skeleton_path, skeleton=True)
    pieces = {k: read_piece(p) for k, p in piece_paths.items()}
    for k, pc in pieces.items():
        if pc["name"] != k: raise RadError(f"piece for {k} holds {pc['name']}")
    L = plan(skel, pieces, act)
    hs = hashlib.sha256()
    fd = os.open(out + ".part", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, L["file_bytes"])
        m = Mover(fd, clone)
        head = L["header"] + skel.pre[HDR:L["pre"]]
        m.write(0, head); hs.update(head); pos = len(head)
        for off, b in L["tables"]:
            _zeros(hs, off - pos); m.write(off, b); hs.update(b); pos = off + len(b)
        for d, pc in zip(L["dirs"], L["picks"]):
            _zeros(hs, d["offset"] - pos)
            sfd = os.open(pc["path"], os.O_RDONLY)
            try: m.copy(sfd, pc["data_off"], d["offset"], d["bytes"], hs)
            finally: os.close(sfd)
            pos = d["offset"] + d["bytes"]
        _zeros(hs, L["file_bytes"] - pos)
    finally: os.close(fd)
    os.replace(out + ".part", out)
    return {"bytes": L["file_bytes"], "sha256": hs.hexdigest(), "cloned": m.cloned, "copied": m.copied}


def mix_size(skeleton_path, piece_paths, act=None, cache=None):
    """the assembled size without writing anything (the server's `bytes` parameter); cache: {path: read_piece()}"""
    cache = {} if cache is None else cache
    skel = cache.get(skeleton_path) or Rad(skeleton_path, skeleton=True); cache[skeleton_path] = skel
    pcs = {}
    for k, p in piece_paths.items():
        if p not in cache: cache[p] = read_piece(p)
        pcs[k] = cache[p]
    return plan(skel, pcs, act)["file_bytes"]
