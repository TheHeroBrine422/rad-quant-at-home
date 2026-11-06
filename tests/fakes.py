# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""CPU-only stand-ins for the GPU side of the tests (stdlib only):

  write_rad(path, weights, ...)   a small, valid RAD2 container (radiance abi/rad_format.h) with the features radmix
                                  touches: strings, metadata (an exl3.act key included), several encodings, multi-plane
                                  entries, a vocab blob, a profile, the 2 MiB / 4 KiB alignment
  ladder(dir, n_levels, ...)      a few levels of one fake model (same names / shapes, different codes)
  record_golden(corpus, out, top) a golden in radiance's --kld-record --kld-top layout from a corpus JSONL: tokens are
                                  the prompt's bytes, every row self-consistent (sorted head, head + tail = 1, the true
                                  token's log p = -nll when it is in the head)
  score(model, golden, out)       --kld-ref: per-row {kl, nll_q, nll_p, top1} as a function of the container's sha256
                                  (nll_p copied from the golden, as radiance does), plus the JSON report
fake_radiance.py and fake_mkgold27.py wrap these as command lines with radiance's / mkgold27's argv."""
import hashlib, json, math, os, random, struct, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import radfile as R
from jobs.golden_common import fnv1a64_lines


def _enc(scheme, salt):
    b = scheme.encode().ljust(32, b"\0") + hashlib.sha256(salt.encode()).digest() * 17
    return b[:R.ENC]


def write_rad(path, weights, extra_strings=(), meta=(), vocab=b"", seed=0):
    """weights: [(name, quantizer, options, enc_key, [plane bytes...], shape)]; meta: [(key, "s"|"i", value)]"""
    rng = random.Random(seed)
    strs = bytearray(b"\0")

    def put(s):
        if not s: return 0
        p = len(strs); strs.extend(s.encode() + b"\0"); return p
    hdr_strs = [put(s) for s in ("qwen35_exl3", "fake-model", "exl3", "recipe: fake", "radiance fake writer")]
    for s in extra_strings: put(s)
    encs, enc_ix = [], {}
    ents, planes, datas = [], [], []
    for name, q, o, ek, pls, shape in weights:
        if ek not in enc_ix: enc_ix[ek] = len(encs); encs.append(_enc("exl3" if q else "plain", ek))
        e = {"name": put(name), "quantizer": put(q), "options": put(o), "enc": enc_ix[ek], "rank": len(shape),
             "layer": int(hashlib.sha256(name.encode()).hexdigest(), 16) % 4, "expert": -1, "slot": 0, "n_planes": len(pls), "plane_first": len(planes),
             "offset": 0, "bytes": 0, "res0": 0, "res1": 0}
        for i in range(R.MAX_RANK): e[f"shape{i}"] = shape[i] if i < len(shape) else 0
        rel, blob = 0, bytearray()
        for n in pls:
            rel = R.al(rel, R.ALIGN_TAB); blob.extend(bytes(rel - len(blob)))
            planes.append([rel, n]); blob.extend(rng.randbytes(n)); rel += n
        e["bytes"] = len(blob); ents.append(e); datas.append(bytes(blob))
    kvs = []
    for k, t, v in meta:
        kvs.append((put(k), R.RAD_P_STR if t == "s" else 1, 0, put(v) if t == "s" else int(v)))
    o = R.al(R.HDR, R.ALIGN_TAB)
    h = {k: 0 for k in R.HDR_FIELDS}
    h.update(magic=R.MAGIC, version=R.VERSION, arch_id=hdr_strs[0], model_name=hdr_strs[1], quant=hdr_strs[2],
             recipe=hdr_strs[3], created_by=hdr_strs[4])
    h["str_off"], h["str_bytes"] = o, len(strs);          o = R.al(o + len(strs), R.ALIGN_TAB)
    h["meta_off"], h["meta_count"] = o, len(kvs);         o = R.al(o + len(kvs) * R.KV, R.ALIGN_TAB)
    h["dir_off"], h["dir_count"] = o, len(ents);          o = R.al(o + len(ents) * R.ENT, R.ALIGN_TAB)
    h["plane_off"], h["plane_count"] = o, len(planes);    o = R.al(o + len(planes) * R.PLANE, R.ALIGN_TAB)
    h["enc_off"], h["enc_count"] = o, len(encs);          o = R.al(o + len(encs) * R.ENC, R.ALIGN_TAB)
    h["vocab_off"], h["vocab_bytes"] = o, len(vocab);     o = R.al(o + len(vocab), R.ALIGN_TAB)
    h["prof_off"], h["prof_count"] = o, 2;                o = R.al(o + 2 * 16, R.ALIGN_TAB)
    h["data_off"] = R.al(o, R.ALIGN_POOL)
    cur = h["data_off"]
    for e in ents:
        cur = R.al(cur, R.ALIGN_UNIT); e["offset"] = cur; cur += e["bytes"]
    h["data_bytes"] = R.al(cur, R.ALIGN_UNIT) - h["data_off"]; h["file_bytes"] = h["data_off"] + h["data_bytes"]
    buf = bytearray(h["file_bytes"])
    buf[0:R.HDR] = struct.pack(R.HDR_FMT, *[h[k] for k in R.HDR_FIELDS])
    buf[h["str_off"]:h["str_off"] + len(strs)] = strs
    for i, kv in enumerate(kvs): buf[h["meta_off"] + i * R.KV:h["meta_off"] + (i + 1) * R.KV] = struct.pack("<QIIQ", *kv)
    for i, e in enumerate(ents):
        for p in range(e["n_planes"]): planes[e["plane_first"] + p][0] += e["offset"]
        buf[h["dir_off"] + i * R.ENT:h["dir_off"] + (i + 1) * R.ENT] = struct.pack(R.ENT_FMT, *[e[k] for k in R.ENT_FIELDS])
        buf[e["offset"]:e["offset"] + e["bytes"]] = datas[i]
    for i, p in enumerate(planes): buf[h["plane_off"] + i * R.PLANE:h["plane_off"] + (i + 1) * R.PLANE] = struct.pack("<QQ", *p)
    for i, x in enumerate(encs): buf[h["enc_off"] + i * R.ENC:h["enc_off"] + (i + 1) * R.ENC] = x
    buf[h["vocab_off"]:h["vocab_off"] + len(vocab)] = vocab
    buf[h["prof_off"]:h["prof_off"] + 32] = struct.pack("<iiff", 0, 0, 0.5, 0) + struct.pack("<iiff", 1, 0, 0.5, 0)
    with open(path + ".part", "wb") as f: f.write(buf)
    os.replace(path + ".part", path)
    return h["file_bytes"]


LAYERS = 3
LINEARS = ["attn_q", "attn_k", "ffn_up", "ffn_down"]


def weight_names():
    return ["token_embd.weight"] + [f"blk.{l}.{x}.weight" for l in range(LAYERS) for x in LINEARS + ["attn_norm"]] + \
           ["output_norm.weight", "output.weight"]


def level_weights(K, seed):
    """one ladder level: linears at K "bits" (planes sized by K), the rest bf16 and identical in every level"""
    out = []
    for n in weight_names():
        lin = any(f".{x}." in n for x in LINEARS) or n == "output.weight"
        if lin:
            words = int(K * 64)
            # "exl3q" sits in every level's strings only inside "xexl3q": radmix interns it anew (not mid-string)
            out.append((n, "exl3" if K == 2 else "exl3q", f"bits={K},cb=i8,seed={seed}", f"exl3-K{K}",
                        [words * 16, 256 + words, 64], (64, 64)))
        else:
            out.append((n, "", "", "bf16", [512 if "norm" in n else 4096], (64,) if "norm" in n else (32, 64)))
    return out


def ladder(d, Ks=(2, 3, 4), seed=0):
    """d/<level>.rad for each K (level names k2, k3, ...); the bf16 weights are byte-identical across levels"""
    os.makedirs(d, exist_ok=True)
    paths = {}
    for i, K in enumerate(Ks):
        lv = f"k{K}".replace(".", "p")
        ws = level_weights(K, seed + i)
        # the same bf16 bytes in every level: those weights' data come from a seed of their own
        p = os.path.join(d, f"{lv}.rad")
        write_rad(p, ws, extra_strings=["xexl3q", "bits=9"], meta=[("exl3.act.blk.1.ffn_down.weight", "s", "f16"),
                  ("general.quant_bits", "i", K * 1000)], vocab=b"vocab-blob" * 100, seed=seed + i)
        _same_bf16(p, seed)
        paths[lv] = p
    return paths


def _same_bf16(path, seed):
    """overwrite the non-linear weights' data with bytes that depend on the name only (identical across levels)"""
    rad = R.Rad(path)
    with open(path, "r+b") as f:
        for e in rad.entries:
            if not rad.strs[e["quantizer"]:e["quantizer"] + 1].strip(b"\0"):
                rng = random.Random(f"{seed}:{e['name_s']}")
                f.seek(e["offset"]); f.write(rng.randbytes(e["bytes"]))


# ---------------------------------------------------------------- goldens
V = 1024         # vocabulary of the fake model (tokens are the prompt's bytes)


def _u(*k):
    return int.from_bytes(hashlib.sha256(":".join(map(str, k)).encode()).digest()[:8], "little") / 2 ** 64


def read_corpus(path):
    docs = []
    for line in open(path, encoding="utf-8"):
        if not line.strip(): continue
        d = json.loads(line)
        docs.append({"prompt": d["prompt"], "source": d.get("source", ""), "score_from": int(d.get("score_from", 0))})
    return docs


def record_golden(corpus, out, top=16, model="fake-bf16", salt=""):
    """radiance --kld-record --kld-top TOP layout: kld.json, tokens.i32, gen.u8, top.bin, rows.f32 (3 per row)"""
    docs = read_corpus(corpus)
    os.makedirs(out, exist_ok=True)
    toks, meta_docs, rows, tops = [], [], [], []
    row0 = 0
    for di, d in enumerate(docs):
        t = list(d["prompt"].encode())
        n = len(t) - 1 - d["score_from"]
        if n < 1: raise ValueError(f"doc {di}: nothing to score")
        for i in range(d["score_from"], len(t) - 1):
            nxt = t[i + 1]
            rng = random.Random(f"{salt}:{di}:{i}")
            ids = rng.sample(range(V), top)
            if rng.random() < 0.8 and nxt not in ids: ids[rng.randrange(top)] = nxt
            lps = sorted((-(k + 1) * 0.7 - rng.random() * 0.3 for k in range(top)), reverse=True)
            head = math.fsum(math.exp(x) for x in lps)
            tail = 1.0 - head
            pairs = list(zip(ids, lps))
            lp_true = dict(pairs).get(nxt)
            nll = -lp_true if lp_true is not None else -lps[-1] + 0.5 + rng.random()
            ent = -math.fsum(math.exp(x) * x for x in lps)
            tops.append(b"".join(struct.pack("<if", a, b) for a, b in pairs)); rows.append(struct.pack("<fff", nll, ent, tail))
        meta_docs.append({"source": d["source"], "tokens": len(t), "score_from": d["score_from"], "row0": row0})
        row0 += n; toks += t
    open(f"{out}/tokens.i32", "wb").write(struct.pack(f"<{len(toks)}i", *toks))
    open(f"{out}/gen.u8", "wb").write(bytes([1]) * row0)
    open(f"{out}/top.bin", "wb").write(b"".join(tops))
    open(f"{out}/rows.f32", "wb").write(b"".join(rows))
    json.dump({"format": "radiance-kld-top-1", "model": model, "corpus": os.path.basename(corpus), "n_vocab": V,
               "corpus_fnv1a64": fnv1a64_lines(corpus),
               "top": top, "rows": row0, "docs": meta_docs, "created": "fake"}, open(f"{out}/kld.json.part", "w"), indent=1)
    os.replace(f"{out}/kld.json.part", f"{out}/kld.json")
    return row0


def score(model, golden, out, bias=0.0, noise=0.0):
    """--kld-ref golden --kld-out out: out (report JSON) + out.rows ([rows, 4] f32 {kl, nll_q, nll_p, top1})"""
    with open(model, "rb") as f:
        if struct.unpack("<II", f.read(8)) != (R.MAGIC, R.VERSION): raise SystemExit(f"{model}: not RAD2")
        f.seek(0); h = hashlib.sha256()
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    seed = h.hexdigest()
    g = json.load(open(f"{golden}/kld.json")); n = g["rows"]
    gr = open(f"{golden}/rows.f32", "rb").read()
    w = 3 if "top" in g else 2
    rng = random.Random()
    out_rows, kls, t1 = [], [], 0
    for r in range(n):
        nll_p = struct.unpack_from("<f", gr, r * w * 4)[0]
        kl = 0.002 + 0.05 * _u(seed, r) ** 3 + bias
        if noise: kl *= 1 + noise * (2 * rng.random() - 1)
        kl = struct.unpack("<f", struct.pack("<f", kl))[0]
        top1 = 1.0 if _u(seed, "t", r) < 0.93 else 0.0
        out_rows.append(struct.pack("<ffff", kl, nll_p + kl, nll_p, top1)); kls.append(kl); t1 += top1
    open(out + ".rows", "wb").write(b"".join(out_rows))
    mean = math.fsum(kls) / n
    rep = {"model": model, "reference": g["model"], "ref_dir": golden, "ref_form": f"top-{g.get('top')} + tail",
           "docs": len(g["docs"]), "all": {"positions": n, "kld": {"mean": mean}, "top1_agreement": t1 / n},
           "devices": {k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if os.environ.get(k) is not None}}
    json.dump(rep, open(out, "w"), indent=1)
    return rep
