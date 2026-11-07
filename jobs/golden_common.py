# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Goldens and corpora as donors see them (stdlib only): the privacy marker, radiance's --kld-record layout and its
consistency checks, and the comparison of two goldens or two per-position score files.

Privacy (AGENTS.md data rules): a golden (goldens/<name>/) or a corpus (corpora/<name>/) goes to donors only if its
marker (golden.json / corpus.json, written by admin.py golden-add / corpus-add) says "privacy": "public" and none of
its documents' sources is one of ours (PRIVATE_SOURCES: the user's agent traces, our session logs). No marker means
private. The server checks this when it builds its manifest (a private golden's files are never listed, so never
served), when a job is enqueued, and again when a file is served; the worker checks the marker before it runs."""
import array, json, math, os, random, struct, sys

PRIVATE_SOURCES = ("agent-trace", "pi-session", "trace/", "traces/", "private/", "user/")
GOLDEN_FILES = ("kld.json", "tokens.i32", "top.bin", "rows.f32")


def fnv1a64_lines(path):
    """radiance's corpus_fnv1a64: FNV-1a over every line's bytes (no newline), chained, from radiance's offset basis
    (1469598103934665603: the standard constant with its last digit missing, as core/kld.cpp has it)"""
    h, M = 1469598103934665603, (1 << 64) - 1
    with open(path, "rb") as f:
        for line in f:
            if line.endswith(b"\n"): line = line[:-1]
            for c in line: h = ((h ^ c) * 1099511628211) & M
    return f"{h:016x}"


def private_sources(sources):
    return sorted({s for s in sources if any(str(s).startswith(p) or f"/{p}" in str(s) for p in PRIVATE_SOURCES)})


def marker_ok(marker):
    """(ok, why) for a golden.json / corpus.json dict (None: no marker = private)"""
    if not isinstance(marker, dict): return False, "no privacy marker (private by default)"
    if marker.get("privacy") != "public": return False, f"marked {marker.get('privacy')!r}"
    bad = private_sources(marker.get("sources", []))
    if bad: return False, f"private sources {bad[:3]}"
    return True, ""


def f32(path, w=None):
    if sys.byteorder != "little": raise RuntimeError("little-endian hosts only")
    a = array.array("f")
    with open(path, "rb") as f: a.frombytes(f.read())
    return a


def doc_rows(meta):
    """(rows, tokens) the documents of a kld.json add up to; raises on a gap"""
    r = t = 0
    for d in meta["docs"]:
        n = d["tokens"] - 1 - d["score_from"]
        if d["row0"] != r or n < 1 or d["score_from"] < 0: raise ValueError(f"kld.json: document rows not contiguous at row {r}")
        r += n; t += d["tokens"]
    if r != meta["rows"]: raise ValueError(f"kld.json: documents add up to {r} rows, not {meta['rows']}")
    return r, t


def check_golden(files, top=None, max_positions=None, corpus=None, sample=256, seed=None):
    """a recorded golden ({file: path}: kld.json tokens.i32 top.bin rows.f32 [gen.u8]): sizes, and on sampled rows the
    head sorted, finite and unique, head + tail = 1, and the true next token's log p = -nll when it is in the head.
    corpus (path): document count, sources and radiance's corpus hash must match it. Returns a summary."""
    meta = json.load(open(files["kld.json"]))
    if meta.get("format") != "radiance-kld-top-1": raise ValueError(f"kld.json: format {meta.get('format')!r}")
    T = int(meta["top"]); rows, ntok = doc_rows(meta); V = int(meta["n_vocab"])
    if top is not None and T != top: raise ValueError(f"top {T} != {top}")
    if max_positions is not None and rows > max_positions: raise ValueError(f"{rows} rows > max_positions {max_positions}")
    exp = {"tokens.i32": 4 * ntok, "top.bin": 8 * T * rows, "rows.f32": 12 * rows, "gen.u8": rows}
    for k, n in exp.items():
        if k in files and os.path.getsize(files[k]) != n: raise ValueError(f"{k}: {os.path.getsize(files[k])} bytes, expected {n}")
    if corpus:
        lines = [json.loads(l) for l in open(corpus, encoding="utf-8") if l.strip()]
        if len(lines) != len(meta["docs"]): raise ValueError(f"{len(meta['docs'])} documents, corpus has {len(lines)}")
        if [d.get("source", "") for d in lines] != [d.get("source", "") for d in meta["docs"]]: raise ValueError("document sources differ from the corpus")
        if [int(d.get("score_from", 0)) for d in lines] != [d["score_from"] for d in meta["docs"]]: raise ValueError("score_from differs from the corpus")
        if meta.get("corpus_fnv1a64") and meta["corpus_fnv1a64"] != fnv1a64_lines(corpus): raise ValueError("corpus hash differs")
    tok = array.array("i"); tok.frombytes(open(files["tokens.i32"], "rb").read())
    rng = random.Random(seed)
    starts, t0 = [], 0
    for d in meta["docs"]: starts.append(t0); t0 += d["tokens"]
    picks = sorted(set([0, rows - 1] + [rng.randrange(rows) for _ in range(min(sample, rows))]))
    worst_sum = worst_nll = 0.0
    with open(files["top.bin"], "rb") as ft, open(files["rows.f32"], "rb") as fr:
        di = 0
        for r in picks:
            while not (meta["docs"][di]["row0"] <= r < meta["docs"][di]["row0"] + meta["docs"][di]["tokens"] - 1 - meta["docs"][di]["score_from"]):
                di += 1
            d = meta["docs"][di]; pos = d["score_from"] + (r - d["row0"])
            nxt = tok[starts[di] + pos + 1]
            ft.seek(8 * T * r); head = ft.read(8 * T)
            fr.seek(12 * r); nll, ent, tail = struct.unpack("<fff", fr.read(12))
            ids = [struct.unpack_from("<i", head, 8 * i)[0] for i in range(T)]
            lps = [struct.unpack_from("<f", head, 8 * i + 4)[0] for i in range(T)]
            if not all(math.isfinite(x) and x <= 1e-6 for x in lps) or any(lps[i] < lps[i + 1] for i in range(T - 1)):
                raise ValueError(f"row {r}: head not sorted / not finite log p")
            if len(set(ids)) != T or not all(0 <= i < V for i in ids): raise ValueError(f"row {r}: bad token ids")
            s = sum(math.exp(x) for x in lps) + tail
            worst_sum = max(worst_sum, abs(s - 1))
            if abs(s - 1) > 1e-4: raise ValueError(f"row {r}: head + tail = {s}")
            if nxt in ids:
                e = abs(nll + lps[ids.index(nxt)]); worst_nll = max(worst_nll, e)
                if e > 1e-4: raise ValueError(f"row {r}: nll {nll} but the true token's log p is {lps[ids.index(nxt)]}")
            elif nll < -lps[-1] - 1e-4: raise ValueError(f"row {r}: nll {nll} below the head's smallest -log p")
    return {"rows": rows, "docs": len(meta["docs"]), "top": T, "sampled": len(picks), "worst_sum": worst_sum, "worst_nll": worst_nll}


def compare_goldens(a, b, tol, sample=512):
    """two recordings of one corpus chunk: identical tokens, per-row nll and the head's argmax close"""
    ta, tb = (open(os.path.join(d, "tokens.i32"), "rb").read() for d in (a, b))
    if ta != tb: return False, {"error": "tokens differ"}
    ra, rb = f32(os.path.join(a, "rows.f32")), f32(os.path.join(b, "rows.f32"))
    if len(ra) != len(rb): return False, {"error": "row counts differ"}
    n = len(ra) // 3
    dn = sum(abs(ra[3 * i] - rb[3 * i]) for i in range(n)) / n
    T = json.load(open(os.path.join(a, "kld.json")))["top"]
    rng = random.Random(0); same = 0; picks = [rng.randrange(n) for _ in range(min(sample, n))]
    with open(os.path.join(a, "top.bin"), "rb") as fa, open(os.path.join(b, "top.bin"), "rb") as fb:
        for r in picks:
            fa.seek(8 * T * r); fb.seek(8 * T * r)
            same += struct.unpack("<i", fa.read(4))[0] == struct.unpack("<i", fb.read(4))[0]
    agree = same / len(picks)
    ok = dn <= tol["nll_mae"] and agree >= tol["top1_agree"]
    return ok, {"nll_mae": dn, "top1_agree": agree, "tol": tol}


def doc_spans(meta, docs):
    """[[document index, row0, rows]] of the given documents of a kld.json, sorted"""
    out = []
    for i in sorted(set(docs)):
        d = meta["docs"][i]
        out.append([i, d["row0"], d["tokens"] - 1 - d["score_from"]])
    return out


def spans_ok(meta, spans):
    """the spans name documents of this kld.json exactly (index, first row, row count), sorted and unique"""
    idx = [s[0] for s in spans]
    if idx != sorted(set(idx)) or not spans: return False
    return all(0 <= i < len(meta["docs"]) and [i, r0, n] == doc_spans(meta, [i])[0] for i, r0, n in spans)


def sub_golden(src, dst, spans):
    """a golden of just these documents ({file: path} -> dir dst), as radiance reads one: kld.json with the documents
    renumbered (row0 contiguous, rows = their sum), tokens.i32 / top.bin / rows.f32 cut to them. Scoring a document
    does not depend on the others, so its rows equal its rows in the whole golden (a spot check's few documents)."""
    meta = json.load(open(src["kld.json"]))
    if not spans_ok(meta, spans): raise ValueError("documents do not match the golden")
    T = int(meta.get("top", 0)); w = 3 if "top" in meta else 2
    tok0, t = [], 0
    for d in meta["docs"]: tok0.append(t); t += d["tokens"]
    os.makedirs(dst, exist_ok=True)
    docs, r = [], 0
    with open(src["tokens.i32"], "rb") as ft, open(src["top.bin"], "rb") as fp, open(src["rows.f32"], "rb") as fr, \
            open(f"{dst}/tokens.i32", "wb") as gt, open(f"{dst}/top.bin", "wb") as gp, open(f"{dst}/rows.f32", "wb") as gr:
        for i, r0, n in spans:
            d = meta["docs"][i]
            ft.seek(4 * tok0[i]); gt.write(ft.read(4 * d["tokens"]))
            fp.seek(8 * T * r0); gp.write(fp.read(8 * T * n))
            fr.seek(4 * w * r0); gr.write(fr.read(4 * w * n))
            docs.append(dict(d, row0=r)); r += n
    sub = {k: v for k, v in meta.items() if k not in ("docs", "rows", "corpus_fnv1a64")}
    sub.update(docs=docs, rows=r, subset_of=[s[0] for s in spans])
    with open(f"{dst}/kld.json", "w") as f: json.dump(sub, f)
    return r


def take_rows(A, spans, width=4):
    """the rows of these spans from a flat [rows, width] array, in span order"""
    out = array.array(A.typecode)
    for _, r0, n in spans: out.extend(A[width * r0:width * (r0 + n)])
    return out


def compare_rows(a, b, tol, spans=None):
    """two [rows, 4] {kl, nll_q, nll_p, top1} score files of one candidate on one golden (a donor's, a trusted
    re-run's): nll_p identical (both copied from the golden), and per position |dKL| <= atol + rtol * max(KL) on at
    least rows_frac of the rows, the mean KL within mean_rel, top-1 flips at most top1_flip. spans: b scored only these
    documents (a spot check): a's rows of them are compared, and also each document's mean KL (doc_mean_rel)"""
    A, B = f32(a), f32(b)
    if spans is not None:
        if len(A) % 4 or any(4 * (r0 + n) > len(A) for _, r0, n in spans): return False, {"error": "documents outside the donor's rows"}
        A = take_rows(A, spans)
    if len(A) != len(B) or len(A) % 4: return False, {"error": f"row counts differ ({len(A) // 4} vs {len(B) // 4})"}
    n = len(A) // 4
    if A[2::4] != B[2::4]: return False, {"error": "reference NLL column differs (another golden?)"}
    ka, kb = A[0::4], B[0::4]
    bad = sum(1 for x, y in zip(ka, kb) if not abs(x - y) <= tol["row_atol"] + tol["row_rtol"] * max(abs(x), abs(y)))
    ma, mb = sum(ka) / n, sum(kb) / n
    flips = sum(1 for x, y in zip(A[3::4], B[3::4]) if x != y) / n
    rel = abs(ma / mb - 1) if mb else float("inf")
    ok = bad <= (1 - tol["rows_frac"]) * n and rel <= tol["mean_rel"] and flips <= tol["top1_flip"]
    det = {"rows": n, "rows_out_of_tol": bad, "mean_donor": ma, "mean_trusted": mb, "mean_rel": rel, "top1_flips": flips, "tol": tol}
    if spans is not None:                                   # per document: the few documents a spot check re-ran
        worst, o = 0.0, 0
        for i, _, m in spans:
            da, db_ = math.fsum(ka[o:o + m]) / m, math.fsum(kb[o:o + m]) / m; o += m
            worst = max(worst, abs(da / db_ - 1) if db_ else (0.0 if da == 0 else math.inf))
        det.update(docs=[x[0] for x in spans], doc_mean_rel=worst)
        ok = ok and worst <= tol.get("doc_mean_rel", 5e-3)
    return ok, det
