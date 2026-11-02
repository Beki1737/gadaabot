"""GadaaBot knowledge base, step 2: clean, chunk (parent-child), embed and index into Qdrant.

Design (D39 retriever, D40 license tiers)
-----------------------------------------
* Documents: every Oromo-check-passing document in data/kb/raw/*.jsonl.gz, minus
  administrative or statistical PDFs (education statistics abstracts, timetables, covers,
  tables of contents), minus exact duplicates across sources (same normalized text).
* Parents: consecutive paragraphs packed up to --parent-words (default 350 words): the
  unit handed to the generator, so it sees whole explanations, not fragments.
* Children: sentence windows of about --child-words (default 80) inside a parent, with a
  one-sentence overlap: the unit that is searched. 80 Oromo words is ~230 AfriE5 tokens,
  well inside its 512-token limit.
* Search keys per child: AfriE5 dense vector (title prefixed), BM25 over 4-letter prefixes
  (best sparse arm on Belebele) and BM25 over Qubee-normalized words (ablation arm). IDF
  is computed by Qdrant from live collection statistics.
* Payload: child text and provenance (doc, parent, source, tier, license, domain, title,
  url); keyword indexes on domain, tier and source enable the authority filter.
* Parents live in SQLite (data/kb/index/parents.sqlite), looked up by parent_id.
* Smoke test: a few Oromo questions through hybrid search (DBSF), cross-encoder rerank
  and parent expansion; the top parents are printed for a native-speaker check.

Usage: python -m gadaabot.kb.kb_index --raw data/kb/raw --out data/kb/index --qdrant-url URL
       --afrie5 DIR [--reranker DIR]
"""

from __future__ import annotations

import argparse
import gzip
import html
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gadaabot.data.qubee import words
from gadaabot.eval.retrieval_bench import E5_TASK, embed
from gadaabot.retrieval.oromo_analyzer import BM25Encoder

COLL = "gadaabot_kb_v1"
SOURCE_PRIORITY = {"wikipedia": 0, "tico19": 0, "ohkb": 0, "mdh": 0, "oeb": 1, "textbooks": 1, "webdomain": 2}
_PUA = re.compile("[\ue000-\uf8ff]")
DROP_TITLE = re.compile(r"statistic|abstract|gabatee wayitii|radio_schedule|schedule|cover[ _]page|^contents|"
                        r"qajeelfama[ _-]bulchiinsa|karoora_barnoota", re.I)
_SENT = re.compile(r"(?<=[.?!])\s+")
SMOKE = [
    "Mallattoon dhukkuba busaa maal fa'a?",
    "Akkamitti dhukkuba busaa ittisuu dandeenya?",
    "Daa'imni garaa kaasaa yoo qabaate maal gochuu qabna?",
    "Talaalliin daa'immaniif maaliif barbaachisaa dha?",
    "Dhukkubni sombaa (tiibii) akkamitti daddarbu?",
    "Foto-sinteesisii jechuun maal jechuu dha?",
    "Sirni Gadaa maal?",
    "Bishaan qulqulluu dhuguun faayidaa maal qaba?",
]


@dataclass
class Child:
    cid: int
    parent_id: str
    text: str
    embed_text: str
    payload: dict[str, object]


def paragraphs(text: str) -> list[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n", text)]
    return [p for p in paras if words(p)]


def pack(paras: list[str], max_words: int) -> list[str]:
    """Greedy paragraph packing; an over-long paragraph is cut at sentence boundaries."""
    units: list[str] = []
    for p in paras:
        if len(p.split()) <= max_words:
            units.append(p)
            continue
        cur: list[str] = []
        for s in _SENT.split(p):
            if cur and len(" ".join(cur + [s]).split()) > max_words:
                units.append(" ".join(cur))
                cur = []
            cur.append(s)
        if cur:
            units.append(" ".join(cur))
    out, cur_w, cur = [], 0, []
    for u in units:
        n = len(u.split())
        if cur and cur_w + n > max_words:
            out.append("\n".join(cur))
            cur, cur_w = [], 0
        cur.append(u)
        cur_w += n
    if cur:
        out.append("\n".join(cur))
    return out


def windows(parent: str, target: int, max_words: int) -> list[str]:
    sents = [s for s in _SENT.split(parent.replace("\n", " ")) if s.strip()]
    out, i = [], 0
    while i < len(sents):
        j, n = i, 0
        while j < len(sents) and (n < target or j == i):
            n += len(sents[j].split())
            j += 1
        chunk = " ".join(sents[i:j])
        if len(chunk.split()) > max_words:  # one very long sentence: hard cut by words
            ws = chunk.split()
            out += [" ".join(ws[k:k + max_words]) for k in range(0, len(ws), max_words)]
        else:
            out.append(chunk)
        if j >= len(sents):
            break
        i = max(i + 1, j - 1)  # one-sentence overlap
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--afrie5", type=Path, required=True)
    ap.add_argument("--reranker", type=Path)
    ap.add_argument("--parent-words", type=int, default=350)
    ap.add_argument("--child-words", type=int, default=80)
    ap.add_argument("--child-max", type=int, default=140)
    ap.add_argument("--min-doc-words", type=int, default=30)
    ap.add_argument("--near-dup", type=float, default=0.7, help="MinHash Jaccard for parent near-duplicates")
    args = ap.parse_args()
    from qdrant_client import QdrantClient, models

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # --- 1) documents --------------------------------------------------------------------
    raw_docs = []
    for f in sorted(args.raw.glob("*.jsonl.gz")):
        raw_docs += [json.loads(line) for line in gzip.open(f, "rt", encoding="utf-8")]
    # authoritative sources first, so duplicates keep the better provenance
    raw_docs.sort(key=lambda d: (SOURCE_PRIORITY.get(d["source"], 3), d["doc_id"]))
    docs, seen, drop = [], set(), Counter()
    for d in raw_docs:
        if not d["oromo_ok"]:
            drop["not_oromo"] += 1
            continue
        text = _PUA.sub("", d["text"])                       # PDF private-use bullet glyphs
        if "\ufffd" in text:  # MADLAD mojibake: in Oromo the lost character is almost always the hudhaa
            drop["mojibake_repaired(kept)"] += 1
            text = text.replace("\ufffd", "'")
        d["text"] = text
        d["title"] = re.sub(r"\s+", " ", html.unescape(d["title"]).replace("\xa0", " ")).strip()
        if d["source"] == "webdomain":  # corpus ids are useless as titles: use the headline
            first = d["text"].strip().split("\n", 1)[0]
            d["title"] = first if 3 <= len(first.split()) <= 20 else " ".join(d["text"].split()[:12]) + " ..."
        if len(words(d["text"])) < args.min_doc_words:
            drop["short"] += 1
            continue
        if DROP_TITLE.search(d["title"]) or DROP_TITLE.search(d["url"].split("/")[-1]):
            drop["admin_or_stats"] += 1
            continue
        key = hashlib.sha1("".join(d["text"].lower().split()).encode()).hexdigest()
        if key in seen:
            drop["exact_dup"] += 1
            continue
        seen.add(key)
        docs.append(d)
    print(f"DOCS: {len(docs):,} kept | dropped {dict(drop)}", flush=True)

    # --- 2) parents and children ------------------------------------------------------------------
    db_path = args.out / "parents.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(str(db_path))
    db.execute("CREATE TABLE parents (parent_id TEXT PRIMARY KEY, doc_id TEXT, source TEXT, tier TEXT, license TEXT, "
               "domain TEXT, title TEXT, url TEXT, position INTEGER, n_words INTEGER, text TEXT)")
    children: list[Child] = []
    stats = Counter()
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=args.near_dup, num_perm=64)
    for d in docs:
        for pi, ptext in enumerate(pack(paragraphs(d["text"]), args.parent_words)):
            pid = f"{d['doc_id']}:{pi}"
            ws = words(ptext)
            mh = MinHash(num_perm=64, seed=1)
            mh.update_batch([" ".join(ws[i:i + 5]).encode() for i in range(max(1, len(ws) - 4))])
            if lsh.query(mh):
                stats["near_dup_parents_dropped"] += 1
                continue
            lsh.insert(pid, mh)
            db.execute("INSERT INTO parents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (pid, d["doc_id"], d["source"], d["tier"], d["license"], d["domain"], d["title"], d["url"], pi,
                        len(ptext.split()), ptext))
            stats[f"parents_{d['source']}"] += 1
            for ci, ctext in enumerate(windows(ptext, args.child_words, args.child_max)):
                if len(words(ctext)) < 5:
                    continue
                payload = {"parent_id": pid, "doc_id": d["doc_id"], "source": d["source"], "tier": d["tier"],
                           "license": d["license"], "domain": d["domain"], "title": d["title"][:200], "url": d["url"],
                           "position": ci, "text": ctext}
                children.append(Child(len(children), pid, ctext, f"{d['title'][:80]}\n{ctext}", payload))
                stats[f"children_{d['source']}"] += 1
    db.commit()
    n_par = db.execute("SELECT COUNT(*) FROM parents").fetchone()[0]
    cw = [len(c.text.split()) for c in children]
    print(f"CHUNKS: {n_par:,} parents, {len(children):,} children (words per child median "
          f"{int(np.median(cw))}, p95 {int(np.percentile(cw, 95))}) | {dict(sorted(stats.items()))}", flush=True)

    # --- 3) keys -------------------------------------------------------------------------------------
    t1 = time.time()
    vecs = embed(args.afrie5, [c.embed_text for c in children], "mean", device, bs=64)
    bm = {a: BM25Encoder(a).fit(c.text for c in children) for a in ("trunc4", "norm")}
    print(f"EMBED: {vecs.shape} on {device} in {time.time() - t1:.0f}s | BM25 hash collisions "
          f"{ {a: e.collisions for a, e in bm.items()} }", flush=True)

    # --- 4) Qdrant ----------------------------------------------------------------------------------
    qc = QdrantClient(url=args.qdrant_url, timeout=300)
    if qc.collection_exists(COLL):
        qc.delete_collection(COLL)
    qc.create_collection(
        COLL, vectors_config={"afrie5": models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE)},
        sparse_vectors_config={f"bm25_{a}": models.SparseVectorParams(modifier=models.Modifier.IDF) for a in bm})
    for field in ("domain", "tier", "source"):
        qc.create_payload_index(COLL, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    t2 = time.time()
    for i in range(0, len(children), 512):
        pts = []
        for c in children[i:i + 512]:
            vec: dict[str, object] = {"afrie5": vecs[c.cid].tolist()}
            for a, e in bm.items():
                idx, val = e.encode_doc(c.text)
                vec[f"bm25_{a}"] = models.SparseVector(indices=idx, values=val)
            pts.append(models.PointStruct(id=c.cid, vector=vec, payload=c.payload))
        qc.upsert(COLL, pts, wait=True)
    print(f"INDEX: {qc.count(COLL).count:,} points in {COLL} | {time.time() - t2:.0f}s", flush=True)

    # --- 5) smoke test: hybrid (DBSF) -> rerank -> parents --------------------------------------------------
    qv = embed(args.afrie5, [f"Instruct: {E5_TASK}\nQuery: {q}" for q in SMOKE], "mean", device)
    rr = rtok = None
    if args.reranker and args.reranker.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        rtok = AutoTokenizer.from_pretrained(str(args.reranker))
        rr = AutoModelForSequenceClassification.from_pretrained(
            str(args.reranker), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    report = []
    for qi, q in enumerate(SMOKE):
        idx, val = bm["trunc4"].encode_query(q)
        res = qc.query_points(COLL, prefetch=[
            models.Prefetch(query=qv[qi].tolist(), using="afrie5", limit=50),
            models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="bm25_trunc4", limit=50)],
            query=models.FusionQuery(fusion=models.Fusion.DBSF), limit=30, with_payload=True).points
        cands = [(p.payload, float(p.score)) for p in res]
        if rr is not None and cands:
            with torch.no_grad():
                b = rtok([q] * len(cands), [c[0]["text"] for c in cands], padding=True, truncation=True,
                         max_length=512, return_tensors="pt").to(device)
                s = rr(**b).logits.view(-1).float().cpu().tolist()
            cands = [(c[0], sc) for c, sc in sorted(zip(cands, s), key=lambda x: -x[1])]
        parents, seen_p = [], set()
        for pl, sc in cands:
            if pl["parent_id"] in seen_p:
                continue
            seen_p.add(pl["parent_id"])
            row = db.execute("SELECT source, tier, title, text FROM parents WHERE parent_id=?", (pl["parent_id"],)).fetchone()
            parents.append({"score": round(sc, 3), "source": row[0], "tier": row[1], "title": row[2], "text": row[3]})
            if len(parents) == 3:
                break
        report.append({"query": q, "top_parents": parents})
        print(f"SMOKE Q: {q}", flush=True)
        for r in parents:
            print(f"   [{r['source']}/{r['tier']}] {r['score']:+.2f} {r['title'][:50]!r}: {r['text'][:220]!r}", flush=True)
    (args.out / "smoke_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out / "index_meta.json").write_text(json.dumps({
        "collection": COLL, "docs": len(docs), "parents": n_par, "children": len(children), "dropped": dict(drop),
        "parent_words": args.parent_words, "child_words": args.child_words, "encoder": str(args.afrie5),
        "sparse": list(bm), "built": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0)}, indent=2))
    print(f"DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
