"""Retrieval arms on Oromo Belebele (gaz_Latn) through a live Qdrant server.

Task: each Belebele question (Oromo) must retrieve its own FLORES passage among all
unique passages. One relevant passage per query, so nDCG@10 = mean 1/log2(rank+1).
This is a public, labelled Oromo retrieval set that lets us choose the BM25 analyzer,
the dense encoder and the fusion before our own health/education dev set exists.

Arms
  sparse   BM25 with each analyzer in oromo_analyzer.ANALYZERS (Qdrant IDF modifier)
  dense    AfriE5-large-instruct (E5 instruct query prefix, mean pooling) and BGE-M3
           (CLS pooling), cosine
  hybrid   dense AfriE5 + each BM25 analyzer, server-side fusion: RRF and DBSF
  rerank   bge-reranker-v2-m3 cross-encoder over the top 50 of the best hybrid
Metrics: nDCG@10, Recall@1/5/10, MRR@10, with 95% bootstrap CIs on nDCG@10, and the
median Qdrant query latency per arm.

Usage: python -m gadaabot.eval.retrieval_bench --belebele FILE --qdrant-url URL
       --afrie5 DIR [--bgem3 DIR] [--reranker DIR] --out FILE
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gadaabot.data.qubee import normalize_qubee
from gadaabot.retrieval.oromo_analyzer import ANALYZERS, BM25Encoder

E5_TASK = "Given a question, retrieve the passage that answers the question"
COLL = "bench_belebele_orm"


def embed(model_dir: Path, texts: list[str], pooling: str, device: str, bs: int = 32) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            h = model(**b).last_hidden_state
            if pooling == "mean":
                m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1)
            else:
                v = h[:, 0]
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def metrics(ranked: list[list[int]], gold: list[int], rng: random.Random) -> dict[str, object]:
    ranks = []
    for r, g in zip(ranked, gold):
        ranks.append(r.index(g) + 1 if g in r else 10**9)
    nd = [1 / math.log2(k + 1) if k <= 10 else 0.0 for k in ranks]
    boots = sorted(statistics.fmean(nd[rng.randrange(len(nd))] for _ in nd) for _ in range(1000))
    return {"ndcg10": round(100 * statistics.fmean(nd), 2), "ndcg10_ci95": [round(100 * boots[25], 1), round(100 * boots[975], 1)],
            "r1": round(100 * statistics.fmean(k <= 1 for k in ranks), 2),
            "r5": round(100 * statistics.fmean(k <= 5 for k in ranks), 2),
            "r10": round(100 * statistics.fmean(k <= 10 for k in ranks), 2),
            "mrr10": round(100 * statistics.fmean(1 / k if k <= 10 else 0.0 for k in ranks), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--belebele", type=Path, required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--afrie5", type=Path, required=True)
    ap.add_argument("--bgem3", type=Path)
    ap.add_argument("--reranker", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    from qdrant_client import QdrantClient, models

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in args.belebele.read_text(encoding="utf-8").splitlines() if l.strip()]
    passages: list[str] = []
    pid: dict[str, int] = {}
    queries, gold = [], []
    for r in rows:
        p = normalize_qubee(r["flores_passage"])
        if p not in pid:
            pid[p] = len(passages)
            passages.append(p)
        queries.append(normalize_qubee(r["question"]))
        gold.append(pid[p])
    print(f"DATA: {len(queries)} Oromo questions, {len(passages)} unique passages | device {device}", flush=True)

    # --- encode -----------------------------------------------------------------------
    t0 = time.time()
    dense = {"afrie5": (embed(args.afrie5, passages, "mean", device),
                        embed(args.afrie5, [f"Instruct: {E5_TASK}\nQuery: {q}" for q in queries], "mean", device))}
    if args.bgem3 and args.bgem3.exists():
        dense["bgem3"] = (embed(args.bgem3, passages, "cls", device), embed(args.bgem3, queries, "cls", device))
    bm25 = {a: BM25Encoder(a).fit(passages) for a in ANALYZERS}
    print(f"ENCODE: dense {list(dense)} in {time.time()-t0:.0f}s | BM25 hash collisions "
          f"{ {a: e.collisions for a, e in bm25.items()} }", flush=True)

    # --- index ---------------------------------------------------------------------------
    qc = QdrantClient(url=args.qdrant_url, timeout=120)
    if qc.collection_exists(COLL):
        qc.delete_collection(COLL)
    qc.create_collection(
        COLL,
        vectors_config={n: models.VectorParams(size=v[0].shape[1], distance=models.Distance.COSINE) for n, v in dense.items()},
        sparse_vectors_config={f"bm25_{a}": models.SparseVectorParams(modifier=models.Modifier.IDF) for a in bm25})
    pts = []
    for i, p in enumerate(passages):
        vec: dict[str, object] = {n: v[0][i].tolist() for n, v in dense.items()}
        for a, e in bm25.items():
            idx, val = e.encode_doc(p)
            vec[f"bm25_{a}"] = models.SparseVector(indices=idx, values=val)
        pts.append(models.PointStruct(id=i, vector=vec, payload={"passage_id": i}))
    qc.upsert(COLL, pts, wait=True)
    print(f"INDEX: {qc.count(COLL).count} points in Qdrant collection {COLL}", flush=True)

    def run(reqs: list[models.QueryRequest]) -> tuple[list[list[int]], float]:
        out, lat = [], []
        for i in range(0, len(reqs), 64):
            s = time.time()
            res = qc.query_batch_points(COLL, requests=reqs[i:i + 64])
            lat.append((time.time() - s) * 1000 / len(reqs[i:i + 64]))
            out += [[int(p.id) for p in r.points] for r in res]
        return out, statistics.median(lat)

    def dq(name: str, j: int) -> list[float]:
        return dense[name][1][j].tolist()

    def sq(a: str, j: int) -> models.SparseVector:
        idx, val = bm25[a].encode_query(queries[j])
        return models.SparseVector(indices=idx, values=val)

    results: dict[str, dict[str, object]] = {}
    rankings: dict[str, list[list[int]]] = {}

    def record(arm: str, reqs: list[models.QueryRequest]) -> None:
        ranked, ms = run(reqs)
        m = metrics(ranked, gold, rng)
        m["latency_ms"] = round(ms, 2)
        results[arm], rankings[arm] = m, ranked
        print(f"ARM {arm:<24} nDCG@10 {m['ndcg10']:6.2f} {m['ndcg10_ci95']}  R@1 {m['r1']:6.2f}  R@10 {m['r10']:6.2f}  "
              f"MRR@10 {m['mrr10']:6.2f}  {ms:.1f} ms/q", flush=True)

    n = len(queries)
    for a in bm25:
        record(f"bm25_{a}", [models.QueryRequest(query=sq(a, j), using=f"bm25_{a}", limit=100) for j in range(n)])
    for name in dense:
        record(f"dense_{name}", [models.QueryRequest(query=dq(name, j), using=name, limit=100) for j in range(n)])
    for a in bm25:
        if a == "raw":
            continue
        for fusion in (models.Fusion.RRF, models.Fusion.DBSF):
            record(f"hyb_afrie5+{a}_{fusion.value}", [models.QueryRequest(
                prefetch=[models.Prefetch(query=dq("afrie5", j), using="afrie5", limit=100),
                          models.Prefetch(query=sq(a, j), using=f"bm25_{a}", limit=100)],
                query=models.FusionQuery(fusion=fusion), limit=100) for j in range(n)])

    # --- rerank the best hybrid ---------------------------------------------------------------
    if args.reranker and args.reranker.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        best = max((k for k in results if k.startswith("hyb_")), key=lambda k: results[k]["ndcg10"])  # type: ignore[arg-type, return-value]
        tok = AutoTokenizer.from_pretrained(str(args.reranker))
        rr = AutoModelForSequenceClassification.from_pretrained(
            str(args.reranker), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        t1 = time.time()
        reranked = []
        with torch.no_grad():
            for j in range(n):
                cand = rankings[best][j][:50]
                b = tok([queries[j]] * len(cand), [passages[c] for c in cand], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
                s = rr(**b).logits.view(-1).float().cpu().tolist()
                reranked.append([c for _, c in sorted(zip(s, cand), key=lambda x: -x[0])])
        m = metrics(reranked, gold, rng)
        m["latency_ms"] = round((time.time() - t1) * 1000 / n, 1)
        m["base_arm"] = best
        results[f"rerank50({best})"] = m
        print(f"ARM rerank50({best}) nDCG@10 {m['ndcg10']:6.2f} {m['ndcg10_ci95']}  R@1 {m['r1']:6.2f}  "
              f"R@10 {m['r10']:6.2f}  MRR@10 {m['mrr10']:6.2f}  {m['latency_ms']} ms/q ({device})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_queries": n, "n_passages": len(passages), "device": device,
                                    "results": results}, indent=2))
    best_all = max(results, key=lambda k: results[k]["ndcg10"])  # type: ignore[arg-type, return-value]
    print(f"BEST: {best_all} nDCG@10 {results[best_all]['ndcg10']} | reference AfriE5 Belebele retrieval "
          f"(AfriMTEB-Lite, Oromo) 69.4", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
