"""Gate R1: retrieval arms on the GadaaBot domain dev set (knowledge base index v1.1).

Gold: the parent section the question was written from (one relevant parent per query).
This under-counts success when another parent also answers the question, so absolute
numbers are a lower bound; arm-to-arm comparisons are fair (same queries, same gold).

Questions: the reviewed set (keep = y; question_edited replaces question when present)
if reviews exist, otherwise all automatic candidates (reported as PRELIMINARY).

Arms (children retrieved, then collapsed to parents in rank order):
  R0 bm25_norm   R1 bm25_trunc4   R2 dense AfriE5
  R3 hybrid AfriE5 + trunc4, RRF    R4 same, DBSF
  R5 best of R3/R4 + bge-reranker-v2-m3 over the top 50 children
Metrics: parent Recall@1/3/5/10, MRR@10, doc-level Hit@5; 95% bootstrap CI for R@5 and
MRR; breakdown by domain and source; and R@5 by question-passage word-overlap tercile
(shows how much lexical copying from the passage favours BM25).

Usage: python -m gadaabot.eval.retrieval_eval --dev DIR --qdrant-url URL --afrie5 DIR [--reranker DIR] --out FILE
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from gadaabot.eval.retrieval_bench import E5_TASK, embed
from gadaabot.kb.kb_index import COLL
from gadaabot.retrieval.oromo_analyzer import BM25Encoder


def load_dev(dev: Path) -> tuple[list[dict[str, object]], bool]:
    cands = {json.loads(l)["id"]: json.loads(l) for l in (dev / "devset_candidates.jsonl").read_text(encoding="utf-8").splitlines()}
    rev = dev / "devset_review.tsv"
    rows = list(csv.DictReader(rev.open(encoding="utf-8"), delimiter="\t")) if rev.exists() else []
    reviewed = [r for r in rows if r.get("keep_y_n", "").strip()]
    if reviewed:
        out = []
        for r in reviewed:
            if r["keep_y_n"].strip().lower().startswith("y") and r["id"] in cands:
                c = dict(cands[r["id"]])
                if r.get("question_edited", "").strip():
                    c["question"] = r["question_edited"].strip()
                out.append(c)
        return out, True
    return list(cands.values()), False


def parent_rank(points: list[tuple[str, str]], k: int = 20) -> list[tuple[str, str]]:
    out, seen = [], set()
    for pid, did in points:
        if pid not in seen:
            seen.add(pid)
            out.append((pid, did))
            if len(out) == k:
                break
    return out


def scores(ranked: list[list[tuple[str, str]]], dev: list[dict[str, object]], rng: random.Random) -> dict[str, object]:
    ranks, doc_hit = [], []
    for r, q in zip(ranked, dev):
        pids = [p for p, _ in r]
        ranks.append(pids.index(q["parent_id"]) + 1 if q["parent_id"] in pids else 999)
        doc_hit.append(any(d == str(q["parent_id"]).rsplit(":", 1)[0] for _, d in r[:5]))
    r5 = [1.0 if k <= 5 else 0.0 for k in ranks]
    mrr = [1 / k if k <= 10 else 0.0 for k in ranks]

    def ci(v: list[float]) -> list[float]:
        b = sorted(statistics.fmean(v[rng.randrange(len(v))] for _ in v) for _ in range(1000))
        return [round(100 * b[25], 1), round(100 * b[975], 1)]

    return {"n": len(ranks), "R@1": round(100 * statistics.fmean(k <= 1 for k in ranks), 1),
            "R@3": round(100 * statistics.fmean(k <= 3 for k in ranks), 1), "R@5": round(100 * statistics.fmean(r5), 1),
            "R@5_ci": ci(r5), "R@10": round(100 * statistics.fmean(k <= 10 for k in ranks), 1),
            "MRR@10": round(100 * statistics.fmean(mrr), 1), "MRR_ci": ci(mrr),
            "docHit@5": round(100 * statistics.fmean(doc_hit), 1), "_ranks": ranks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--afrie5", type=Path, required=True)
    ap.add_argument("--reranker", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from qdrant_client import QdrantClient, models

    rng = random.Random(11)
    dev, reviewed = load_dev(args.dev)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEV: {len(dev)} questions ({'REVIEWED' if reviewed else 'PRELIMINARY, unreviewed candidates'}) | "
          f"device {device}", flush=True)
    qs = [str(q["question"]) for q in dev]
    qv = embed(args.afrie5, [f"Instruct: {E5_TASK}\nQuery: {q}" for q in qs], "mean", device)
    enc = {a: BM25Encoder(a) for a in ("norm", "trunc4")}
    qc = QdrantClient(url=args.qdrant_url, timeout=300)

    def sq(a: str, q: str) -> models.SparseVector:
        i, v = enc[a].encode_query(q)
        return models.SparseVector(indices=i, values=v)

    def run(reqs: list[models.QueryRequest]) -> tuple[list[list[dict[str, object]]], float]:
        out, t = [], time.time()
        for i in range(0, len(reqs), 64):
            for r in qc.query_batch_points(COLL, requests=reqs[i:i + 64]):
                out.append([dict(p.payload or {}) for p in r.points])
        return out, (time.time() - t) * 1000 / max(1, len(reqs))

    def req(kind: str, j: int, fusion: models.Fusion | None = None) -> models.QueryRequest:
        if kind in ("norm", "trunc4"):
            return models.QueryRequest(query=sq(kind, qs[j]), using=f"bm25_{kind}", limit=100, with_payload=True)
        if kind == "dense":
            return models.QueryRequest(query=qv[j].tolist(), using="afrie5", limit=100, with_payload=True)
        return models.QueryRequest(prefetch=[models.Prefetch(query=qv[j].tolist(), using="afrie5", limit=100),
                                             models.Prefetch(query=sq("trunc4", qs[j]), using="bm25_trunc4", limit=100)],
                                   query=models.FusionQuery(fusion=fusion), limit=100, with_payload=True)

    n = len(dev)
    results: dict[str, dict[str, object]] = {}
    hits: dict[str, list[list[dict[str, object]]]] = {}
    arms = {"R0_bm25_norm": ("norm", None), "R1_bm25_trunc4": ("trunc4", None), "R2_dense_afrie5": ("dense", None),
            "R3_hybrid_rrf": ("hyb", models.Fusion.RRF), "R4_hybrid_dbsf": ("hyb", models.Fusion.DBSF)}
    for name, (kind, fus) in arms.items():
        res, ms = run([req(kind, j, fus) for j in range(n)])
        hits[name] = res
        s = scores([parent_rank([(str(p["parent_id"]), str(p["doc_id"])) for p in r]) for r in res], dev, rng)
        s["ms_per_query"] = round(ms, 1)
        results[name] = s

    best = max(("R3_hybrid_rrf", "R4_hybrid_dbsf"), key=lambda k: results[k]["MRR@10"])  # type: ignore[arg-type,return-value]
    if args.reranker and args.reranker.exists():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(args.reranker))
        rr = AutoModelForSequenceClassification.from_pretrained(
            str(args.reranker), dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        t0, rer = time.time(), []
        with torch.no_grad():
            for j in range(n):
                cand = hits[best][j][:50]
                b = tok([qs[j]] * len(cand), [str(c["text"]) for c in cand], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
                sc = rr(**b).logits.view(-1).float().cpu().tolist()
                rer.append([c for _, c in sorted(zip(sc, cand), key=lambda x: -x[0])])
        s = scores([parent_rank([(str(p["parent_id"]), str(p["doc_id"])) for p in r]) for r in rer], dev, rng)
        s["ms_per_query"] = round((time.time() - t0) * 1000 / n, 1)
        results[f"R5_rerank50({best})"] = s

    print(f"{'ARM':<30}{'R@1':>6}{'R@3':>6}{'R@5':>6}{'R@5 95%CI':>14}{'R@10':>6}{'MRR':>6}{'docHit5':>8}{'ms/q':>7}")
    for k, s in results.items():
        print(f"ARM {k:<26}{s['R@1']:>6}{s['R@3']:>6}{s['R@5']:>6}{str(s['R@5_ci']):>14}{s['R@10']:>6}"
              f"{s['MRR@10']:>6}{s['docHit@5']:>8}{s['ms_per_query']:>7}", flush=True)

    top = max(results, key=lambda k: results[k]["MRR@10"])  # type: ignore[arg-type,return-value]
    ranks = results[top]["_ranks"]
    by: dict[str, list[int]] = defaultdict(list)
    for q, k in zip(dev, ranks):  # type: ignore[arg-type]
        by[f"domain={q['domain']}"].append(k)
        by[f"source={q['source']}"].append(k)
    print(f"BREAKDOWN ({top}) R@5 by group:", "; ".join(f"{g} {100 * statistics.fmean(k <= 5 for k in v):.0f} (n={len(v)})"
                                                    for g, v in sorted(by.items())), flush=True)
    ov = sorted(range(n), key=lambda j: float(dev[j].get("q_overlap", 0)))
    terc = [ov[: n // 3], ov[n // 3: 2 * n // 3], ov[2 * n // 3:]]
    for arm in ("R1_bm25_trunc4", "R2_dense_afrie5", top):
        rk = results[arm]["_ranks"]
        print(f"OVERLAP {arm:<26} R@5 by question-passage overlap tercile (low/mid/high): " + " / ".join(
            f"{100 * statistics.fmean(rk[j] <= 5 for j in t):.0f}" for t in terc), flush=True)  # type: ignore[index]

    for s in results.values():
        s.pop("_ranks", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reviewed": reviewed, "n": n, "results": results, "best": top}, indent=2))
    print(f"BEST: {top} | {'REVIEWED' if reviewed else 'PRELIMINARY'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
