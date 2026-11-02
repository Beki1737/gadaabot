"""GadaaBot production retriever (Gate R1 winner, preliminary T024):
hybrid AfriE5 dense + BM25 over 4-letter prefixes, server-side DBSF fusion in Qdrant,
bge-reranker-v2-m3 over the top children, then expansion to parent sections (SQLite).

An optional authority prior adds  boost * PRIOR[source]  to the reranker score, so
public-health and textbook sources win ties against web news for health questions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch

from gadaabot.retrieval.oromo_analyzer import BM25Encoder

COLL = "gadaabot_kb_v1"
E5_TASK = "Given a question, retrieve the passage that answers the question"
PRIOR = {"mdh": 1.0, "tico19": 1.0, "ohkb": 1.0, "wikipedia": 0.7, "oeb": 0.7, "textbooks": 0.7, "webdomain": 0.0}


@dataclass
class Evidence:
    parent_id: str
    source: str
    tier: str
    license: str
    domain: str
    title: str
    url: str
    text: str
    score: float
    child: str


class Retriever:
    def __init__(self, qdrant_url: str, afrie5: Path, reranker: Path, parents_db: Path, device: str | None = None):
        from qdrant_client import QdrantClient
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if self.device == "cuda" else torch.float32
        self.qc = QdrantClient(url=qdrant_url, timeout=60)
        self.etok = AutoTokenizer.from_pretrained(str(afrie5))
        self.enc = AutoModel.from_pretrained(str(afrie5), dtype=dt).to(self.device).eval()
        self.rtok = AutoTokenizer.from_pretrained(str(reranker))
        self.rr = AutoModelForSequenceClassification.from_pretrained(str(reranker), dtype=dt).to(self.device).eval()
        self.bm = BM25Encoder("trunc4")
        self.db = sqlite3.connect(str(parents_db), check_same_thread=False)

    @torch.no_grad()
    def _qvec(self, q: str) -> list[float]:
        b = self.etok([f"Instruct: {E5_TASK}\nQuery: {q}"], truncation=True, max_length=512,
                      return_tensors="pt").to(self.device)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = torch.nn.functional.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)
        return v[0].cpu().tolist()

    @torch.no_grad()
    def search(self, query: str, k_parents: int = 4, k_children: int = 50, authority_boost: float = 0.0) -> list[Evidence]:
        from qdrant_client import models

        idx, val = self.bm.encode_query(query)
        pts = self.qc.query_points(COLL, prefetch=[
            models.Prefetch(query=self._qvec(query), using="afrie5", limit=100),
            models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="bm25_trunc4", limit=100)],
            query=models.FusionQuery(fusion=models.Fusion.DBSF), limit=k_children, with_payload=True).points
        if not pts:
            return []
        texts = [str(p.payload["text"]) for p in pts]
        b = self.rtok([query] * len(texts), texts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(self.device)
        scores = self.rr(**b).logits.view(-1).float().cpu().tolist()
        best: dict[str, tuple[float, str]] = {}
        for p, s in zip(pts, scores):
            pl = p.payload or {}
            s += authority_boost * PRIOR.get(str(pl.get("source")), 0.0)
            pid = str(pl["parent_id"])
            if pid not in best or s > best[pid][0]:
                best[pid] = (s, str(pl["text"]))
        out = []
        for pid, (s, child) in sorted(best.items(), key=lambda x: -x[1][0])[:k_parents]:
            row = self.db.execute("SELECT source, tier, license, domain, title, url, text FROM parents WHERE parent_id=?",
                                  (pid,)).fetchone()
            if row:
                out.append(Evidence(pid, row[0], row[1], row[2], row[3], row[4], row[5], row[6], round(s, 3), child))
        return out
