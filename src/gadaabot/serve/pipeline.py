"""Production answer pipeline of GadaaBot 2.0 (configuration chosen by the held-out study, decision D54/D55).

Single-pass native RAG: ingress danger-sign screen -> hybrid retrieval (AfriE5 dense + 4-letter-prefix BM25,
DBSF fusion) -> bge-reranker-v2-m3 -> top-4 parent sections -> one generation with [n] citations ->
sources + health disclaimer, or an honest abstention. The agentic nodes (refine, grade, verify) are off:
they did not improve correctness and verification hurt it (paper, Section 6.3).

Because refine is off, the health disclaimer is attached when the top retrieved evidence is health content.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gadaabot.agent.graph import OROMO_TEXTS, answer, build_graph
from gadaabot.agent.llm import LLM
from gadaabot.agent.retriever import Retriever


@dataclass
class Settings:
    llm_url: str = "http://127.0.0.1:48000"
    llm_model: str = "gadaabot-gen"
    qdrant_url: str = "http://127.0.0.1:46333"
    afrie5: Path = Path("models/McGill-NLP/AfriE5-Large-instruct")
    reranker: Path = Path("models/BAAI/bge-reranker-v2-m3")
    parents_db: Path = Path("data/kb/index/parents.sqlite")
    max_question_chars: int = 1000


@dataclass
class Reply:
    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)
    abstained: bool = False
    emergency: bool = False
    seconds: float = 0.0


class GadaaBot:
    """Thread-safe wrapper: one retriever (GPU) behind a lock, one HTTP client to the vLLM server."""

    def __init__(self, cfg: Settings, retriever: Any | None = None, llm: Any | None = None) -> None:  # noqa: ANN401
        self.cfg = cfg
        self.llm = llm or LLM(cfg.llm_url, cfg.llm_model)
        self._ret = retriever or Retriever(cfg.qdrant_url, cfg.afrie5, cfg.reranker, cfg.parents_db)
        self._lock = threading.Lock()
        self.app = build_graph(self.llm, self, use_refine=False, use_grade=False, use_verify=False)

    # the graph calls retriever.search(...); serialize GPU access
    def search(self, query: str, k_parents: int = 4, k_children: int = 50, authority_boost: float = 0.0) -> list:
        with self._lock:
            return self._ret.search(query, k_parents=k_parents, k_children=k_children, authority_boost=authority_boost)

    def ask(self, question: str) -> Reply:
        q = (question or "").strip()[: self.cfg.max_question_chars]
        if not q:
            return Reply(answer="Gaaffii kee barreessi.", abstained=True)
        t0 = time.time()
        out = answer(self.app, q)
        text = out["answer"]
        doms = out.get("evidence_domains") or []
        if (not out["abstained"] and not out["emergency"] and "health" in doms[:2]
                and OROMO_TEXTS["disclaimer_health"] not in text):
            text = text + "\n\n" + OROMO_TEXTS["disclaimer_health"]
        return Reply(answer=text, sources=out.get("sources", []), abstained=bool(out["abstained"]),
                     emergency=bool(out["emergency"]), seconds=round(time.time() - t0, 2))
