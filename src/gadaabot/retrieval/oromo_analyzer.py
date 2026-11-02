"""Oromo lexical analyzers and a BM25 sparse-vector encoder for Qdrant.

Analyzers (all start from the shared Qubee normalizer, so ta'e / ta’e are one term):
  raw        lowercase whitespace split, punctuation kept (no normalization): floor arm
  norm       Qubee-normalized word tokens (hudhaa kept inside words)
  stem       norm + light suffix stripping (one pass, longest match, stem >= 3 letters)
  trunc{k}   norm + prefix truncation to k letters (language-independent stemming,
             strong for agglutinative languages)
Which one wins is decided on data (retrieval_bench), not assumed.

BM25 in Qdrant: each document is stored as a sparse vector whose values are the BM25
term-frequency component  tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)); the collection is
created with the IDF modifier, so Qdrant multiplies by IDF at query time from live
collection statistics. Query vectors carry weight 1.0 per distinct term. Term ids are
a stable 32-bit hash of the term, so no vocabulary file has to be kept in sync.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

from gadaabot.data.qubee import normalize_qubee, words

# Light inflectional suffixes (plural, case, definiteness, common verb endings).
# Longest match first; one suffix removed at most. Heuristic, evaluated empirically.
OROMO_SUFFIXES: tuple[str, ...] = tuple(sorted({
    "oolee", "oota", "ota", "wwan", "lee", "yyii",              # plural
    "ittii", "icha", "itti",                                    # singulative / definite
    "dhaan", "tiin", "iin",                                     # instrumental
    "dhaaf", "tiif", "iif", "dhaa",                             # dative
    "irraa", "rraa",                                            # ablative
    "tti",                                                      # locative
    "nii", "tii",                                               # genitive / linkers
    "achuu", "amuu", "sisuu", "suu", "uun", "uu",               # verbal nouns / infinitive
    "anii", "ani", "ane", "ama", "ame", "tani", "tan", "te", "ti", "ne", "nu", "an",  # verb agreement
}, key=lambda s: (-len(s), s)))
MIN_STEM = 3


def light_stem(word: str) -> str:
    for suf in OROMO_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= MIN_STEM:
            return word[: -len(suf)]
    return word


def _raw(text: str) -> list[str]:
    return text.lower().split()


def _norm(text: str) -> list[str]:
    return words(normalize_qubee(text))


def _stem(text: str) -> list[str]:
    return [light_stem(w) for w in _norm(text)]


def _trunc(k: int) -> Callable[[str], list[str]]:
    def f(text: str) -> list[str]:
        return [w[:k] for w in _norm(text)]
    return f


ANALYZERS: dict[str, Callable[[str], list[str]]] = {
    "raw": _raw, "norm": _norm, "stem": _stem,
    "trunc4": _trunc(4), "trunc5": _trunc(5), "trunc6": _trunc(6),
}


def term_id(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "little")


@dataclass
class BM25Encoder:
    """Document-side BM25 TF weights and query-side term sets for one analyzer."""

    analyzer: str
    k1: float = 1.2
    b: float = 0.75
    avgdl: float = 0.0
    n_docs: int = 0
    collisions: int = 0
    _seen: dict[int, str] = field(default_factory=dict, repr=False)

    def fit(self, docs: Iterable[str]) -> "BM25Encoder":
        f = ANALYZERS[self.analyzer]
        total = 0
        for d in docs:
            terms = f(d)
            total += len(terms)
            self.n_docs += 1
            for t in set(terms):
                h = term_id(t)
                prev = self._seen.setdefault(h, t)
                self.collisions += prev != t
        self.avgdl = total / max(1, self.n_docs)
        return self

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        terms = ANALYZERS[self.analyzer](text)
        tf = Counter(term_id(t) for t in terms)
        dl = len(terms)
        norm = self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        idx = sorted(tf)
        return idx, [tf[i] * (self.k1 + 1) / (tf[i] + norm) for i in idx]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        idx = sorted({term_id(t) for t in ANALYZERS[self.analyzer](text)})
        return idx, [1.0] * len(idx)
