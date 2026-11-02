"""Held-out system comparison for the paper: native vs translation cascade, the agent ablation, and two
stress conditions (orthographic noise, missing gold evidence).

Question set: every dev candidate that was NOT used for agent iteration (the 48 dev items of T025 to T028
are excluded; the 6 chat prompts too). Final tables are later restricted to the items the native-speaker
review keeps (a filter on ids, no rerun needed).

Base arms (one LLM for all: AfriqueQwen3.5-9B-it, non-thinking; only the processing language and pipeline change)
  closed           native closed book (question in Oromo, answer in Oromo)
  closed_tt_nllb   translate-test cascade: Oromo -> English (NLLB-200 3.3B), answer in English, back to Oromo
  closed_tt_self   self-translate cascade: the same 9B model translates both ways
  rag_naive        retrieve top-4 parents for the question -> generate with [n] (no refine, grade, verify)
  rag_tt_nllb      translate-everything RAG: question and the same 4 passages to English (NLLB), answer, back
  rag_tt_self      as rag_tt_nllb, but the 9B model translates question, passages and answer
  agent_noverify   refine + grade / reformulate loop + generate (no claim verification)
  agent_full       full cyclic agent v4 (adds sentence verification, regeneration, extractive fallback)
Conditions (suffix), for closed and the three graph arms:
  @noisy   the question is rewritten with informal Qubee spelling: no hudhaa ('), random degemination and
           vowel shortening (p = 0.5 each), lower case. Tests the refine node and the lexical index.
  @nogold  the gold parent passage is removed from retrieval. Tests whether the pipeline abstains or
           hallucinates when its best evidence is missing (the judge still sees the gold passage).

Scoring: the calibrated judge (gadaabot.eval.judge, variant ref_ctx, D11 PASS in T028), with the same
background passages for every arm (top-3 parents retrieved for the clean question) and the dev reference
answer; the judge always sees the clean question. Judge scores persist in judge.jsonl and are reused.
Also: abstain %, confident-wrong % (answered and judge <= 2), latency, answer length, English leakage.
Paired differences use a bootstrap over questions; p-values are Holm-adjusted within the planned family.

Outputs (out-dir): arm_<arm>.jsonl checkpoints, answers.jsonl, judge.jsonl, summary.json, table_arms.tsv,
table_pairs.tsv, blind_human_v2.tsv + blind_key_v2.json (30 questions x 5 systems for human validation).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gadaabot.agent.graph import _passages, answer, build_graph
from gadaabot.agent.llm import JUDGE, LLM
from gadaabot.agent.retriever import Evidence, Retriever
from gadaabot.eval.judge import Judge, answer_body, boot_ci

BASES = ("closed", "closed_tt_nllb", "closed_tt_self", "rag_naive", "rag_tt_nllb", "rag_tt_self", "agent_noverify",
         "agent_full")
CONDITIONS = ("noisy", "nogold")
CONDITIONED = ("closed", "rag_naive", "agent_noverify", "agent_full")
GRAPHS = {"rag_naive": dict(use_refine=False, use_grade=False, use_verify=False),
          "agent_noverify": dict(use_verify=False), "agent_full": {}}
DEFAULT_ARMS = ("closed,closed_tt_nllb,closed_tt_self,rag_naive,rag_tt_nllb,rag_tt_self,agent_noverify,agent_full,"
                "closed@noisy,rag_naive@noisy,agent_noverify@noisy,rag_naive@nogold,agent_noverify@nogold,agent_full@nogold")
PAIRS = [  # planned family (Holm); a - b
    ("rag_naive", "closed"), ("agent_full", "closed"),
    ("closed", "closed_tt_nllb"), ("closed", "closed_tt_self"),
    ("rag_naive", "rag_tt_nllb"), ("rag_naive", "rag_tt_self"), ("rag_tt_self", "rag_tt_nllb"),
    ("agent_noverify", "rag_naive"), ("agent_full", "agent_noverify"), ("agent_full", "rag_naive"),
    ("closed@noisy", "closed"), ("rag_naive@noisy", "rag_naive"), ("agent_noverify@noisy", "agent_noverify"),
    ("agent_noverify@noisy", "rag_naive@noisy"),
    ("rag_naive@nogold", "closed"), ("agent_noverify@nogold", "rag_naive@nogold"),
    ("agent_full@nogold", "agent_noverify@nogold"), ("rag_naive@nogold", "rag_naive")]
CLOSED_SYS = "Ati gargaaraa Afaan Oromoo ti. Afaan Oromootiin qofa deebisi."
CLOSED_SYS_EN = "You are a helpful assistant. Answer in English."
SYS_ANSWER_EN = (
    "You are a helpful assistant for health and education questions. Write in English. Use ONLY facts stated in "
    "the numbered passages and put the passage number in brackets, like [1], after every sentence that uses it. "
    "If the passages do not cover part of the question, say so plainly instead of guessing. Never invent facts, "
    "never give medicine doses. Explain in your own words; do not repeat the question.")
TR_SYS = "You are a professional translator. Output only the translation, nothing else."
EN_WORDS = re.compile(r"\b(the|and|of|is|are|to|in|for|with|that|this|you|your|it)\b", re.I)


def split_arm(arm: str) -> tuple[str, str]:
    base, _, cond = arm.partition("@")
    return base, cond


def qubee_noise(text: str, seed: str) -> str:
    """Informal Qubee spelling: drop hudhaa, randomly degeminate doubled letters and shorten long vowels
    (p = 0.5 each), lower case. Deterministic per item; guarantees at least one change when possible."""
    rng = random.Random(seed)
    t = text.lower().replace("'", "").replace("’", "").replace("`", "")
    out, i = [], 0
    while i < len(t):
        if i + 1 < len(t) and t[i] == t[i + 1] and t[i].isalpha() and rng.random() < 0.5:
            out.append(t[i])
            i += 2
            continue
        out.append(t[i])
        i += 1
    noisy = "".join(out)
    if noisy == text:  # nothing changed: degeminate the first doubled letter
        noisy = re.sub(r"([a-z])\1", r"\1", noisy, count=1)
    return noisy


class LockedRetriever:
    """Serializes GPU retrieval calls coming from worker threads; a thread-local exclusion set removes given
    parent ids from the results (the @nogold condition)."""

    def __init__(self, r: Retriever) -> None:
        self.r = r
        self.lock = threading.Lock()
        self.local = threading.local()

    def search(self, query: str, k_parents: int = 4, k_children: int = 50, authority_boost: float = 0.0) -> list[Evidence]:
        excl: set[str] = getattr(self.local, "exclude", set())
        with self.lock:
            res = self.r.search(query, k_parents=k_parents + len(excl), k_children=k_children,
                                authority_boost=authority_boost)
        return [e for e in res if e.parent_id not in excl][:k_parents]


class NLLB:
    """NLLB-200 sentence-level translation (gaz_Latn = West Central Oromo, eng_Latn = English)."""

    def __init__(self, path: Path, device: str = "cuda") -> None:  # device: e.g. cuda:1 for a second card
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.path = str(path)
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.toks = {lang: AutoTokenizer.from_pretrained(self.path, src_lang=lang) for lang in ("gaz_Latn", "eng_Latn")}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path, dtype=torch.bfloat16).to(device).eval()
        self.seconds = 0.0
        self.ooms = 0

    @staticmethod
    def _segments(line: str) -> list[str]:
        out: list[str] = []
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            w = s.split()
            if len(w) > 80:  # unpunctuated runs (lists, tables): translate in 60-word pieces
                out.extend(" ".join(w[i:i + 60]) for i in range(0, len(w), 60))
            elif w:
                out.append(s)
        return out

    def _gen(self, tok: Any, texts: list[str], bos: int) -> list[str]:  # noqa: ANN401
        """Translate one batch; on CUDA OOM split the batch in half and retry (the GPU may be shared with vLLM).
        The retry happens OUTSIDE the except block: inside it the traceback keeps the failed forward pass's
        activations alive, so a retry there cannot get the memory back (T030b crash)."""
        oom = False
        try:
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            gen = self.model.generate(**enc, forced_bos_token_id=bos, num_beams=2,
                                      max_new_tokens=min(400, int(enc["input_ids"].shape[1] * 2) + 16))
            return [d.strip() for d in tok.batch_decode(gen, skip_special_tokens=True)]
        except self.torch.OutOfMemoryError:
            oom = True
        if oom:
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
            self.ooms += 1
            if len(texts) == 1:
                raise RuntimeError("NLLB out of memory on a single segment")
            h = len(texts) // 2
            return self._gen(tok, texts[:h], bos) + self._gen(tok, texts[h:], bos)
        return []

    def translate(self, texts: list[str], src: str, tgt: str, budget: int = 2048) -> list[str]:
        """budget = padded source tokens per batch (beam 2 doubles decoder memory)."""
        t0 = time.time()
        segs: list[str] = []
        layout: list[list[list[int]]] = []
        for t in texts:
            lines = []
            for line in (t or "").split("\n"):
                ss = self._segments(line)
                lines.append(list(range(len(segs), len(segs) + len(ss))))
                segs.extend(ss)
            layout.append(lines)
        uniq = list(dict.fromkeys(segs))
        order = sorted(range(len(uniq)), key=lambda i: len(uniq[i]))
        tok, done = self.toks[src], {}
        bos = tok.convert_tokens_to_ids(tgt)
        lens = [min(256, len(x)) for x in tok([uniq[i] for i in order], add_special_tokens=True)["input_ids"]]
        batches: list[list[int]] = []
        cur: list[int] = []
        for i, n in zip(order, lens):  # sorted by length, so the last item sets the padded width
            if cur and (len(cur) + 1) * n > budget:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        with self.torch.no_grad():
            for idx in batches:
                for i, d in zip(idx, self._gen(tok, [uniq[i] for i in idx], bos)):
                    done[uniq[i]] = d
        self.seconds += time.time() - t0
        return ["\n".join(" ".join(done[segs[k]] for k in line) for line in lay).strip() for lay in layout]


def strip_cites(text: str) -> str:
    return re.sub(r"\s*\[\d+\](?:\s*\[\d+\])*", "", text)


def english_ratio(text: str) -> float:
    w = re.findall(r"[^\W\d_]+", answer_body(text))
    return round(len(EN_WORDS.findall(answer_body(text))) / len(w), 3) if w else 0.0



def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = run
    return adj


def boot_p(d: list[float], reps: int = 4000, seed: int = 13) -> float:
    """Two-sided bootstrap p-value for mean(d) = 0 (percentile method)."""
    rng = random.Random(seed)
    n = len(d)
    m = [statistics.fmean(d[rng.randrange(n)] for _ in range(n)) for _ in range(reps)]
    lo = sum(x <= 0 for x in m) / reps
    hi = sum(x >= 0 for x in m) / reps
    return min(1.0, 2 * min(lo, hi))


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", default="gadaabot-gen")
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--afrie5", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True, help="dir with devset_candidates.jsonl")
    ap.add_argument("--exclude-jsonl", type=Path, required=True, help="iteration run whose ids are excluded")
    ap.add_argument("--nllb", type=Path, required=True)
    ap.add_argument("--nllb-device", default="cuda:0", help="cuda:1 = a second visible card (no sharing with vLLM)")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge-samples", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        b, c = split_arm(a)
        if b not in BASES or (c and (c not in CONDITIONS or b not in CONDITIONED)) or (c == "nogold" and b == "closed"):
            raise SystemExit(f"unknown or unsupported arm {a}")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    excl = {json.loads(l)["id"] for l in args.exclude_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()}
    dev = [json.loads(l) for l in (args.dev / "devset_candidates.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [d for d in dev if d["id"] not in excl]
    if args.limit:
        items = items[: args.limit]
    for d in items:
        d["q_noisy"] = qubee_noise(d["question"], d["id"])
    print(f"HELDOUT: {len(items)} questions (excluded {len(excl)} iteration ids) | arms {arms}", flush=True)
    print(f"NOISE e.g. {items[0]['question'][:70]!r} -> {items[0]['q_noisy'][:70]!r}", flush=True)

    llm = LLM(args.llm_url, args.llm_model)
    ret = LockedRetriever(Retriever(args.qdrant_url, args.afrie5, args.reranker, args.parents))
    pool = ThreadPoolExecutor(args.workers)

    t0 = time.time()
    ev4 = {d["id"]: ret.search(d["question"], k_parents=4) for d in items}
    print(f"RETRIEVAL: {time.time() - t0:.0f}s | gold parent in top-4 "
          f"{sum(d['parent_id'] in [e.parent_id for e in ev4[d['id']]] for d in items)}/{len(items)}", flush=True)

    res: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in arms}
    ids_all = [d["id"] for d in items]

    def save_arm(a: str) -> None:
        """Checkpoint one finished arm, so a crash later never costs the GPU hours already spent."""
        with (out / f"arm_{a}.jsonl").open("w", encoding="utf-8") as fh:
            for i in ids_all:
                fh.write(json.dumps({"id": i, **res[a][i]}, ensure_ascii=False) + "\n")

    for a in arms:  # resume: reuse an arm whose checkpoint covers exactly this question set
        f = out / f"arm_{a}.jsonl"
        if f.exists():
            rows = {r.pop("id"): r for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
            if set(rows) == set(ids_all):
                res[a] = rows
                print(f"RESUME {a}: {len(rows)} cached answers", flush=True)

    def todo(a: str) -> bool:
        return a in arms and not res[a]

    # ---------------- closed book (clean and noisy question)
    for a in [x for x in arms if split_arm(x)[0] == "closed" and todo(x)]:
        noisy = split_arm(a)[1] == "noisy"

        def closed(d: dict[str, Any], noisy: bool = noisy) -> dict[str, Any]:
            t = time.time()
            q = d["q_noisy"] if noisy else d["question"]
            ans = llm.chat([{"role": "system", "content": CLOSED_SYS}, {"role": "user", "content": q}], max_tokens=500)
            return {"answer": ans, "seconds": time.time() - t, "abstained": False, "asked": q}
        for d, r in zip(items, pool.map(closed, items)):
            res[a][d["id"]] = r
        save_arm(a)
        print(f"ARM {a} done", flush=True)

    # ---------------- graph arms (same code path, nodes switched off) x conditions
    for a in [x for x in arms if split_arm(x)[0] in GRAPHS and todo(x)]:
        base, cond = split_arm(a)
        app = build_graph(llm, ret, **GRAPHS[base])  # type: ignore[arg-type]

        def run(d: dict[str, Any], app: Any = app, cond: str = cond) -> dict[str, Any]:  # noqa: ANN401
            ret.local.exclude = {d["parent_id"]} if cond == "nogold" else set()
            q = d["q_noisy"] if cond == "noisy" else d["question"]
            try:
                r = answer(app, q)
            except Exception as exc:  # noqa: BLE001
                r = {"answer": f"ERROR {type(exc).__name__}: {exc}", "abstained": False, "seconds": 0.0, "trace": [],
                     "evidence_ids": [], "contexts": [], "drafts": [], "abstain_reason": ""}
            finally:
                ret.local.exclude = set()
            r["asked"] = q
            return r
        c0 = llm.calls
        for d, r in zip(items, pool.map(run, items)):
            res[a][d["id"]] = {"answer": r["answer"], "seconds": r["seconds"], "abstained": bool(r["abstained"]),
                               "abstain_reason": r.get("abstain_reason", ""), "trace": r.get("trace", []),
                               "evidence_ids": r.get("evidence_ids", []), "contexts": r.get("contexts", []),
                               "drafts": r.get("drafts", []), "asked": r["asked"],
                               "gold_in_evidence": d["parent_id"] in r.get("evidence_ids", [])}
        save_arm(a)
        print(f"ARM {a} done | LLM calls/question {(llm.calls - c0) / max(1, len(items)):.2f}", flush=True)

    # ---------------- translation cascades
    def self_tr(text: str, to_en: bool, max_tokens: int = 700) -> str:
        msg = (f"Translate this Afaan Oromoo text into English:\n\n{text}" if to_en
               else f"Translate this English text into Afaan Oromoo (Qubee script):\n\n{text}")
        return llm.chat([{"role": "system", "content": TR_SYS}, {"role": "user", "content": msg}], max_tokens=max_tokens,
                        sampling=JUDGE)

    def en_closed(q_en: str) -> str:
        return llm.chat([{"role": "system", "content": CLOSED_SYS_EN}, {"role": "user", "content": q_en}], max_tokens=500)

    def rag_en(d: dict[str, Any], qe: str, psg: list[Evidence]) -> str:
        return llm.chat([{"role": "system", "content": SYS_ANSWER_EN}, {"role": "user", "content":
                          f"Passages:\n{_passages(psg)}\n\nQuestion: {qe}"}], max_tokens=700)

    def with_text(e: Evidence, text: str) -> Evidence:
        return Evidence(e.parent_id, e.source, e.tier, e.license, e.domain, e.title, e.url, text, e.score, e.child)

    need_nllb = any(todo(a) for a in ("closed_tt_nllb", "rag_tt_nllb"))
    nllb = NLLB(args.nllb, args.nllb_device) if need_nllb else None
    q_en_nllb: list[str] = nllb.translate([d["question"] for d in items], "gaz_Latn", "eng_Latn") if nllb else []

    if todo("closed_tt_nllb") and nllb:
        t = time.time()
        en = list(pool.map(en_closed, q_en_nllb))
        back = nllb.translate(en, "eng_Latn", "gaz_Latn")
        per = (time.time() - t) / len(items)
        for d, a, e in zip(items, back, en):
            res["closed_tt_nllb"][d["id"]] = {"answer": a, "answer_en": e, "seconds": per, "abstained": False}
        save_arm("closed_tt_nllb")
        print("ARM closed_tt_nllb done", flush=True)

    if todo("closed_tt_self"):
        def tt_self(d: dict[str, Any]) -> dict[str, Any]:
            t = time.time()
            qe = self_tr(d["question"], True)
            e = en_closed(qe)
            return {"answer": self_tr(e, False), "answer_en": e, "question_en": qe, "seconds": time.time() - t,
                    "abstained": False}
        for d, r in zip(items, pool.map(tt_self, items)):
            res["closed_tt_self"][d["id"]] = r
        save_arm("closed_tt_self")
        print("ARM closed_tt_self done", flush=True)

    if todo("rag_tt_nllb") and nllb:
        t = time.time()
        flat_en = nllb.translate([e.text for d in items for e in ev4[d["id"]]], "gaz_Latn", "eng_Latn")
        k, psg_en = 0, {}
        for d in items:
            psg_en[d["id"]] = [with_text(e, flat_en[k + j]) for j, e in enumerate(ev4[d["id"]])]
            k += len(ev4[d["id"]])
        en = list(pool.map(lambda p: rag_en(p[0], p[1], psg_en[p[0]["id"]]), zip(items, q_en_nllb)))
        back = nllb.translate([strip_cites(x) for x in en], "eng_Latn", "gaz_Latn")
        per = (time.time() - t) / len(items)
        for d, a, e in zip(items, back, en):
            res["rag_tt_nllb"][d["id"]] = {"answer": a, "answer_en": e, "seconds": per, "abstained": False,
                                           "evidence_ids": [x.parent_id for x in ev4[d["id"]]]}
        save_arm("rag_tt_nllb")
        print(f"ARM rag_tt_nllb done | NLLB {nllb.seconds:.0f}s, OOM splits {nllb.ooms}", flush=True)

    if todo("rag_tt_self"):
        cached_q = {i: r.get("question_en", "") for i, r in res.get("closed_tt_self", {}).items()}

        def tt_rag_self(d: dict[str, Any]) -> dict[str, Any]:
            t = time.time()
            qe = cached_q.get(d["id"]) or self_tr(d["question"], True)
            psg = [with_text(e, self_tr(e.text, True, max_tokens=1500)) for e in ev4[d["id"]]]
            e = rag_en(d, qe, psg)
            return {"answer": self_tr(strip_cites(e), False), "answer_en": e, "question_en": qe,
                    "seconds": time.time() - t, "abstained": False, "evidence_ids": [x.parent_id for x in ev4[d["id"]]]}
        for d, r in zip(items, pool.map(tt_rag_self, items)):
            res["rag_tt_self"][d["id"]] = r
        save_arm("rag_tt_self")
        print("ARM rag_tt_self done", flush=True)

    with (out / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for d in items:
            for a in arms:
                r = res[a].get(d["id"])
                if r is not None:
                    fh.write(json.dumps({"id": d["id"], "arm": a, "domain": d["domain"], "source": d["source"],
                                         "question": d["question"], "gold": d["answer"], **r}, ensure_ascii=False) + "\n")

    # ---------------- judge (calibrated ref_ctx variant, identical background for all arms; persistent cache)
    judge = Judge(llm, args.judge_samples, workers=max(8, args.workers * 2))
    qtext = {d["id"]: d["question"] for d in items}
    jpath = out / "judge.jsonl"
    old: dict[tuple[str, str], dict[str, Any]] = {}
    if jpath.exists():
        for l in jpath.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                old[(r["id"], r["arm"])] = r
    reused = 0
    for (i, a), r in old.items():
        if a in res and i in res[a] and r.get("score") is not None:
            judge.cache[("ref_ctx", qtext[i], answer_body(res[a][i]["answer"]))] = \
                {"score": r["score"], "mean": r["mean"], "raw": r.get("raw", [])}
            reused += 1
    bg = {d["id"]: [e.text for e in ev4[d["id"]][:3]] for d in items}
    jobs, keys = [], []
    for d in items:
        for a in arms:
            if d["id"] in res[a]:
                jobs.append({"variant": "ref_ctx", "question": d["question"], "answer": res[a][d["id"]]["answer"],
                             "reference": d["answer"], "passages": bg[d["id"]]})
                keys.append((d["id"], a))
    t = time.time()
    scores = judge.score_many(jobs)
    print(f"JUDGE: {len(jobs)} answers ({reused} reused from cache) in {time.time() - t:.0f}s", flush=True)
    sc: dict[str, dict[str, float]] = {a: {} for a in arms}
    with jpath.open("w", encoding="utf-8") as fh:
        for (i, a), s in zip(keys, scores):
            if s["mean"] is not None:
                sc[a][i] = s["mean"]
            fh.write(json.dumps({"id": i, "arm": a, **s}) + "\n")

    # ---------------- tables
    dom = {d["id"]: d["domain"] for d in items}
    srcof = {d["id"]: d["source"] for d in items}
    srcs = sorted(set(srcof.values()))

    def mean_or_nan(v: list[float]) -> float:
        return round(statistics.fmean(v), 3) if v else float("nan")

    rows = []
    for a in arms:
        ids = list(sc[a])
        if not ids:
            continue
        r = res[a]
        ci = boot_ci(lambda s, a=a, ids=ids: statistics.fmean(sc[a][ids[t]] for t in s), len(ids))
        answered = [i for i in ids if not r[i]["abstained"]]
        row: dict[str, Any] = {
            "arm": a, "n": len(ids), "judge": mean_or_nan([sc[a][i] for i in ids]), "lo": round(ci[0], 3), "hi": round(ci[1], 3),
            "health": mean_or_nan([sc[a][i] for i in ids if dom[i] == "health"]),
            "education": mean_or_nan([sc[a][i] for i in ids if dom[i] == "education"]),
            "abstain_pct": round(100 * (len(ids) - len(answered)) / len(ids), 1),
            "confident_wrong_pct": round(100 * sum(sc[a][i] <= 2 for i in answered) / len(ids), 1),
            "fallback_pct": round(100 * sum(any("extractive" in x for x in r[i].get("trace", [])) for i in ids) / len(ids), 1),
            "answered_judge": mean_or_nan([sc[a][i] for i in answered]),
            "median_s": round(statistics.median(r[i]["seconds"] for i in ids), 2),
            "words": round(statistics.fmean(len(answer_body(r[i]["answer"]).split()) for i in ids), 1),
            "english_ratio": round(statistics.fmean(english_ratio(r[i]["answer"]) for i in ids), 3),
            "gold_in_ev_pct": round(100 * statistics.fmean(float(r[i].get("gold_in_evidence", 0)) for i in ids), 1)
            if split_arm(a)[0] in GRAPHS else ""}
        for s_ in srcs:
            v = [sc[a][i] for i in ids if srcof[i] == s_]
            row[f"src_{s_}"] = round(statistics.fmean(v), 2) if v else ""
        rows.append(row)
        print("ARM-RESULT " + " | ".join(f"{k} {v}" for k, v in row.items()), flush=True)
    with (out / "table_arms.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    prow = []
    for a, b in PAIRS:
        if a not in sc or b not in sc:
            continue
        common = [i for i in sc[a] if i in sc[b]]
        if not common:
            continue
        dlt = [sc[a][i] - sc[b][i] for i in common]
        ci = boot_ci(lambda s, dlt=dlt: statistics.fmean(dlt[t] for t in s), len(dlt))
        prow.append({"a": a, "b": b, "n": len(common), "diff": round(statistics.fmean(dlt), 3), "lo": round(ci[0], 3),
                     "hi": round(ci[1], 3), "p": round(boot_p(dlt), 4), "wins": sum(x > 0 for x in dlt),
                     "ties": sum(x == 0 for x in dlt), "losses": sum(x < 0 for x in dlt)})
    for pr, ph in zip(prow, holm([p["p"] for p in prow])):
        pr["p_holm"] = round(ph, 4)
        print(f"PAIR {pr['a']} - {pr['b']}: {pr['diff']:+.2f} [{pr['lo']:+.2f}, {pr['hi']:+.2f}] n={pr['n']} "
              f"p_holm={pr['p_holm']:.4f} | W/T/L {pr['wins']}/{pr['ties']}/{pr['losses']}", flush=True)
    with (out / "table_pairs.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0]), delimiter="\t")
        w.writeheader()
        w.writerows(prow)

    # ---------------- blind sheet v2 for human validation (30 questions x 5 systems, shuffled)
    five = [a for a in ("closed", "closed_tt_self", "rag_naive", "rag_tt_nllb", "agent_full") if a in arms and res[a]]
    rng = random.Random(31)
    pick = rng.sample(items, min(30, len(items)))
    key = {}
    with (out / "blind_human_v2.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["row", "question", "reference_answer", "answer", "correct_1to5", "fluent_1to5", "question_ok_y_n", "notes"])
        n = 0
        for d in pick:
            order = five[:]
            rng.shuffle(order)
            for a in order:
                n += 1
                key[str(n)] = {"id": d["id"], "arm": a}
                w.writerow([n, d["question"], d["answer"],
                            answer_body(res[a][d["id"]]["answer"]).replace("\t", " ").replace("\n", " | "), "", "", "", ""])
    (out / "blind_key_v2.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"arms": rows, "pairs": prow, "n_items": len(items), "llm_calls": llm.calls,
                                                  "nllb_seconds": nllb.seconds if nllb else 0}, indent=1), encoding="utf-8")
    print(f"DONE: LLM calls {llm.calls} | blind sheet v2 {len(key)} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
