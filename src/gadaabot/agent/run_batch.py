"""Run the GadaaBot agent on a question batch next to a closed-book baseline (same LLM,
no retrieval), for native-speaker scoring and, later, RAGAS.

Questions: the 6 fixed chat prompts of T020 plus a stratified sample of the dev set
(6 per stratum; unreviewed unless the review file has keep = y). Outputs:
  <out-jsonl>  full records (answer, sources, trace, evidence ids, scores, context texts for RAGAS)
  <out-tsv>    side-by-side table with empty scoring columns
It also prints automatic failure flags for the error types found in the T027 human scoring:
answer body is only citations, answer repeats the question, textbook exercise debris, empty body.

Usage: python -m gadaabot.agent.run_batch --llm-url URL --dev DIR --out-jsonl FILE --out-tsv FILE ...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from gadaabot.agent.graph import answer, build_graph
from gadaabot.agent.llm import LLM
from gadaabot.agent.retriever import Retriever

CHAT_PROMPTS = [
    ("health", "Mallattoon dhukkuba busaa maal fa'a? Akkamitti of eeguu dandeenya?"),
    ("health", "Daa'imni koo garaa kaasaa qaba. Mana keessatti maal gochuu qaba, yoom gara buufata fayyaa deemuu qaba?"),
    ("health", "Talaalliin daa'immaniif maaliif barbaachisaa dha?"),
    ("education", "Foto-sinteesisii (photosynthesis) jechuun maal jechuu dha? Akka barataa kutaa 6ffaaf ibsi."),
    ("education", "Sirni Gadaa maal? Seenaa isaa gabaabsii ibsi."),
    ("education", "Bishaan qulqulluu dhuguun fayyaa keenyaaf faayidaa maal qaba?"),
]
CLOSED_SYS = "Ati gargaaraa Afaan Oromoo ti. Afaan Oromootiin qofa deebisi."
_DEBRIS = re.compile(r"_{3,}|\.{5,}|(?:\b\d+\s*){6,}")


def answer_body(text: str) -> str:
    """The answer without the appended source list and disclaimer (both start on their own line)."""
    return re.split(r"\n\s*(?:Maddeewwan:|Hubachiisa:)", text, maxsplit=1)[0].strip()


def failure_flags(question: str, text: str, abstained: bool) -> list[str]:
    """Cheap detectors for the T027 failure modes; empty list means no flag."""
    if abstained:
        return []
    body = answer_body(text)
    words = re.findall(r"[^\W\d_]{2,}", body)
    flags = []
    if not words:
        flags.append("only_citations" if re.search(r"\[\d+\]", body) else "empty")
    q = re.sub(r"\W+", " ", question.lower()).strip()
    b = re.sub(r"\W+", " ", re.sub(r"\[\d+\]", " ", body.lower())).strip()
    if q and b and (b == q or (b.startswith(q) and len(b) - len(q) < 15)):
        flags.append("question_echo")
    if _DEBRIS.search(body):
        flags.append("debris")
    if 0 < len(words) < 6:
        flags.append("too_short")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", default="gadaabot-gen")
    ap.add_argument("--qdrant-url", required=True)
    ap.add_argument("--afrie5", type=Path, required=True)
    ap.add_argument("--reranker", type=Path, required=True)
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=6)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    args = ap.parse_args()

    llm = LLM(args.llm_url, args.llm_model)
    ret = Retriever(args.qdrant_url, args.afrie5, args.reranker, args.parents)
    app = build_graph(llm, ret)

    dev = [json.loads(l) for l in (args.dev / "devset_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(23)
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in dev:
        by[(d["domain"], d["source"])].append(d)
    sample = [x for k in sorted(by) for x in rng.sample(by[k], min(args.per_stratum, len(by[k])))]
    items = [{"id": f"chat{i}", "set": "chat", "domain": d, "question": q, "gold": "", "gold_parent": ""}
             for i, (d, q) in enumerate(CHAT_PROMPTS)]
    items += [{"id": d["id"], "set": "dev", "domain": d["domain"], "question": d["question"], "gold": d["answer"],
               "gold_parent": d["parent_id"]} for d in sample]
    print(f"BATCH: {len(items)} questions ({len(CHAT_PROMPTS)} chat prompts + {len(sample)} dev)", flush=True)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for it in items:
            t0 = time.time()
            closed = llm.chat([{"role": "system", "content": CLOSED_SYS}, {"role": "user", "content": it["question"]}],
                              max_tokens=500)
            t_closed = time.time() - t0
            try:
                r = answer(app, it["question"])
            except Exception as exc:  # noqa: BLE001  (one failure must not stop the batch)
                r = {"answer": f"ERROR {type(exc).__name__}: {exc}", "sources": [], "abstained": False, "emergency": False,
                     "trace": [], "seconds": 0.0, "evidence_ids": [], "attempts": 0, "domain": ""}
            rec = {**it, "closed_book": closed, "closed_seconds": round(t_closed, 2), **{f"rag_{k}": v for k, v in r.items()
                                                                                         if k != "question"}}
            rec["gold_in_evidence"] = bool(it["gold_parent"]) and it["gold_parent"] in r.get("evidence_ids", [])
            rec["flags"] = failure_flags(it["question"], str(r.get("answer", "")), bool(r.get("abstained")))
            recs.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"Q {it['id']:<8} rag {r['seconds']:5.1f}s abst={int(r['abstained'])} tries={r.get('attempts')} "
                  f"gold_in_ev={int(rec['gold_in_evidence'])} flags={','.join(rec['flags']) or '-'} | {it['question'][:60]}", flush=True)

    with args.out_tsv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "set", "domain", "question", "reference_answer", "closed_book_answer", "rag_answer",
                    "rag_abstained", "closed_fluent_1to5", "closed_correct_1to5", "rag_fluent_1to5", "rag_correct_1to5",
                    "rag_grounded_y_n", "notes"])
        for r in recs:
            def flat(x: str) -> str:
                return x.replace("\t", " ").replace("\n", " | ")
            w.writerow([r["id"], r["set"], r["domain"], flat(r["question"]), flat(r["gold"]), flat(r["closed_book"]),
                        flat(r["rag_answer"]), int(r["rag_abstained"]), "", "", "", "", "", ""])
    dev_recs = [r for r in recs if r["set"] == "dev"]
    from collections import Counter
    reasons = Counter(r.get("rag_abstain_reason") or "?" for r in recs if r["rag_abstained"])
    fallback = sum(1 for r in recs if any("extractive" in t for t in r.get("rag_trace", [])))
    print(f"ABSTAIN REASONS: {dict(reasons)} | extractive fallbacks {fallback} | abstained although gold passage retrieved "
          f"{sum(1 for r in dev_recs if r['rag_abstained'] and r['gold_in_evidence'])}", flush=True)
    print(f"SUMMARY: abstain {sum(r['rag_abstained'] for r in recs)}/{len(recs)} | emergency {sum(r['rag_emergency'] for r in recs)} | "
          f"retries {sum(1 for r in recs if r.get('rag_attempts', 0) > 1)} | median RAG time "
          f"{statistics.median(r['rag_seconds'] for r in recs):.1f}s, closed-book {statistics.median(r['closed_seconds'] for r in recs):.1f}s | "
          f"gold parent in evidence {sum(r['gold_in_evidence'] for r in dev_recs)}/{len(dev_recs)} | LLM calls {llm.calls}", flush=True)
    flagged = Counter(f for r in recs for f in r["flags"])
    print(f"FLAGS: {dict(flagged) or 'none'} | flagged answers {sum(1 for r in recs if r['flags'])}/{len(recs)} | "
          f"ids {[r['id'] for r in recs if r['flags']]}", flush=True)
    for r in recs[:6]:
        print(f"CHAT-RAG [{r['domain']}] {r['question'][:50]} -> {r['rag_answer'][:300]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
