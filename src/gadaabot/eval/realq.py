"""Real-question set (RQ) for GadaaBot 2.0: leakage screen and blind human scoring sheet.

The RQ set holds questions written by 4 native speakers (W01 to W04) on paper or phone, from memory, without
seeing the knowledge base (KB); the first author transcribed them. The writers agreed on a shared topic blueprint
(10 health, 8 education, 2 culture), so topics are stratified by design.

Subcommands
  screen  Pre-stated exclusion rules, applied before any system answers exist:
            R1  the question shares a word 6-gram (normalised Qubee tokens) with any KB child chunk (copied wording);
                shared 5-grams are reported but not excluded (formulaic phrases such as "irraa akkamitti of ittisuu
                dandeenya" are natural Oromo, not leakage)
            R2  token-set Jaccard >= 0.6 with any generated dev question
            R3  token-set Jaccard >= 0.6 with an earlier RQ question (keep the first)
          Writes data/eval/real_questions/devset_candidates.jsonl (the ablation.py input format) and a screen TSV.
          Diagnostic: lexical overlap between each question and its top-1 BM25 child (share of question content
          tokens found in that child), RQ vs the 184 generated test questions. Generated questions are expected to
          overlap more, which is the leakage the RQ set is meant to remove.
  sheet   Blind human scoring workbook from the answered arms: every kept question x 4 systems (native RAG,
          closed-book, NLLB cascade, agent without verifier), labels A to D shuffled per question, citation markers,
          source lists and the health disclaimer stripped. One copy per annotator, same order (for agreement).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SHEET_ARMS = ("rag_naive", "closed", "rag_tt_nllb", "agent_noverify")
DISCLAIMER = re.compile(r"\s*Hubachiisa:.*?deemaa\.?", re.S)
CITE = re.compile(r"\s*\[\d+\](?:\s*\[\d+\])*")


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ngrams(tok: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tok[i:i + n]) for i in range(len(tok) - n + 1)}


def jacc(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cmd_screen(a: argparse.Namespace) -> int:
    from qdrant_client import QdrantClient

    from gadaabot.agent.retriever import COLL
    from gadaabot.eval.extra import Data, bm25_index
    from gadaabot.retrieval.oromo_analyzer import ANALYZERS
    norm = ANALYZERS["norm"]
    raw = read_jsonl(a.raw)
    D = Data(a.root)
    qc = QdrantClient(url=a.qdrant_url, timeout=120)
    texts, nxt = [], None
    while True:
        pts, nxt = qc.scroll(COLL, limit=2000, offset=nxt, with_payload=["text"], with_vectors=False)
        texts += [str(p.payload["text"]) for p in pts]
        if nxt is None:
            break
    kb5: set[tuple[str, ...]] = set()
    kb6: set[tuple[str, ...]] = set()
    for t in texts:
        tt = norm(t)
        kb5 |= ngrams(tt, 5)
        kb6 |= ngrams(tt, 6)
    gen = {i: set(norm(d["question"])) for i, d in D.dev.items()}
    print(f"RQ-SCREEN KB {len(texts)} children, {len(kb5)} distinct word 5-grams | generated dev questions {len(gen)} | RQ raw {len(raw)}", flush=True)
    kept, rows, seen = [], [], []
    for d in raw:
        tok = norm(d["question"])
        hit5 = sorted(" ".join(g) for g in ngrams(tok, 5) & kb5)
        hit6 = sorted(" ".join(g) for g in ngrams(tok, 6) & kb6)
        jg = max(((jacc(set(tok), s), i) for i, s in gen.items()), default=(0.0, ""))
        js = max(((jacc(set(tok), s), i) for i, s in seen), default=(0.0, ""))
        rule = "R1" if hit6 else "R2" if jg[0] >= 0.6 else "R3" if js[0] >= 0.6 else ""
        rows.append({"id": d["id"], "writer": d["writer"], "domain": d["domain"], "excluded": rule, "kb_5gram_hits": len(hit5), "kb_6gram_hits": len(hit6),
                     "kb_5gram_example": hit5[0] if hit5 else "", "max_jacc_generated": round(jg[0], 3), "nearest_generated": jg[1],
                     "max_jacc_rq": round(js[0], 3), "nearest_rq": js[1], "question": d["question"]})
        seen.append((d["id"], set(tok)))
        if not rule:
            kept.append({"id": d["id"], "question": d["question"], "answer": d.get("key_points") or "", "domain": d["domain"],
                         "source": d["writer"], "parent_id": "", "topic": d.get("topic", ""), "question_type": d.get("qtype", ""),
                         "asked_by": d.get("asked_by", ""), "confidence": d.get("conf")})
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with (a.out_dir / "devset_candidates.jsonl").open("w", encoding="utf-8") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    with (a.out_dir / "rq_screen.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["excluded"]:
            print(f"RQ-EXCLUDE {r['id']} {r['excluded']} | 6gram hits {r['kb_6gram_hits']} | 5gram {r['kb_5gram_example']!r} | jacc_gen {r['max_jacc_generated']} "
                  f"| jacc_rq {r['max_jacc_rq']} | {r['question'][:80]}", flush=True)
    print(f"RQ-5GRAM questions sharing a word 5-gram with the KB (reported, not excluded): "
          f"{sum(r['kb_5gram_hits'] > 0 for r in rows)}/{len(rows)}", flush=True)
    by = {}
    for d in kept:
        by[d["domain"]] = by.get(d["domain"], 0) + 1
    print(f"RQ-KEPT {len(kept)}/{len(raw)} | " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())), flush=True)
    # diagnostic: lexical overlap with the top-1 BM25 child, RQ vs generated test questions
    sc = bm25_index(texts, norm)

    def ov(q: str) -> float:
        tok = {t for t in norm(q) if len(t) >= 3}
        s = sc(q)
        if not tok or not s:
            return 0.0
        best = max(s.items(), key=lambda x: x[1])[0]
        return len(tok & set(norm(texts[best]))) / len(tok)
    rq_ov = [ov(d["question"]) for d in kept]
    gen_ov = [ov(D.dev[i]["question"]) for i in D.test]
    import sqlite3
    db = sqlite3.connect(str(a.parents))
    gold_ov = []
    for i in D.test:
        tok = {t for t in norm(D.dev[i]["question"]) if len(t) >= 3}
        row = db.execute("SELECT text FROM parents WHERE parent_id=?", (D.dev[i]["parent_id"],)).fetchone()
        ref = set(norm(row[0])) if row else set()
        if tok and ref:
            gold_ov.append(len(tok & ref) / len(tok))
    print(f"RQ-OVERLAP top-1 BM25 child, share of question content tokens: real {100 * statistics.fmean(rq_ov):.1f}% "
          f"(median {100 * statistics.median(rq_ov):.1f}) | generated test {100 * statistics.fmean(gen_ov):.1f}% "
          f"(median {100 * statistics.median(gen_ov):.1f})" + (f" | generated vs own source passage {100 * statistics.fmean(gold_ov):.1f}%" if gold_ov else ""),
          flush=True)
    print("DONE screen", flush=True)
    return 0


def clean(ans: str) -> str:
    from gadaabot.eval.judge import answer_body
    return DISCLAIMER.sub("", CITE.sub("", answer_body(ans))).strip()


INSTR = [
    ("GadaaBot: blind scoring of answers to real questions", True),
    ("Each question below was written by a native speaker. Each question has 4 answers (A, B, C, D) from different "
     "systems, in random order. You do not know which system wrote which answer; please do not try to guess.", False),
    ("For each answer give two scores, the same scale as the earlier sheet:", True),
    ("correct_1to5: 5 = fully correct and complete; 4 = correct, small gap; 3 = partly correct or missing important "
     "parts; 2 = mostly wrong or off-topic; 1 = wrong, harmful, or no answer. Use the key points and your own knowledge.",
     False),
    ("fluent_1to5: 5 = natural Afaan Oromoo; 3 = understandable but unnatural; 1 = hard to understand or not Oromo.", False),
    ("An honest 'I do not know' answer is not wrong: score correct 2 if the question is answerable and important, "
     "3 if the answer then points to a sensible place to ask.", False),
    ("Rules", True),
    ("- No AI tools, no web search, no discussion with the other annotator until both are finished.", False),
    ("- Questions 1 to 50 (yellow) are the priority. If time runs out, finish those first.", False),
    ("- Save often. Send the file back with your annotator code in the name.", False),
]


def cmd_sheet(a: argparse.Namespace) -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    items = read_jsonl(a.items)
    arms = {x: {r["id"]: r for r in read_jsonl(a.arms_dir / f"arm_{x}.jsonl")} for x in SHEET_ARMS}
    items = [d for d in items if all(d["id"] in arms[x] for x in SHEET_ARMS)]
    rng = random.Random(41)
    order_q = items[:]
    rng.shuffle(order_q)
    plan, key = [], []
    for qn, d in enumerate(order_q, 1):
        o = list(SHEET_ARMS)
        rng.shuffle(o)
        for lab, x in zip("ABCD", o):
            plan.append((qn, d, lab, clean(arms[x][d["id"]]["answer"])))
            key.append({"q_no": qn, "id": d["id"], "label": lab, "arm": x, "writer": d["source"], "domain": d["domain"]})
    wrap, bold = Alignment(wrap_text=True, vertical="top"), Font(bold=True)
    head, prio = PatternFill("solid", fgColor="DDEBF7"), PatternFill("solid", fgColor="FFF2CC")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for ann in a.annotators:
        wb = Workbook()
        ws = wb.active
        ws.title = "Instructions"
        ws.column_dimensions["A"].width = 120
        for r, (t, b) in enumerate(INSTR, 1):
            c = ws.cell(row=r, column=1, value=t)
            c.alignment = wrap
            if b:
                c.font = bold
        sc = wb.create_sheet("Score")
        cols = [("row", 6), ("q_no", 6), ("question", 40), ("key_points", 40), ("answer", 7), ("answer text", 80),
                ("correct_1to5", 10), ("fluent_1to5", 10), ("notes", 35)]
        for j, (n, wdt) in enumerate(cols, 1):
            c = sc.cell(row=1, column=j, value=n)
            c.font, c.fill, c.alignment = bold, head, wrap
            sc.column_dimensions[c.column_letter].width = wdt
        sc.freeze_panes = "A2"
        for r, (qn, d, lab, txt) in enumerate(plan, 2):
            first = lab == "A"
            vals = [r - 1, qn, d["question"] if first else "", d["answer"] if first else "", lab, txt]
            for j, v in enumerate(vals, 1):
                c = sc.cell(row=r, column=j, value=v)
                c.alignment = wrap
                if qn <= 50 and j <= 5:
                    c.fill = prio
        last = len(plan) + 1
        for col in ("G", "H"):
            dv = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
            sc.add_data_validation(dv)
            dv.add(f"{col}2:{col}{last}")
        wb.save(a.out_dir / f"GadaaBot_realq_blind_{ann}.xlsx")
    with (a.out_dir / "realq_key_PRIVATE.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]), delimiter="\t")
        w.writeheader()
        w.writerows(key)
    print(f"RQ-SHEET {len(order_q)} questions x {len(SHEET_ARMS)} systems = {len(plan)} rows | files "
          + ", ".join(f"GadaaBot_realq_blind_{x}.xlsx" for x in a.annotators), flush=True)
    print("DONE sheet", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["screen", "sheet"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--raw", type=Path, help="screen: transcribed RQ jsonl")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:46333")
    ap.add_argument("--parents", type=Path, help="screen: parents.sqlite (source passages of the generated questions)")
    ap.add_argument("--items", type=Path, help="sheet: kept RQ jsonl")
    ap.add_argument("--arms-dir", type=Path, help="sheet: run dir with arm_*.jsonl")
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    a = ap.parse_args()
    return cmd_screen(a) if a.cmd == "screen" else cmd_sheet(a)


if __name__ == "__main__":
    sys.exit(main())
