#!/usr/bin/env bash
# =============================================================================
# T029_diag.sh | GadaaBot 2.0 | Batch B24 | why do v3 answers fall back / abstain?
# -----------------------------------------------------------------------------
# What:   reads T027 + T028 outputs and the judge summary, prints
#         A) failure flags v2 (T027) vs v3 (T028) with the same detector
#         B) every T028 extractive fallback / abstain: trace, evidence scores
#         C) items where the judge says v3 got worse by >= 1 point
#         D) full chat0 / chat1 answers (malaria, diarrhea)
# GPU:    NO (CPU only, < 1 min)
# Run:    bash ~/gadaabot/tasks/T029_diag.sh
# Result: cat ~/gadaabot/reports/T029.txt
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
source "$HOME/gadaabot/scripts/common.sh"
gb_init T029 gb-core
cd "$GB_ROOT"
python - "$GB_ROOT" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, re, sys
from collections import Counter
from pathlib import Path
R = Path(sys.argv[1])
def load(p): return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}
v2, v3 = load(R / "runs/T027_agent/answers.jsonl"), load(R / "runs/T028_agent/answers.jsonl")
js = json.load(open(R / "runs/T028_judge/summary.json", encoding="utf-8"))["runs"]
DEB = re.compile(r"_{3,}|\.{5,}|(?:\b\d+\s*){6,}")
def body(t): return re.split(r"\n\s*(?:Maddeewwan:|Hubachiisa:)", t or "", maxsplit=1)[0].strip()
def flags(q, t, ab):
    if ab: return []
    b = body(t); w = re.findall(r"[^\W\d_]{2,}", b); f = []
    if not w: f.append("only_citations" if re.search(r"\[\d+\]", b) else "empty")
    qq = re.sub(r"\W+", " ", q.lower()).strip(); bb = re.sub(r"\W+", " ", re.sub(r"\[\d+\]", " ", b.lower())).strip()
    if qq and bb and (bb == qq or (bb.startswith(qq) and len(bb) - len(qq) < 15)): f.append("question_echo")
    if DEB.search(b): f.append("debris")
    if 0 < len(w) < 6: f.append("too_short")
    return f
for name, run in (("v2 T027", v2), ("v3 T028", v3)):
    c = Counter(x for r in run.values() for x in flags(r["question"], r["rag_answer"], r["rag_abstained"]))
    fb = sum(1 for r in run.values() if any("extractive" in t for t in r.get("rag_trace", [])))
    print(f"A {name}: flags {dict(c) or 'none'} | flagged {sum(1 for r in run.values() if flags(r['question'], r['rag_answer'], r['rag_abstained']))}/54 | fallback {fb} | abstain {sum(r['rag_abstained'] for r in run.values())}")
print()
for i, r in v3.items():
    tr = r.get("rag_trace", [])
    if r["rag_abstained"] or any("extractive" in t for t in tr):
        a = js.get("T027", {}).get(i, [None])[0]; b = js.get("T028", {}).get(i, [None])[0]
        kind = "ABSTAIN " + (r.get("rag_abstain_reason") or "") if r["rag_abstained"] else "FALLBACK"
        print(f"B {i:<8} {kind} | judge v2 {a} -> v3 {b} | gold_in_ev={int(r.get('gold_in_evidence', 0))} | Q: {r['question'][:90]}")
        for t in tr:
            if t.startswith(("retrieve", "grade", "generate", "verify", "reformulate")):
                print(f"     {t[:260]}")
print()
for i in v3:
    a = js.get("T027", {}).get(i, [None])[0]; b = js.get("T028", {}).get(i, [None])[0]
    if a is not None and b is not None and b <= a - 1:
        print(f"C {i:<8} judge {a:.1f} -> {b:.1f} | Q: {v3[i]['question'][:80]}")
        print(f"     v2: {body(v2[i]['rag_answer'])[:220]!r}")
        print(f"     v3: {body(v3[i]['rag_answer'])[:220]!r}")
print()
for i in ("chat0", "chat1"):
    print(f"D {i} trace: " + " | ".join(t[:200] for t in v3[i].get("rag_trace", [])))
    print(f"D {i} answer: {v3[i]['rag_answer'][:900]!r}")
PY
gb_say "STATUS: PASS"
gb_report_print
