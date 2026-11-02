"""Draft the GadaaBot domain dev set (retrieval Gate R1 and later RAGAS) for native-speaker review.

Method: stratified sample of knowledge-base parents (health / education, every source),
then AfriqueQwen3.5-9B-it writes ONE question a real user could ask that this passage
answers, plus a short answer drawn only from the passage. Automatic filters drop items
that are not Oromo, copy the passage verbatim, are too short or too long, or whose
answer is not supported by the passage. A human then accepts, edits or rejects each item.

Known bias, reported in the paper: questions written from a passage share its wording
more than real user questions do; the review step asks the reviewer to rephrase
questions the way a user would actually ask.

Output: devset_candidates.jsonl and devset_review.tsv (columns to fill: keep y/n,
question_edited, answer_edited, notes).

Usage: python -m gadaabot.eval.devset_gen --parents parents.sqlite --model DIR --out DIR --n 420
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from gadaabot.data.qubee import words
from gadaabot.kb.kb_acquire import is_oromo_sent

SYSTEM = ("You write evaluation questions for an Afaan Oromoo health and education assistant. "
          "Write ONLY in Afaan Oromoo (Qubee script).")
USER_TMPL = """Passage (source: {source}, topic: {domain}):
\"\"\"{text}\"\"\"

Task: write ONE question that an ordinary Oromo speaker (a student, a parent or a patient) might ask, which this passage answers. Use everyday wording, do not copy a sentence from the passage, and do not mention "the passage". Then give a short, correct answer (1 to 3 sentences) using only facts in the passage.
Return only JSON: {{"question": "...", "answer": "..."}}"""

# per-stratum quotas (proportions of --n); every source and both domains are represented
QUOTA = {("health", "mdh"): .12, ("health", "tico19"): .08, ("health", "webdomain"): .22, ("health", "oeb"): .05,
         ("health", "wikipedia"): .03, ("education", "textbooks"): .12, ("education", "oeb"): .10,
         ("education", "webdomain"): .13, ("encyclopedia", "wikipedia"): .15}


def prefix4(text: str) -> set[str]:
    return {w[:4] for w in words(text) if len(w) > 3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    db = sqlite3.connect(str(args.parents))
    rows = db.execute("SELECT parent_id, source, tier, domain, title, url, n_words, text FROM parents "
                      "WHERE n_words BETWEEN 80 AND 360").fetchall()
    strata: dict[tuple[str, str], list[tuple]] = {}
    for r in rows:
        strata.setdefault((r[3], r[1]), []).append(r)
    picked = []
    for key, frac in QUOTA.items():
        pool = strata.get(key, [])
        k = min(len(pool), round(frac * args.n))
        picked += rng.sample(pool, k)
        print(f"STRATUM {key[0]:<12} {key[1]:<10} pool {len(pool):>6,} picked {k}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(args.model))
    prompts = []
    for r in picked:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(source=r[1], domain=r[3], text=r[7])}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_mem, seed=7)
    sp = SamplingParams(temperature=0.4, top_p=0.9, top_k=20, presence_penalty=0.5, max_tokens=300, seed=7)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"GENERATED {len(outs)} in {time.time() - t0:.0f}s", flush=True)

    kept, why = [], Counter()
    for r, o in zip(picked, outs):
        raw = o.outputs[0].text
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            qa = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            qa = None
        if not qa or not isinstance(qa.get("question"), str) or not isinstance(qa.get("answer"), str):
            why["bad_json"] += 1
            continue
        q, a, passage = qa["question"].strip(), qa["answer"].strip(), r[7]
        nq = len(words(q))
        if not 4 <= nq <= 30:
            why["length"] += 1
            continue
        if not is_oromo_sent(q):
            why["not_oromo"] += 1
            continue
        if q.lower().rstrip("?") in passage.lower():
            why["copied"] += 1
            continue
        ap4, pp4 = prefix4(a), prefix4(passage)
        if not ap4 or len(ap4 & pp4) / len(ap4) < 0.5:
            why["answer_unsupported"] += 1
            continue
        kept.append({"id": f"dev{len(kept):04d}", "parent_id": r[0], "source": r[1], "tier": r[2], "domain": r[3],
                     "title": r[4], "url": r[5], "question": q, "answer": a,
                     "q_overlap": round(len(prefix4(q) & pp4) / max(1, len(prefix4(q))), 2)})
    print(f"FILTER: kept {len(kept)} of {len(outs)} | dropped {dict(why)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "devset_candidates.jsonl").open("w", encoding="utf-8") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    with (args.out / "devset_review.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "domain", "source", "question", "answer", "passage_start", "keep_y_n", "question_edited",
                    "answer_edited", "notes"])
        texts = dict(db.execute("SELECT parent_id, text FROM parents").fetchall())
        for k in kept:
            w.writerow([k["id"], k["domain"], k["source"], k["question"], k["answer"],
                        texts[k["parent_id"]][:700].replace("\n", " | ").replace("\t", " "), "", "", "", ""])
    by = Counter((k["domain"], k["source"]) for k in kept)
    print("KEPT BY STRATUM: " + "; ".join(f"{d}/{s} {n}" for (d, s), n in sorted(by.items())), flush=True)
    for k in kept[:12]:
        print(f"SAMPLE [{k['domain']}/{k['source']}] Q: {k['question']} | A: {k['answer'][:140]}", flush=True)
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    sys.stdout.flush()
    os._exit(rc)
