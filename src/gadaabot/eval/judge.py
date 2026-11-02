"""LLM-as-judge for answer correctness (1 to 5), calibrated against native-speaker labels (decision D11).

Why: human scoring does not scale to the held-out evaluation and to RAGAS-style sweeps. An automatic
judge may replace the human only after it agrees with the human labels (target: quadratic weighted
kappa >= 0.6) and reproduces the human system-level difference (RAG minus closed book).

Two judge variants are compared (ablation):
  ref      question + reference answer (dev items) or the judge's own knowledge (chat items, no reference)
  ref_ctx  as `ref` plus the top-3 retrieved parent passages of the RAG run as background
Each answer is scored blind (no system name), the source list and disclaimer are stripped first so
citations are not rewarded, and the judge is sampled `--samples` times (temperature 0.6); the median
is the integer score, the mean the continuous one.

Caveat recorded for the paper: the judge is the generator model itself (AfriqueQwen3.5-9B-it), so
self-preference bias is possible; the human calibration is the control for it.

Usage:
  python -m gadaabot.eval.judge --llm-url URL --parents parents.sqlite \
      --calib-tsv reports/tables/T027_rag_vs_closed.tsv --calib-jsonl runs/T027_agent/answers.jsonl \
      --runs T027=runs/T027_agent/answers.jsonl T028=runs/T028_agent/answers.jsonl \
      --out-dir runs/T028_judge --out-table reports/tables/T028_judge_calibration.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gadaabot.agent.llm import LLM

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
VARIANTS = ("ref", "ref_ctx")
PASSAGES = 3
PASSAGE_CHARS = 1500

SYSTEM = ("You are a strict expert grader of answers written in Afaan Oromo about health and school subjects. "
          "You read Afaan Oromo fluently. You judge factual correctness and completeness only, not style.")
RUBRIC = """Score the ANSWER for correctness on a 1 to 5 scale.
5 = fully correct and complete for what the question asks; no false or misleading claim
4 = correct; small omission or slight imprecision
3 = partly correct; an important point is missing or there is one minor error
2 = mostly wrong or seriously incomplete; contains important errors or misleading advice
1 = wrong, irrelevant, empty, garbled, only repeats the question, only lists sources, or declines to answer
Rules: a claim that contradicts the reference or established medical/scientific knowledge is an error.
Unsafe health advice is at most 2. Do not reward length. Ignore spelling and style."""


def answer_body(text: str) -> str:
    """The answer without the appended source list and disclaimer."""
    return re.split(r"\n\s*(?:Maddeewwan:|Hubachiisa:)", text or "", maxsplit=1)[0].strip()


def build_prompt(variant: str, question: str, answer: str, reference: str, passages: list[str]) -> list[dict[str, str]]:
    parts = [RUBRIC, f"\nQUESTION:\n{question}"]
    if reference.strip():
        parts.append(f"\nREFERENCE ANSWER (written from the source text; may be shorter than a full answer):\n{reference}")
    else:
        parts.append("\nNo reference answer is available; use your own expert knowledge.")
    if variant == "ref_ctx" and passages:
        ctx = "\n\n".join(f"[{i + 1}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(passages[:PASSAGES]))
        parts.append(f"\nBACKGROUND PASSAGES (may or may not contain the answer):\n{ctx}")
    parts.append(f"\nANSWER TO GRADE:\n{answer if answer.strip() else '(empty)'}")
    parts.append('\nReturn only JSON: {"reason": "<one short English sentence>", "score": <integer 1-5>}')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def parse_score(raw: str) -> int | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            v = int(json.loads(m.group(0)).get("score"))
            return v if 1 <= v <= 5 else None
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    m = re.search(r'"score"\s*:\s*([1-5])', raw)
    return int(m.group(1)) if m else None


class Judge:
    def __init__(self, llm: LLM, samples: int = 3, workers: int = 8) -> None:
        self.llm = llm
        self.samples = samples
        self.workers = workers
        self.cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _one(self, variant: str, question: str, answer: str, reference: str, passages: list[str]) -> dict[str, Any]:
        msgs = build_prompt(variant, question, answer, reference, passages)
        got: list[int] = []
        failures = 0
        while len(got) < self.samples and failures < self.samples + 2:
            try:
                s = parse_score(self.llm.chat(msgs, max_tokens=160, sampling=SAMPLING))
            except RuntimeError:
                s = None
            if s is None:
                failures += 1
            else:
                got.append(s)
        if not got:
            return {"score": None, "mean": None, "raw": []}
        return {"score": int(round(statistics.median(got))), "mean": statistics.fmean(got), "raw": got}

    def score_many(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """jobs: dicts with variant, question, answer, reference, passages. Cached by (variant, question, answer)."""
        todo = []
        for j in jobs:
            key = (j["variant"], j["question"], answer_body(j["answer"]))
            if key not in self.cache and key not in {k for k, _ in todo}:
                todo.append((key, j))
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(lambda kj: self._one(kj[0][0], kj[1]["question"], kj[0][2], kj[1]["reference"],
                                                   kj[1]["passages"]), todo))
        for (key, _), r in zip(todo, res):
            self.cache[key] = r
        return [self.cache[(j["variant"], j["question"], answer_body(j["answer"]))] for j in jobs]


# ----------------------------------------------------------------------------------- statistics
def qwk(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic weighted Cohen's kappa for ratings 1..k."""
    n = len(a)
    if n == 0:
        return float("nan")
    o = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        o[x - 1][y - 1] += 1
    ra = [sum(r) for r in o]
    cb = [sum(o[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * o[i][j]
            den += w * ra[i] * cb[j] / n
    return 1.0 - num / den if den else float("nan")


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def boot_ci(fn: Any, n_items: int, reps: int = 2000, seed: int = 11) -> tuple[float, float]:  # noqa: ANN401
    rng = random.Random(seed)
    vals = sorted(v for v in (fn([rng.randrange(n_items) for _ in range(n_items)]) for _ in range(reps)) if v == v)
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


# ----------------------------------------------------------------------------------- data
def load_jsonl(p: Path) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def passages_for(rec: dict[str, Any], db: sqlite3.Connection) -> list[str]:
    ctx = rec.get("rag_contexts")
    if ctx:
        return [str(c) for c in ctx[:PASSAGES]]
    out = []
    for pid in (rec.get("rag_evidence_ids") or [])[:PASSAGES]:
        row = db.execute("SELECT text FROM parents WHERE parent_id=?", (pid,)).fetchone()
        if row:
            out.append(str(row[0]))
    return out


def jobs_for(rec: dict[str, Any], variant: str, db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Two jobs (rag, closed) for one record; both see the same reference and background passages."""
    psg = passages_for(rec, db)
    rag_text = str(rec.get("rag_answer", ""))  # an abstain message is graded as is; the rubric scores it 1
    base = {"variant": variant, "question": rec["question"], "reference": rec.get("gold", "") or "", "passages": psg}
    return [{**base, "answer": rag_text, "system": "rag"},
            {**base, "answer": str(rec.get("closed_book", "")), "system": "closed"}]


def to_int(x: str) -> int | None:
    try:
        v = int(str(x).strip())
        return v if 1 <= v <= 5 else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-url", required=True)
    ap.add_argument("--llm-model", default="gadaabot-gen")
    ap.add_argument("--parents", type=Path, required=True)
    ap.add_argument("--calib-tsv", type=Path, required=True)
    ap.add_argument("--calib-jsonl", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="name=answers.jsonl, scored with the picked variant")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-table", type=Path, required=True)
    args = ap.parse_args()

    llm = LLM(args.llm_url, args.llm_model)
    judge = Judge(llm, args.samples, args.workers)
    db = sqlite3.connect(str(args.parents))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    human = {r["id"]: r for r in csv.DictReader(args.calib_tsv.open(encoding="utf-8"), delimiter="\t")}
    recs = load_jsonl(args.calib_jsonl)
    ids = [i for i in human if i in recs and to_int(human[i]["rag_correct_1to5"]) and to_int(human[i]["closed_correct_1to5"])]
    missing = [i for i in human if i not in ids]
    print(f"CALIB: {len(ids)} scored questions x 2 systems | skipped {missing or 'none'}", flush=True)

    rows_out: list[list[Any]] = []
    calib: dict[str, Any] = {}
    for variant in VARIANTS:
        jobs = [j for i in ids for j in jobs_for(recs[i], variant, db)]
        res = judge.score_many(jobs)
        # per question: (human_rag, human_closed, judge_rag, judge_closed, judge_rag_mean, judge_closed_mean, set)
        per: list[tuple[int, int, int, int, float, float, str]] = []
        dump = []
        for k, i in enumerate(ids):
            jr, jc = res[2 * k], res[2 * k + 1]
            if jr["score"] is None or jc["score"] is None:
                continue
            per.append((to_int(human[i]["rag_correct_1to5"]), to_int(human[i]["closed_correct_1to5"]),  # type: ignore[arg-type]
                        jr["score"], jc["score"], jr["mean"], jc["mean"], human[i]["set"]))
            dump.append(json.dumps({"id": i, "variant": variant, "human_rag": per[-1][0], "human_closed": per[-1][1],
                                    "judge_rag": jr, "judge_closed": jc}, ensure_ascii=False))
        (args.out_dir / f"calib_{variant}.jsonl").write_text("\n".join(dump) + "\n", encoding="utf-8")

        def pooled(idx: list[int], p: list[tuple] = per) -> tuple[list[int], list[int], list[float]]:
            h = [p[t][0] for t in idx] + [p[t][1] for t in idx]
            j = [p[t][2] for t in idx] + [p[t][3] for t in idx]
            jm = [p[t][4] for t in idx] + [p[t][5] for t in idx]
            return h, j, jm

        allidx = list(range(len(per)))
        h, j, jm = pooled(allidx)
        k_all = qwk(j, h)
        k_ci = boot_ci(lambda s: qwk(pooled(s)[1], pooled(s)[0]), len(per))
        rho = spearman(jm, [float(x) for x in h])
        exact = sum(a == b for a, b in zip(h, j)) / len(h)
        within1 = sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(h)
        bias = statistics.fmean(b - a for a, b in zip(h, j))
        hd = statistics.fmean(p[0] - p[1] for p in per)
        jd = statistics.fmean(p[4] - p[5] for p in per)
        jd_ci = boot_ci(lambda s: statistics.fmean(per[t][4] - per[t][5] for t in s), len(per))
        by_set = {}
        for st in ("chat", "dev"):
            sidx = [t for t in allidx if per[t][6] == st]
            if sidx:
                hs, js, _ = pooled(sidx)
                by_set[st] = round(qwk(js, hs), 3)
        by_sys = {"rag": round(qwk([p[2] for p in per], [p[0] for p in per]), 3),
                  "closed": round(qwk([p[3] for p in per], [p[1] for p in per]), 3)}
        calib[variant] = {"n_questions": len(per), "qwk": round(k_all, 3), "qwk_ci": [round(x, 3) for x in k_ci],
                          "spearman": round(rho, 3), "exact": round(exact, 3), "within1": round(within1, 3),
                          "bias": round(bias, 3), "qwk_by_set": by_set, "qwk_by_system": by_sys,
                          "human_rag_minus_closed": round(hd, 3), "judge_rag_minus_closed": round(jd, 3),
                          "judge_diff_ci": [round(x, 3) for x in jd_ci],
                          "human_means": [round(statistics.fmean(p[0] for p in per), 2), round(statistics.fmean(p[1] for p in per), 2)],
                          "judge_means": [round(statistics.fmean(p[4] for p in per), 2), round(statistics.fmean(p[5] for p in per), 2)]}
        c = calib[variant]
        print(f"JUDGE-CALIB {variant:<7} n={c['n_questions']} QWK {c['qwk']:.3f} [{k_ci[0]:.2f}, {k_ci[1]:.2f}] | "
              f"by set {by_set} | by system {by_sys} | Spearman {rho:.3f} | exact {exact:.2f} | within-1 {within1:.2f} | "
              f"bias {bias:+.2f} | RAG-closed human {hd:+.2f} judge {jd:+.2f} [{jd_ci[0]:+.2f}, {jd_ci[1]:+.2f}] | "
              f"means human {c['human_means']} judge {c['judge_means']}", flush=True)
        rows_out.append([variant, c["n_questions"], c["qwk"], f"{k_ci[0]:.3f}", f"{k_ci[1]:.3f}", c["spearman"], c["exact"],
                         c["within1"], c["bias"], by_set.get("chat", ""), by_set.get("dev", ""), by_sys["rag"],
                         by_sys["closed"], c["human_rag_minus_closed"], c["judge_rag_minus_closed"]])

    pick = max(VARIANTS, key=lambda v: (calib[v]["qwk"] if calib[v]["qwk"] == calib[v]["qwk"] else -9))
    passed = calib[pick]["qwk"] >= 0.6 and (calib[pick]["judge_rag_minus_closed"] > 0) == (calib[pick]["human_rag_minus_closed"] > 0)
    print(f"JUDGE-PICK {pick} | D11 gate (QWK >= 0.6 and same sign of RAG-closed): {'PASS' if passed else 'FAIL'}", flush=True)

    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    with args.out_table.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["variant", "n_questions", "qwk", "qwk_lo", "qwk_hi", "spearman", "exact", "within1", "bias",
                    "qwk_chat", "qwk_dev", "qwk_rag", "qwk_closed", "human_rag_minus_closed", "judge_rag_minus_closed"])
        w.writerows(rows_out)

    # --- automatic scoring of agent runs with the picked variant (relative comparison, same judge) ---
    run_scores: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        rr = load_jsonl(Path(path))
        rids = list(rr)
        res = judge.score_many([j for i in rids for j in jobs_for(rr[i], pick, db)])
        run_scores[name] = {}
        for k, i in enumerate(rids):
            a, b = res[2 * k]["mean"], res[2 * k + 1]["mean"]
            if a is not None and b is not None:
                run_scores[name][i] = (a, b)
        sc = run_scores[name]
        dev = [i for i in sc if rr[i]["set"] == "dev"]
        print(f"JUDGE-RUN {name:<6} n={len(sc)} | RAG {statistics.fmean(v[0] for v in sc.values()):.2f} closed "
              f"{statistics.fmean(v[1] for v in sc.values()):.2f} | chat RAG "
              f"{statistics.fmean(sc[i][0] for i in sc if rr[i]['set'] == 'chat'):.2f} | dev RAG "
              f"{statistics.fmean(sc[i][0] for i in dev):.2f}", flush=True)
    names = list(run_scores)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            common = [i for i in run_scores[a] if i in run_scores[b]]
            if not common:
                continue
            d = [run_scores[b][i][0] - run_scores[a][i][0] for i in common]
            ci = boot_ci(lambda s, d=d: statistics.fmean(d[t] for t in s), len(d))
            print(f"JUDGE-DIFF RAG {b} - {a}: {statistics.fmean(d):+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] over {len(common)} "
                  f"questions | wins {sum(x > 0 for x in d)} ties {sum(x == 0 for x in d)} losses {sum(x < 0 for x in d)}",
                  flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps({"calibration": calib, "pick": pick, "d11_pass": passed,
                                                            "runs": {k: {i: list(v) for i, v in s.items()} for k, s in run_scores.items()},
                                                            "llm_calls": llm.calls}, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print(f"JUDGE-DONE LLM calls {llm.calls}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
