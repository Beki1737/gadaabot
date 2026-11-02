"""Gate M1 screen: log-likelihood multiple-choice evaluation of one model on Oromo.

Tasks (Oromo slices, files staged by T006):
  belebele   reading comprehension, 4 options, 3-shot (shots excluded from scoring)
  afrimmlu   knowledge, 4 options, 5-shot from the dev split, scored on test
  afrixnli   inference, 3 labels, 5-shot from dev, scored on test
  sib200     topic classification, 7 labels, 5-shot from train, scored on test

Scoring is the lm-evaluation-harness convention: each option is a continuation; we
sum its token log-probabilities given the prompt (acc) and also divide by the
option's UTF-8 byte length (acc_norm). Prompts and options are tokenized
separately and concatenated as token ids, so the option boundary is exact.
The same procedure applies to base and instruct models, so they are comparable.

Run one model per process (vLLM spawns workers; the entry point is guarded):
  python -m gadaabot.eval.m1_screen --model PATH --label NAME --eval-root DIR --out DIR
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from gadaabot.data.qubee import normalize_qubee

SEED = 1234
XNLI_LABELS = ["entailment", "neutral", "contradiction"]


@dataclass
class Item:
    task: str
    uid: str
    context: str             # prompt text for this item (without few-shot prefix)
    options: list[str]        # continuations, each starting with a space
    gold: int
    shots: str = ""           # few-shot prefix
    meta: dict[str, str] = field(default_factory=dict)


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _pick(rec: dict[str, str], *names: str) -> str:
    for n in names:
        for k, v in rec.items():
            if k and k.strip().lower() == n:
                return v or ""
    raise KeyError(f"none of {names} in columns {list(rec)}")


def _split_file(folder: Path, *stems: str) -> Path | None:
    for s in stems:
        p = folder / f"{s}.tsv"
        if p.exists():
            return p
    return None


def n(text: str) -> str:
    return normalize_qubee(text)


# ---------------------------------------------------------------------------
# task builders
# ---------------------------------------------------------------------------
def build_belebele(root: Path, k: int = 3) -> list[Item]:
    path = root / "facebook/belebele/data/gaz_Latn.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def fmt(r: dict[str, str]) -> tuple[str, list[str], int]:
        opts = [n(r[f"mc_answer{i}"]) for i in range(1, 5)]
        ctx = f"Passage: {n(r['flores_passage'])}\nQuestion: {n(r['question'])}\nAnswer:"
        return ctx, [" " + o for o in opts], int(r["correct_answer_num"]) - 1

    rng = random.Random(SEED)
    shot_idx = set(rng.sample(range(len(rows)), k))
    shots = "".join(f"{c}{o[g]}\n\n" for c, o, g in (fmt(rows[i]) for i in sorted(shot_idx)))
    items = []
    for i, r in enumerate(rows):
        if i in shot_idx:
            continue
        c, o, g = fmt(r)
        items.append(Item("belebele", f"bb{i}", c, o, g, shots))
    return items


def build_afrimmlu(root: Path, k: int = 5) -> list[Item]:
    folder = root / "masakhane/afrimmlu/data/orm"
    test, dev = _split_file(folder, "test"), _split_file(folder, "dev", "validation", "val")
    if not test or not dev:
        raise FileNotFoundError(f"afrimmlu splits missing in {folder}: {sorted(p.name for p in folder.glob('*'))}")

    def fmt(r: dict[str, str]) -> tuple[str, list[str], int]:
        raw = _pick(r, "choices", "options")
        try:
            ch = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            ch = [c.strip() for c in raw.split("|")]
        ans = _pick(r, "answer", "label").strip()
        gold = "ABCD".index(ans.upper()) if ans.upper() in "ABCD" and len(ans) == 1 else int(ans)
        ctx = f"Question: {n(_pick(r, 'question'))}\nAnswer:"
        return ctx, [" " + n(str(c)) for c in ch], gold

    dev_rows = _tsv(dev)
    shots = "".join(f"{c}{o[g]}\n\n" for c, o, g in (fmt(r) for r in dev_rows[:k]))
    return [Item("afrimmlu", f"mm{i}", *fmt(r), shots=shots) for i, r in enumerate(_tsv(test))]


def build_afrixnli(root: Path, k: int = 5) -> list[Item]:
    folder = root / "masakhane/afrixnli/data/orm"
    test, dev = _split_file(folder, "test"), _split_file(folder, "dev", "validation", "val")
    if not test or not dev:
        raise FileNotFoundError(f"afrixnli splits missing in {folder}: {sorted(p.name for p in folder.glob('*'))}")

    def fmt(r: dict[str, str]) -> tuple[str, list[str], int]:
        lab = _pick(r, "label").strip().lower()
        gold = XNLI_LABELS.index(lab) if lab in XNLI_LABELS else int(lab)
        ctx = f"Premise: {n(_pick(r, 'premise'))}\nHypothesis: {n(_pick(r, 'hypothesis'))}\nRelation:"
        return ctx, [" " + l for l in XNLI_LABELS], gold

    rng = random.Random(SEED)
    dev_rows = _tsv(dev)
    by_label: dict[int, list[dict[str, str]]] = {}
    for r in dev_rows:
        by_label.setdefault(fmt(r)[2], []).append(r)
    picks = [rng.choice(v) for _, v in sorted(by_label.items())]
    picks += rng.sample(dev_rows, max(0, k - len(picks)))
    shots = "".join(f"{c}{o[g]}\n\n" for c, o, g in (fmt(r) for r in picks[:k]))
    return [Item("afrixnli", f"nli{i}", *fmt(r), shots=shots) for i, r in enumerate(_tsv(test))]


def build_sib200(root: Path, k: int = 5) -> list[Item]:
    folder = root / "Davlan/sib200/data/gaz_Latn"
    test, train = _split_file(folder, "test"), _split_file(folder, "train", "dev")
    if not test or not train:
        raise FileNotFoundError(f"sib200 splits missing in {folder}: {sorted(p.name for p in folder.glob('*'))}")
    tr, te = _tsv(train), _tsv(test)
    cats = sorted({_pick(r, "category").strip() for r in tr + te})

    def fmt(r: dict[str, str]) -> tuple[str, list[str], int]:
        return (f"Text: {n(_pick(r, 'text'))}\nTopic:", [" " + c for c in cats],
                cats.index(_pick(r, "category").strip()))

    rng = random.Random(SEED)
    shots = "".join(f"{c}{o[g]}\n\n" for c, o, g in (fmt(r) for r in rng.sample(tr, k)))
    return [Item("sib200", f"sib{i}", *fmt(r), shots=shots) for i, r in enumerate(te)]


BUILDERS = {"belebele": build_belebele, "afrimmlu": build_afrimmlu,
            "afrixnli": build_afrixnli, "sib200": build_sib200}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default="belebele,afrimmlu,afrixnli,sib200")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(args.model)
    items: list[Item] = []
    for t in args.tasks.split(","):
        try:
            built = BUILDERS[t](args.eval_root)
            items += built
            print(f"TASK {t}: {len(built)} items, {len(built[0].options)} options", flush=True)
        except Exception as exc:  # noqa: BLE001  (one bad schema must not stop the others)
            print(f"TASK {t}: SKIPPED ({type(exc).__name__}: {str(exc)[:300]})", flush=True)

    # one request per (item, option); left-truncate the few-shot prefix if too long
    reqs, index = [], []
    trunc = 0
    for it in items:
        ctx_ids = tok(it.shots + it.context, add_special_tokens=False)["input_ids"]
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        for j, opt in enumerate(it.options):
            opt_ids = tok(opt, add_special_tokens=False)["input_ids"]
            room = args.max_len - 1 - len(bos) - len(opt_ids)
            if len(ctx_ids) > room:
                trunc += 1
            ids = bos + ctx_ids[-room:] + opt_ids
            reqs.append(TokensPrompt(prompt_token_ids=ids))
            index.append((it, j, len(ids) - len(opt_ids), len(opt.encode("utf-8"))))
    print(f"REQUESTS: {len(reqs)} (truncated contexts: {trunc})", flush=True)

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=False, seed=SEED)
    load_s = time.time() - t0
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
    score_s = time.time() - t0 - load_s

    scores: dict[str, list[tuple[float, float]]] = {}
    for out, (it, j, start, nbytes) in zip(outs, index):
        ids = out.prompt_token_ids
        lp = sum(out.prompt_logprobs[p][ids[p]].logprob for p in range(start, len(ids)))
        scores.setdefault(it.uid, [(0.0, 0.0)] * len(it.options))[j] = (lp, lp / max(1, nbytes))

    args.out.mkdir(parents=True, exist_ok=True)
    per_task: dict[str, dict[str, float]] = {}
    with (args.out / f"{args.label}.items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            s = scores[it.uid]
            pred = max(range(len(s)), key=lambda q: s[q][0])
            pred_n = max(range(len(s)), key=lambda q: s[q][1])
            fh.write(json.dumps({"task": it.task, "uid": it.uid, "gold": it.gold,
                                 "pred": pred, "pred_norm": pred_n}) + "\n")
            d = per_task.setdefault(it.task, {"n": 0, "acc": 0, "acc_norm": 0})
            d["n"] += 1
            d["acc"] += int(pred == it.gold)
            d["acc_norm"] += int(pred_n == it.gold)
    summary = {"label": args.label, "model": args.model, "load_s": round(load_s),
               "score_s": round(score_s), "tasks": {}}
    for t, d in per_task.items():
        summary["tasks"][t] = {"n": d["n"], "acc": round(100 * d["acc"] / d["n"], 2),
                               "acc_norm": round(100 * d["acc_norm"] / d["n"], 2)}
        print(f"RESULT {args.label} {t}: n={d['n']} acc={summary['tasks'][t]['acc']} "
              f"acc_norm={summary['tasks'][t]['acc_norm']}", flush=True)
    (args.out / f"{args.label}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"TIMING {args.label}: load {load_s:.0f}s, scoring {score_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
