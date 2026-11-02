"""Tokenizer audit on parallel Oromo / English text (FLORES+ devtest).

For every tokenizer it reports, on the same 1,012 parallel sentences:
  fertility       tokens per word (word = regex word incl. internal hudhaa)
  premium         Oromo fertility / English fertility (same tokenizer)
  chars/token     characters per token (higher = better compression)
  STRR            single-token retention rate: share of word types kept as one token
  raw-vs-norm     fertility on raw text vs Qubee-normalized text (hudhaa effect)
95% CIs come from a sentence-level bootstrap (1,000 resamples, fixed seed).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from gadaabot.data.qubee import normalize_qubee, words


def load_flores(path: Path) -> list[str]:
    """Read a FLORES+ jsonl file (field 'text')."""
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def n_tokens(tok, text: str) -> int:  # type: ignore[no-untyped-def]
    return len(tok(text, add_special_tokens=False)["input_ids"])


def bootstrap_ci(num: list[int], den: list[int], reps: int = 1000, seed: int = 13) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(num)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(num[i] for i in idx) / max(1, sum(den[i] for i in idx)))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orm", type=Path, required=True, help="FLORES+ gaz_Latn devtest jsonl")
    ap.add_argument("--eng", type=Path, required=True, help="FLORES+ eng_Latn devtest jsonl")
    ap.add_argument("--tokenizers", type=Path, required=True, help="json: {label: path}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    orm_raw = load_flores(args.orm)
    eng = load_flores(args.eng)
    if len(orm_raw) != len(eng):
        print(f"WARNING: sentence counts differ (orm {len(orm_raw)}, eng {len(eng)})")
    orm = [normalize_qubee(s) for s in orm_raw]
    changed = sum(a != b for a, b in zip(orm_raw, orm))
    print(f"DATA: {len(orm)} Oromo / {len(eng)} English sentences; normalization changed {changed} Oromo sentences")

    w_orm = [len(words(s)) for s in orm]
    w_eng = [len(words(s)) for s in eng]
    orm_types = sorted({w for s in orm for w in words(s)})

    rows = []
    for label, path in json.loads(args.tokenizers.read_text()).items():
        try:
            tok = AutoTokenizer.from_pretrained(path)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {label}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        t_orm = [n_tokens(tok, s) for s in orm]
        t_raw = [n_tokens(tok, s) for s in orm_raw]
        t_eng = [n_tokens(tok, s) for s in eng]
        fert_o = sum(t_orm) / sum(w_orm)
        fert_e = sum(t_eng) / sum(w_eng)
        lo, hi = bootstrap_ci(t_orm, w_orm)
        single = sum(n_tokens(tok, " " + w) == 1 for w in orm_types) / len(orm_types)
        row = {
            "tokenizer": label,
            "vocab": len(tok),
            "fertility_orm": round(fert_o, 3),
            "fertility_orm_ci95": [round(lo, 3), round(hi, 3)],
            "fertility_eng": round(fert_e, 3),
            "premium": round(fert_o / fert_e, 3),
            "chars_per_token_orm": round(sum(len(s) for s in orm) / sum(t_orm), 3),
            "strr_orm": round(single, 3),
            "fertility_orm_raw": round(sum(t_raw) / sum(w_orm), 3),
            "tokens_orm_total": sum(t_orm),
        }
        rows.append(row)
        print(f"DONE {label}")

    rows.sort(key=lambda r: r["fertility_orm"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    with args.out.with_suffix(".csv").open("w", encoding="utf-8") as fh:
        keys = [k for k in rows[0] if k != "fertility_orm_ci95"] if rows else []
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")

    print(f"{'tokenizer':<26}{'vocab':>8}{'fert_orm':>9}{'95% CI':>15}{'fert_eng':>9}{'premium':>8}"
          f"{'chr/tok':>8}{'STRR':>7}{'raw_fert':>9}")
    for r in rows:
        ci = f"[{r['fertility_orm_ci95'][0]:.2f},{r['fertility_orm_ci95'][1]:.2f}]"
        print(f"{r['tokenizer']:<26}{r['vocab']:>8}{r['fertility_orm']:>9.3f}{ci:>15}{r['fertility_eng']:>9.3f}"
              f"{r['premium']:>8.2f}{r['chars_per_token_orm']:>8.2f}{r['strr_orm']:>7.3f}{r['fertility_orm_raw']:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
