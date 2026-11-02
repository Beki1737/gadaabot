"""Aggregate Gate M1 results: per-task accuracy with 95% bootstrap CIs and paired
bootstrap differences against a reference model.

Usage: python -m gadaabot.eval.m1_summarize --dir RUNS --ref AfriqueQwen-8B
Writes RUNS/m1_table.csv and RUNS/m1_table.json; prints a compact table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPS = 2000
SEED = 99


def load(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """task -> uid -> (correct acc, correct acc_norm)."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.setdefault(r["task"], {})[r["uid"]] = (int(r["pred"] == r["gold"]), int(r["pred_norm"] == r["gold"]))
    return out


def ci(values: list[int], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    boots = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(REPS))
    return 100 * boots[int(0.025 * REPS)], 100 * boots[int(0.975 * REPS)]


def paired(a: list[int], b: list[int], rng: random.Random) -> tuple[float, float, float, float]:
    """Mean difference a-b (points), 95% CI, one-sided p for H0: diff <= 0."""
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(REPS))
    p = sum(v <= 0 for v in boots) / REPS
    return 100 * sum(d) / n, 100 * boots[int(0.025 * REPS)], 100 * boots[int(0.975 * REPS)], p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", choices=["acc", "acc_norm"], default="acc_norm")
    args = ap.parse_args()
    rng = random.Random(SEED)
    k = 0 if args.metric == "acc" else 1

    runs = {p.name[: -len(".items.jsonl")]: load(p) for p in sorted(args.dir.glob("*.items.jsonl"))}
    if args.ref not in runs:
        print(f"reference {args.ref} not found among {list(runs)}")
        return 1
    tasks = sorted({t for r in runs.values() for t in r})
    rows = []
    for label, res in runs.items():
        row: dict[str, object] = {"model": label}
        means = []
        for t in tasks:
            if t not in res:
                continue
            uids = sorted(res[t])
            v = [res[t][u][k] for u in uids]
            lo, hi = ci(v, rng)
            m = 100 * sum(v) / len(v)
            means.append(m)
            row[t] = f"{m:.1f} [{lo:.1f},{hi:.1f}]"
            if label != args.ref and t in runs[args.ref]:
                common = [u for u in uids if u in runs[args.ref][t]]
                diff, dlo, dhi, p = paired([res[t][u][k] for u in common],
                                           [runs[args.ref][t][u][k] for u in common], rng)
                row[f"{t}_vs_ref"] = f"{diff:+.1f} [{dlo:+.1f},{dhi:+.1f}] p={p:.3f}"
        row["mean"] = round(sum(means) / len(means), 2) if means else None
        rows.append(row)
    rows.sort(key=lambda r: -(r["mean"] or 0))  # type: ignore[operator]

    (args.dir / "m1_table.json").write_text(json.dumps(rows, indent=2))
    cols = ["model", "mean"] + tasks + [f"{t}_vs_ref" for t in tasks]
    with (args.dir / "m1_table.csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(f"\"{r.get(c, '')}\"" for c in cols) + "\n")

    print(f"GATE M1 ({args.metric}, 95% bootstrap CI; diffs paired vs {args.ref}, one-sided p)")
    for r in rows:
        print(f"  {r['model']:<28} mean {r['mean']}")
        for t in tasks:
            if t in r:
                extra = f"   vs ref {r[f'{t}_vs_ref']}" if f"{t}_vs_ref" in r else ""
                print(f"      {t:<10} {r[t]}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
