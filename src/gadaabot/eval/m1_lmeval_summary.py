"""Gate M1 summary from lm-evaluation-harness per-sample logs.

For each model and task family (belebele, afrimmlu, sib, afrixnli) the family
score is the MEAN over its prompt variants (no best-prompt selection). Per-item
correctness is first averaged across prompt variants; models are then compared
with a paired bootstrap on those item-level means against a reference model.

Usage: python -m gadaabot.eval.m1_lmeval_summary --root RUNS --ref AfriqueQwen-8B
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FAMILIES = {"belebele": ("belebele", "acc"), "afrimmlu": ("afrimmlu", "acc"),
            "sib": ("sib", "acc"), "afrixnli": ("afrixnli", "acc")}
PRIMARY = ("belebele", "afrimmlu")
REPS, SEED = 2000, 7


def family_of(task: str) -> str | None:
    t = task.lower()
    if "mgsm" in t:
        return None
    for fam, (key, _) in FAMILIES.items():
        if key in t:
            return fam
    return None


def load_model(dir_: Path) -> dict[str, dict[str, dict[int, float]]]:
    """family -> task -> doc_id -> correctness (0/1)."""
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
    for f in dir_.rglob("samples_*.jsonl"):
        m = re.match(r"samples_(.+?)_\d{4}-\d{2}-\d{2}T", f.name)
        task = m.group(1) if m else f.stem[len("samples_"):]
        fam = family_of(task)
        if fam is None:
            continue
        metric = FAMILIES[fam][1]
        vals: dict[int, float] = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if metric in r:
                vals[int(r["doc_id"])] = float(r[metric])
        if vals:
            out[fam][task] = vals
    return out


def item_means(tasks: dict[str, dict[int, float]]) -> dict[int, float]:
    docs = set.intersection(*(set(v) for v in tasks.values()))
    return {d: sum(v[d] for v in tasks.values()) / len(tasks) for d in docs}


def boot_ci(vals: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(vals)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(REPS))
    return 100 * b[int(.025 * REPS)], 100 * b[int(.975 * REPS)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)

    models = {d.name: load_model(d) for d in sorted(args.root.iterdir()) if d.is_dir()}
    models = {k: v for k, v in models.items() if v}
    if args.ref not in models:
        print(f"reference {args.ref} missing; have {list(models)}")
        return 1
    rows = []
    for name, fams in models.items():
        row: dict[str, object] = {"model": name}
        prim = []
        for fam in FAMILIES:
            if fam not in fams:
                continue
            per_prompt = [100 * sum(v.values()) / len(v) for v in fams[fam].values()]
            im = item_means(fams[fam])
            lo, hi = boot_ci(list(im.values()), rng)
            mean = statistics.mean(per_prompt)
            row[fam] = {"mean": round(mean, 2), "ci95": [round(lo, 1), round(hi, 1)],
                        "prompt_min": round(min(per_prompt), 1), "prompt_max": round(max(per_prompt), 1),
                        "n_prompts": len(per_prompt), "n_items": len(im)}
            if fam in PRIMARY:
                prim.append(mean)
            if name != args.ref and fam in models[args.ref]:
                ref_im = item_means(models[args.ref][fam])
                common = sorted(set(im) & set(ref_im))
                d = [im[c] - ref_im[c] for c in common]
                b = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(REPS))
                row[fam]["vs_ref"] = {"diff": round(100 * sum(d) / len(d), 2),  # type: ignore[index]
                                      "ci95": [round(100 * b[int(.025 * REPS)], 1), round(100 * b[int(.975 * REPS)], 1)],
                                      "p_le_0": round(sum(x <= 0 for x in b) / REPS, 4)}
        row["primary_mean"] = round(statistics.mean(prim), 2) if len(prim) == len(PRIMARY) else None
        rows.append(row)
    rows.sort(key=lambda r: -(r["primary_mean"] or -1))  # type: ignore[operator]
    (args.root / "m1_summary.json").write_text(json.dumps(rows, indent=2))

    print(f"GATE M1 ranking by primary = mean(Belebele, AfriMMLU); family = mean over prompt variants")
    print(f"paired bootstrap vs {args.ref}: diff [95% CI], one-sided p(diff<=0)")
    for r in rows:
        print(f"  {r['model']:<26} primary {r['primary_mean']}")
        for fam in FAMILIES:
            if fam in r:
                f = r[fam]  # type: ignore[index]
                s = (f"      {fam:<9} {f['mean']:6.2f} [{f['ci95'][0]},{f['ci95'][1]}] "
                     f"prompts {f['prompt_min']}-{f['prompt_max']} (n={f['n_items']})")
                if "vs_ref" in f:
                    v = f["vs_ref"]
                    s += f"  vs ref {v['diff']:+.2f} [{v['ci95'][0]:+},{v['ci95'][1]:+}] p={v['p_le_0']}"
                print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
