"""Final paper analysis on the native-speaker-reviewed held-out set (CPU only).

Inputs
  runs/T030_ablation/arm_*.jsonl, judge.jsonl, blind_key_v2.json   (T030 to T031 outputs)
  devset_review.tsv      keep_y_n per dev item (native-speaker review)
  blind_human_v2.tsv     human correctness for 30 questions x 5 systems (blind)
Outputs (out-dir)
  final_arms.tsv    judge score per arm on reviewed held-out items, 95% CI, abstain, confident-wrong
  final_pairs.tsv   planned paired differences, bootstrap CI, Holm-adjusted p
  human_arms.tsv    human correctness per system (blind sheet), 95% CI
  human_pairs.tsv   human paired differences for the main claims
  judge_human.tsv   judge vs human agreement on the 150 blind rows (QWK, Spearman, bias), overall,
                    per system, native vs translated outputs (is the judge biased on MT output?)
  devset_stats.tsv  review keep rate by domain and source (dataset table)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from gadaabot.eval.judge import boot_ci, qwk, spearman

PAIRS = [("rag_naive", "closed"), ("agent_full", "closed"), ("closed", "closed_tt_nllb"), ("closed", "closed_tt_self"),
         ("rag_naive", "rag_tt_nllb"), ("rag_naive", "rag_tt_self"), ("rag_tt_self", "rag_tt_nllb"),
         ("agent_noverify", "rag_naive"), ("agent_full", "agent_noverify"), ("agent_full", "rag_naive"),
         ("closed@noisy", "closed"), ("rag_naive@noisy", "rag_naive"), ("agent_noverify@noisy", "agent_noverify"),
         ("agent_noverify@noisy", "rag_naive@noisy"), ("rag_naive@nogold", "closed"),
         ("agent_noverify@nogold", "rag_naive@nogold"), ("agent_full@nogold", "agent_noverify@nogold"),
         ("rag_naive@nogold", "rag_naive")]
HUMAN_PAIRS = [("rag_naive", "closed"), ("rag_naive", "rag_tt_nllb"), ("closed", "closed_tt_self"),
               ("agent_full", "rag_naive"), ("rag_tt_nllb", "closed")]
TRANSLATED = {"closed_tt_nllb", "closed_tt_self", "rag_tt_nllb", "rag_tt_self"}


def holm(p: list[float]) -> list[float]:
    order = sorted(range(len(p)), key=lambda i: p[i])
    adj, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def boot_p(d: list[float], reps: int = 4000, seed: int = 13) -> float:
    rng = random.Random(seed)
    n = len(d)
    m = [statistics.fmean(d[rng.randrange(n)] for _ in range(n)) for _ in range(reps)]
    return min(1.0, 2 * min(sum(x <= 0 for x in m) / reps, sum(x >= 0 for x in m) / reps))


def mean_ci(v: list[float]) -> tuple[float, float, float]:
    lo, hi = boot_ci(lambda s: statistics.fmean(v[t] for t in s), len(v))
    return round(statistics.fmean(v), 3), round(lo, 3), round(hi, 3)


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def paired(sc: dict[str, dict[str, float]], pairs: list[tuple[str, str]], ids: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for a, b in pairs:
        if a not in sc or b not in sc:
            continue
        common = [i for i in sc[a] if i in sc[b] and (ids is None or i in ids)]
        if len(common) < 5:
            continue
        d = [sc[a][i] - sc[b][i] for i in common]
        m, lo, hi = mean_ci(d)
        out.append({"a": a, "b": b, "n": len(common), "diff": m, "lo": lo, "hi": hi, "p": round(boot_p(d), 4),
                    "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)})
    for r, ph in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = round(ph, 4)
    return out


def main() -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--blind", type=Path, default=None, help="scored blind sheet; omit to skip the human section")
    ap.add_argument("--dev", type=Path, required=True, help="devset_candidates.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    review = {r["id"]: r for r in csv.DictReader(args.review.open(encoding="utf-8"), delimiter="\t")}
    keep = {i for i, r in review.items() if r["keep_y_n"].strip().lower().startswith("y")}
    dev = {json.loads(l)["id"]: json.loads(l) for l in args.dev.read_text(encoding="utf-8").splitlines() if l.strip()}

    # ---------- dataset table: keep rate by domain x source
    by: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for i, r in review.items():
        k = (r["domain"], r["source"])
        by[k][0] += 1
        by[k][1] += int(i in keep)
    ds = [{"domain": d, "source": s, "candidates": n, "kept": k, "keep_pct": round(100 * k / n, 1)}
          for (d, s), (n, k) in sorted(by.items())]
    ds.append({"domain": "all", "source": "all", "candidates": len(review), "kept": len(keep),
               "keep_pct": round(100 * len(keep) / len(review), 1)})
    write(out / "devset_stats.tsv", ds)
    for r in ds:
        print(f"DEVSET {r['domain']:<12} {r['source']:<10} kept {r['kept']:>3}/{r['candidates']:<3} ({r['keep_pct']}%)")

    # ---------- arms on reviewed held-out items
    abl = args.ablation_dir
    res: dict[str, dict[str, dict[str, Any]]] = {}
    for f in sorted(abl.glob("arm_*.jsonl")):
        arm = f.stem[4:]
        res[arm] = {r["id"]: r for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}
    sc: dict[str, dict[str, float]] = defaultdict(dict)
    for l in (abl / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("mean") is not None:
                sc[r["arm"]][r["id"]] = r["mean"]
    held = set(res.get("closed", {}))
    final = held & keep
    print(f"FINAL SET: {len(final)} reviewed held-out questions (held-out {len(held)}, kept {len(keep)})")
    order = ["closed", "closed_tt_nllb", "closed_tt_self", "rag_naive", "rag_tt_nllb", "rag_tt_self", "agent_noverify",
             "agent_full", "closed@noisy", "rag_naive@noisy", "agent_noverify@noisy", "rag_naive@nogold",
             "agent_noverify@nogold", "agent_full@nogold"]
    rows = []
    for a in [x for x in order if x in res and sc.get(x)]:
        ids = sorted(i for i in sc[a] if i in final)
        m, lo, hi = mean_ci([sc[a][i] for i in ids])
        ab = [i for i in ids if res[a][i].get("abstained")]
        ans = [i for i in ids if i not in ab]
        dom = defaultdict(list)
        for i in ids:
            dom[dev[i]["domain"]].append(sc[a][i])
        row = {"arm": a, "n": len(ids), "judge": m, "lo": lo, "hi": hi,
               **{f"dom_{k}": round(statistics.fmean(v), 2) for k, v in sorted(dom.items())},
               "abstain_pct": round(100 * len(ab) / len(ids), 1),
               "confident_wrong_pct": round(100 * sum(sc[a][i] <= 2 for i in ans) / len(ids), 1),
               "answered_judge": round(statistics.fmean(sc[a][i] for i in ans), 3) if ans else "",
               "median_s": round(statistics.median(res[a][i].get("seconds", 0.0) for i in ids), 2)}
        rows.append(row)
        print("FINAL-ARM " + " | ".join(f"{k} {v}" for k, v in row.items()))
    write(out / "final_arms.tsv", rows)
    pr = paired(sc, PAIRS, final)
    write(out / "final_pairs.tsv", pr)
    for r in pr:
        print(f"FINAL-PAIR {r['a']} - {r['b']}: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] n={r['n']} "
              f"p_holm={r['p_holm']:.4f} W/T/L {r['wins']}/{r['ties']}/{r['losses']}")

    hrows: list[dict[str, Any]] = []
    hp: list[dict[str, Any]] = []
    jh: list[dict[str, Any]] = []
    if args.blind is None:
        (out / "summary.json").write_text(json.dumps({"final_n": len(final), "arms": rows, "pairs": pr, "devset": ds},
                                                     indent=1), encoding="utf-8")
        print("DONE (no human section)")
        return 0
    # ---------- blind human sheet
    key = json.loads((abl / "blind_key_v2.json").read_text(encoding="utf-8"))
    hum: dict[str, dict[str, float]] = defaultdict(dict)
    pairs_jh: list[tuple[int, int, str, str]] = []  # human, judge(int), arm, id
    judge_int: dict[tuple[str, str], int] = {}
    for l in (abl / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("score") is not None:
                judge_int[(r["id"], r["arm"])] = int(r["score"])
    for r in csv.DictReader(args.blind.open(encoding="utf-8"), delimiter="\t"):
        k = key.get(str(r["row"]).strip())
        v = r["correct_1to5"].strip()
        if not k or not v.isdigit():
            continue
        hum[k["arm"]][k["id"]] = float(v)
        if (k["id"], k["arm"]) in judge_int:
            pairs_jh.append((int(v), judge_int[(k["id"], k["arm"])], k["arm"], k["id"]))
    qids = {i for v in hum.values() for i in v}
    print(f"HUMAN: {sum(len(v) for v in hum.values())} scored rows, {len(qids)} questions "
          f"({len(qids & keep)} kept in review)")
    for a in [x for x in order if x in hum]:
        v = list(hum[a].values())
        m, lo, hi = mean_ci(v)
        hrows.append({"arm": a, "n": len(v), "human": m, "lo": lo, "hi": hi,
                      "judge_same_items": round(statistics.fmean(sc[a][i] for i in hum[a] if i in sc[a]), 3)})
        print(f"HUMAN-ARM {a:<16} n={len(v)} human {m:.2f} [{lo:.2f}, {hi:.2f}] | judge on same items {hrows[-1]['judge_same_items']:.2f}")
    write(out / "human_arms.tsv", hrows)
    hp = paired(hum, HUMAN_PAIRS)
    hpk = paired(hum, HUMAN_PAIRS, keep)  # only questions the review kept (reference answer trusted)
    for r in hp:
        r["subset"] = "all"
    for r in hpk:
        r["subset"] = "kept"
    write(out / "human_pairs.tsv", hp + hpk)
    for r in hp + hpk:
        print(f"HUMAN-PAIR[{r['subset']}] {r['a']} - {r['b']}: {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] n={r['n']} "
              f"p_holm={r['p_holm']:.4f} W/T/L {r['wins']}/{r['ties']}/{r['losses']}")

    groups = {"all": pairs_jh, "native": [p for p in pairs_jh if p[2] not in TRANSLATED],
              "translated": [p for p in pairs_jh if p[2] in TRANSLATED],
              "kept_only": [p for p in pairs_jh if p[3] in keep]}
    for a in sorted({p[2] for p in pairs_jh}):
        groups[a] = [p for p in pairs_jh if p[2] == a]
    for g, ps in groups.items():
        if len(ps) < 5:
            continue
        h, j = [p[0] for p in ps], [p[1] for p in ps]
        lo, hi = boot_ci(lambda s, h=h, j=j: qwk([j[t] for t in s], [h[t] for t in s]), len(ps))
        row = {"group": g, "n": len(ps), "qwk": round(qwk(j, h), 3), "qwk_lo": round(lo, 3), "qwk_hi": round(hi, 3),
               "spearman": round(spearman([float(x) for x in j], [float(x) for x in h]), 3),
               "exact": round(sum(a == b for a, b in zip(h, j)) / len(ps), 3),
               "within1": round(sum(abs(a - b) <= 1 for a, b in zip(h, j)) / len(ps), 3),
               "bias_judge_minus_human": round(statistics.fmean(b - a for a, b in zip(h, j)), 3)}
        jh.append(row)
        print("JUDGE-HUMAN " + " | ".join(f"{k} {v}" for k, v in row.items()))
    write(out / "judge_human.tsv", jh)
    (out / "summary.json").write_text(json.dumps({"final_n": len(final), "arms": rows, "pairs": pr, "human_arms": hrows,
                                                  "human_pairs": hp, "judge_human": jh, "devset": ds}, indent=1),
                                      encoding="utf-8")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
