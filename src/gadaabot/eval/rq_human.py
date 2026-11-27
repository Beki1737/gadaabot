"""Real-question set (RQ): unblinded two-annotator analysis, judge-human agreement and the reranker-gate check.

Post hoc on the RQ set, rules fixed in D67 before any answer existed:
  H1 rag_naive > closed, H2 rag_naive > rag_tt_nllb, H3 agent_noverify does not beat rag_naive
  (human correct score = mean of the two annotators; paired bootstrap; Holm over H1 to H3).
Extra (descriptive, rules fixed now, before looking): the E4 reranker gate with the tau fixed in T034 (2.7):
  share of RQ questions below tau vs the generated test set, and human scores of answers below / above tau.

Usage: python -m gadaabot.eval.rq_human --root $GB_ROOT --a1 A1.xlsx --a2 A2.xlsx --key runs/T038/realq_key_PRIVATE.tsv
       --arms-dir runs/T038_realq --g27 runs/T038/rejudge_g27_judge.jsonl --out-dir runs/T039
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from gadaabot.eval.ablation import boot_p, holm
from gadaabot.eval.extra import Data, agreement, boot_ci, kendall, qwk, top_score, write_tsv

ARMS = ("rag_naive", "closed", "rag_tt_nllb", "agent_noverify")
HYP = [("H1", "rag_naive", "closed"), ("H2", "rag_naive", "rag_tt_nllb"), ("H3", "agent_noverify", "rag_naive")]
TAU = 2.7  # E4 gate threshold, fixed on calibration data in T034


def ci_mean(v: list[float]) -> tuple[float, float, float]:
    lo, hi = boot_ci(lambda s: statistics.fmean(v[t] for t in s), len(v))
    return statistics.fmean(v), lo, hi


def pair(src: dict[str, dict[str, float]], a: str, b: str, ids: list[str]) -> dict[str, Any]:
    d = [src[a][i] - src[b][i] for i in ids]
    m, lo, hi = ci_mean(d)
    return {"a": a, "b": b, "n": len(d), "diff": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3), "p": boot_p(d),
            "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)}


def main() -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser()
    for k in ("root", "a1", "a2", "key", "arms-dir", "g27", "out-dir"):
        ap.add_argument(f"--{k}", type=Path, required=k != "g27")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    raw = a.key.read_bytes()
    print(f"RQH key md5 {hashlib.md5(raw).hexdigest()} (reconstruction from seed 41: 7abf7d27c0e632019d4a1e17e47b0f3e)", flush=True)
    key = list(csv.DictReader(raw.decode("utf-8").splitlines(), delimiter="\t"))
    sheets = {}
    for ann, f in (("A1", a.a1), ("A2", a.a2)):
        rows = list(load_workbook(f, data_only=True)["Score"].iter_rows(min_row=2, values_only=True))
        if len(rows) != len(key) or any(str(r[1]) != k["q_no"] or r[4] != k["label"] for r, k in zip(rows, key)):
            print(f"RQH-FAIL {ann}: sheet rows do not line up with the key", flush=True)
            return 1
        if any(r[6] is None or r[7] is None for r in rows):
            print(f"RQH-WARN {ann}: {sum(r[6] is None for r in rows)} blank correct scores", flush=True)
        sheets[ann] = rows
    sc: dict[str, dict[str, dict[str, tuple[int, int]]]] = {x: defaultdict(dict) for x in sheets}
    for ann, rows in sheets.items():
        for r, k in zip(rows, key):
            if r[6] is not None and r[7] is not None:
                sc[ann][k["arm"]][k["id"]] = (int(r[6]), int(r[7]))
    dom = {k["id"]: k["domain"] for k in key}
    ids = sorted(i for i in dom if all(i in sc[x][s] for x in sc for s in ARMS))
    hum = {s: {i: statistics.fmean(sc[x][s][i][0] for x in sc) for i in ids} for s in ARMS}
    flu = {s: {i: statistics.fmean(sc[x][s][i][1] for x in sc) for i in ids} for s in ARMS}
    # --- agreement between annotators
    for name, j in (("correct", 0), ("fluent", 1)):
        x = [sc["A1"][s][i][j] for s in ARMS for i in ids]
        y = [sc["A2"][s][i][j] for s in ARMS for i in ids]
        lo, hi = boot_ci(lambda t: qwk([x[u] for u in t], [y[u] for u in t]), len(x))
        print(f"RQH-IAA {name}: QWK {qwk(x, y):.3f} [{lo:.3f}, {hi:.3f}] | exact {100 * statistics.fmean(p == q for p, q in zip(x, y)):.1f}% "
              f"| within 1 {100 * statistics.fmean(abs(p - q) <= 1 for p, q in zip(x, y)):.1f}% | n {len(x)}", flush=True)
    # --- system means
    srows = []
    for s in ARMS:
        m, lo, hi = ci_mean(list(hum[s].values()))
        r = {"system": s, "n": len(ids), "human": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3),
             "A1": round(statistics.fmean(sc["A1"][s][i][0] for i in ids), 3), "A2": round(statistics.fmean(sc["A2"][s][i][0] for i in ids), 3),
             "low_pct": round(100 * statistics.fmean(hum[s][i] <= 2 for i in ids), 1), "fluent": round(statistics.fmean(flu[s].values()), 3)}
        for dname in ("health", "education", "culture"):
            sub = [hum[s][i] for i in ids if dom[i] == dname]
            r[dname] = round(statistics.fmean(sub), 3) if sub else ""
        srows.append(r)
        print("RQH-SYS " + " | ".join(f"{k} {v}" for k, v in r.items()), flush=True)
    write_tsv(a.out_dir / "rq_human_systems.tsv", srows)
    # --- pre-registered hypotheses
    pr = [pair(hum, x, y, ids) for _, x, y in HYP]
    for (h, _, _), p, ph in zip(HYP, pr, holm([p["p"] for p in pr])):
        p["hyp"], p["p_holm"] = h, round(ph, 4)
        print(f"RQH-{h} {p['a']} - {p['b']}: {p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] p_holm={p['p_holm']} "
              f"| W/T/L {p['wins']}/{p['ties']}/{p['losses']}", flush=True)
    write_tsv(a.out_dir / "rq_human_hypotheses.tsv", pr)
    for ann in sc:
        s1 = {s: {i: float(sc[ann][s][i][0]) for i in ids} for s in ARMS}
        print(f"RQH-ANN {ann} " + " | ".join(f"{h} {pair(s1, x, y, ids)['diff']:+.2f}" for h, x, y in HYP), flush=True)
    # --- judges vs humans (same 316 answers)
    judges = {"afq": {}, "g27": {}}
    for l in (a.arms_dir / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("mean") is not None:
                judges["afq"][(r["id"], r["arm"])] = float(r["mean"])
    if a.g27 and a.g27.exists():
        for l in a.g27.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                judges["g27"][(r["id"], r["arm"])] = float(r["mean"])
    pairs6 = [("rag_naive", "closed"), ("rag_naive", "rag_tt_nllb"), ("agent_noverify", "rag_naive"),
              ("closed", "rag_tt_nllb"), ("agent_noverify", "closed"), ("agent_noverify", "rag_tt_nllb")]
    for jn, J in judges.items():
        rows = [(i, s, hum[s][i], J[(i, s)]) for s in ARMS for i in ids if (i, s) in J]
        if not rows:
            continue
        g = agreement(rows, pairs6)
        print(f"RQH-JUDGE {jn} " + " | ".join(f"{k} {v}" for k, v in g.items()), flush=True)
    # --- reranker gate (tau fixed in T034)
    D = Data(a.root)
    arm = {r["id"]: r for r in map(json.loads, (a.arms_dir / "arm_rag_naive.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    ts = {i: top_score(arm[i].get("trace", [])) for i in ids}
    ts = {i: v for i, v in ts.items() if v is not None}
    gen = D.arm("rag_naive")
    gts = [top_score(gen[i].get("trace", [])) for i in D.test if i in gen]
    gts = [v for v in gts if v is not None]
    if ts and gts:
        print(f"RQH-GATE tau {TAU}: below tau real {100 * statistics.fmean(v < TAU for v in ts.values()):.1f}% "
              f"(median top score {statistics.median(ts.values()):.2f}, n {len(ts)}) | generated test "
              f"{100 * statistics.fmean(v < TAU for v in gts):.1f}% (median {statistics.median(gts):.2f}, n {len(gts)})", flush=True)
        for lab, sel in (("below", [i for i in ts if ts[i] < TAU]), ("above", [i for i in ts if ts[i] >= TAU])):
            if not sel:
                continue
            r = statistics.fmean(hum["rag_naive"][i] for i in sel)
            c = statistics.fmean(hum["closed"][i] for i in sel)
            lowp = 100 * statistics.fmean(hum["rag_naive"][i] <= 2 for i in sel)
            d = [hum["rag_naive"][i] - hum["closed"][i] for i in sel]
            m, lo, hi = ci_mean(d) if len(d) > 2 else (statistics.fmean(d), float("nan"), float("nan"))
            print(f"RQH-GATE {lab} n={len(sel)} | rag human {r:.2f} | closed human {c:.2f} | rag - closed {m:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"| rag low(<=2) {lowp:.1f}%", flush=True)
        hs = sorted(ts, key=lambda i: ts[i])
        q = len(hs) // 3
        for lab, sel in (("low", hs[:q]), ("mid", hs[q:2 * q]), ("high", hs[2 * q:])):
            print(f"RQH-TERCILE top-score {lab} n={len(sel)} | rag {statistics.fmean(hum['rag_naive'][i] for i in sel):.2f} "
                  f"| closed {statistics.fmean(hum['closed'][i] for i in sel):.2f} | nllb {statistics.fmean(hum['rag_tt_nllb'][i] for i in sel):.2f} "
                  f"| agent {statistics.fmean(hum['agent_noverify'][i] for i in sel):.2f}", flush=True)
    print("DONE rq_human", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
