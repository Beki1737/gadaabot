#!/usr/bin/env python3
"""T041: unblind and analyse the clinician safety review (T037 sheet), as pre-registered in D71.

Primary (pre-registered): overall safety S5, native RAG (rag_naive) minus closed book and native RAG
minus NLLB RAG cascade (rag_tt_nllb); paired bootstrap over questions (2000 resamples for CIs, 4000 for
two-sided p-values); Holm over the two comparisons.
Secondary (descriptive): harm rate (S1 > 0), wrong-fact rate (S2 = yes), mean missing-information
score S3, referral answers S4; primary contrast repeated on the priority questions 1 to 30.

Usage:
  python T041_safety_analysis.py --sheet RETURNED.xlsx --key safety_key_PRIVATE.tsv --out report.txt
The key is read locally and never printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import openpyxl
except ImportError:  # pragma: no cover
    sys.exit("openpyxl missing: run inside the gb-core environment")

SYSTEMS = ("rag_naive", "closed", "rag_tt_nllb")
PAIRS = (("rag_naive", "closed"), ("rag_naive", "rag_tt_nllb"))
SEED = 37
OUT: list[str] = []


def say(line: str) -> None:
    print(line, flush=True)
    OUT.append(line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(h: object) -> str:
    return str(h or "").strip().lower().replace(" ", "_")


def read_sheet(path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    if "Review" not in wb.sheetnames:
        sys.exit(f"STATUS: FAIL no 'Review' tab in {path.name}; tabs: {wb.sheetnames}")
    ws = wb["Review"]
    rows = list(ws.iter_rows(values_only=True))
    head = [norm(h) for h in rows[0]]

    def col(prefix: str) -> int:
        for i, h in enumerate(head):
            if h.startswith(prefix):
                return i
        sys.exit(f"STATUS: FAIL sheet column '{prefix}' not found; header: {head}")

    iq, ia = col("q_no"), head.index("answer") if "answer" in head else col("answer")
    i1, i2, i3, i4, i5 = col("s1"), col("s2"), col("s3"), col("s4"), col("s5")
    out, last_q = [], None
    for r in rows[1:]:
        if all(v is None for v in r):
            continue
        q = r[iq] if r[iq] not in (None, "") else last_q
        last_q = q
        out.append({
            "q": int(str(q).strip()),
            "label": str(r[ia]).strip().upper(),
            "s1": None if r[i1] in (None, "") else int(float(r[i1])),
            "s2": None if r[i2] in (None, "") else str(r[i2]).strip().lower(),
            "s3": None if r[i3] in (None, "") else int(float(r[i3])),
            "s4": None if r[i4] in (None, "") else str(r[i4]).strip().lower(),
            "s5": None if r[i5] in (None, "") else int(float(r[i5])),
        })
    return out


def read_key(path: Path) -> dict[tuple[int, str], str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        head = [norm(h) for h in (rd.fieldnames or [])]
        rows = [{norm(k): v for k, v in r.items()} for r in rd]
    qcol = next((h for h in head if h in ("q_no", "q", "qno", "question_no", "q_num", "qid_no")), None)
    lcol = next((h for h in head if h in ("label", "answer", "letter", "answer_label")), None)
    scol = next((h for h in head if h in ("system", "arm", "system_name")), None)
    if not (qcol and lcol and scol):
        say(f"KEY header (column names only): {head}")
        sys.exit("STATUS: NEEDS_KEY_FORMAT  paste the header line above; no key values were printed")
    key = {}
    for r in rows:
        key[(int(str(r[qcol]).strip()), str(r[lcol]).strip().upper())] = str(r[scol]).strip()
    return key


def boot_ci(diffs: list[float], b: int, rng: random.Random) -> tuple[float, float]:
    n = len(diffs)
    bs = sorted(st.fmean(rng.choices(diffs, k=n)) for _ in range(b))
    return bs[int(0.025 * b)], bs[int(0.975 * b) - 1]


def boot_p(diffs: list[float], b: int, rng: random.Random) -> float:
    """Two-sided bootstrap p-value: resample the centred differences (null of zero mean)."""
    n, obs = len(diffs), st.fmean(diffs)
    centred = [d - obs for d in diffs]
    extreme = sum(1 for _ in range(b) if abs(st.fmean(rng.choices(centred, k=n))) >= abs(obs))
    return (extreme + 1) / (b + 1)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, type=Path)
    ap.add_argument("--key", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    say(f"T041 sheet {a.sheet.name} sha256 {sha256(a.sheet)}")
    say(f"T041 key   {a.key.name} sha256 {sha256(a.key)}")
    rows = read_sheet(a.sheet)
    key = read_key(a.key)

    # integrity checks before any statistic
    missing = [r for r in rows if r["s5"] is None]
    labels_per_q = defaultdict(set)
    for r in rows:
        labels_per_q[r["q"]].add(r["label"])
    bad = [q for q, ls in labels_per_q.items() if ls != {"A", "B", "C"}]
    unmatched = [r for r in rows if (r["q"], r["label"]) not in key]
    say(f"CHECK rows {len(rows)} | questions {len(labels_per_q)} | missing S5 {len(missing)} | "
        f"questions without exactly A/B/C {len(bad)} | rows not in key {len(unmatched)}")
    if len(rows) != 120 or bad or unmatched:
        a.out.write_text("\n".join(OUT) + "\nSTATUS: FAIL integrity\n")
        sys.exit("STATUS: FAIL integrity")

    by = defaultdict(dict)  # system -> q -> row
    for r in rows:
        by[key[(r["q"], r["label"])]][r["q"]] = r
    systems = sorted(by)
    say(f"SYSTEMS in key: {systems}")
    if set(SYSTEMS) - set(systems):
        sys.exit(f"STATUS: FAIL expected systems {SYSTEMS}")

    rng = random.Random(SEED)
    qs = sorted(labels_per_q)
    for s in SYSTEMS:
        v5 = [by[s][q]["s5"] for q in qs]
        harm = [int((by[s][q]["s1"] or 0) > 0) for q in qs]
        wrong = [int(by[s][q]["s2"] == "yes") for q in qs]
        s3 = [by[s][q]["s3"] for q in qs if by[s][q]["s3"] is not None]
        s4 = Counter(by[s][q]["s4"] for q in qs)
        lo, hi = boot_ci([float(x) for x in v5], 2000, rng)
        say(f"SYS {s:12s} | n {len(qs)} | S5 mean {st.fmean(v5):.3f} [{lo:.2f}, {hi:.2f}] | "
            f"harm>0 {100 * st.fmean(harm):.1f}% | wrong fact {100 * st.fmean(wrong):.1f}% | "
            f"S3 mean {st.fmean(s3):.2f} | S4 {dict(s4)}")

    ps, res = [], []
    for x, y in PAIRS:
        d = [float(by[x][q]["s5"] - by[y][q]["s5"]) for q in qs]
        lo, hi = boot_ci(d, 2000, rng)
        p = boot_p(d, 4000, rng)
        w = sum(v > 0 for v in d); t = sum(v == 0 for v in d); l_ = sum(v < 0 for v in d)
        res.append((x, y, st.fmean(d), lo, hi, w, t, l_))
        ps.append(p)
    for (x, y, m, lo, hi, w, t, l_), p, ph in zip(res, ps, holm(ps)):
        say(f"PRIMARY S5 {x} - {y}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] p {p:.4f} p_holm {ph:.4f} W/T/L {w}/{t}/{l_}")

    for x, y in PAIRS:
        for name, f in (("harm>0", lambda r: int((r["s1"] or 0) > 0)), ("wrong fact", lambda r: int(r["s2"] == "yes"))):
            d = [float(f(by[x][q]) - f(by[y][q])) for q in qs]
            lo, hi = boot_ci(d, 2000, rng)
            say(f"SECONDARY {name} rate {x} - {y}: {100 * st.fmean(d):+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] (descriptive)")

    pq = [q for q in qs if q <= 30]
    for x, y in PAIRS:
        d = [float(by[x][q]["s5"] - by[y][q]["s5"]) for q in pq]
        lo, hi = boot_ci(d, 2000, rng)
        say(f"PRIORITY q1-30 S5 {x} - {y}: {st.fmean(d):+.2f} [{lo:+.2f}, {hi:+.2f}] (n {len(pq)}, descriptive)")
    say("STATUS: PASS")
    a.out.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
