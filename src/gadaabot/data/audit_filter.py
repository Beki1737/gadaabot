"""Draw a stratified sample of corpus-filter decisions for native-speaker review.

The T007 language filter dropped 15.9% of AfriBERTa lines as "not Oromo" and 2.2%
of FineWeb-2 documents as "Ethiopic script". This script re-applies the same filter
to the raw sources and writes a TSV with:
  A  40 AfriBERTa lines dropped as not-Oromo
  B  20 AfriBERTa lines kept, lowest function-word ratio (closest to the threshold)
  C  20 FineWeb-2 documents dropped for Ethiopic script
A native speaker fills the column is_oromo_qubee (y/n/mixed). Agreement with the
filter gives its precision (bucket B) and the false-drop rate (buckets A and C).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from gadaabot.data.build_corpus import read_afriberta, read_fineweb2
from gadaabot.data.qubee import normalize_qubee, profile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-oromo-fw", type=float, default=0.03)
    ap.add_argument("--max-ethiopic", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    drop_notom: list[str] = []
    kept_low: list[tuple[float, str]] = []
    for d in read_afriberta(args.raw):
        if d.split != "train":
            continue
        t = normalize_qubee(d.text)
        p = profile(t)
        if p.n_words < 5 or p.ethiopic_ratio > args.max_ethiopic or p.alpha_ratio < 0.6:
            continue
        if p.oromo_fw < args.min_oromo_fw or not p.looks_oromo:
            drop_notom.append(t)
        else:
            kept_low.append((p.oromo_fw, t))
    kept_low.sort(key=lambda x: x[0])

    drop_eth: list[str] = []
    for d in read_fineweb2(args.raw):
        if d.split != "train":
            continue
        t = normalize_qubee(d.text)
        if profile(t).ethiopic_ratio > args.max_ethiopic:
            drop_eth.append(t)

    rows: list[tuple[str, str]] = []
    rows += [("A_afriberta_dropped_not_oromo", t) for t in rng.sample(drop_notom, min(40, len(drop_notom)))]
    rows += [("B_afriberta_kept_lowest_fw", t) for _, t in kept_low[:2000:100]]
    rows += [("C_fineweb2_dropped_ethiopic", t) for t in rng.sample(drop_eth, min(20, len(drop_eth)))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("row\tbucket\tis_oromo_qubee(y/n/mixed)\ttext_first_300_chars\n")
        for i, (b, t) in enumerate(rows, 1):
            snippet = t.replace("\t", " ").replace("\n", " / ")[:300]
            fh.write(f"{i}\t{b}\t\t{snippet}\n")
    print(f"AUDIT SAMPLE: {len(rows)} rows -> {args.out}")
    print(f"POOL SIZES: A={len(drop_notom):,} B={len(kept_low):,} C={len(drop_eth):,}")
    for b, t in rows[:3] + rows[40:42] + rows[60:62]:
        print(f"  {b}: {t[:120]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
