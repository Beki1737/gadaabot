"""Near-duplicate removal and evaluation-set decontamination (corpus IN -> OUT tag,
v1 -> v2 by default; v3pre -> v3 for filter v3).

Steps
-----
1. Decontamination. Every training and held-out document is checked against all
   Oromo evaluation texts (FLORES+, Belebele, SIB-200, AfriMMLU, AfriXNLI,
   AfriMGSM). A document is removed if it shares any 13-word n-gram with an
   evaluation text (the GPT-3 convention). 8-gram hits are counted but not removed,
   as a sensitivity figure for the paper.
2. Held-out isolation. Training documents that are near-duplicates (MinHash Jaccard
   >= threshold) of a held-out document are removed from training.
3. Near-duplicate removal inside training: MinHash LSH over 5-word shingles; the
   first document of each near-duplicate cluster (by source priority) is kept.

Output: oromo_corpus.OUT.{train,heldout}.jsonl.gz and oromo_corpus.OUT.stats.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator

from gadaabot.data.qubee import normalize_qubee, words

NUM_PERM = 128
SHINGLE = 5
SOURCE_PRIORITY = {"wura": 0, "fineweb2": 1, "madlad400": 2, "afriberta": 3}


# ---------------------------------------------------------------------------
# evaluation texts
# ---------------------------------------------------------------------------
def _jsonl_texts(path: Path, fields: Iterable[str]) -> Iterator[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for f in fields:
            v = rec.get(f)
            if isinstance(v, str) and v.strip():
                yield v


def _tsv_texts(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rec in reader:
            for k, v in rec.items():
                if k and v and isinstance(v, str) and len(v.split()) >= 4:
                    yield v


def load_eval_texts(eval_root: Path) -> dict[str, list[str]]:
    """Collect every Oromo evaluation string, grouped by benchmark."""
    out: dict[str, list[str]] = {}
    fl = eval_root / "openlanguagedata/flores_plus"
    out["flores"] = [t for p in sorted(fl.glob("*/gaz_Latn.jsonl")) for t in _jsonl_texts(p, ["text"])]
    bb = eval_root / "facebook/belebele/data/gaz_Latn.jsonl"
    out["belebele"] = list(_jsonl_texts(bb, ["flores_passage", "question", "mc_answer1",
                                             "mc_answer2", "mc_answer3", "mc_answer4"])) if bb.exists() else []
    for name, rel in (("sib200", "Davlan/sib200/data/gaz_Latn"),
                      ("afrimmlu", "masakhane/afrimmlu/data/orm"),
                      ("afrixnli", "masakhane/afrixnli/data/orm"),
                      ("afrimgsm", "masakhane/afrimgsm/data/orm")):
        out[name] = [t for p in sorted((eval_root / rel).glob("*.tsv")) for t in _tsv_texts(p)]
    return out


def ngrams(ws: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


# ---------------------------------------------------------------------------
# MinHash helpers (worker side)
# ---------------------------------------------------------------------------
def _minhash(text: str):  # type: ignore[no-untyped-def]
    """Worker: LeanMinHash of 5-word shingles (picklable, returned as is)."""
    from datasketch import LeanMinHash, MinHash

    ws = words(text)
    m = MinHash(num_perm=NUM_PERM, seed=1)
    shingles = {" ".join(ws[i:i + SHINGLE]) for i in range(max(1, len(ws) - SHINGLE + 1))}
    m.update_batch([s.encode("utf-8") for s in shingles])
    return LeanMinHash(m)


def read_jsonl_gz(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interim", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--in-tag", default="v1")
    ap.add_argument("--out-tag", default="v2")
    args = ap.parse_args()

    from datasketch import MinHashLSH

    t0 = time.time()
    train = read_jsonl_gz(args.interim / f"oromo_corpus.{args.in_tag}.train.jsonl.gz")
    held = read_jsonl_gz(args.interim / f"oromo_corpus.{args.in_tag}.heldout.jsonl.gz")
    print(f"LOADED: train {len(train):,} docs | heldout {len(held):,} docs")

    # --- 1) decontamination --------------------------------------------------
    ev = load_eval_texts(args.eval_root)
    ev13: dict[tuple[str, ...], str] = {}
    ev8: set[tuple[str, ...]] = set()
    for bench, texts in ev.items():
        for t in texts:
            ws = words(normalize_qubee(t))
            for g in ngrams(ws, 13):
                ev13.setdefault(g, bench)
            ev8 |= ngrams(ws, 8)
    print("EVAL TEXTS:", json.dumps({k: len(v) for k, v in ev.items()}),
          f"| 13-grams {len(ev13):,} | 8-grams {len(ev8):,}")

    contam = Counter()
    hit8 = Counter()

    def clean(docs: list[dict[str, object]], split: str) -> list[dict[str, object]]:
        kept = []
        for d in docs:
            ws = words(str(d["text"]))
            g13 = ngrams(ws, 13)
            bench = next((ev13[g] for g in g13 if g in ev13), None)
            if bench:
                contam[f"{split}:{d['source']} x {bench}"] += 1
                continue
            if ngrams(ws, 8) & ev8:
                hit8[f"{split}:{d['source']}"] += 1
            kept.append(d)
        return kept

    held = clean(held, "heldout")
    train = clean(train, "train")
    print("DECONTAM removed (13-gram):", json.dumps(dict(contam.most_common())) or "{}")
    print("8-gram-only hits kept (sensitivity):", json.dumps(dict(hit8.most_common())))

    # --- 2) + 3) MinHash ------------------------------------------------------
    train.sort(key=lambda d: SOURCE_PRIORITY.get(str(d["source"]), 9))
    with Pool(args.workers) as pool:
        h_held = pool.map(_minhash, [str(d["text"]) for d in held], chunksize=256)
        h_train = pool.map(_minhash, [str(d["text"]) for d in train], chunksize=256)

    lsh_held = MinHashLSH(threshold=args.jaccard, num_perm=NUM_PERM)
    for i, hv in enumerate(h_held):
        lsh_held.insert(f"h{i}", hv)

    lsh = MinHashLSH(threshold=args.jaccard, num_perm=NUM_PERM)
    kept_train: list[dict[str, object]] = []
    near_held = Counter()
    near_dup = Counter()
    for i, (d, hv) in enumerate(zip(train, h_train)):
        m = hv
        if lsh_held.query(m):
            near_held[str(d["source"])] += 1
            continue
        if lsh.query(m):
            near_dup[str(d["source"])] += 1
            continue
        lsh.insert(f"t{i}", m)
        kept_train.append(d)

    print(f"NEAR-DUP of heldout removed from train (J>={args.jaccard}):", json.dumps(dict(near_held)))
    print(f"NEAR-DUP inside train removed (J>={args.jaccard}):", json.dumps(dict(near_dup)))

    # --- write -----------------------------------------------------------------
    per_src: dict[str, Counter[str]] = {}
    for split, docs in (("train", kept_train), ("heldout", held)):
        p = args.interim / f"oromo_corpus.{args.out_tag}.{split}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            for d in docs:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
                c = per_src.setdefault(f"{d['source']}/{split}", Counter())
                c["docs"] += 1
                c["words"] += int(d["n_words"])  # type: ignore[arg-type]
    print(f"{args.out_tag.upper()} SOURCE/SPLIT              docs        words")
    tot = Counter()
    for k, c in sorted(per_src.items()):
        print(f"   {k:<22}{c['docs']:>9,}{c['words']:>13,}")
        tot[k.split("/")[1]] += c["words"]
    print(f"{args.out_tag.upper()} TOTAL words: train {tot['train']:,} | heldout {tot['heldout']:,}")
    stats = {"decontam_13gram": dict(contam), "hits_8gram_only": dict(hit8),
             "near_dup_of_heldout": dict(near_held), "near_dup_in_train": dict(near_dup),
             "per_source": {k: dict(v) for k, v in per_src.items()},
             "jaccard": args.jaccard, "num_perm": NUM_PERM, "shingle": SHINGLE,
             "seconds": round(time.time() - t0, 1)}
    (args.interim / f"oromo_corpus.{args.out_tag}.stats.json").write_text(json.dumps(stats, indent=2))
    print(f"SECONDS: {stats['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
