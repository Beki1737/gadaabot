"""Build the cleaned Oromo corpus (v1) from the raw sources staged by T005.

Output (data/interim/):
  oromo_corpus.v1.train.jsonl.gz     cleaned training documents
  oromo_corpus.v1.heldout.jsonl.gz   cleaned held-out documents (source eval/test splits)
  oromo_corpus.v1.stats.json         per-source counts, drop reasons, overlaps

Held-out documents are processed first; any training document that is an exact
(normalized) duplicate of a held-out document is dropped, so held-out text never
leaks into training. Near-duplicate removal and evaluation-set decontamination are
done separately (T008).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from gadaabot.data.qubee import count_hudhaa_variants, normalize_qubee, profile


@dataclass(frozen=True)
class RawDoc:
    source: str
    split: str  # "train" or "heldout"
    origin: str  # file the document came from
    text: str


def _text_field(rec: dict[str, object]) -> str:
    for key in ("text", "content", "document", "body"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def read_fineweb2(root: Path) -> Iterator[RawDoc]:
    import pyarrow.parquet as pq

    for path in sorted(root.glob("HuggingFaceFW/fineweb-2/data/gaz_Latn/*/*.parquet")):
        split = "heldout" if path.parent.name == "test" else "train"
        table = pq.read_table(path, columns=["text"])
        for txt in table.column("text").to_pylist():
            if txt:
                yield RawDoc("fineweb2", split, str(path.relative_to(root)), txt)


def read_madlad(root: Path) -> Iterator[RawDoc]:
    for path in sorted(root.glob("allenai/MADLAD-400/data/om/om_clean_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                txt = _text_field(json.loads(line))
                # MADLAD stores line breaks as the two characters backslash + n
                txt = txt.replace("\\n", "\n")
                if txt:
                    yield RawDoc("madlad400", "train", str(path.relative_to(root)), txt)


def read_wura(root: Path) -> Iterator[RawDoc]:
    # documents-v1.0 only; passages-v1.0 (if present) re-splits the same documents
    for path in sorted(root.glob("castorini/wura/documents-v1.0/*/orm.jsonl")):
        split = "heldout" if path.parent.name in {"eval", "test", "validation"} else "train"
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                rec = json.loads(line)
                txt = _text_field(rec)
                head = rec.get("headline")
                if isinstance(head, str) and head.strip() and not txt.startswith(head.strip()):
                    txt = f"{head.strip()}\n{txt}"
                if txt:
                    yield RawDoc("wura", split, str(path.relative_to(root)), txt)


def read_afriberta(root: Path) -> Iterator[RawDoc]:
    for path in sorted(root.glob("castorini/afriberta-corpus/afaanoromoo/*.zip")):
        split = "heldout" if "eval" in path.stem else "train"
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                with zf.open(name) as raw:
                    for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                        if line.strip():
                            yield RawDoc("afriberta", split, f"{path.name}:{name}", line)


READERS = {
    "fineweb2": read_fineweb2,
    "madlad400": read_madlad,
    "wura": read_wura,
    "afriberta": read_afriberta,
}


def doc_key(text: str) -> str:
    """Exact-duplicate key on normalized, case-folded, whitespace-free text."""
    return hashlib.sha1("".join(text.lower().split()).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--min-words-afriberta", type=int, default=5,
                    help="AfriBERTa is line-based (sentences or paragraphs)")
    ap.add_argument("--min-alpha", type=float, default=0.60)
    ap.add_argument("--max-ethiopic", type=float, default=0.05)
    ap.add_argument("--min-oromo-fw", type=float, default=0.03)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}  # key -> "source/split" that kept it
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    overlap: Counter[str] = Counter()
    hudhaa_raw: Counter[str] = Counter()
    fw_values: list[float] = []
    t0 = time.time()

    out_paths = {s: args.out / f"oromo_corpus.v1.{s}.jsonl.gz" for s in ("train", "heldout")}
    writers = {s: gzip.open(p, "wt", encoding="utf-8") for s, p in out_paths.items()}
    try:
        # held-out first, so train duplicates of held-out text are dropped
        for want_split in ("heldout", "train"):
            for name, reader in READERS.items():
                for doc in reader(args.raw):
                    if doc.split != want_split:
                        continue
                    st = stats[f"{name}/{doc.split}"]
                    st["raw"] += 1
                    hudhaa_raw.update(count_hudhaa_variants(doc.text))
                    text = normalize_qubee(doc.text)
                    prof = profile(text)
                    min_w = args.min_words_afriberta if name == "afriberta" else args.min_words
                    if prof.n_words < min_w:
                        st["drop_short"] += 1
                        continue
                    if prof.ethiopic_ratio > args.max_ethiopic:
                        st["drop_ethiopic"] += 1
                        continue
                    if prof.alpha_ratio < args.min_alpha:
                        st["drop_low_alpha"] += 1
                        continue
                    fw_values.append(prof.oromo_fw)
                    if prof.oromo_fw < args.min_oromo_fw or not prof.looks_oromo:
                        st["drop_not_oromo"] += 1
                        continue
                    key = doc_key(text)
                    if key in seen:
                        st["drop_exact_dup"] += 1
                        overlap[f"{name}/{doc.split} dup-of {seen[key]}"] += 1
                        continue
                    seen[key] = f"{name}/{doc.split}"
                    st["kept"] += 1
                    st["kept_words"] += prof.n_words
                    st["kept_chars"] += prof.n_chars
                    rec = {"id": key[:16], "source": name, "split": doc.split,
                           "origin": doc.origin, "n_words": prof.n_words,
                           "oromo_fw": round(prof.oromo_fw, 4), "text": text}
                    writers[doc.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            w.close()

    fw_sorted = sorted(fw_values)

    def pct(q: float) -> float:
        return round(fw_sorted[int(q * (len(fw_sorted) - 1))], 4) if fw_sorted else 0.0

    summary = {
        "per_source": {k: dict(v) for k, v in sorted(stats.items())},
        "exact_overlap": dict(overlap.most_common()),
        "hudhaa_variants_raw": dict(hudhaa_raw.most_common()),
        "oromo_fw_percentiles": {f"p{int(q*100)}": pct(q) for q in (0.01, 0.05, 0.25, 0.5, 0.75)},
        "thresholds": vars(args) | {"raw": str(args.raw), "out": str(args.out)},
        "seconds": round(time.time() - t0, 1),
    }
    (args.out / "oromo_corpus.v1.stats.json").write_text(json.dumps(summary, indent=2))

    # compact report lines
    print("SOURCE/SPLIT            raw     kept   short  ethio  alpha  notOm  exdup   kept_words")
    tot_words = {"train": 0, "heldout": 0}
    for k, v in sorted(stats.items()):
        print(f"{k:<20} {v['raw']:>7} {v['kept']:>7} {v['drop_short']:>6} {v['drop_ethiopic']:>6} "
              f"{v['drop_low_alpha']:>6} {v['drop_not_oromo']:>6} {v['drop_exact_dup']:>6} {v['kept_words']:>12,}")
        tot_words[k.split("/")[1]] += v["kept_words"]
    print(f"TOTAL kept words: train {tot_words['train']:,} | heldout {tot_words['heldout']:,}")
    print("EXACT OVERLAP:", json.dumps(dict(overlap.most_common(8))))
    print("HUDHAA VARIANTS (raw):", json.dumps(dict(hudhaa_raw.most_common())))
    print("OROMO FUNCTION-WORD RATIO percentiles:", json.dumps(summary["oromo_fw_percentiles"]))
    print(f"OUTPUT: {out_paths['train']} ({out_paths['train'].stat().st_size/1e6:.1f} MB), "
          f"{out_paths['heldout']} ({out_paths['heldout'].stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
