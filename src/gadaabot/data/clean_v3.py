"""Corpus filter v3 (decision D26): rebuild the cleaned Oromo corpus from raw sources.

Why v3. The T008 native-speaker audit of v1 showed:
  * the function-word minimum (oromo_fw >= 0.03) dropped real Oromo: about 85% of
    AfriBERTa lines it removed were clean Qubee (short lines have no function words);
  * document-level Ge'ez removal discarded Qubee documents that only quoted a few
    Ethiopic lines;
  * kept text still carried BBC navigation boilerplate, image-caption lines, and SEO
    spam padded with Lithuanian/Czech/Polish words.

v3 rules, in order (each logged as a separate drop reason / line counter):
  1. Normalize Qubee (unchanged from v1: NFC, hudhaa -> U+0027, whitespace).
  2. Line level:
       a. drop lines that start with a known caption/navigation marker;
       b. drop lines dominated by Ge'ez or non-Qubee letters (foreign script);
       c. drop corpus-repeated lines: a normalized line seen in >= --min-line-docs
          distinct documents is boilerplate (C4/Gopher-style repeated-line removal).
  3. Document level on the remaining text:
       a. min words (20; AfriBERTa 5, it is line-based), alpha ratio >= 0.6;
       b. Qubee alphabet: drop if > 5% of words contain a non-ASCII letter;
       c. foreign dominance (replaces the v1 Oromo function-word minimum): drop if
          Somali or English function words number >= 2, make up >= 10% of words, and
          outnumber Oromo function words. A line with no function words is KEPT.
  4. Exact dedup (held-out first, so held-out text never enters training).

Outputs (data/interim/): oromo_corpus.v3pre.{train,heldout}.jsonl.gz, *.stats.json,
and an 80-row audit sheet for native-speaker labeling (reports/tables/T011_v3_audit.tsv):
  R  30 train docs rescued by v3 (v1 dropped them, v3 keeps them)
  K  20 train docs kept by both v1 and v3 (random)
  D  20 train docs dropped by the new v3 document rules (alphabet / foreign dominance)
  L  10 distinct lines removed as corpus-repeated boilerplate
Near-duplicate removal and decontamination run next (dedup.py --in-tag v3pre).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from gadaabot.data.build_corpus import READERS, doc_key
from gadaabot.data.qubee import (function_word_counts, line_is_foreign_script, non_qubee_word_ratio,
                                 normalize_qubee, profile, words)

# lines starting with these (case-insensitive) are captions / navigation, not content
LINE_PREFIXES = (
    "madda suuraa",          # image source credit (BBC Afaan Oromoo)
    "ibsa waa'ee suuraa",    # image caption label
    "qabiyyeetti darbi",     # "skip to content" navigation
)
_PREFIX_RE = re.compile(r"^\W*(?:" + "|".join(re.escape(p) for p in LINE_PREFIXES) + r")", re.I)
_DIGITS = re.compile(r"\d+")


def line_key(line: str) -> int:
    """Hash of a line for repeated-line counting: lowercased, digits and spaces removed."""
    return hash(_DIGITS.sub("", "".join(line.lower().split())))


class Reservoir:
    """Fixed-size uniform sample over a stream (Algorithm R)."""

    def __init__(self, k: int, rng: random.Random) -> None:
        self.k, self.rng, self.n, self.items = k, rng, 0, []  # type: ignore[var-annotated]

    def add(self, item: object) -> None:
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self.items[j] = item


def v1_keeps(text: str, source: str) -> bool:
    """The v1 document rule (T007 defaults), for the rescued/kept audit pools."""
    p = profile(text)
    min_w = 5 if source == "afriberta" else 20
    return (p.n_words >= min_w and p.ethiopic_ratio <= 0.05 and p.alpha_ratio >= 0.60
            and p.oromo_fw >= 0.03 and p.looks_oromo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--tag", default="v3pre")
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--min-words-afriberta", type=int, default=5)
    ap.add_argument("--min-alpha", type=float, default=0.60)
    ap.add_argument("--max-non-qubee", type=float, default=0.05)
    ap.add_argument("--foreign-min-count", type=int, default=2)
    ap.add_argument("--foreign-min-ratio", type=float, default=0.10)
    ap.add_argument("--min-line-docs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(args.seed)

    # --- pass 1: count each normalized line once per document -----------------
    line_docs: Counter[int] = Counter()
    line_text: dict[int, str] = {}
    n_docs = 0
    for name, reader in READERS.items():
        for doc in reader(args.raw):
            n_docs += 1
            keys = set()
            for ln in normalize_qubee(doc.text).split("\n"):
                if words(ln):
                    k = line_key(ln)
                    keys.add(k)
                    if k not in line_text and len(line_text) < 5_000_000:
                        line_text[k] = ln[:300]
            line_docs.update(keys)
    boiler = {k for k, c in line_docs.items() if c >= args.min_line_docs}
    print(f"PASS1: {n_docs:,} raw docs | distinct lines {len(line_docs):,} | "
          f"repeated-line patterns (>= {args.min_line_docs} docs) {len(boiler):,} | {time.time()-t0:.0f}s",
          flush=True)

    # --- pass 2: clean, filter, exact dedup, write --------------------------------
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    line_stats: Counter[str] = Counter()
    seen: set[str] = set()
    pools = {p: Reservoir(k, rng) for p, k in (("R", 30), ("K", 20), ("D", 20))}
    boiler_removed: Counter[int] = Counter()
    out_paths = {s: args.out / f"oromo_corpus.{args.tag}.{s}.jsonl.gz" for s in ("train", "heldout")}
    writers = {s: gzip.open(p, "wt", encoding="utf-8") for s, p in out_paths.items()}
    try:
        for want_split in ("heldout", "train"):
            for name, reader in READERS.items():
                for doc in reader(args.raw):
                    if doc.split != want_split:
                        continue
                    st = stats[f"{name}/{doc.split}"]
                    st["raw"] += 1
                    norm = normalize_qubee(doc.text)
                    kept_lines = []
                    for ln in norm.split("\n"):
                        if not words(ln):
                            kept_lines.append(ln)
                            continue
                        if _PREFIX_RE.match(ln):
                            line_stats["prefix"] += 1
                            continue
                        if line_is_foreign_script(ln):
                            line_stats["foreign_script"] += 1
                            continue
                        k = line_key(ln)
                        if k in boiler:
                            line_stats["repeated"] += 1
                            boiler_removed[k] += 1
                            continue
                        kept_lines.append(ln)
                    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
                    ws = words(text)
                    prof = profile(text)
                    min_w = args.min_words_afriberta if name == "afriberta" else args.min_words
                    om, so, en = function_word_counts(ws)
                    foreign = max(so, en)
                    reason = None
                    if len(ws) < min_w:
                        reason = "short"
                    elif prof.alpha_ratio < args.min_alpha:
                        reason = "low_alpha"
                    elif non_qubee_word_ratio(ws) > args.max_non_qubee:
                        reason = "non_qubee_alphabet"
                    elif (foreign >= args.foreign_min_count and foreign / len(ws) >= args.foreign_min_ratio
                          and foreign > om):
                        reason = "foreign_dominance"
                    if reason is None:
                        key = doc_key(text)
                        if key in seen:
                            reason = "exact_dup"
                        else:
                            seen.add(key)
                    if doc.split == "train" and reason in (None, "non_qubee_alphabet", "foreign_dominance"):
                        v1 = v1_keeps(norm, name)
                        row = (name, reason or "kept", len(ws), text)
                        if reason is None:
                            pools["K" if v1 else "R"].add(row)
                            if not v1:
                                st["rescued_vs_v1"] += 1
                        else:
                            pools["D"].add(row)
                    if reason:
                        st[f"drop_{reason}"] += 1
                        continue
                    st["kept"] += 1
                    st["kept_words"] += len(ws)
                    rec = {"id": doc_key(text)[:16], "source": name, "split": doc.split,
                           "origin": doc.origin, "n_words": len(ws),
                           "oromo_fw": round(prof.oromo_fw, 4), "text": text}
                    writers[doc.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for w in writers.values():
            w.close()

    # --- audit sheet ------------------------------------------------------------------
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    top_boiler = [k for k, _ in boiler_removed.most_common(200)]
    l_pick = rng.sample(top_boiler, min(10, len(top_boiler)))

    def cell(s: str) -> str:
        return s.replace("\t", " ").replace("\n", " | ")[:600]

    with args.audit.open("w", encoding="utf-8") as fh:
        fh.write("row\tpool\tsource\treason\tn_words\tlabel\ttext\n")
        r = 0
        for pool in ("R", "K", "D"):
            for src, why, nw, txt in pools[pool].items:  # type: ignore[misc]
                r += 1
                fh.write(f"{r}\t{pool}\t{src}\t{why}\t{nw}\t\t{cell(txt)}\n")
        for k in l_pick:
            r += 1
            fh.write(f"{r}\tL\t-\trepeated_in_{line_docs[k]}_docs\t{len(words(line_text.get(k, '')))}\t\t"
                     f"{cell(line_text.get(k, '?'))}\n")
    print(f"AUDIT: {args.audit} ({r} rows). label: y = clean Oromo Qubee, mixed = Oromo plus junk "
          f"or foreign text, n = not Oromo. Pool L: y = boilerplate (right to remove), n = real content")

    summary = {"per_source": {k: dict(v) for k, v in sorted(stats.items())},
               "lines_removed": dict(line_stats), "repeated_line_patterns": len(boiler),
               "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "seconds": round(time.time() - t0, 1)}
    (args.out / f"oromo_corpus.{args.tag}.stats.json").write_text(json.dumps(summary, indent=2))

    print("SOURCE/SPLIT          raw     kept  short alpha nonQb foreign exdup  rescued   kept_words")
    tot = Counter()
    for k, v in sorted(stats.items()):
        print(f"{k:<18} {v['raw']:>8} {v['kept']:>8} {v['drop_short']:>6} {v['drop_low_alpha']:>5} "
              f"{v['drop_non_qubee_alphabet']:>5} {v['drop_foreign_dominance']:>7} {v['drop_exact_dup']:>5} "
              f"{v['rescued_vs_v1']:>8} {v['kept_words']:>12,}")
        tot[k.split("/")[1]] += v["kept_words"]
    print(f"LINES REMOVED: {json.dumps(dict(line_stats))}")
    print(f"TOTAL kept words ({args.tag}, before near-dup): train {tot['train']:,} | heldout {tot['heldout']:,}")
    print(f"SECONDS: {summary['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
