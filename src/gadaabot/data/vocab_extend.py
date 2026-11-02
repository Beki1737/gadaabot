"""Gate T1: extend the Qwen3.5 byte-level BPE tokenizer with Oromo merges (continued BPE).

Method (continued BPE from the base tokenizer's own state)
---------------------------------------------------------
1. Pre-tokenize the Oromo training corpus (corpus v3, train split only) with the
   base tokenizer's own normalizer and pre-tokenizer (regex split + byte-level map).
2. Segment every distinct pre-token with the UNCHANGED base BPE model. This is the
   exact state the base tokenizer ends in.
3. Run standard greedy BPE from that state: repeatedly merge the most frequent
   adjacent token pair (frequency weighted by pre-token counts), K times.
4. Append the learned merges AFTER all base merges (lowest priority) and add the new
   tokens to the vocabulary with fresh ids after every existing id.

Why this is exact. A base merge only involves base tokens, and a new merge always
produces a token that no base merge consumes, so base merges are exhausted before any
appended merge fires, and appended merges then fire in learned order. The extended
tokenizer therefore reproduces the training-time segmentation; this script verifies
that on a sample of pre-tokens. Base behaviour on text the new merges never touch is
unchanged by construction, and no base token is removed or renumbered, so the base
model's embeddings stay valid. Merges whose result is already a base token (reachable
through another merge path) add no vocabulary entry and are counted separately.

Outputs
-------
  <out>/qwen35-orm-ext-K{K}/   tokenizer.json + tokenizer config files, per K
  <out>/continued_bpe_merges.jsonl   every learned merge: a, b, is_new, pair count
  <table>.json / .csv          fertility (FLORES+ devtest and corpus held-out),
                               English fertility, premium, chars/token, STRR, 95% CIs
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from gadaabot.data.qubee import normalize_qubee, words
from gadaabot.data.tok_audit import bootstrap_ci, load_flores

TOKENIZER_SIDE_FILES = ("tokenizer_config.json", "special_tokens_map.json",
                        "chat_template.jinja", "chat_template.json")
_W_TOK = None  # worker-global tokenizer


# ---------------------------------------------------------------------------
# corpus -> pre-token counts (parallel)
# ---------------------------------------------------------------------------
def _init_worker(path: str) -> None:
    global _W_TOK
    from tokenizers import Tokenizer

    _W_TOK = Tokenizer.from_file(path)


def _count_chunk(texts: list[str]) -> Counter[str]:
    tok = _W_TOK
    assert tok is not None
    c: Counter[str] = Counter()
    for t in texts:
        if tok.normalizer is not None:
            t = tok.normalizer.normalize_str(t)
        c.update(p for p, _ in tok.pre_tokenizer.pre_tokenize_str(t))
    return c


def read_texts(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh]


# ---------------------------------------------------------------------------
# continued BPE
# ---------------------------------------------------------------------------
def continued_bpe(segs: list[list[str]], freqs: list[int], base_vocab: set[str], k_max: int,
                  min_freq: int, log_every: int = 2000) -> list[tuple[str, str, bool, int]]:
    """Greedy BPE from given segmentations until k_max NEW tokens exist.

    Returns merges in learned order as (a, b, is_new_token, pair_count). Mutates segs
    in place into the final segmentation (used for the exactness check).
    """
    pc: dict[tuple[str, str], int] = defaultdict(int)
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, w in enumerate(segs):
        for p in zip(w, w[1:]):
            pc[p] += freqs[i]
            where[p].add(i)
    heap = [(-c, p) for p, c in pc.items()]
    heapq.heapify(heap)
    vocab = set(base_vocab)
    merges: list[tuple[str, str, bool, int]] = []
    n_new, t0 = 0, time.time()
    while heap and n_new < k_max:
        negc, p = heapq.heappop(heap)
        c = pc.get(p, 0)
        if c != -negc:                       # stale heap entry
            if c > 0:
                heapq.heappush(heap, (-c, p))
            continue
        if c < min_freq:
            break
        a, b = p
        ab = a + b
        is_new = ab not in vocab
        merges.append((a, b, is_new, c))
        if is_new:
            vocab.add(ab)
            n_new += 1
            if n_new % log_every == 0:
                print(f"  continued-BPE: {n_new:,} new tokens, {len(merges):,} merges, "
                      f"last pair count {c:,}, {time.time()-t0:.0f}s", flush=True)
        touched: set[tuple[str, str]] = set()
        for i in list(where.pop(p, ())):
            w, f = segs[i], freqs[i]
            old = list(zip(w, w[1:]))
            nw: list[str] = []
            j = 0
            while j < len(w):
                if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                    nw.append(ab)
                    j += 2
                else:
                    nw.append(w[j])
                    j += 1
            new = list(zip(nw, nw[1:]))
            for q in old:
                pc[q] -= f
                touched.add(q)
            for q in new:
                pc[q] += f
                touched.add(q)
            for q in set(old) - set(new):
                if q in where:
                    where[q].discard(i)
            for q in set(new):
                where[q].add(i)
            segs[i] = nw
        pc.pop(p, None)
        for q in touched:
            if q != p and pc.get(q, 0) > 0:
                heapq.heappush(heap, (-pc[q], q))
    return merges


# ---------------------------------------------------------------------------
# tokenizer.json writing
# ---------------------------------------------------------------------------
def write_extended(base_json: dict, merges: list[tuple[str, str, bool, int]], out_dir: Path,
                   base_dir: Path) -> int:
    """Write tokenizer.json with appended merges / new tokens; return number of new tokens."""
    tj = json.loads(json.dumps(base_json))  # deep copy
    model = tj["model"]
    vocab: dict[str, int] = model["vocab"]
    # The tokenizers library re-numbers added (special) tokens from len(model vocab) on load.
    # Base Qwen3.5: model vocab 0..248043, specials 248044..248076. Appending new tokens after
    # the specials would shift the specials onto the last new ids. Registering every special
    # token in the model vocab under its own id keeps the id space contiguous and unambiguous.
    for t in tj.get("added_tokens", []):
        if vocab.setdefault(t["content"], t["id"]) != t["id"]:
            raise ValueError(f"special token {t['content']!r} has id {vocab[t['content']]} in vocab, {t['id']} in added_tokens")
    next_id = 1 + max(vocab.values())
    as_pairs = bool(model["merges"]) and isinstance(model["merges"][0], list)
    n_new = 0
    specials = {t["content"] for t in tj.get("added_tokens", [])}
    for a, b, _is_new, _c in merges:
        ab = a + b
        if ab in specials:  # never let plain text merge into a special token
            continue
        if ab not in vocab:
            vocab[ab] = next_id
            next_id += 1
            n_new += 1
        model["merges"].append([a, b] if as_pairs else f"{a} {b}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    for name in TOKENIZER_SIDE_FILES:
        if (base_dir / name).exists():
            shutil.copyfile(base_dir / name, out_dir / name)
    return n_new


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(tok, orm: list[str], eng: list[str], held: list[str]) -> dict[str, object]:  # type: ignore[no-untyped-def]
    def lens(xs: list[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(xs, add_special_tokens=False)]

    w_o, w_e, w_h = ([len(words(s)) for s in xs] for xs in (orm, eng, held))
    t_o, t_e, t_h = lens(orm), lens(eng), lens(held)
    f_o, f_e, f_h = sum(t_o) / sum(w_o), sum(t_e) / sum(w_e), sum(t_h) / sum(w_h)
    lo, hi = bootstrap_ci(t_o, w_o)
    types = sorted({w for s in orm for w in words(s)})
    single = sum(len(e.ids) == 1 for e in tok.encode_batch([" " + w for w in types], add_special_tokens=False))
    return {"fertility_flores_orm": round(f_o, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "fertility_heldout_orm": round(f_h, 3), "fertility_flores_eng": round(f_e, 3),
            "premium": round(f_o / f_e, 3), "chars_per_token_orm": round(sum(map(len, orm)) / sum(t_o), 3),
            "strr_orm": round(single / len(types), 3), "tokens_flores_orm": sum(t_o)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="dir with the base tokenizer.json")
    ap.add_argument("--train", type=Path, required=True, help="corpus v3 train jsonl.gz")
    ap.add_argument("--heldout", type=Path, required=True, help="corpus v3 heldout jsonl.gz")
    ap.add_argument("--flores-orm", type=Path, required=True)
    ap.add_argument("--flores-eng", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--ks", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--min-word-freq", type=int, default=2)
    ap.add_argument("--min-pair-freq", type=int, default=20)
    ap.add_argument("--target", type=float, default=2.30)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    t0 = time.time()
    base_path = args.base / "tokenizer.json"
    base_json = json.loads(base_path.read_text(encoding="utf-8"))
    if base_json["model"].get("type") != "BPE":
        print(f"ABORT: base model type {base_json['model'].get('type')} is not BPE")
        return 1
    base_tok = Tokenizer.from_file(str(base_path))
    base_vocab = set(base_json["model"]["vocab"])
    print(f"BASE: {base_path} | BPE vocab {len(base_vocab):,} | merges {len(base_json['model']['merges']):,} | "
          f"added tokens {len(base_json.get('added_tokens', [])):,} | ignore_merges "
          f"{base_json['model'].get('ignore_merges')}", flush=True)

    # 1) pre-token counts on the training corpus
    texts = read_texts(args.train)
    chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
    counts: Counter[str] = Counter()
    with Pool(args.workers, initializer=_init_worker, initargs=(str(base_path),)) as pool:
        for c in pool.imap_unordered(_count_chunk, chunks):
            counts.update(c)
    n_words = sum(len(words(t)) for t in texts)
    del texts
    kept = [(p, f) for p, f in counts.items() if f >= args.min_word_freq]
    print(f"PRETOKENS: {sum(counts.values()):,} occurrences, {len(counts):,} distinct, "
          f"{len(kept):,} with freq >= {args.min_word_freq} | corpus words {n_words:,} | {time.time()-t0:.0f}s",
          flush=True)

    # 2) base segmentation, 3) continued BPE
    pieces = [p for p, _ in kept]
    freqs = [f for _, f in kept]
    segs = [[t.value for t in base_tok.model.tokenize(p)] for p in pieces]
    base_tokens = sum(len(s) * f for s, f in zip(segs, freqs))
    ks = sorted(int(k) for k in args.ks.split(","))
    merges = continued_bpe(segs, freqs, base_vocab, ks[-1], args.min_pair_freq)
    final_tokens = sum(len(s) * f for s, f in zip(segs, freqs))
    n_new_total = sum(m[2] for m in merges)
    print(f"CONTINUED-BPE: {len(merges):,} merges, {n_new_total:,} new tokens, "
          f"{len(merges) - n_new_total:,} merges onto existing tokens | training-corpus tokens "
          f"{base_tokens:,} -> {final_tokens:,} ({(final_tokens/base_tokens-1)*100:+.1f}%) | {time.time()-t0:.0f}s",
          flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "continued_bpe_merges.jsonl").open("w", encoding="utf-8") as fh:
        for a, b, is_new, c in merges:
            fh.write(json.dumps({"a": a, "b": b, "new": is_new, "count": c}, ensure_ascii=False) + "\n")

    # 4) write tokenizers per K, evaluate
    orm = [normalize_qubee(s) for s in load_flores(args.flores_orm)]
    eng = load_flores(args.flores_eng)
    held = read_texts(args.heldout)
    rows = [{"K": 0, "vocab_added": 0, **evaluate(base_tok, orm, eng, held)}]
    exts: dict[int, Path] = {}
    for k in ks:
        n, cut = 0, len(merges)
        for idx, m in enumerate(merges):
            n += m[2]
            if n == k:
                cut = idx + 1
                break
        d = args.out / f"qwen35-orm-ext-K{k}"
        added = write_extended(base_json, merges[:cut], d, args.base)
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        rows.append({"K": k, "vocab_added": added, **evaluate(tok, orm, eng, held)})
        exts[k] = d
        print(f"DONE K={k}: added {added:,} tokens ({cut:,} merges)", flush=True)

    # 5) exactness + losslessness checks on the largest K
    big = Tokenizer.from_file(str(exts[ks[-1]] / "tokenizer.json"))
    vocab_ok = rows[-1]["vocab_added"] == min(n_new_total, ks[-1])
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(pieces)), min(5000, len(pieces)))
    mism = sum([t.value for t in big.model.tokenize(pieces[i])] != segs[i] for i in idx)
    sample = rng.sample(held, min(2000, len(held)))
    lossless = sum(big.decode(e.ids, skip_special_tokens=False) == s
                   for e, s in zip(big.encode_batch(sample, add_special_tokens=False), sample))
    print(f"CHECK exactness: {len(idx) - mism}/{len(idx)} sampled pre-tokens segment exactly as in training "
          f"| lossless round-trip {lossless}/{len(sample)} held-out docs | vocab count matches {vocab_ok}")

    base_f = rows[0]["fertility_flores_orm"]
    reach = next((r["K"] for r in rows if r["fertility_flores_orm"] <= args.target), None)
    for r in rows:
        r["fertility_reduction_pct"] = round((1 - r["fertility_flores_orm"] / base_f) * 100, 1)  # type: ignore[operator]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps({"rows": rows, "target": args.target, "first_K_meeting_target": reach,
                                      "exact_mismatches": mism, "lossless": [lossless, len(sample)]}, indent=2))
    keys = [k for k in rows[0] if k != "ci95"]
    with args.table.with_suffix(".csv").open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"{'K':>6}{'added':>7}{'fert_FL':>9}{'95% CI':>15}{'fert_HO':>9}{'fert_EN':>9}{'premium':>8}"
          f"{'chr/tok':>8}{'STRR':>7}{'reduct%':>8}")
    for r in rows:
        ci = f"[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"  # type: ignore[index]
        print(f"{r['K']:>6}{r['vocab_added']:>7}{r['fertility_flores_orm']:>9.3f}{ci:>15}"
              f"{r['fertility_heldout_orm']:>9.3f}{r['fertility_flores_eng']:>9.3f}{r['premium']:>8.2f}"
              f"{r['chars_per_token_orm']:>8.2f}{r['strr_orm']:>7.3f}{r['fertility_reduction_pct']:>8.1f}")
    print(f"GATE T1 (FLORES+ Oromo fertility <= {args.target}): "
          f"{'first met at K=' + str(reach) if reach else 'NOT met at any K tested'} | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
