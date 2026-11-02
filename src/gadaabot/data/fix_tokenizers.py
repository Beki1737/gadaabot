"""Repair extended tokenizer.json files written before the special-token fix (T015c).

Bug: the tokenizers library assigns added (special) tokens ids from len(model vocab) when
it loads a tokenizer.json. In base Qwen3.5 the 33 specials sit at 248044..248076 right
after the 248,044-entry model vocab, so this is invisible. Our extension appended new
tokens at 248077.. without registering the specials in the model vocab, so on load the
specials were re-numbered onto the LAST 33 new-token ids: <|endoftext|> no longer had id
248044, and those 33 new ids decoded as special tokens.

Fix: register every special token in model.vocab under its original id (contiguous id
space). Checks, per file:
  * every special keeps its base id; every vocab id maps back to its own string;
  * encoding of held-out Oromo text gives IDENTICAL ids before and after the repair
    (so fertility, the T014 checkpoints and T014/T015a probes are unaffected);
  * lossless round-trip on all held-out docs.
The original file is kept as tokenizer.json.pre_T015c.

Usage: python -m gadaabot.data.fix_tokenizers --base DIR --heldout FILE DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    base = json.loads((args.base / "tokenizer.json").read_text(encoding="utf-8"))
    base_special = {t["content"]: t["id"] for t in base.get("added_tokens", [])}
    held = [json.loads(l)["text"] for l in gzip.open(args.heldout, "rt", encoding="utf-8")]
    plain = [h for h in held if not any(s in h for s in base_special)][:5000]
    ok_all = True
    for d in args.dirs:
        p = d / "tokenizer.json"
        if not p.exists():
            print(f"SKIP {d}: no tokenizer.json")
            continue
        old_tok = Tokenizer.from_file(str(p))
        tj = json.loads(p.read_text(encoding="utf-8"))
        vocab: dict[str, int] = tj["model"]["vocab"]
        for t in tj.get("added_tokens", []):
            have = vocab.setdefault(t["content"], t["id"])
            if have != t["id"]:
                print(f"FAIL {d}: special {t['content']!r} id {have} in vocab vs {t['id']}")
                ok_all = False
                break
        ids = sorted(vocab.values())
        contiguous = ids == list(range(len(ids)))
        bak = d / "tokenizer.json.pre_T015c"
        if not bak.exists():
            shutil.copyfile(p, bak)
        p.write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
        tok = Tokenizer.from_file(str(p))
        sp_ok = all(tok.token_to_id(s) == i for s, i in base_special.items())
        inv_bad = sum(tok.id_to_token(i) != s for s, i in vocab.items())
        same_ids = sum(a.ids == b.ids for a, b in zip(old_tok.encode_batch(plain, add_special_tokens=False),
                                                       tok.encode_batch(plain, add_special_tokens=False)))
        rt_fail = sum(tok.decode(e.ids, skip_special_tokens=False) != h
                      for e, h in zip(tok.encode_batch(held, add_special_tokens=False), held))
        good = sp_ok and inv_bad == 0 and contiguous and same_ids == len(plain) and rt_fail == 0
        ok_all &= good
        print(f"{'OK  ' if good else 'FAIL'} {d.name}: vocab {len(vocab):,} contiguous {contiguous} | specials keep "
              f"base ids {sp_ok} | id->token mismatches {inv_bad} | identical ids on plain text "
              f"{same_ids}/{len(plain)} | round-trip failures {rt_fail}/{len(held)}", flush=True)
    print("TOKENIZER REPAIR:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
