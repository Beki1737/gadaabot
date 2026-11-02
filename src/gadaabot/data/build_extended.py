"""Build an AfriqueQwen3.5-9B-it checkpoint with an Oromo-extended vocabulary (Gate T2 input).

For each new token t (present in the extended tokenizer, absent from the base one):
    E_in[t]  = mean of E_in[p]  over p in base_tokenize(t)
    E_out[t] = mean of E_out[p] over p in base_tokenize(t)
(Fast Vocabulary Transfer style mean-of-pieces initialization). Base rows, all other
weights and all base token ids are unchanged, so the model is bit-identical to the
base model on any text whose segmentation does not use a new token.

The output is a plain text-only Qwen3_5ForCausalLM checkpoint:
  * tensor names "model.language_model.*" are written as "model.*" (standard causal-LM
    layout, loadable by both transformers and vLLM);
  * embedding rows padded to a multiple of 64; config vocab_size updated;
  * sharded safetensors (<= 5 GB per shard) with an index;
  * tokenizer files taken from the extended tokenizer directory.

Usage: python -m gadaabot.data.build_extended --base DIR --tokenizer DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 5 * 1024**3
COPY_FILES = ("generation_config.json",)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "chat_template.json")


def rename(key: str) -> str:
    return "model." + key[len("model.language_model."):] if key.startswith("model.language_model.") else key


def tensor_files(d: Path) -> list[Path]:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return sorted({d / f for f in json.loads(idx.read_text())["weight_map"].values()})
    return sorted(d.glob("*.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad-to", type=int, default=64)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    t0 = time.time()
    base_json = json.loads((args.base / "tokenizer.json").read_text(encoding="utf-8"))
    ext_json = json.loads((args.tokenizer / "tokenizer.json").read_text(encoding="utf-8"))
    base_tok = Tokenizer.from_file(str(args.base / "tokenizer.json"))
    bv, ev = base_json["model"]["vocab"], ext_json["model"]["vocab"]
    specials = {t["content"] for t in base_json.get("added_tokens", [])}
    new = sorted(((i, s) for s, i in ev.items() if s not in bv and s not in specials))
    if any(bv[s] != ev[s] for s in bv):
        print("ABORT: extended tokenizer renumbers base tokens")
        return 1
    pieces = {i: [t.id for t in base_tok.model.tokenize(s)] for i, s in new}
    max_id = max((i for i, _ in new), default=-1)
    if new:
        print(f"TOKENIZER: {len(new):,} new tokens, ids {new[0][0]:,}..{max_id:,}; pieces per new token "
              f"mean {sum(map(len, pieces.values())) / len(pieces):.2f}, max {max(map(len, pieces.values()))}",
              flush=True)
    else:
        print("TOKENIZER: no new tokens (format conversion only: control checkpoint)", flush=True)

    cfg = json.loads((args.base / "config.json").read_text())
    tied = bool(cfg.get("tie_word_embeddings", False))
    files = tensor_files(args.base)
    names: dict[str, Path] = {}
    for f in files:
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                names[k] = f
    emb_key = next(k for k in names if k.endswith("embed_tokens.weight"))
    head_key = "lm_head.weight" if "lm_head.weight" in names else None
    if head_key is None and not tied:
        print("ABORT: no lm_head.weight and tie_word_embeddings is false")
        return 1

    def load(k: str) -> torch.Tensor:
        with safe_open(str(names[k]), "pt") as fh:
            return fh.get_tensor(k)

    def extend(w: torch.Tensor, label: str) -> torch.Tensor:
        rows, hid = w.shape
        n_rows = max(rows, max_id + 1)
        n_rows = ((n_rows + args.pad_to - 1) // args.pad_to) * args.pad_to
        out = torch.zeros(n_rows, hid, dtype=w.dtype)
        out[:rows] = w
        w32 = w.float()
        for i, ps in pieces.items():
            out[i] = w32[ps].mean(0).to(w.dtype)
        reused = sum(i < rows for i in pieces)
        keep = torch.ones(rows, dtype=torch.bool)
        reuse_ids = torch.tensor([i for i in pieces if i < rows], dtype=torch.long)
        if reuse_ids.numel():
            keep[reuse_ids] = False
        same = torch.equal(out[:rows][keep], w[keep])
        print(f"{label}: {tuple(w.shape)} -> {tuple(out.shape)}; {reused} new ids reuse padding rows; "
              f"base rows unchanged: {same}", flush=True)
        if not same:
            raise RuntimeError(f"{label}: base rows changed")
        return out

    emb = extend(load(emb_key), "embed_tokens")
    head = extend(load(head_key), "lm_head") if head_key else None
    n_rows = emb.shape[0]

    # write shards
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.safetensors"):
        old.unlink()
    shard: dict[str, torch.Tensor] = {}
    size, n_shard, weight_map, total = 0, 0, {}, 0

    def flush() -> None:
        nonlocal shard, size, n_shard
        if not shard:
            return
        n_shard += 1
        fname = f"model-{n_shard:05d}.safetensors"
        save_file(shard, str(args.out / fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        shard, size = {}, 0

    order = sorted(names, key=lambda k: (k != emb_key, k))
    for k in order:
        t = emb if k == emb_key else head if k == head_key else load(k)
        nk = rename(k)
        nbytes = t.numel() * t.element_size()
        if size + nbytes > SHARD_BYTES and shard:
            flush()
        shard[nk] = t.contiguous()
        size += nbytes
        total += nbytes
    flush()
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))

    cfg["vocab_size"] = n_rows
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in COPY_FILES:
        if (args.base / name).exists():
            shutil.copyfile(args.base / name, args.out / name)
    for name in TOKENIZER_FILES:
        if (args.tokenizer / name).exists():
            shutil.copyfile(args.tokenizer / name, args.out / name)
    (args.out / "gadaabot_extension.json").write_text(json.dumps({
        "base": str(args.base), "tokenizer": str(args.tokenizer), "n_new": len(new),
        "new_id_min": new[0][0] if new else None, "new_id_max": max_id if new else None, "vocab_rows": n_rows, "init": "mean_of_base_pieces",
        "renamed_prefix": "model.language_model. -> model.", "tied": tied}, indent=2))
    print(f"WROTE {args.out}: {n_shard} shards, {total / 1024**3:.1f} GiB, vocab_size {n_rows:,} | {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
