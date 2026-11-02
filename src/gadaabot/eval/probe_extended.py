"""Probe extended-vocabulary checkpoints before any training (GPU, one card).

For every extended checkpoint it measures, with transformers (bf16, no training):
  1. Loading: missing / unexpected keys must both be empty.
  2. Bits per word (BPW) on corpus held-out Oromo docs and FLORES+ English sentences:
       base   = base tokenizer, logits of the new ids masked out of the softmax
                (the base model's own behaviour; body and base rows are unchanged)
       init   = extended tokenizer, full softmax (mean-of-pieces init, untrained)
     BPW = total NLL in bits / number of words, so it is comparable across tokenizers.
     Every document is prefixed with <|endoftext|> so all of its tokens are scored.
  3. Training feasibility for embedding-only adaptation: gradient checkpointing, only
     embed_tokens and lm_head trainable, forward+backward on 2 x 2048 tokens; reports
     seconds per step, tokens/s and peak memory, and the fast-kernel status.

Usage: python -m gadaabot.eval.probe_extended --models K4096=DIR,K16384=DIR --base-tok DIR
       --heldout FILE --flores-eng FILE --out FILE
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from gadaabot.data.qubee import words


def bpw(model, ids_list: list[list[int]], n_words: list[int], mask: torch.Tensor | None) -> float:  # type: ignore[no-untyped-def]
    total_bits = 0.0
    for ids in ids_list:
        x = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        if mask is not None:
            logits[:, mask] = float("-inf")
        lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None]).squeeze(1)
        total_bits += float(-lp.sum()) / math.log(2)
        del logits, lp
    return total_bits / sum(n_words)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True)
    ap.add_argument("--base-tok", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--flores-eng", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-docs", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--skip-bpw", action="store_true", help="only the training probe")
    ap.add_argument("--train-shapes", default="2x2048", help="micro-batch shapes, e.g. 1x2048,2x2048")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    rng = random.Random(args.seed)
    held = [json.loads(l)["text"] for l in gzip.open(args.heldout, "rt", encoding="utf-8")]
    held = [h for h in held if len(words(h)) >= 50]
    docs = []
    for h in rng.sample(held, min(args.n_docs, len(held))):
        parts = h.split(" ")
        docs.append(" ".join(parts[: args.max_words]))
    eng = [json.loads(l)["text"] for l in args.flores_eng.read_text(encoding="utf-8").splitlines() if l.strip()]
    eng_docs = [" ".join(eng[i:i + 10]) for i in range(0, 300, 10)]
    w_orm = [len(words(d)) for d in docs]
    w_eng = [len(words(d)) for d in eng_docs]
    base_tok = Tokenizer.from_file(str(args.base_tok / "tokenizer.json"))
    eot = base_tok.token_to_id("<|endoftext|>")
    print(f"DATA: {len(docs)} held-out Oromo docs ({sum(w_orm):,} words), {len(eng_docs)} English docs "
          f"({sum(w_eng):,} words); prefix id {eot}", flush=True)

    kernels = {}
    for m in ("fla", "causal_conv1d", "flash_attn"):
        try:
            __import__(m)
            kernels[m] = "ok"
        except Exception as exc:  # noqa: BLE001
            kernels[m] = f"missing ({type(exc).__name__})"
    print("KERNELS:", kernels, flush=True)

    results = {"kernels": kernels, "models": {}}
    for spec in args.models.split(","):
        label, path = spec.split("=", 1)
        t0 = time.time()
        model, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                                           output_loading_info=True)
        model.eval()
        miss, unexp = info.get("missing_keys", []), info.get("unexpected_keys", [])
        print(f"LOAD {label}: {time.time()-t0:.0f}s | missing {len(miss)} {list(miss)[:3]} | "
              f"unexpected {len(unexp)} {list(unexp)[:3]}", flush=True)
        ext_tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        meta = json.loads((Path(path) / "gadaabot_extension.json").read_text())
        new_ids = torch.arange(meta["new_id_min"], meta["new_id_max"] + 1, device=model.device)
        vocab_rows = model.get_output_embeddings().weight.shape[0]
        pad = torch.zeros(vocab_rows, dtype=torch.bool, device=model.device)
        pad[meta["new_id_max"] + 1:] = True      # zero padding rows added by the extension: never targets
        mask = pad.clone()
        mask[new_ids] = True                      # base view: new tokens removed from the softmax

        def enc(tok, xs: list[str]) -> list[list[int]]:  # type: ignore[no-untyped-def]
            return [[eot] + e.ids for e in tok.encode_batch(xs, add_special_tokens=False)]

        r = {"missing_keys": len(miss), "unexpected_keys": len(unexp)}
        ob, oe = enc(base_tok, docs), enc(ext_tok, docs)
        eb, ee = enc(base_tok, eng_docs), enc(ext_tok, eng_docs)
        r["tokens_orm_base"], r["tokens_orm_ext"] = sum(map(len, ob)), sum(map(len, oe))
        if not args.skip_bpw:
            r["bpw_orm_base"] = round(bpw(model, ob, w_orm, mask), 4)
            r["bpw_orm_init"] = round(bpw(model, oe, w_orm, pad), 4)
            r["bpw_eng_base"] = round(bpw(model, eb, w_eng, mask), 4)
            r["bpw_eng_init"] = round(bpw(model, ee, w_eng, pad), 4)
            print(f"BPW {label}: Oromo base {r['bpw_orm_base']} -> init {r['bpw_orm_init']} "
                  f"(tokens {r['tokens_orm_base']:,} -> {r['tokens_orm_ext']:,}) | English base {r['bpw_eng_base']} "
                  f"-> init {r['bpw_eng_init']}", flush=True)
        torch.cuda.empty_cache()

        # training feasibility probe (embedding-only)
        for p in model.parameters():
            p.requires_grad_(False)
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
        flat = [i for ids in oe for i in ids]
        for shape in args.train_shapes.split(","):
            mb, sl = (int(v) for v in shape.lower().split("x"))
            batch = torch.tensor([flat[i * sl:(i + 1) * sl] for i in range(mb)], device=model.device)
            torch.cuda.empty_cache()
            free, tot = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            times = []
            try:
                for step in range(4):
                    torch.cuda.synchronize()
                    s0 = time.time()
                    loss = model(input_ids=batch, labels=batch).loss
                    loss.backward()
                    torch.cuda.synchronize()
                    times.append(time.time() - s0)
                    model.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"TRAIN-PROBE {label} {shape}: FAILED at step {len(times)} ({str(exc)[:120]}) | "
                      f"GPU free before {free / 1024**3:.1f} of {tot / 1024**3:.1f} GiB", flush=True)
                r[f"train_{shape}"] = "failed"
                continue
            step_s = sum(times[2:]) / len(times[2:])
            r[f"train_{shape}"] = {"s_per_step": round(step_s, 2), "tok_per_s": round(batch.numel() / step_s),
                                   "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
                                   "loss": round(float(loss), 3)}
            print(f"TRAIN-PROBE {label} {shape}: {step_s:.2f} s/step, {batch.numel() / step_s:,.0f} tok/s, "
                  f"peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB, loss {float(loss):.3f} | "
                  f"GPU free before {free / 1024**3:.1f} GiB | steps {[round(t, 1) for t in times]}", flush=True)
        results["models"][label] = r
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
