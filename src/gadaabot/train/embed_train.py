"""Stage-1 adaptation: train ONLY the new Oromo token rows of an extended checkpoint.

Everything else is frozen: all transformer weights and every base row of the input
embedding (embed_tokens) and output embedding (lm_head). Base-token behaviour is
therefore preserved exactly; the model only learns to read and write the new tokens.

Mechanics
---------
* The model weights stay bf16. The new rows [lo, hi) of embed_tokens and lm_head have
  fp32 master copies optimized by AdamW; after every optimizer step the masters are
  written back (cast to bf16) into the model. Gradients are taken from the full
  weight-gradient tensors and sliced to [lo, hi), so base rows can never move.
* Data: corpus v3 train docs, tokenized with the extended tokenizer, shuffled (seeded),
  joined with <|endoftext|>, packed into fixed-length sequences. Held-out text is
  never trained on.
* Schedule: linear warm-up, cosine decay to 10% of peak. Token budget and wall-clock
  limit, whichever comes first.
* Evaluation every --eval-every optimizer steps: bits per word (BPW) on a fixed set of
  held-out Oromo docs (same sampling as probe_extended, first --eval-docs of them),
  compared with the base model's BPW on the same docs. The best checkpoint (lowest
  BPW) is the one written.
* Output: a new model directory that symlinks every unchanged shard of the extended
  checkpoint and rewrites only the shard(s) holding embed_tokens / lm_head.

Usage: python -m gadaabot.train.embed_train --model DIR --base-tok DIR --train FILE
       --heldout FILE --flores-eng FILE --out DIR [--max-tokens N --time-limit-h H]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gadaabot.data.qubee import words
from gadaabot.eval.probe_extended import bpw

EMB_KEY, HEAD_KEY = "model.embed_tokens.weight", "lm_head.weight"


def eval_docs(heldout: Path, n_docs: int, max_words: int, seed: int) -> list[str]:
    """Same selection as probe_extended (seed 3, docs >= 50 words, first max_words words)."""
    rng = random.Random(seed)
    held = [json.loads(l)["text"] for l in gzip.open(heldout, "rt", encoding="utf-8")]
    held = [h for h in held if len(words(h)) >= 50]
    return [" ".join(h.split(" ")[:max_words]) for h in rng.sample(held, min(150, len(held)))][:n_docs]


def pack_corpus(tok, path: Path, eot: int, seq_len: int, seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    texts = [json.loads(l)["text"] for l in gzip.open(path, "rt", encoding="utf-8")]
    random.Random(seed).shuffle(texts)
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), 4096):
        for e in tok.encode_batch(texts[i:i + 4096], add_special_tokens=False):
            parts.append(np.asarray(e.ids + [eot], dtype=np.int32))
    flat = np.concatenate(parts)
    n = len(flat) // seq_len
    return flat[: n * seq_len].reshape(n, seq_len)


def save_checkpoint(src: Path, dst: Path, new_tensors: dict[str, torch.Tensor], meta: dict[str, object]) -> None:
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm: dict[str, str] = idx["weight_map"]
    targets = {wm[k] for k in new_tensors}
    for f in sorted(set(wm.values())):
        out = dst / f
        if out.exists() or out.is_symlink():
            out.unlink()
        if f in targets:
            tensors = load_file(str(src / f))
            for k, v in new_tensors.items():
                if wm[k] == f:
                    if tensors[k].shape != v.shape:
                        raise RuntimeError(f"shape mismatch for {k}: {tensors[k].shape} vs {v.shape}")
                    tensors[k] = v.contiguous()
            save_file(tensors, str(out), metadata={"format": "pt"})
        else:
            os.symlink((src / f).resolve(), out)
    for p in src.iterdir():
        if p.suffix != ".safetensors" and p.is_file():
            shutil.copyfile(p, dst / p.name)
    (dst / "gadaabot_stage1.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--base-tok", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--flores-eng", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    ap.add_argument("--time-limit-h", type=float, default=6.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-docs", type=int, default=50)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    t_start = time.time()
    torch.manual_seed(args.seed)
    meta_ext = json.loads((args.model / "gadaabot_extension.json").read_text())
    lo, hi = int(meta_ext["new_id_min"]), int(meta_ext["new_id_max"]) + 1
    ext_tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    base_tok = Tokenizer.from_file(str(args.base_tok / "tokenizer.json"))
    eot = ext_tok.token_to_id("<|endoftext|>")
    assert eot == base_tok.token_to_id("<|endoftext|>")

    data = pack_corpus(ext_tok, args.train, eot, args.seq_len, args.seed)
    tokens_per_step = args.micro_batch * args.accum * args.seq_len
    total_steps = min(args.max_tokens // tokens_per_step, len(data) // (args.micro_batch * args.accum))
    print(f"DATA: {data.size:,} train tokens in {len(data):,} sequences of {args.seq_len} | "
          f"{tokens_per_step:,} tokens/step, {total_steps:,} steps planned "
          f"({total_steps * tokens_per_step / data.size:.2f} epochs) | {time.time()-t_start:.0f}s", flush=True)

    docs = eval_docs(args.heldout, args.eval_docs, args.max_words, seed=3)
    w_docs = [len(words(d)) for d in docs]
    ev_ext = [[eot] + e.ids for e in ext_tok.encode_batch(docs, add_special_tokens=False)]
    ev_base = [[eot] + e.ids for e in base_tok.encode_batch(docs, add_special_tokens=False)]

    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16, device_map="cuda")
    dev = model.device
    E, H = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    rows = H.shape[0]
    pad = torch.zeros(rows, dtype=torch.bool, device=dev)
    pad[hi:] = True
    base_mask = pad.clone()
    base_mask[lo:hi] = True

    model.eval()
    bpw_base = bpw(model, ev_base, w_docs, base_mask)
    bpw_init = bpw(model, ev_ext, w_docs, pad)
    print(f"EVAL step 0: BPW base {bpw_base:.4f} | extended init {bpw_init:.4f}", flush=True)
    torch.cuda.empty_cache()

    for p in model.parameters():
        p.requires_grad_(False)
    E.requires_grad_(True)
    H.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    m_in = torch.nn.Parameter(E.data[lo:hi].float().clone())
    m_out = torch.nn.Parameter(H.data[lo:hi].float().clone())
    opt = torch.optim.AdamW([m_in, m_out], lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    args.log.parent.mkdir(parents=True, exist_ok=True)
    logf = args.log.open("w", encoding="utf-8")
    logf.write(json.dumps({"step": 0, "tokens": 0, "bpw": bpw_init, "bpw_base": bpw_base}) + "\n")
    best = (bpw_init, 0, m_in.detach().cpu().clone(), m_out.detach().cpu().clone())
    order = np.random.default_rng(args.seed).permutation(len(data))
    cursor, seen, step = 0, 0, 0
    t_train = time.time()
    stop_reason = "token budget"
    model.train()
    while step < total_steps:
        loss_sum = 0.0
        for _ in range(args.accum):
            idx = order[cursor:cursor + args.micro_batch]
            cursor += args.micro_batch
            x = torch.from_numpy(data[idx].astype(np.int64)).to(dev)
            loss = model(input_ids=x, labels=x).loss / args.accum
            loss.backward()
            loss_sum += float(loss)
        m_in.grad = E.grad[lo:hi].float()
        m_out.grad = H.grad[lo:hi].float()
        gnorm = float(torch.nn.utils.clip_grad_norm_([m_in, m_out], 1.0))
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        E.grad = None
        H.grad = None
        with torch.no_grad():
            E.data[lo:hi] = m_in.data.to(E.dtype)
            H.data[lo:hi] = m_out.data.to(H.dtype)
        step += 1
        seen += tokens_per_step
        el = time.time() - t_train
        if step % 10 == 0:
            print(f"step {step}/{total_steps} loss {loss_sum:.4f} gnorm {gnorm:.3f} lr {lr_at(step):.2e} "
                  f"tok/s {seen / el:,.0f} elapsed {el / 3600:.2f} h", flush=True)
        if step % args.eval_every == 0 or step == total_steps:
            model.eval()
            b = bpw(model, ev_ext, w_docs, pad)
            torch.cuda.empty_cache()
            model.train()
            logf.write(json.dumps({"step": step, "tokens": seen, "loss": loss_sum, "bpw": b,
                                   "bpw_base": bpw_base, "elapsed_h": el / 3600}) + "\n")
            logf.flush()
            print(f"EVAL step {step}: BPW {b:.4f} (base {bpw_base:.4f}, gap {b - bpw_base:+.4f}) | "
                  f"tokens {seen:,}", flush=True)
            if b < best[0]:
                best = (b, step, m_in.detach().cpu().clone(), m_out.detach().cpu().clone())
        if time.time() - t_start > args.time_limit_h * 3600:
            stop_reason = "time limit"
            break
    logf.close()

    # restore best rows, final metrics on the full 150-doc probe set and English
    with torch.no_grad():
        E.data[lo:hi] = best[2].to(dev, E.dtype)
        H.data[lo:hi] = best[3].to(dev, H.dtype)
    model.eval()
    full = eval_docs(args.heldout, 150, args.max_words, seed=3)
    wf = [len(words(d)) for d in full]
    fb = [[eot] + e.ids for e in base_tok.encode_batch(full, add_special_tokens=False)]
    fe = [[eot] + e.ids for e in ext_tok.encode_batch(full, add_special_tokens=False)]
    eng = [json.loads(l)["text"] for l in args.flores_eng.read_text(encoding="utf-8").splitlines() if l.strip()]
    eng_docs = [" ".join(eng[i:i + 10]) for i in range(0, 300, 10)]
    we = [len(words(d)) for d in eng_docs]
    ee = [[eot] + e.ids for e in ext_tok.encode_batch(eng_docs, add_special_tokens=False)]
    eb = [[eot] + e.ids for e in base_tok.encode_batch(eng_docs, add_special_tokens=False)]
    final = {"bpw_orm_base": bpw(model, fb, wf, base_mask), "bpw_orm_trained": bpw(model, fe, wf, pad),
             "bpw_eng_base": bpw(model, eb, we, base_mask), "bpw_eng_trained": bpw(model, ee, we, pad),
             "tokens_orm_base": sum(map(len, fb)), "tokens_orm_ext": sum(map(len, fe))}
    final = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()}
    print(f"FINAL (150 docs, best step {best[1]}): Oromo BPW base {final['bpw_orm_base']} | trained "
          f"{final['bpw_orm_trained']} | English base {final['bpw_eng_base']} | trained {final['bpw_eng_trained']} | "
          f"Oromo tokens {final['tokens_orm_base']:,} -> {final['tokens_orm_ext']:,}", flush=True)

    meta = {"source_model": str(args.model), "trained_rows": [lo, hi], "best_step": best[1], "best_eval_bpw": best[0],
            "steps_run": step, "tokens_seen": seen, "stop_reason": stop_reason, "final": final,
            "hparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "hours": round((time.time() - t_start) / 3600, 2)}
    emb_full = E.data.detach().cpu()
    head_full = H.data.detach().cpu()
    save_checkpoint(args.model, args.out, {EMB_KEY: emb_full, HEAD_KEY: head_full}, meta)
    print(f"SAVED {args.out} | stop: {stop_reason} | steps {step} | tokens {seen:,} | "
          f"{meta['hours']} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
