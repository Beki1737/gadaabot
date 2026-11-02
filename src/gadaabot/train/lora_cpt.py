"""Stage-2 adaptation: LoRA continued pre-training on Oromo (+ new token rows, if any).

Used for two arms that see the SAME Oromo text (same shuffled doc order, same fraction
of the corpus), so their difference isolates the vocabulary extension:
  control   AfriqueQwen3.5-9B-it with its original tokenizer (K0 checkpoint) + LoRA
  extended  the stage-1 K4096 checkpoint + LoRA + continued training of the new rows
Both are scored in bits per word (tokenizer-independent) on held-out Oromo and on English
(forgetting canary), against the untouched base model.

Trainable: LoRA (rank r) on every linear layer of every decoder block (attention, linear
attention, MLP; not the embeddings or lm_head), and, for the extended arm, fp32 masters of
the new rows of embed_tokens and lm_head (base rows frozen, as in stage 1). Base weights
stay bf16 and frozen.

Output dir: adapter/ (PEFT LoRA), new_rows.safetensors (extended arm), curve.jsonl,
stage2.json (metrics + hyper-parameters). Merging for vLLM evaluation is a later step.

Usage: python -m gadaabot.train.lora_cpt --model DIR --base-tok DIR --train FILE
       --heldout FILE --flores-eng FILE --out DIR --data-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gadaabot.data.qubee import words
from gadaabot.eval.probe_extended import bpw
from gadaabot.train.embed_train import eval_docs, pack_corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--base-tok", type=Path, required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--flores-eng", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-fraction", type=float, default=0.5)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-rows", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--time-limit-h", type=float, default=9.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-docs", type=int, default=50)
    ap.add_argument("--max-words", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    t_start = time.time()
    torch.manual_seed(args.seed)
    meta_ext = json.loads((args.model / "gadaabot_extension.json").read_text())
    has_new = meta_ext.get("new_id_min") is not None
    lo, hi = (int(meta_ext["new_id_min"]), int(meta_ext["new_id_max"]) + 1) if has_new else (0, 0)
    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    base_tok = Tokenizer.from_file(str(args.base_tok / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    if eot != base_tok.token_to_id("<|endoftext|>"):
        print("ABORT: <|endoftext|> id differs between model tokenizer and base tokenizer")
        return 1

    # same doc order for both arms (seeded shuffle of docs); same fraction of the text
    data = pack_corpus(tok, args.train, eot, args.seq_len, args.seed)
    n_seq = int(len(data) * args.data_fraction)
    data = data[:n_seq]
    tps = args.micro_batch * args.accum * args.seq_len
    total_steps = n_seq // (args.micro_batch * args.accum)
    print(f"DATA: {n_seq:,} sequences x {args.seq_len} = {data.size:,} tokens ({args.data_fraction:.0%} of corpus "
          f"text) | {tps:,} tokens/step, {total_steps:,} steps | new rows {hi - lo:,} | {time.time()-t_start:.0f}s",
          flush=True)

    docs = eval_docs(args.heldout, args.eval_docs, args.max_words, seed=3)
    w_docs = [len(words(d)) for d in docs]
    ev_m = [[eot] + e.ids for e in tok.encode_batch(docs, add_special_tokens=False)]

    model, info = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16, device_map="cuda",
                                                       output_loading_info=True)
    if info.get("missing_keys") or info.get("unexpected_keys"):
        print(f"ABORT: loading keys missing {list(info['missing_keys'])[:3]} unexpected {list(info['unexpected_keys'])[:3]}")
        return 1
    dev = model.device
    rows = model.get_output_embeddings().weight.shape[0]
    pad = torch.zeros(rows, dtype=torch.bool, device=dev)
    pad[(hi if has_new else len(base_tok.get_vocab())):] = True  # rows that are never targets
    model.eval()
    bpw0 = bpw(model, ev_m, w_docs, pad)
    print(f"EVAL step 0: BPW {bpw0:.4f}", flush=True)
    torch.cuda.empty_cache()

    for p in model.parameters():
        p.requires_grad_(False)
    E, H = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    lora_params = []
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
            lora_params.append(p)
    n_lora = sum(p.numel() for p in lora_params)
    groups = [{"params": lora_params, "lr": args.lr, "base_lr": args.lr}]
    masters: list[torch.nn.Parameter] = []
    if has_new:
        E.requires_grad_(True)
        H.requires_grad_(True)
        masters = [torch.nn.Parameter(E.data[lo:hi].float().clone()), torch.nn.Parameter(H.data[lo:hi].float().clone())]
        groups.append({"params": masters, "lr": args.lr_rows, "base_lr": args.lr_rows})
    else:
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    print(f"TRAINABLE: LoRA r={args.rank} {n_lora / 1e6:.1f}M params on all decoder linears"
          f"{' + new rows ' + format(2 * (hi - lo) * E.shape[1] / 1e6, '.1f') + 'M' if has_new else ''}", flush=True)

    def sched(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    args.out.mkdir(parents=True, exist_ok=True)
    logf = (args.out / "curve.jsonl").open("w", encoding="utf-8")
    logf.write(json.dumps({"step": 0, "tokens": 0, "bpw": bpw0}) + "\n")
    order = np.random.default_rng(args.seed).permutation(n_seq)
    cursor, seen, step, stop = 0, 0, 0, "data budget"
    t_train = time.time()
    model.train()
    while step < total_steps:
        loss_sum = 0.0
        for _ in range(args.accum):
            x = torch.from_numpy(data[order[cursor:cursor + args.micro_batch]].astype(np.int64)).to(dev)
            cursor += args.micro_batch
            loss = model(input_ids=x, labels=x).loss / args.accum
            loss.backward()
            loss_sum += float(loss)
        if has_new:
            masters[0].grad = E.grad[lo:hi].float()
            masters[1].grad = H.grad[lo:hi].float()
        gnorm = float(torch.nn.utils.clip_grad_norm_(lora_params + masters, 1.0))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * sched(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if has_new:
            E.grad = None
            H.grad = None
            with torch.no_grad():
                E.data[lo:hi] = masters[0].data.to(E.dtype)
                H.data[lo:hi] = masters[1].data.to(H.dtype)
        step += 1
        seen += tps
        el = time.time() - t_train
        if step % 10 == 0:
            print(f"step {step}/{total_steps} loss {loss_sum:.4f} gnorm {gnorm:.3f} lr x{sched(step):.2f} "
                  f"tok/s {seen / el:,.0f} elapsed {el / 3600:.2f} h", flush=True)
        if step % args.eval_every == 0 or step == total_steps:
            model.eval()
            b = bpw(model, ev_m, w_docs, pad)
            torch.cuda.empty_cache()
            model.train()
            logf.write(json.dumps({"step": step, "tokens": seen, "loss": loss_sum, "bpw": b,
                                   "elapsed_h": el / 3600}) + "\n")
            logf.flush()
            print(f"EVAL step {step}: BPW {b:.4f} (start {bpw0:.4f}) | tokens {seen:,}", flush=True)
        if time.time() - t_start > args.time_limit_h * 3600:
            stop = "time limit"
            break
    logf.close()

    # final: 150-doc Oromo set + English canary
    model.eval()
    full = eval_docs(args.heldout, 150, args.max_words, seed=3)
    wf = [len(words(d)) for d in full]
    fm = [[eot] + e.ids for e in tok.encode_batch(full, add_special_tokens=False)]
    eng = [json.loads(l)["text"] for l in args.flores_eng.read_text(encoding="utf-8").splitlines() if l.strip()]
    eng_docs = [" ".join(eng[i:i + 10]) for i in range(0, 300, 10)]
    we = [len(words(d)) for d in eng_docs]
    em = [[eot] + e.ids for e in tok.encode_batch(eng_docs, add_special_tokens=False)]
    final = {"bpw_orm": round(bpw(model, fm, wf, pad), 4), "bpw_eng": round(bpw(model, em, we, pad), 4),
             "tokens_orm": sum(map(len, fm))}
    print(f"FINAL (150 docs): Oromo BPW {final['bpw_orm']} | English BPW {final['bpw_eng']} | "
          f"Oromo tokens {final['tokens_orm']:,}", flush=True)

    model.save_pretrained(str(args.out / "adapter"))
    if has_new:
        save_file({"embed_rows": E.data[lo:hi].detach().cpu().contiguous(),
                   "lm_head_rows": H.data[lo:hi].detach().cpu().contiguous()}, str(args.out / "new_rows.safetensors"))
    (args.out / "stage2.json").write_text(json.dumps({
        "source_model": str(args.model), "new_rows": [lo, hi], "steps": step, "tokens_seen": seen, "stop": stop,
        "lora_params_m": round(n_lora / 1e6, 1), "final": final, "bpw_start_50docs": bpw0,
        "hparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "hours": round((time.time() - t_start) / 3600, 2)}, indent=2))
    print(f"SAVED {args.out} | stop: {stop} | steps {step} | tokens {seen:,} | "
          f"{(time.time() - t_start) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
