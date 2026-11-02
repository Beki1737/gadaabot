"""Chat sanity probe: does continued pre-training keep the instruct model's ability to
answer a user in Afaan Oromoo? Same prompts, chat template, non-thinking sampling (D23).
Writes a TSV for native-speaker judgement (fluency, correctness, follows the question).

Usage: python -m gadaabot.eval.chat_probe --model DIR --label NAME --out FILE
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROMPTS = [
    ("health", "Mallattoon dhukkuba busaa maal fa'a? Akkamitti of eeguu dandeenya?"),
    ("health", "Daa'imni koo garaa kaasaa qaba. Mana keessatti maal gochuu qaba, yoom gara buufata fayyaa deemuu qaba?"),
    ("health", "Talaalliin daa'immaniif maaliif barbaachisaa dha?"),
    ("education", "Foto-sinteesisii (photosynthesis) jechuun maal jechuu dha? Akka barataa kutaa 6ffaaf ibsi."),
    ("education", "Sirni Gadaa maal? Seenaa isaa gabaabsii ibsi."),
    ("education", "Bishaan qulqulluu dhuguun fayyaa keenyaaf faayidaa maal qaba?"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.80)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(str(args.model))
    msgs = [[{"role": "system", "content": "Ati gargaaraa Afaan Oromoo ti. Afaan Oromootiin qofa deebisi."},
             {"role": "user", "content": q}] for _, q in PROMPTS]
    texts = []
    for m in msgs:
        try:
            texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
    llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=args.gpu_mem, seed=7)
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=320, seed=7)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        if new:
            w.writerow(["model", "domain", "prompt", "answer", "n_tokens", "fluent_1to5", "correct_1to5", "notes"])
        for (dom, q), o in zip(PROMPTS, outs):
            a = o.outputs[0].text.strip().replace("\t", " ")
            w.writerow([args.label, dom, q, a.replace("\n", " | "), len(o.outputs[0].token_ids), "", "", ""])
            print(f"CHAT {args.label} [{dom}] {q[:50]}... -> {len(o.outputs[0].token_ids)} tokens: {a[:160]!r}", flush=True)
    print(f"CHAT-DONE {args.label} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    sys.stdout.flush()
    os._exit(rc)  # vLLM can hang at interpreter shutdown (seen in T010)
