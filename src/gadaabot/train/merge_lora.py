"""Merge a stage-2 LoRA adapter (and, for the extended arm, its trained new rows) into a
full bf16 checkpoint that vLLM / lm-eval can load directly. Runs on CPU.

Usage: python -m gadaabot.train.merge_lora --base DIR --run DIR --out DIR
  --base  checkpoint the adapter was trained on (K0 control, or K4096-s1 extended)
  --run   runs/T016_<arm> (adapter/, optional new_rows.safetensors, stage2.json)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(args.base), dtype=torch.bfloat16, device_map="cpu")
    st = json.loads((args.run / "stage2.json").read_text())
    rows = args.run / "new_rows.safetensors"
    if rows.exists():
        lo, hi = st["new_rows"]
        t = load_file(str(rows))
        with torch.no_grad():
            model.get_input_embeddings().weight[lo:hi] = t["embed_rows"].to(torch.bfloat16)
            model.get_output_embeddings().weight[lo:hi] = t["lm_head_rows"].to(torch.bfloat16)
        print(f"ROWS: restored trained new rows {lo}..{hi - 1}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.run / "adapter"))
    model = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True, max_shard_size="5GB")
    for p in args.base.iterdir():
        if p.is_file() and (p.name.startswith("tokenizer") or p.name in (
                "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "generation_config.json",
                "gadaabot_extension.json")):
            shutil.copyfile(p, args.out / p.name)
    (args.out / "gadaabot_stage2_merged.json").write_text(json.dumps(
        {"base": str(args.base), "run": str(args.run), "final_bpw": st.get("final"), "merged_on": "cpu"}, indent=2))
    print(f"MERGED {args.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
