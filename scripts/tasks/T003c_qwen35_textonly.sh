#!/usr/bin/env bash
# =============================================================================
# T003c_qwen35_textonly.sh | GadaaBot 2.0 | Batch B02 (fix for the 9B model)
# -----------------------------------------------------------------------------
# Facts (from the header check): the McGill checkpoint holds ONLY text weights
#   (426 x model.language_model.* + lm_head; 0 visual, 0 MTP), but config.json
#   names the vision-language class Qwen3_5ForConditionalGeneration, so vLLM
#   builds a vision tower that has nothing to load.
# Fix A (zero extra disk): a "text view" folder = flattened text_config with
#   architectures=Qwen3_5ForCausalLM + symlinks to the original files.
# Fix B (only if A fails on weight names): write a renamed copy of the weights
#   (model.language_model.X -> model.X), about +18 GiB, original left untouched.
# Then: chat-mode smoke test (Oromo question, chat template, thinking off),
#   reporting whether the instruct model stops by itself (finish_reason).
# Tool:   MobaXterm terminal, inside tmux        Time: 5 to 20 minutes
# GPU:    YES: 1 card, >= 40 GiB free, <= 30% util (auto-picked). Building the
#         view and the optional rename use NO GPU.
# Run:    bash ~/gadaabot/tasks/T003c_qwen35_textonly.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T003c gb-serve
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR

SRC="$GB_ROOT/models/McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
VIEW="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
CONV="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-text"
SMOKE="$GB_ROOT/runs/T003c_chat_smoke.py"

# ---------------------------------------------------------------------------
# helper: build a text-only folder (NO GPU)
#   $1 = target dir, $2 = "link" (symlink weights) or "rename" (rewrite keys)
# ---------------------------------------------------------------------------
build_textonly() {
python - "$SRC" "$1" "$2" <<'PY'
import json, os, sys, pathlib
src, dst, mode = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
dst.mkdir(parents=True, exist_ok=True)
cfg = json.loads((src / "config.json").read_text())
tc = dict(cfg["text_config"])
tc["architectures"] = ["Qwen3_5ForCausalLM"]
tc["model_type"] = tc.get("model_type", "qwen3_5_text")
tc["tie_word_embeddings"] = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))
tc["mtp_num_hidden_layers"] = 0          # checkpoint carries no MTP weights
for k in ("eos_token_id", "pad_token_id", "dtype"):
    if k in cfg and k not in tc:
        tc[k] = cfg[k]
(dst / "config.json").write_text(json.dumps(tc, indent=2))
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "generation_config.json", "special_tokens_map.json", "README.md"):
    s, d = src / name, dst / name
    if s.exists() and not d.exists():
        d.symlink_to(s)
w = dst / "model.safetensors"
if mode == "link":
    if not w.exists():
        w.symlink_to(src / "model.safetensors")
    print(f"VIEW built (symlinked weights): {dst}")
else:
    from safetensors import safe_open
    from safetensors.torch import save_file
    if w.is_symlink():
        w.unlink()
    out = {}
    with safe_open(str(src / "model.safetensors"), framework="pt") as fh:
        for k in fh.keys():
            nk = k.replace("model.language_model.", "model.", 1)
            out[nk] = fh.get_tensor(k)
    save_file(out, str(w), metadata={"format": "pt"})
    print(f"RENAMED copy written: {w} ({w.stat().st_size/1024**3:.1f} GiB, {len(out)} tensors)")
PY
}

# ---------------------------------------------------------------------------
# chat smoke test script (entry point guarded for vLLM spawn)
# ---------------------------------------------------------------------------
cat > "$SMOKE" <<'PY'
"""Chat-mode smoke test for an instruct model under vLLM."""
import sys
import time


def main() -> None:
    from vllm import LLM, SamplingParams

    model_path, label = sys.argv[1], sys.argv[2]
    questions = [
        "Mallattoon busaa maal fa'a?",
        "Fotosinteesiin maali? Akka barsiisaatti salphaatti ibsi.",
        "Sirni Gadaa maali?",
        "Dhukkuba sukkaaraa akkamitti ittisuun danda'ama?",
    ] * 4
    msgs = [[{"role": "user", "content": q}] for q in questions]
    t0 = time.time()
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, trust_remote_code=False)
    load_s = time.time() - t0
    params = SamplingParams(temperature=0.0, max_tokens=384)
    t1 = time.time()
    outs = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False})
    gen_s = time.time() - t1
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    stopped = sum(o.outputs[0].finish_reason == "stop" for o in outs)
    think = sum("<think>" in o.outputs[0].text for o in outs)
    print(f"SMOKE OK {label}: load {load_s:.0f}s, {len(msgs)} chats, {n_tok} tokens in "
          f"{gen_s:.1f}s = {n_tok / gen_s:.0f} tok/s; stopped by itself {stopped}/{len(msgs)}; "
          f"<think> blocks {think}", flush=True)
    for q, o in zip(questions[:4], outs[:4]):
        a = o.outputs[0].text.replace("\n", " ").strip()[:220]
        print(f"QA {label}: Q: {q} | A: {a}", flush=True)


if __name__ == "__main__":
    main()
PY

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
build_textonly "$VIEW" link | tee -a "$GB_REPORT"

if ! GPU="$(gb_pick_gpus 1 40 30)"; then
  gb_say "SMOKE: skipped, no free GPU. Re-run later (view build is instant)."
  gb_say "STATUS: PARTIAL"; gb_report_print; exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

try_smoke() {  # $1 = model dir, $2 = label; returns 0 on success
  local run_log="$GB_ROOT/logs/T003c_${2}_${GB_TS}.log"
  echo ">>> chat smoke test $2 on GPU $GPU"
  if python "$SMOKE" "$1" "$2" > "$run_log" 2>&1; then
    grep -E '^(SMOKE|QA)' "$run_log" | tee -a "$GB_REPORT" >&2
    grep -E 'GPU KV cache size' "$run_log" | tail -n 1 | sed 's/^.*\] /  vllm: /' | tee -a "$GB_REPORT" >&2
    return 0
  fi
  gb_say "SMOKE FAIL $2 (log: $run_log); key errors:"
  grep -E 'Error|error' "$run_log" | grep -v '^\s*File' | tail -n 4 | cut -c1-260 | tee -a "$GB_REPORT" >&2
  return 1
}

if try_smoke "$VIEW" "Afrique9B-textview"; then
  gb_say "RESULT: Fix A works (zero extra disk). Serve from: $VIEW"
  gb_say "STATUS: PASS"
else
  LAST="$(ls -t "$GB_ROOT"/logs/T003c_Afrique9B-textview_*.log | head -n 1)"
  if grep -qiE 'not initialized|unexpected key|KeyError|language_model|weight' "$LAST"; then
    FOOT="$(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.0f", $1/1024^3}')"
    if [ "$FOOT" -gt 70 ]; then
      gb_say "FIX B skipped: project at ${FOOT} GiB, a +18 GiB copy would pass 90"
      gb_say "STATUS: PARTIAL"; gb_report_print; exit 0
    fi
    gb_say "FIX B: weight names mismatch; writing renamed copy (no GPU, about 3 to 8 min)"
    CUDA_VISIBLE_DEVICES="" build_textonly "$CONV" rename | tee -a "$GB_REPORT"
    if try_smoke "$CONV" "Afrique9B-renamed"; then
      gb_say "RESULT: Fix B works. Serve from: $CONV"
      gb_say "CLEANUP CANDIDATE: original $SRC/model.safetensors (about 17 GiB) once you confirm"
      gb_say "STATUS: PASS"
    else
      gb_say "STATUS: PARTIAL (both fixes failed; send the two logs)"
    fi
  else
    gb_say "Failure is not about weight names; Fix B not attempted"
    gb_say "STATUS: PARTIAL"
  fi
fi
gb_say "QUOTA: project at $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
gb_report_print
