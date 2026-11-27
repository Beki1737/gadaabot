#!/usr/bin/env bash
# =============================================================================
# T036_encoders_gemma.sh | GadaaBot 2.0 | Batch B29 | council round 3: E7 + E2b retry
# -----------------------------------------------------------------------------
# E7  retrieval components on the 223 reviewed questions (same set as final Gate R1):
#     BM25 analyzers (full words, prefix 3/4/5/6, light stemmer) and 5 dense encoders
#     (AfriE5, its base multilingual-e5-large-instruct, BGE-M3 dense, gte-multilingual,
#     LaBSE), each alone, fused with prefix-4 BM25 (RRF), and + cross-encoder rerank.
#     Answers "why AfriE5" and "why 4 letters".
# E2b Gemma-3-12B-it backbone, retry. T035 died at engine start: 24 GB of weights
#     left no KV cache at 60% of the card. Now vLLM gets one card alone (85%,
#     12k context); retriever + NLLB run on a second card. Answers judged by the
#     calibrated AfriqueQwen judge (a different family from Gemma).
# GPU:    YES. E7: 1 card (>= 20 GiB). E2b: 1 card (>= 44 GiB) + 1 card (>= 14 GiB).
#         Judge: 1 card (>= 40 GiB). ~3 to 4 h, queued if busy
# Disk:   +2.2 GB (intfloat/multilingual-e5-large-instruct download)
# Needs:  extra.py (new) uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T036_encoders_gemma.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T036.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T036_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T036.pid"; disown -a
  echo "T036 started in background, pid $(cat "$HOME/gadaabot/runs/T036.pid")"
  echo "Result: cat ~/gadaabot/reports/T036.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T036 gb-core
cd "$GB_ROOT"
SP=""
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "$SP" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
E="$GB_ROOT/src/gadaabot/eval"; OUTD="$GB_ROOT/runs/T036"; mkdir -p "$OUTD"
PORT=48000
[ -f "$GB_ROOT/tasks/extra.py" ] || { gb_say "MISSING: upload extra.py"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/extra.py" "$E/"; python -m py_compile "$E/extra.py"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
NLLB="$GB_ROOT/models/facebook/nllb-200-3.3B"
G12="$SHARED_MODELS/Google/gemma-3-12b-Instruct"; AFQ="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
RET=(--qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct"
     --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite")
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
show() { { grep -E "$2" "$1" || grep -E 'Error' "$1" | tail -n 6; } | cut -c1-500 | tee -a "$GB_REPORT" || true; }

# --- 0) download mE5-large-instruct (NO GPU) --------------------------------------------------------------
ME5="$GB_ROOT/models/intfloat/multilingual-e5-large-instruct"
if [ ! -f "$ME5/config.json" ]; then
  HF_HUB_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as s; s('intfloat/multilingual-e5-large-instruct', local_dir='$ME5', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])" \
    2>&1 | tail -n 1 | tee -a "$GB_REPORT" || gb_say "mE5 download failed (E7 will skip it)"
fi

# --- 1) E7 retrieval components (1 GPU, no vLLM) -------------------------------------------------------------
GPU="$(gb_wait_gpus 1 20 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
LOG="$GB_ROOT/logs/T036_e7_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.extra encoders --root "$GB_ROOT" --out-dir "$OUTD" "${RET[@]}" \
  --shared-models "$SHARED_MODELS" > "$LOG" 2>&1 || gb_say "E7 exited non-zero (see $LOG)"
show "$LOG" '^(E7|DONE)'

# --- 2) E2b Gemma-3-12B backbone (vLLM alone on card A; retriever + NLLB on card B) ------------------------
OLD="$(ls -t "$GB_ROOT"/logs/T035_vllm_gemma-3-12b-Instruct_*.log 2>/dev/null | head -n 1 || true)"
[ -n "$OLD" ] && grep -iE "memory|kv cache|cuda|error" "$OLD" | grep -v Traceback | tail -n 3 | sed 's/^/T035-CAUSE /' | cut -c1-300 | tee -a "$GB_REPORT" || true
GA="$(gb_wait_gpus 1 44 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
SLOG="$GB_ROOT/logs/T036_vllm_gemma12_${GB_TS}.log"
( conda activate gb-serve; cd "$GB_ROOT"
  CUDA_VISIBLE_DEVICES="$GA" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve "$G12" \
    --host 127.0.0.1 --port "$PORT" --served-model-name gadaabot-gen --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0}' --seed 7 ) > "$SLOG" 2>&1 &
SP=$!
for i in $(seq 1 90); do
  curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SP" 2>/dev/null || { gb_say "vLLM died again (see $SLOG)"; grep -iE "memory|error" "$SLOG" | tail -n 4 | tee -a "$GB_REPORT"; SP=""; break; }
  sleep 10
done
if [ -n "$SP" ]; then
  gb_say "served gemma-3-12b on GPU $GA"
  GB="$(gb_wait_gpus 1 14 30 240)" || GB=""
  if [ -n "$GB" ] && [ "$GB" != "$GA" ]; then
    AD="$GB_ROOT/runs/T035_gemma3_12b"; ALOG="$GB_ROOT/logs/T036_ablation_gemma12_${GB_TS}.log"
    CUDA_VISIBLE_DEVICES="$GB" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.ablation \
      --llm-url "http://127.0.0.1:$PORT" "${RET[@]}" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
      --exclude-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" --nllb "$NLLB" --nllb-device cuda:0 \
      --arms "closed,closed_tt_self,rag_naive,rag_tt_nllb,rag_tt_self,agent_noverify" --workers 8 --judge-samples 1 \
      --out-dir "$AD" > "$ALOG" 2>&1 || gb_say "ablation gemma12 exited non-zero (see $ALOG)"
    show "$ALOG" '^(HELDOUT|ARM [a-z]|DONE)'
    gb_say "SELFJUDGE gemma3_12b below (self-preference check only):"
    show "$ALOG" '^ARM-RESULT arm'
    MLOG="$GB_ROOT/logs/T036_mt_gemma12_${GB_TS}.log"
    CUDA_VISIBLE_DEVICES="$GB" python -m gadaabot.eval.extra mt --root "$GB_ROOT" --out-dir "$OUTD" --mt-systems self \
      --tag gemma3_12b --llm-url "http://127.0.0.1:$PORT" > "$MLOG" 2>&1 || true
    show "$MLOG" '^(E3|DONE)'
  else
    gb_say "no second card for retriever/NLLB within 4 h; gemma12 skipped"
  fi
  kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true; SP=""
fi

# --- 3) AfriqueQwen judge on the Gemma-12B answers ---------------------------------------------------------
if ls "$GB_ROOT/runs/T035_gemma3_12b"/arm_rag_naive.jsonl >/dev/null 2>&1; then
  GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU; STATUS: PARTIAL"; gb_report_print; exit 0; }
  SLOG="$GB_ROOT/logs/T036_vllm_afq_${GB_TS}.log"
  ( conda activate gb-serve; cd "$GB_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve "$AFQ" \
      --host 127.0.0.1 --port "$PORT" --served-model-name gadaabot-gen --max-model-len 16384 \
      --gpu-memory-utilization 0.72 --seed 7 ) > "$SLOG" 2>&1 &
  SP=$!
  for i in $(seq 1 60); do curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  JLOG="$GB_ROOT/logs/T036_rejudge_gemma12_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.extra rejudge --root "$GB_ROOT" --out-dir "$OUTD" "${RET[@]}" \
    --llm-url "http://127.0.0.1:$PORT" --arms-dir "$GB_ROOT/runs/T035_gemma3_12b" --tag afq_judge \
    --arm-list closed closed_tt_self rag_naive rag_tt_nllb rag_tt_self agent_noverify > "$JLOG" 2>&1 || true
  show "$JLOG" '^(E2|DONE)'
  kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true; SP=""
fi
gb_say "vLLM stopped"
git add src reports/T036.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T036: retrieval components, Gemma-3-12B backbone

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^E7-PAIR' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
