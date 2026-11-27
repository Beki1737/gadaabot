#!/usr/bin/env bash
# =============================================================================
# T038_realq.sh | GadaaBot 2.0 | Batch B31 | real-question set (RQ): screen, answer, judge, blind sheet
# -----------------------------------------------------------------------------
# 1) Leakage screen (pre-stated rules R1 to R3, before any answer exists) + lexical
#    overlap diagnostic: real vs generated questions.
# 2) AfriqueQwen answers all kept RQ questions with the 8 base systems; calibrated
#    judge (ref_ctx, writers' key points as reference, 3 samples).
# 3) Second judge family: Gemma-3-27B re-judges the same answers.
# 4) Blind human sheet: every kept question x 4 systems, one file per annotator.
# GPU:    YES. Step 2: 1 card (>= 60 GiB), ~1 to 1.5 h. Step 3: 2 cards (>= 42 GiB each),
#         ~30 min. Queued if busy. Steps 1 and 4: CPU.
# Needs:  realq.py, extra.py (new), real_questions_raw.jsonl uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T038_realq.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T038.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T038_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T038.pid"; disown -a
  echo "T038 started in background, pid $(cat "$HOME/gadaabot/runs/T038.pid")"
  echo "Result: cat ~/gadaabot/reports/T038.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T038 gb-core
cd "$GB_ROOT"
SP=""
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "$SP" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
E="$GB_ROOT/src/gadaabot/eval"; OUTD="$GB_ROOT/runs/T038"; AD="$GB_ROOT/runs/T038_realq"; mkdir -p "$OUTD" "$AD"
RQD="$GB_ROOT/data/eval/real_questions"; PORT=48000
for f in realq.py extra.py real_questions_raw.jsonl; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f"; gb_report_print; exit 1; }
done
cp -f "$GB_ROOT/tasks/realq.py" "$GB_ROOT/tasks/extra.py" "$E/"; python -m py_compile "$E/realq.py" "$E/extra.py"
mkdir -p "$RQD"; cp -f "$GB_ROOT/tasks/real_questions_raw.jsonl" "$GB_ROOT/data/eval/real_questions_raw.jsonl"
python -c "import openpyxl" 2>/dev/null || pip install -q openpyxl
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
NLLB="$GB_ROOT/models/facebook/nllb-200-3.3B"
AFQ="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"; G27="$SHARED_MODELS/Google/gemma-3-27b-Instruct"
RET=(--qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct"
     --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite")
ARMS="closed,closed_tt_nllb,closed_tt_self,rag_naive,rag_tt_nllb,rag_tt_self,agent_noverify,agent_full"
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
show() { { grep -E "$2" "$1" || grep -E 'Error' "$1" | tail -n 6; } | cut -c1-500 | tee -a "$GB_REPORT" || true; }
serve() {  # serve <model_path> <gpus> <tp> <util> [extra vllm args...]
  local path="$1" gpus="$2" tp="$3" util="$4"; shift 4
  local log="$GB_ROOT/logs/T038_vllm_$(basename "$path")_${GB_TS}.log"
  ( conda activate gb-serve; cd "$GB_ROOT"
    CUDA_VISIBLE_DEVICES="$gpus" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve "$path" \
      --host 127.0.0.1 --port "$PORT" --served-model-name gadaabot-gen --max-model-len 16384 \
      --tensor-parallel-size "$tp" --gpu-memory-utilization "$util" --seed 7 "$@" ) > "$log" 2>&1 &
  SP=$!
  for i in $(seq 1 90); do
    curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { gb_say "served $(basename "$path") on GPU $gpus"; return 0; }
    kill -0 "$SP" 2>/dev/null || { gb_say "vLLM died for $(basename "$path") (see $log)"; tail -n 4 "$log" | tee -a "$GB_REPORT"; SP=""; return 1; }
    sleep 10
  done
  gb_say "vLLM not ready for $(basename "$path")"; kill -KILL -- "-$SP" 2>/dev/null || true; SP=""; return 1
}
unserve() { [ -n "$SP" ] && { kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true; }; SP=""; sleep 5; }

# --- 1) leakage screen (NO GPU) ------------------------------------------------------------------------------
SLOG="$GB_ROOT/logs/T038_screen_${GB_TS}.log"
python -m gadaabot.eval.realq screen --root "$GB_ROOT" --out-dir "$RQD" --raw "$GB_ROOT/data/eval/real_questions_raw.jsonl" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --parents "$GB_ROOT/data/kb/index/parents.sqlite" > "$SLOG" 2>&1 \
  || { gb_say "screen failed (see $SLOG)"; show "$SLOG" 'RQ-'; gb_report_print; exit 1; }
show "$SLOG" '^(RQ-|DONE)'
: > "$OUTD/no_exclusions.jsonl"

# --- 2) AfriqueQwen: 8 systems + calibrated judge (1 GPU) ----------------------------------------------------
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
G2="$(GB_REPORT=/dev/null gb_pick_gpus 1 12 50 2>/dev/null || true)"
if [ -n "$G2" ] && [ "$G2" != "$GPU" ]; then VIS="$GPU,$G2"; NDEV="cuda:1"; else VIS="$GPU"; NDEV="cuda:0"; fi
serve "$AFQ" "$GPU" 1 0.60 || { gb_report_print; exit 1; }
ALOG="$GB_ROOT/logs/T038_ablation_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$VIS" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.ablation \
  --llm-url "http://127.0.0.1:$PORT" "${RET[@]}" --dev "$RQD" --exclude-jsonl "$OUTD/no_exclusions.jsonl" \
  --nllb "$NLLB" --nllb-device "$NDEV" --arms "$ARMS" --workers 8 --judge-samples 3 --out-dir "$AD" > "$ALOG" 2>&1 \
  || gb_say "ablation exited non-zero (see $ALOG)"
show "$ALOG" '^(HELDOUT|RESUME|ARM [a-z]|JUDGE|DONE)'
gb_say "RQ judge AfriqueQwen (ref_ctx, key points as reference):"
show "$ALOG" '^(ARM-RESULT arm|PAIR )'
unserve

# --- 3) Gemma-3-27B second judge (2 GPUs) --------------------------------------------------------------------
GP="$(gb_wait_gpus 2 42 30 720)" || { gb_say "no 2 GPUs in 12 h; second judge skipped"; GP=""; }
if [ -n "$GP" ] && serve "$G27" "$GP" 2 0.80 --limit-mm-per-prompt '{"image":0}'; then
  JLOG="$GB_ROOT/logs/T038_g27_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="${GP%%,*}" HF_HUB_OFFLINE=1 python -m gadaabot.eval.extra rejudge --root "$GB_ROOT" --out-dir "$OUTD" "${RET[@]}" \
    --judge-url "http://127.0.0.1:$PORT" --judge-model gadaabot-gen --arms-dir "$AD" --items-jsonl "$RQD/devset_candidates.jsonl" \
    --tag g27_judge --arm-list closed closed_tt_nllb closed_tt_self rag_naive rag_tt_nllb rag_tt_self agent_noverify agent_full \
    > "$JLOG" 2>&1 || gb_say "Gemma judge exited non-zero (see $JLOG)"
  show "$JLOG" '^(E2|DONE)'
  unserve
fi

# --- 4) blind human sheets (NO GPU) --------------------------------------------------------------------------
HLOG="$GB_ROOT/logs/T038_sheet_${GB_TS}.log"
python -m gadaabot.eval.realq sheet --root "$GB_ROOT" --out-dir "$OUTD" --items "$RQD/devset_candidates.jsonl" --arms-dir "$AD" \
  --annotators A1 A2 > "$HLOG" 2>&1 || gb_say "sheet failed (see $HLOG)"
show "$HLOG" '^(RQ-|DONE)'
gb_say "Download for annotators: $OUTD/GadaaBot_realq_blind_A1.xlsx and _A2.xlsx (never send realq_key_PRIVATE.tsv)"
git add src reports/T038.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T038: real-question set screen, answers, judges, blind sheet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^RQ-SHEET' "$GB_REPORT" && grep -q '^PAIR ' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
