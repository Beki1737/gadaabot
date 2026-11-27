#!/usr/bin/env bash
# =============================================================================
# T010c_gate_m1.sh | GadaaBot 2.0 | GATE M1 (model selection), validated harness
# -----------------------------------------------------------------------------
# What:   runs the lm-eval protocol validated in T010b (vLLM backend, 5-shot,
#         same Oromo task list from runs/T010b_tasks.txt, MGSM excluded) on 5
#         models, one at a time, with --log_samples (per-item results), then a
#         summary: family score = MEAN over prompt variants, 95% bootstrap CIs,
#         paired bootstrap differences vs AfriqueQwen-8B.
#           ref:    AfriqueQwen-8B (rerun: T010b had no per-item logs)
#           core:   AfriqueQwen-14B, AfriqueQwen3.5-9B-it (text view)
#           refs:   Qwen3.5-9B (same family, no Afrique CPT), Gemma-4-E4B-it
#         No chat template for any model (log-likelihood scoring, same as the
#         AfriqueLLM paper), so base and instruct models are scored identically.
# GPU:    YES, 1 card at a time: >= 40 GiB free, <= 30% util, picked fresh per
#         model. If none is free it waits (every 5 min, up to 60 min per model),
#         then skips that model. Nothing else uses GPU.
# Safety: per model: killed 90 s after its results file appears (vLLM shutdown
#         hang) or after 3 h at most. One model failing does not stop the rest.
# Time:   about 2.5 to 4 h in total; detaches itself (nohup).
# Needs:  m1_lmeval_summary.py uploaded to ~/gadaabot/tasks/
# Resume: rerun the same command; models with a results file are skipped.
# Run:    bash ~/gadaabot/tasks/T010c_gate_m1.sh
# Watch:  tail -f ~/gadaabot/logs/T010c_nohup_*.log
# Result: cat ~/gadaabot/reports/T010c.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T010c_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T010c.pid"; disown -a
  echo "T010c started in background, pid $(cat "$HOME/gadaabot/runs/T010c.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T010c.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T010c gb-serve
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- 0) preflight (NO GPU) -----------------------------------------------------
DST="$GB_ROOT/src/gadaabot/eval"; mkdir -p "$DST"; [ -f "$DST/__init__.py" ] || : > "$DST/__init__.py"
[ -f "$GB_ROOT/tasks/m1_lmeval_summary.py" ] || { gb_say "MISSING: upload m1_lmeval_summary.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/m1_lmeval_summary.py" "$DST/"
python -m py_compile "$DST/m1_lmeval_summary.py"
python -c "import lm_eval" || { gb_say "lm_eval missing in gb-serve (run T010b first)"; gb_report_print; exit 1; }
python -c "import torchcodec" 2>/dev/null && { gb_say "WARNING: torchcodec present; removing"; python -m pip uninstall -y torchcodec; } || true

TASKS_FILE="$GB_ROOT/runs/T010b_tasks.txt"
[ -s "$TASKS_FILE" ] || { gb_say "MISSING $TASKS_FILE (run T010b first)"; gb_report_print; exit 1; }
TASKS="$(grep -viE 'mgsm' "$TASKS_FILE" | grep . | paste -sd, -)"
gb_say "TASKS: $(echo "$TASKS" | tr ',' '\n' | grep -c .) Oromo variants (MGSM excluded: not decision-relevant)"

OUTD="$GB_ROOT/runs/T010c_m1"; mkdir -p "$OUTD"
SM="$SHARED_MODELS"
MODELS=(
  "AfriqueQwen-8B|$GB_ROOT/models/McGill-NLP/AfriqueQwen-8B|40|0.85"
  "AfriqueQwen-14B|$GB_ROOT/models/McGill-NLP/AfriqueQwen-14B|40|0.85"
  "AfriqueQwen3.5-9B-it|$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview|40|0.85"
  "ref-Qwen3.5-9B|$SM/Qwen/Qwen3.5-9B|40|0.85"
  "ref-Gemma-4-E4B-it|$SM/Google/gemma-4-E4B-it|36|0.80"
)

run_lmeval() {  # $1 model path, $2 out dir, $3 gpu, $4 mem fraction, $5 log
  CUDA_VISIBLE_DEVICES="$3" lm_eval --model vllm \
    --model_args "pretrained=$1,dtype=bfloat16,gpu_memory_utilization=$4,max_model_len=4096,seed=1234" \
    --tasks "$TASKS" --num_fewshot 5 --batch_size auto --seed 1234 \
    --log_samples --output_path "$2" >> "$5" 2>&1 &
  local pid=$! start; start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if [ -n "$(find "$2" -name 'results_*.json' -newermt "@$start" 2>/dev/null | head -n 1)" ]; then
      sleep 90; kill -0 "$pid" 2>/dev/null && { echo "results written; forcing exit"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; }
      break
    fi
    if [ $(( $(date +%s) - start )) -gt 10800 ]; then
      echo "3 h limit hit; killing"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; break
    fi
  done
  wait "$pid" 2>/dev/null || true
  sleep 5; pkill -KILL -P "$pid" 2>/dev/null || true
}

# --- 1) one model at a time (GPU: 1 card) ---------------------------------------
for spec in "${MODELS[@]}"; do
  IFS='|' read -r label path need mem <<< "$spec"
  MD="$OUTD/$label"; mkdir -p "$MD"
  if [ -n "$(find "$MD" -name 'results_*.json' | head -n 1)" ]; then
    gb_say "SKIP $label: already done (delete $MD to redo)"; continue
  fi
  [ -f "$path/config.json" ] || { gb_say "FAIL $label: no config.json at $path"; continue; }
  GPU=""
  for attempt in $(seq 1 13); do
    if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 "$need" 30 2>/dev/null)"; then break; fi
    GPU=""; echo "$(date +%H:%M) no free GPU for $label (need ${need} GiB); waiting 5 min"; sleep 300
  done
  [ -n "$GPU" ] || { gb_say "SKIP $label: no GPU free for 60 min"; continue; }
  RUN_LOG="$GB_ROOT/logs/T010c_${label}_${GB_TS}.log"
  t0=$(date +%s)
  gb_say "RUN $label on GPU $GPU ($(date +%H:%M))"
  run_lmeval "$path" "$MD" "$GPU" "$mem" "$RUN_LOG"
  if [ -n "$(find "$MD" -name 'results_*.json' | head -n 1)" ]; then
    gb_say "DONE $label in $(( ($(date +%s) - t0) / 60 )) min"
  else
    gb_say "FAIL $label (log: $RUN_LOG):"
    { grep -E 'Error|error' "$RUN_LOG" | grep -v '^\s*File' | tail -n 4 | cut -c1-240 || true; } | sed 's/^/  /' | tee -a "$GB_REPORT"
  fi
done

# --- 2) summary (NO GPU) -----------------------------------------------------------
python -m gadaabot.eval.m1_lmeval_summary --root "$OUTD" --ref AfriqueQwen-8B 2>&1 | tee -a "$GB_REPORT" || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r p m; do ps -o user= -p "$p" 2>/dev/null | grep -q "^$USER" && echo "LEFTOVER GPU PROCESS $p ($m)"; done \
  | tee -a "$GB_REPORT" || true

cd "$GB_ROOT"
git add src reports/T010c.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T010c: Gate M1 screen with validated lm-eval protocol (mean over prompt variants)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -qE '^(FAIL|SKIP .*no GPU)' "$GB_REPORT"; then gb_say "STATUS: PARTIAL"; else gb_say "STATUS: PASS"; fi
gb_report_print
