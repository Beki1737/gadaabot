#!/usr/bin/env bash
# =============================================================================
# T010_gate_m1.sh | GadaaBot 2.0 | Batch B04 part 2 | GATE M1 (model selection)
# -----------------------------------------------------------------------------
# What:   log-likelihood screen on Oromo Belebele, AfriMMLU, AfriXNLI, SIB-200
#         for 3 candidates + 2 references, one model at a time:
#           core:  AfriqueQwen-8B (base), AfriqueQwen-14B (base),
#                  AfriqueQwen3.5-9B-Instruct (text view)
#           refs:  Qwen3.5-9B (same family before Afrique CPT), Gemma-4-E4B-it
#         then a summary table with 95% CIs and paired differences vs 8B.
# GPU:    YES, 1 card at a time: >= 40 GiB free (>= 43 for the 14B), <= 30% util.
#         Picked fresh for each model; if none is free it waits (checks every
#         5 min, up to 60 min per model) and then skips that model.
# Time:   about 40 to 90 minutes in total; detaches itself (nohup), you can close
#         MobaXterm.
# Needs:  m1_screen.py, m1_summarize.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T010_gate_m1.sh
# Watch:  tail -f ~/gadaabot/logs/T010_nohup_*.log
# Result: cat ~/gadaabot/reports/T010.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T010_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T010.pid"; disown -a
  echo "T010 started in background, pid $(cat "$HOME/gadaabot/runs/T010.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T010.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T010 gb-serve
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DST="$GB_ROOT/src/gadaabot/eval"
mkdir -p "$DST"; [ -f "$DST/__init__.py" ] || : > "$DST/__init__.py"
for f in m1_screen.py m1_summarize.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$DST/$f"
done
python -m py_compile "$DST/m1_screen.py" "$DST/m1_summarize.py"

OUTD="$GB_ROOT/runs/T010_m1"; mkdir -p "$OUTD"
SM="$SHARED_MODELS"
MODELS=(
  "AfriqueQwen-8B|$GB_ROOT/models/McGill-NLP/AfriqueQwen-8B|40|0.85"
  "AfriqueQwen-14B|$GB_ROOT/models/McGill-NLP/AfriqueQwen-14B|43|0.92"
  "AfriqueQwen3.5-9B-it|$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview|40|0.85"
  "ref-Qwen3.5-9B|$SM/Qwen/Qwen3.5-9B|40|0.85"
  "ref-Gemma-4-E4B-it|$SM/Google/gemma-4-E4B-it|30|0.80"
)

for spec in "${MODELS[@]}"; do
  IFS='|' read -r label path need mem <<< "$spec"
  if [ -f "$OUTD/$label.summary.json" ]; then
    gb_say "SKIP $label: already scored (delete $OUTD/$label.* to redo)"; continue
  fi
  GPU=""
  for attempt in $(seq 1 13); do
    if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 "$need" 30 2>/dev/null)"; then break; fi
    GPU=""; echo "$(date +%H:%M) no free GPU for $label (need ${need} GiB); waiting 5 min"; sleep 300
  done
  if [ -z "$GPU" ]; then gb_say "SKIP $label: no GPU free for 60 min"; continue; fi
  gb_say "RUN $label on GPU $GPU"
  RUN_LOG="$GB_ROOT/logs/T010_${label}_${GB_TS}.log"
  if CUDA_VISIBLE_DEVICES="$GPU" python -m gadaabot.eval.m1_screen --model "$path" --label "$label" \
       --eval-root "$GB_ROOT/data/eval" --out "$OUTD" --gpu-mem "$mem" > "$RUN_LOG" 2>&1; then
    { grep -E '^(TASK|RESULT|TIMING|REQUESTS)' "$RUN_LOG" || true; } | tee -a "$GB_REPORT"
  else
    gb_say "FAIL $label (log: $RUN_LOG):"
    { grep -E 'Error|error' "$RUN_LOG" | grep -v '^\s*File' | tail -n 3 | cut -c1-240 || true; } | tee -a "$GB_REPORT"
  fi
done

if [ -f "$OUTD/AfriqueQwen-8B.items.jsonl" ]; then
  python -m gadaabot.eval.m1_summarize --dir "$OUTD" --ref AfriqueQwen-8B --metric acc_norm | tee -a "$GB_REPORT"
  { python -m gadaabot.eval.m1_summarize --dir "$OUTD" --ref AfriqueQwen-8B --metric acc | grep -E '^\s+\S+\s+mean' || true; } \
    | sed 's/^/  (acc) /' | tee -a "$GB_REPORT"
fi

cd "$GB_ROOT"
git add src reports/T010.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T010: Gate M1 screen (Oromo Belebele, AfriMMLU, AfriXNLI, SIB-200)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^FAIL\|^SKIP .*no GPU' "$GB_REPORT"; then gb_say "STATUS: PARTIAL"; else gb_say "STATUS: PASS"; fi
gb_report_print
