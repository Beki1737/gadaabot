#!/usr/bin/env bash
# =============================================================================
# T010d_m1_14b.sh | GadaaBot 2.0 | Gate M1 completion: AfriqueQwen-14B
# -----------------------------------------------------------------------------
# Why:    in T010c the 14B hit the 3 h safety limit (no error: lm-eval log-
#         likelihood scoring through vLLM ran at ~5k input tok/s, the 8B already
#         needed 137 min, the 14B needs about twice that).
# What:   same protocol as T010c, task list split in two halves that run AT THE
#         SAME TIME on two GPUs (part A = Belebele variants, part B = the rest),
#         5 h limit per half. Then the full Gate M1 summary over all 5 models.
#         Also prints (no GPU) whether ref Qwen3.5-9B is a base or post-trained
#         checkpoint, needed to interpret the Afrique-vs-Qwen3.5 gap.
# GPU:    YES, 2 cards at once: >= 40 GiB free each, <= 30% util. Waits up to
#         60 min for 2 free cards; if only 1 is free it runs the halves one after
#         the other on that card (slower, about 4.5 h).
# Time:   about 2 to 3 h with 2 cards; detaches itself (nohup).
# Needs:  nothing new (uses T010c files)
# Run:    bash ~/gadaabot/tasks/T010d_m1_14b.sh
# Watch:  tail -f ~/gadaabot/logs/T010d_nohup_*.log
# Result: cat ~/gadaabot/reports/T010d.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T010d_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T010d.pid"; disown -a
  echo "T010d started in background, pid $(cat "$HOME/gadaabot/runs/T010d.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T010d.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T010d gb-serve
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# --- 0) preflight + checkpoint type check (NO GPU) ------------------------------
python - "$SHARED_MODELS/Qwen/Qwen3.5-9B" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
cfg = json.loads((p / "config.json").read_text())
tc = p / "tokenizer_config.json"
tmpl = "chat_template" in json.loads(tc.read_text()) if tc.exists() else False
tmpl = tmpl or (p / "chat_template.jinja").exists()
readme = (p / "README.md").read_text(errors="replace")[:3000].lower() if (p / "README.md").exists() else ""
kind = "BASE" if ("base" in p.name.lower() or "pretrained (base)" in readme or "base model" in readme[:600]) else "POST-TRAINED (instruct/hybrid-thinking)"
print(f"CHECKPOINT Qwen3.5-9B: architectures={cfg.get('architectures')} chat_template={tmpl} -> likely {kind}")
PY

OUTD="$GB_ROOT/runs/T010c_m1"
MD="$OUTD/AfriqueQwen-14B"
MODEL="$GB_ROOT/models/McGill-NLP/AfriqueQwen-14B"
if [ -n "$(find "$MD" -name 'results_*.json' 2>/dev/null | head -n 1)" ]; then
  gb_say "NOTE: $MD already has results; delete it to redo. Going to summary."
else
  rm -rf "$MD"; mkdir -p "$MD"
  ALL="$(grep -viE 'mgsm' "$GB_ROOT/runs/T010b_tasks.txt" | grep .)"
  PART_A="$(echo "$ALL" | grep -iE 'belebele' | paste -sd, -)"
  PART_B="$(echo "$ALL" | grep -viE 'belebele' | paste -sd, -)"
  gb_say "SPLIT: part A $(echo "$PART_A" | tr ',' '\n' | grep -c .) tasks (Belebele), part B $(echo "$PART_B" | tr ',' '\n' | grep -c .) tasks"

  run_part() {  # $1 tasks, $2 out dir, $3 gpu, $4 log
    mkdir -p "$2"
    CUDA_VISIBLE_DEVICES="$3" lm_eval --model vllm \
      --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,seed=1234" \
      --tasks "$1" --num_fewshot 5 --batch_size auto --seed 1234 \
      --log_samples --output_path "$2" >> "$4" 2>&1 &
    local pid=$! start; start=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
      if [ -n "$(find "$2" -name 'results_*.json' -newermt "@$start" 2>/dev/null | head -n 1)" ]; then
        sleep 90; kill -0 "$pid" 2>/dev/null && { echo "results written ($2); forcing exit"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; }
        break
      fi
      if [ $(( $(date +%s) - start )) -gt 18000 ]; then
        echo "5 h limit hit ($2); killing"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; break
      fi
    done
    wait "$pid" 2>/dev/null || true
    sleep 5; pkill -KILL -P "$pid" 2>/dev/null || true
  }

  GPUS=""
  for attempt in $(seq 1 13); do
    if GPUS="$(GB_REPORT=/dev/null gb_pick_gpus 2 40 30 2>/dev/null)"; then break; fi
    GPUS=""; echo "$(date +%H:%M) 2 free GPUs not available; waiting 5 min"; sleep 300
  done
  t0=$(date +%s)
  if [ -n "$GPUS" ]; then
    GPUS="$(echo "$GPUS" | tr -s ' ;' ',,' | sed 's/^,//; s/,$//')"
    G1="${GPUS%%,*}"; G2="${GPUS##*,}"
    gb_say "RUN 14B in parallel: part A on GPU $G1, part B on GPU $G2 ($(date +%H:%M))"
    run_part "$PART_A" "$MD/partA" "$G1" "$GB_ROOT/logs/T010d_partA_${GB_TS}.log" &
    PA=$!
    sleep 60   # stagger model loading (disk read, CPU)
    run_part "$PART_B" "$MD/partB" "$G2" "$GB_ROOT/logs/T010d_partB_${GB_TS}.log" &
    PB=$!
    wait "$PA" || true; wait "$PB" || true
  else
    G1=""
    for attempt in $(seq 1 13); do
      if G1="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
      G1=""; echo "$(date +%H:%M) no free GPU; waiting 5 min"; sleep 300
    done
    [ -n "$G1" ] || { gb_say "SKIP 14B: no GPU free; STATUS: PARTIAL"; gb_report_print; exit 0; }
    gb_say "RUN 14B sequentially on GPU $G1 (2 cards were not free) ($(date +%H:%M))"
    run_part "$PART_A" "$MD/partA" "$G1" "$GB_ROOT/logs/T010d_partA_${GB_TS}.log"
    run_part "$PART_B" "$MD/partB" "$G1" "$GB_ROOT/logs/T010d_partB_${GB_TS}.log"
  fi
  for p in partA partB; do
    if [ -n "$(find "$MD/$p" -name 'results_*.json' | head -n 1)" ]; then gb_say "DONE 14B $p"
    else
      gb_say "FAIL 14B $p:"
      { grep -E 'Error|error' "$GB_ROOT/logs/T010d_${p}_${GB_TS}.log" | grep -v '^\s*File' | tail -n 4 | cut -c1-240 || true; } | sed 's/^/  /' | tee -a "$GB_REPORT"
    fi
  done
  gb_say "14B wall time $(( ($(date +%s) - t0) / 60 )) min"
fi

# --- summary over all models (NO GPU) ------------------------------------------------
python -m gadaabot.eval.m1_lmeval_summary --root "$OUTD" --ref AfriqueQwen-8B 2>&1 | tee -a "$GB_REPORT" || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r p m; do ps -o user= -p "$p" 2>/dev/null | grep -q "^$USER" && echo "LEFTOVER GPU PROCESS $p ($m)"; done \
  | tee -a "$GB_REPORT" || true
cd "$GB_ROOT"
git add reports/T010d.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T010d: Gate M1 AfriqueQwen-14B (split across two GPUs)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -qE '^(FAIL|SKIP)' "$GB_REPORT"; then gb_say "STATUS: PARTIAL"; else gb_say "STATUS: PASS"; fi
gb_report_print
