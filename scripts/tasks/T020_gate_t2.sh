#!/usr/bin/env bash
# =============================================================================
# T020_gate_t2.sh | GadaaBot 2.0 | Batch B15 | Gate T2 decision run
# -----------------------------------------------------------------------------
# Waits for T016 (control arm) to finish, then:
#   1) merges each stage-2 adapter into a full checkpoint (CPU, ~10 min each)
#        control  = original tokenizer + LoRA        -> ...-cpt-control
#        extended = K4096 + new rows + LoRA          -> ...-cpt-ext
#   2) Gate M1/T2 benchmark protocol (same 23 Oromo lm-eval tasks as T010c,
#      mean over prompt variants, paired bootstrap) on both merged models,
#      compared with the untouched AfriqueQwen3.5-9B-it (T010c results reused)
#   3) chat sanity probe (6 Oromo health/education questions, chat template,
#      D23 sampling) on base, control, extended -> TSV for your judgement:
#      did continued pre-training on raw text damage instruction following?
# GPU:    YES. Step 2: 2 cards at once (one per model), >= 40 GiB free each,
#         ~2.5 h. Step 3: 1 card, ~10 min. Step 1: NO GPU.
# Disk:   +34 GiB (two merged checkpoints). Stops if the project would pass
#         140 GiB: delete the K16384 models first (see the chat message).
# Needs:  merge_lora.py, chat_probe.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T020_gate_t2.sh   (detaches; may wait for T016)
# Result: cat ~/gadaabot/reports/T020.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/logs" "$HOME/gadaabot/runs"
  OUT="$HOME/gadaabot/logs/T020_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T020.pid"; disown -a
  echo "T020 started in background, pid $(cat "$HOME/gadaabot/runs/T020.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T020.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T020 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export HF_HUB_OFFLINE=1   # local models only

for spec in "merge_lora.py|train" "chat_probe.py|eval"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"; python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"
done

# --- 0) wait for T016 ------------------------------------------------------------------
for i in $(seq 1 72); do
  [ -f "$GB_ROOT/runs/T016_ext/stage2.json" ] && [ -f "$GB_ROOT/runs/T016_control/stage2.json" ] && break
  echo "$(date +%H:%M) waiting for T016 control to finish"; sleep 300
done
[ -f "$GB_ROOT/runs/T016_control/stage2.json" ] || { gb_say "T016 control not finished after 6 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
python - "$GB_ROOT/runs" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, sys
from pathlib import Path
for arm in ("control", "ext"):
    s = json.loads((Path(sys.argv[1]) / f"T016_{arm}" / "stage2.json").read_text())
    f = s["final"]
    print(f"T016 {arm:<8} Oromo BPW {f['bpw_orm']} | English BPW {f['bpw_eng']} | Oromo tokens {f['tokens_orm']:,} | "
          f"steps {s['steps']} | {s['hours']} h | stop {s['stop']}")
print("REFERENCE base (150 docs): Oromo BPW 7.7774 | English BPW 5.3217 | Oromo tokens 114,537")
PY

# --- 1) merge (NO GPU) ------------------------------------------------------------------------
USED="$(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%d", $1/1024^3}')"
L="$GB_ROOT/models/local"
NEED=0; for a in control ext; do [ -f "$L/AfriqueQwen3.5-9B-it-cpt-$a/gadaabot_stage2_merged.json" ] || NEED=$((NEED + 17)); done
[ $((USED + NEED)) -le 140 ] || { gb_say "STOP: ${USED} + ${NEED} GiB would pass 140 GiB; delete the K16384 models first"; gb_report_print; exit 1; }
merge() {  # $1 arm, $2 base
  local O="$L/AfriqueQwen3.5-9B-it-cpt-$1"
  [ -f "$O/gadaabot_stage2_merged.json" ] && { gb_say "SKIP merge $1 (exists)"; return; }
  CUDA_VISIBLE_DEVICES="" python -m gadaabot.train.merge_lora --base "$2" --run "$GB_ROOT/runs/T016_$1" --out "$O" 2>&1 \
    | grep -E '^(ROWS|MERGED)|Error' | tee -a "$GB_REPORT"
}
merge control "$L/AfriqueQwen3.5-9B-it-K0"
merge ext "$L/AfriqueQwen3.5-9B-it-orm-K4096-s1"

# --- 2) lm-eval, 2 cards in parallel (gb-serve) ------------------------------------------------
conda activate gb-serve
export VLLM_WORKER_MULTIPROC_METHOD=spawn
TASKS="$(grep -viE 'mgsm' "$GB_ROOT/runs/T010b_tasks.txt" | grep . | paste -sd, -)"
R="$GB_ROOT/runs/T020_m1"; mkdir -p "$R"
[ -e "$R/base-9B-it" ] || ln -s "$GB_ROOT/runs/T010c_m1/AfriqueQwen3.5-9B-it" "$R/base-9B-it"
run_lmeval() {  # $1 label, $2 model, $3 gpu
  local MD="$R/$1" LOG="$GB_ROOT/logs/T020_$1_${GB_TS}.log"; mkdir -p "$MD"
  [ -n "$(find "$MD" -name 'results_*.json' | head -n 1)" ] && { echo "skip $1"; return; }
  CUDA_VISIBLE_DEVICES="$3" lm_eval --model vllm \
    --model_args "pretrained=$2,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,seed=1234" \
    --tasks "$TASKS" --num_fewshot 5 --batch_size auto --seed 1234 --log_samples --output_path "$MD" >> "$LOG" 2>&1 &
  local pid=$! start; start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if [ -n "$(find "$MD" -name 'results_*.json' -newermt "@$start" 2>/dev/null | head -n 1)" ]; then
      sleep 90; kill -0 "$pid" 2>/dev/null && { pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; }; break
    fi
    [ $(( $(date +%s) - start )) -gt 14400 ] && { echo "4 h limit $1"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; break; }
  done
  wait "$pid" 2>/dev/null || true; sleep 5; pkill -KILL -P "$pid" 2>/dev/null || true
}
GPUS=""
for attempt in $(seq 1 13); do
  if GPUS="$(GB_REPORT=/dev/null gb_pick_gpus 2 40 30 2>/dev/null)"; then break; fi
  GPUS=""; echo "$(date +%H:%M) 2 free GPUs not available; waiting 5 min"; sleep 300
done
if [ -n "$GPUS" ]; then
  GPUS="$(echo "$GPUS" | tr -s ' ;' ',,' | sed 's/^,//; s/,$//')"; G1="${GPUS%%,*}"; G2="${GPUS##*,}"
  gb_say "LM-EVAL in parallel: control on GPU $G1, ext on GPU $G2 ($(date +%H:%M))"
  run_lmeval cpt-control "$L/AfriqueQwen3.5-9B-it-cpt-control" "$G1" & P1=$!
  sleep 60
  run_lmeval cpt-ext "$L/AfriqueQwen3.5-9B-it-cpt-ext" "$G2" & P2=$!
  wait "$P1" || true; wait "$P2" || true
else
  G1="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null || true)"
  [ -n "$G1" ] || { gb_say "SKIP lm-eval: no GPU; STATUS: PARTIAL"; gb_report_print; exit 0; }
  gb_say "LM-EVAL sequential on GPU $G1"
  run_lmeval cpt-control "$L/AfriqueQwen3.5-9B-it-cpt-control" "$G1"; run_lmeval cpt-ext "$L/AfriqueQwen3.5-9B-it-cpt-ext" "$G1"
fi

# --- 3) chat probe (1 card) ------------------------------------------------------------------------
G1="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null || true)"
CHAT="$GB_ROOT/reports/tables/T020_chat.tsv"; rm -f "$CHAT"
if [ -n "$G1" ]; then
  for spec in "base|$L/AfriqueQwen3.5-9B-Instruct-textview" "control|$L/AfriqueQwen3.5-9B-it-cpt-control" "ext|$L/AfriqueQwen3.5-9B-it-cpt-ext"; do
    IFS='|' read -r lab mp <<< "$spec"
    CUDA_VISIBLE_DEVICES="$G1" timeout 1200 python "$GB_ROOT/src/gadaabot/eval/chat_probe.py" --model "$mp" --label "$lab" --out "$CHAT" \
      > "$GB_ROOT/logs/T020_chat_${lab}_${GB_TS}.log" 2>&1 || gb_say "chat probe $lab failed (log logs/T020_chat_${lab}_${GB_TS}.log)"
    { grep -E '^CHAT' "$GB_ROOT/logs/T020_chat_${lab}_${GB_TS}.log" || true; } | tee -a "$GB_REPORT"
  done
else
  gb_say "SKIP chat probe: no GPU free"
fi

# --- 4) summary ------------------------------------------------------------------------------------
PYTHONPATH="$GB_ROOT/src" python -m gadaabot.eval.m1_lmeval_summary --root "$R" --ref base-9B-it 2>&1 | tee -a "$GB_REPORT" || true
cd "$GB_ROOT"
git add src reports/T020.txt reports/tables/T020_chat.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T020: Gate T2 run (merged stage-2 models, lm-eval, chat probe)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q 'primary' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
