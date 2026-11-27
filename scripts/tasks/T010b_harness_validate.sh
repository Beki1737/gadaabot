#!/usr/bin/env bash
# =============================================================================
# T010b_harness_validate.sh | GadaaBot 2.0 | Gate M1, step 0: validate harness
# -----------------------------------------------------------------------------
# Why:    T010 numbers for AfriqueQwen-8B missed the published AfriqueLLM
#         numbers by up to 26 points (prompt format differs). Rule: a harness
#         must reproduce a published number before we trust it.
# What:   1) installs EleutherAI lm-evaluation-harness into gb-serve WITHOUT
#            touching torch/vLLM/transformers (pinned by the env lock file)
#         2) lists every Oromo task variant the harness really has for
#            Belebele, AfriMMLU, AfriXNLI, SIB-200, AfriMGSM (no guessing names)
#         3) runs ALL of them on AfriqueQwen-8B (vLLM backend, 5-shot,
#            AfriMGSM 8-shot, same as the AfriqueLLM paper)
#         4) prints each score next to the paper's Oromo number
# GPU:    YES, 1 card, >= 40 GiB free, <= 30% util (auto-picked; waits up to
#         60 min). Steps 1 and 2 use NO GPU.
# Safety: hard stop: the job is killed 90 s after its results file appears
#         (vLLM can hang at shutdown, as T010 did) or after 3 h at most.
# Time:   30 to 90 minutes; detaches itself (nohup).
# Run:    bash ~/gadaabot/tasks/T010b_harness_validate.sh
# Watch:  tail -f ~/gadaabot/logs/T010b_nohup_*.log
# Result: cat ~/gadaabot/reports/T010b.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T010b_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T010b.pid"; disown -a
  echo "T010b started in background, pid $(cat "$HOME/gadaabot/runs/T010b.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T010b.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T010b gb-serve
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR

# --- 1) install lm-eval, freezing everything already installed (NO GPU) ------
LOCK="$GB_ROOT/envs/gb-serve.lock.txt"
python -m pip freeze > "$LOCK"
# name==version for every installed package (no URL lines, which pip refuses as constraints)
python -m pip list --format=freeze | grep -viE '^(torchcodec|pip|setuptools|wheel)==' > "$GB_ROOT/envs/gb-serve.constraints.full.txt"
if ! python -c "import lm_eval" 2>/dev/null; then
  python -m pip install -c "$GB_ROOT/envs/gb-serve.constraints.full.txt" "lm_eval" 2>&1 | tail -n 3
fi
python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
from importlib.metadata import version
print("VERSIONS: lm_eval==%s vllm==%s torch==%s transformers==%s" % tuple(
    version(p) for p in ("lm_eval", "vllm", "torch", "transformers")))
PY
python -c "import torchcodec" 2>/dev/null && { gb_say "WARNING: torchcodec came back; removing"; python -m pip uninstall -y torchcodec; } || true

# --- 2) discover Oromo task names (NO GPU) -----------------------------------
TASKS_FILE="$GB_ROOT/runs/T010b_tasks.txt"
python - "$TASKS_FILE" <<'PY' 2>&1 | grep -v "^WARNING" | tee -a "$GB_REPORT"
import re, sys
from lm_eval.tasks import TaskManager
names = sorted(TaskManager().all_tasks)
fam = re.compile(r"(belebele|afrimmlu|afrixnli|sib|afrimgsm)", re.I)
lang = re.compile(r"(^|_)(orm|gaz|gaz_latn|oromo)(_|$)", re.I)
skip = re.compile(r"(translate|_trans|_en_|_in_|cot)", re.I)
picked = [n for n in names if fam.search(n) and lang.search(n) and not skip.search(n)]
open(sys.argv[1], "w").write("\n".join(picked))
print(f"HARNESS TASKS: {len(names)} total; Oromo variants picked ({len(picked)}):")
for n in picked:
    print("   ", n)
PY
NTASK="$(grep -c . "$TASKS_FILE" || true)"
if [ "${NTASK:-0}" -eq 0 ]; then
  gb_say "NO Oromo tasks found in this lm-eval version; STATUS: FAIL"; gb_report_print; exit 1
fi

# --- 3) run on AfriqueQwen-8B (GPU: 1 card) ----------------------------------
GPU=""
for attempt in $(seq 1 13); do
  if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
  GPU=""; echo "$(date +%H:%M) no free GPU; waiting 5 min"; sleep 300
done
[ -n "$GPU" ] || { gb_say "SKIP: no GPU free for 60 min; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "RUN AfriqueQwen-8B on GPU $GPU, $NTASK tasks"

OUTD="$GB_ROOT/runs/T010b_lmeval"; mkdir -p "$OUTD"
MODEL="$GB_ROOT/models/McGill-NLP/AfriqueQwen-8B"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
RUN_LOG="$GB_ROOT/logs/T010b_run_${GB_TS}.log"

run_lmeval() {  # $1 = comma task list, $2 = num_fewshot
  CUDA_VISIBLE_DEVICES="$GPU" lm_eval --model vllm \
    --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,seed=1234" \
    --tasks "$1" --num_fewshot "$2" --batch_size auto --seed 1234 \
    --output_path "$OUTD" >> "$RUN_LOG" 2>&1 &
  local pid=$! start; start=$(date +%s)
  local before; before="$(find "$OUTD" -name 'results_*.json' | wc -l)"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    local now; now="$(find "$OUTD" -name 'results_*.json' | wc -l)"
    if [ "$now" -gt "$before" ]; then          # results written: allow 90 s to exit
      sleep 90; kill -0 "$pid" 2>/dev/null && { echo "results written; forcing exit"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; }
      break
    fi
    if [ $(( $(date +%s) - start )) -gt 10800 ]; then
      echo "3 h limit hit; killing"; pkill -TERM -P "$pid" || true; kill -TERM "$pid" || true; break
    fi
  done
  wait "$pid" 2>/dev/null || true
}

MAIN_TASKS="$(grep -viE 'mgsm' "$TASKS_FILE" | paste -sd, - || true)"
MGSM_TASKS="$(grep -iE 'mgsm' "$TASKS_FILE" | paste -sd, - || true)"
[ -n "$MAIN_TASKS" ] && run_lmeval "$MAIN_TASKS" 5
[ -n "$MGSM_TASKS" ] && run_lmeval "$MGSM_TASKS" 8

# --- 4) compare with the paper -------------------------------------------------
python - "$OUTD" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, re, sys
from pathlib import Path
paper = {"belebele": 60.64, "afrimmlu": 46.44, "afrixnli": 50.07, "sib": 69.51, "afrimgsm": 37.76}
files = sorted(Path(sys.argv[1]).rglob("results_*.json"))
if not files:
    print("NO RESULTS FILE"); sys.exit(0)
print("TASK                                         metric      ours   paper    diff")
for f in files:
    res = json.loads(f.read_text()).get("results", {})
    for task, m in sorted(res.items()):
        fam = next((k for k in paper if k in task.lower()), None)
        for key in ("acc_norm,none", "acc,none", "exact_match,flexible-extract", "exact_match,strict-match", "exact_match,none"):
            if key in m and isinstance(m[key], (int, float)):
                ours = 100 * m[key]
                ref = paper.get(fam) if fam else None
                d = f"{ours - ref:+6.1f}" if ref is not None else "     -"
                rs = f"{ref:6.2f}" if ref is not None else "     -"
                print(f"{task:<44} {key.split(',')[0]:<10} {ours:6.2f} {rs}  {d}")
PY
{ grep -E "Traceback|Error" "$RUN_LOG" | grep -v "^\s*File" | tail -n 5 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"

gb_say "VALIDATION RULE: harness accepted if Belebele and AfriMMLU are within 3 points of the paper"
gb_say "STATUS: DONE"
gb_report_print
