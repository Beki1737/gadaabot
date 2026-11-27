#!/usr/bin/env bash
# =============================================================================
# T015b_fla_debug.sh | GadaaBot 2.0 | Batch B09 | fix the fla backward OOM
# -----------------------------------------------------------------------------
# Facts from T015a: fla is installed and IS used (the crash is inside fla's
# backward kernel); forward results match T014 to 4 decimals (kernels correct);
# the backward kernel failed at launch with "Triton Error [CUDA]: out of memory"
# while autotuning, after the probe had cached several GiB from the BPW passes.
# What:   1) fla kernel alone (no model) with Qwen3.5's exact head shapes on
#            an empty GPU: tells "kernel cannot run on L40S" apart from
#            "GPU was too full"
#         2) training probe again on K4096, BPW skipped, cache freed first,
#            expandable memory segments on, micro-batch 1x2048 and 2x2048
#         3) installs the updated probe_extended.py and embed_train.py
#            (micro-batch 1 x accumulation 16, same tokens per step; cache freed
#            after each evaluation)
# GPU:    YES: 1 card, >= 40 GiB free, <= 30% util, about 10 min
# Run:    bash ~/gadaabot/tasks/T015b_fla_debug.sh
# Result: cat ~/gadaabot/reports/T015b.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T015b_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T015b.pid"; disown -a
  echo "T015b started in background, pid $(cat "$HOME/gadaabot/runs/T015b.pid")"
  echo "Result: cat ~/gadaabot/reports/T015b.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T015b gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for spec in "probe_extended.py|eval" "embed_train.py|train" "fla_check.py|"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  if [ -n "$sub" ]; then mkdir -p "$GB_ROOT/src/gadaabot/$sub"; cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"; python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"; fi
done
[ -f "$GB_ROOT/src/gadaabot/train/__init__.py" ] || : > "$GB_ROOT/src/gadaabot/train/__init__.py"

GPU=""
for attempt in $(seq 1 13); do
  if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
  GPU=""; echo "$(date +%H:%M) no free GPU; waiting 5 min"; sleep 300
done
[ -n "$GPU" ] || { gb_say "SKIP: no GPU free for 60 min; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "GPU $GPU | others on it: $(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';' || true)"
MODEL="$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K4096"

# --- 1) kernel alone ---------------------------------------------------------------
CUDA_VISIBLE_DEVICES="$GPU" timeout 900 python "$GB_ROOT/tasks/fla_check.py" "$MODEL/config.json" \
  > "$GB_ROOT/logs/T015b_fla_${GB_TS}.log" 2>&1 || true
{ grep -E '^FLA-CHECK' "$GB_ROOT/logs/T015b_fla_${GB_TS}.log" || tail -n 5 "$GB_ROOT/logs/T015b_fla_${GB_TS}.log"; } | tee -a "$GB_REPORT"

# --- 2) training probe in the model --------------------------------------------------------
PL="$GB_ROOT/logs/T015b_probe_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" timeout 2400 python -m gadaabot.eval.probe_extended --models "K4096=$MODEL" \
  --base-tok "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" \
  --heldout "$GB_ROOT/data/interim/oromo_corpus.v3.heldout.jsonl.gz" \
  --flores-eng "$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest/eng_Latn.jsonl" \
  --out "$GB_ROOT/reports/tables/T015b_probe.json" --skip-bpw --train-shapes 1x2048,2x2048 > "$PL" 2>&1 \
  || gb_say "PROBE exited non-zero (see $PL)"
{ grep -E '^(LOAD|TRAIN-PROBE)' "$PL" || true; } | tee -a "$GB_REPORT"
{ grep -E 'Error' "$PL" | grep -v '^\s*File' | sort | uniq -c | head -n 4 | cut -c1-200 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"
gb_say "REFERENCE T014 (no fla, 2x2048): 3.69 s/step, 1,110 tok/s, peak 29.4 GiB"
if grep -qE '^TRAIN-PROBE K4096 [12]x2048: [0-9]' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: FAIL"; fi
gb_report_print
