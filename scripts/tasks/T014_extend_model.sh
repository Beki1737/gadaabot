#!/usr/bin/env bash
# =============================================================================
# T014_extend_model.sh | GadaaBot 2.0 | Batch B07 | Gate T2 setup
# -----------------------------------------------------------------------------
# What:   1) installs build_extended.py (data/) and probe_extended.py (eval/)
#         2) builds 2 extended checkpoints of AfriqueQwen3.5-9B-it (CPU):
#              K=4096  (Oromo fertility 3.07 -> 1.96, -36% tokens)
#              K=16384 (3.07 -> 1.70, -45% tokens)
#            new token rows = mean of their base pieces (input and output
#            embeddings); every base weight and id unchanged
#         3) probes both on GPU (no training): load check, bits-per-word on
#            held-out Oromo and English (base vs untrained init), and a
#            feasibility probe for embedding-only training (speed, memory)
# GPU:    YES in step 3 only: 1 card, >= 40 GiB free, <= 30% util, about
#         15 to 30 min. Steps 1 and 2 use NO GPU.
# Disk:   +37 GiB (2 x 18.5 GiB checkpoints); stops if the project would pass
#         140 GiB
# Time:   about 40 to 70 min in total; detaches itself (nohup)
# Needs:  build_extended.py, probe_extended.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T014_extend_model.sh
# Watch:  tail -f ~/gadaabot/logs/T014_nohup_*.log
# Result: cat ~/gadaabot/reports/T014.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T014_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T014.pid"; disown -a
  echo "T014 started in background, pid $(cat "$HOME/gadaabot/runs/T014.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T014.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T014 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 1) install (NO GPU) -------------------------------------------------------
mkdir -p "$GB_ROOT/src/gadaabot/eval"; [ -f "$GB_ROOT/src/gadaabot/eval/__init__.py" ] || : > "$GB_ROOT/src/gadaabot/eval/__init__.py"
for spec in "build_extended.py|data" "probe_extended.py|eval"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"
  python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"
done
USED="$(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%d", $1/1024^3}')"
gb_say "QUOTA before: ${USED} GiB"
[ $((USED + 37)) -le 140 ] || { gb_say "STOP: ${USED} + 37 GiB would pass 140 GiB; free space first"; gb_report_print; exit 1; }

# --- 2) build checkpoints (NO GPU) ----------------------------------------------------
BASE="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
SPECS=""
for K in 4096 16384; do
  OUTM="$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K$K"
  if [ -f "$OUTM/gadaabot_extension.json" ]; then
    gb_say "SKIP build K=$K (exists)"
  else
    python -m gadaabot.data.build_extended --base "$BASE" \
      --tokenizer "$GB_ROOT/models/tokenizers/qwen35-orm-ext-K$K" --out "$OUTM" 2>&1 | tee -a "$GB_REPORT"
  fi
  SPECS="${SPECS:+$SPECS,}K$K=$OUTM"
done

# --- 3) GPU probe (1 card) --------------------------------------------------------------
GPU=""
for attempt in $(seq 1 13); do
  if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
  GPU=""; echo "$(date +%H:%M) no free GPU; waiting 5 min"; sleep 300
done
[ -n "$GPU" ] || { gb_say "SKIP probe: no GPU free for 60 min; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "PROBE on GPU $GPU ($(date +%H:%M))"
PROBE_LOG="$GB_ROOT/logs/T014_probe_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" timeout 5400 python -m gadaabot.eval.probe_extended --models "$SPECS" \
  --base-tok "$BASE" --heldout "$GB_ROOT/data/interim/oromo_corpus.v3.heldout.jsonl.gz" \
  --flores-eng "$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest/eng_Latn.jsonl" \
  --out "$GB_ROOT/reports/tables/T014_probe.json" > "$PROBE_LOG" 2>&1 || gb_say "PROBE exited non-zero (see $PROBE_LOG)"
{ grep -E '^(DATA|KERNELS|LOAD|BPW|TRAIN-PROBE)' "$PROBE_LOG" || true; } | tee -a "$GB_REPORT"
{ grep -E 'Error|error|warn' "$PROBE_LOG" | grep -v '^\s*File' | sort | uniq -c | sort -rn | head -n 6 | cut -c1-240 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"

cd "$GB_ROOT"
git add src reports/T014.txt reports/tables/T014_probe.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T014: extended-vocabulary checkpoints (K=4096, 16384) with mean-of-pieces init; GPU probe

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "QUOTA after: $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
if grep -q '^BPW K16384' "$GB_REPORT" && grep -q '^TRAIN-PROBE K16384' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
