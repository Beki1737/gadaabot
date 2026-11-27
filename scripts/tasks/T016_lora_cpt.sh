#!/usr/bin/env bash
# =============================================================================
# T016_lora_cpt.sh | GadaaBot 2.0 | Batch B11 | Gate T2, stage 2 (controlled)
# -----------------------------------------------------------------------------
# Why:    stage 1 (new rows only) closed most of the gap but not all: held-out
#         Oromo bits/word 7.78 base -> 8.28 (K4096) / 8.54 (K16384). A frozen
#         body cannot fully learn to read fused tokens. Stage 2 adds LoRA.
# Design: two arms on the SAME Oromo text (same doc order, 50% of corpus v3):
#           control   original tokenizer (K0 checkpoint) + LoRA
#           extended  K4096 stage-1 checkpoint + LoRA + new rows keep training
#         The difference isolates the vocabulary extension from "more Oromo
#         training". K16384 is dropped from stage 2 (worse at equal budget;
#         stays as an ablation point).
# What:   1) installs lora_cpt.py and the updated build_extended.py
#         2) builds the K0 control checkpoint (CPU, ~4 min, +16.6 GiB): same
#            weights, standard tensor names, original tokenizer
#         3) trains both arms in parallel, LoRA r=64 on all decoder linears,
#            lr 1e-4 cosine, eval every 100 steps, English canary at the end
# GPU:    YES, 2 cards at once (one per arm), >= 40 GiB free each. If only 1
#         card is free they run one after the other. Limit 9 h per arm.
#         Expected: extended ~3.5 h, control ~5.5 h (it needs 1.6x more
#         tokens for the same text: that difference IS the efficiency claim).
# Disk:   +17 GiB (K0) + ~1 GiB adapters -> about 102 GiB of 150
# Needs:  lora_cpt.py, build_extended.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T016_lora_cpt.sh
# Watch:  tail -f ~/gadaabot/logs/T016_ext_*.log
# Result: cat ~/gadaabot/reports/T016.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T016_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T016.pid"; disown -a
  echo "T016 started in background, pid $(cat "$HOME/gadaabot/runs/T016.pid")"
  echo "Watch:  tail -f ~/gadaabot/logs/T016_ext_*.log"
  echo "Result: cat ~/gadaabot/reports/T016.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T016 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for spec in "lora_cpt.py|train" "build_extended.py|data"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"
  python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"
done
python -c "import peft; print('peft', peft.__version__)" | tee -a "$GB_REPORT"

BASE="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
K0="$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-K0"
S1="$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K4096-s1"
[ -f "$S1/gadaabot_stage1.json" ] || { gb_say "MISSING stage-1 checkpoint $S1"; gb_report_print; exit 1; }

# --- K0 control checkpoint (NO GPU) ---------------------------------------------------
if [ ! -f "$K0/gadaabot_extension.json" ]; then
  python -m gadaabot.data.build_extended --base "$BASE" --tokenizer "$BASE" --out "$K0" 2>&1 | tee -a "$GB_REPORT"
else
  gb_say "SKIP K0 build (exists)"
fi

INT="$GB_ROOT/data/interim"
ENG="$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest/eng_Latn.jsonl"
run_arm() {  # $1 label, $2 model dir, $3 gpu
  local LOG="$GB_ROOT/logs/T016_$1_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="$3" timeout 33000 python -m gadaabot.train.lora_cpt --model "$2" --base-tok "$BASE" \
    --train "$INT/oromo_corpus.v3.train.jsonl.gz" --heldout "$INT/oromo_corpus.v3.heldout.jsonl.gz" \
    --flores-eng "$ENG" --out "$GB_ROOT/runs/T016_$1" --data-fraction 0.5 --time-limit-h 9 > "$LOG" 2>&1 \
    || echo "$1 exited non-zero" >> "$LOG"
}

GPUS=""
for attempt in $(seq 1 13); do
  if GPUS="$(GB_REPORT=/dev/null gb_pick_gpus 2 40 30 2>/dev/null)"; then break; fi
  GPUS=""; echo "$(date +%H:%M) 2 free GPUs not available; waiting 5 min"; sleep 300
done
if [ -n "$GPUS" ]; then
  GPUS="$(echo "$GPUS" | tr -s ' ;' ',,' | sed 's/^,//; s/,$//')"
  G1="${GPUS%%,*}"; G2="${GPUS##*,}"
  gb_say "TRAIN in parallel: extended on GPU $G1, control on GPU $G2 ($(date +%H:%M))"
  run_arm ext "$S1" "$G1" & P1=$!
  sleep 90
  run_arm control "$K0" "$G2" & P2=$!
  wait "$P1" || true; wait "$P2" || true
else
  G1=""
  for attempt in $(seq 1 13); do
    if G1="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
    G1=""; sleep 300
  done
  [ -n "$G1" ] || { gb_say "SKIP: no GPU free; STATUS: PARTIAL"; gb_report_print; exit 0; }
  gb_say "TRAIN sequentially on GPU $G1 ($(date +%H:%M))"
  run_arm ext "$S1" "$G1"; run_arm control "$K0" "$G1"
fi

for a in ext control; do
  LOG="$(ls -t "$GB_ROOT"/logs/T016_${a}_*.log | head -n 1)"
  gb_say "=== $a ==="
  { grep -E '^(DATA|TRAINABLE|EVAL|FINAL|SAVED|ABORT)|exited non-zero' "$LOG" || true; } | tee -a "$GB_REPORT"
  { grep -E 'step [0-9]+/' "$LOG" | tail -n 1 || true; } | tee -a "$GB_REPORT"
  { grep -E 'Error|error|out of memory' "$LOG" | grep -v '^\s*File' | tail -n 3 | cut -c1-240 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"
done
gb_say "REFERENCE (150 docs): base Oromo BPW 7.7774, English 5.3217 | K4096 stage 1: Oromo 8.2757, English 5.3386"

cd "$GB_ROOT"
git add src reports/T016.txt runs/T016_*/curve.jsonl runs/T016_*/stage2.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T016: stage-2 LoRA continued pre-training, extended (K4096) vs original-tokenizer control

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "QUOTA: $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
if [ "$(grep -c '^SAVED' "$GB_REPORT" || true)" -eq 2 ]; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
