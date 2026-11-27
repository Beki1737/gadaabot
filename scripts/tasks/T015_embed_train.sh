#!/usr/bin/env bash
# =============================================================================
# T015_embed_train.sh | GadaaBot 2.0 | Batch B08 | Gate T2, stage 1 training
# -----------------------------------------------------------------------------
# RUN AFTER T015a PASSES (fast kernels), otherwise it is ~4x slower.
# What:   trains ONLY the new Oromo token rows (input + output embeddings) of
#         the two extended checkpoints; every other weight frozen, so base
#         behaviour is preserved exactly. Corpus v3 train only; held-out BPW
#         tracked every 50 steps; best checkpoint kept.
#           K4096  and  K16384, 20M training tokens each (about 0.35 epoch)
# GPU:    YES, 2 cards at once (one per model): >= 40 GiB free each, <= 30%
#         util. If only 1 card is free they run one after the other.
#         Hard limit 6 h per model.
# Time:   about 1.5 to 3 h with fast kernels; detaches itself (nohup)
# Disk:   about +10 GiB (only the embedding shard is rewritten; the rest are
#         symlinks to the T014 checkpoints, which must therefore be kept)
# Needs:  embed_train.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T015_embed_train.sh
# Watch:  tail -f ~/gadaabot/logs/T015_K4096_*.log
# Result: cat ~/gadaabot/reports/T015.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T015_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T015.pid"; disown -a
  echo "T015 started in background, pid $(cat "$HOME/gadaabot/runs/T015.pid")"
  echo "Watch:  tail -f ~/gadaabot/logs/T015_K4096_*.log"
  echo "Result: cat ~/gadaabot/reports/T015.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T015 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DST="$GB_ROOT/src/gadaabot/train"; mkdir -p "$DST"; [ -f "$DST/__init__.py" ] || : > "$DST/__init__.py"
[ -f "$GB_ROOT/tasks/embed_train.py" ] || { gb_say "MISSING: upload embed_train.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/embed_train.py" "$DST/"
python -m py_compile "$DST/embed_train.py"
python -c "import fla" 2>/dev/null && gb_say "KERNELS: fla present" || gb_say "WARNING: fla missing; training will be slow (run T015a first)"

BASE="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
INT="$GB_ROOT/data/interim"
ENG="$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest/eng_Latn.jsonl"

train_one() {  # $1 K, $2 gpu
  local K=$1 G=$2
  local LOG="$GB_ROOT/logs/T015_K${K}_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="$G" timeout 23400 python -m gadaabot.train.embed_train \
    --model "$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K$K" --base-tok "$BASE" \
    --train "$INT/oromo_corpus.v3.train.jsonl.gz" --heldout "$INT/oromo_corpus.v3.heldout.jsonl.gz" \
    --flores-eng "$ENG" --out "$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K$K-s1" \
    --log "$GB_ROOT/runs/T015_K$K/curve.jsonl" --max-tokens 20000000 --time-limit-h 6 > "$LOG" 2>&1 \
    || echo "K$K exited non-zero" >> "$LOG"
}

GPUS=""
for attempt in $(seq 1 13); do
  if GPUS="$(GB_REPORT=/dev/null gb_pick_gpus 2 40 30 2>/dev/null)"; then break; fi
  GPUS=""; echo "$(date +%H:%M) 2 free GPUs not available; waiting 5 min"; sleep 300
done
if [ -n "$GPUS" ]; then
  GPUS="$(echo "$GPUS" | tr -s ' ;' ',,' | sed 's/^,//; s/,$//')"
  G1="${GPUS%%,*}"; G2="${GPUS##*,}"
  gb_say "TRAIN in parallel: K4096 on GPU $G1, K16384 on GPU $G2 ($(date +%H:%M))"
  train_one 4096 "$G1" & P1=$!
  sleep 90
  train_one 16384 "$G2" & P2=$!
  wait "$P1" || true; wait "$P2" || true
else
  G1=""
  for attempt in $(seq 1 13); do
    if G1="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
    G1=""; sleep 300
  done
  [ -n "$G1" ] || { gb_say "SKIP: no GPU free; STATUS: PARTIAL"; gb_report_print; exit 0; }
  gb_say "TRAIN sequentially on GPU $G1 ($(date +%H:%M))"
  train_one 4096 "$G1"; train_one 16384 "$G1"
fi

for K in 4096 16384; do
  LOG="$(ls -t "$GB_ROOT"/logs/T015_K${K}_*.log | head -n 1)"
  gb_say "=== K$K ==="
  { grep -E '^(DATA|EVAL|FINAL|SAVED)|exited non-zero' "$LOG" || true; } | tee -a "$GB_REPORT"
  { grep -E 'step [0-9]+/' "$LOG" | tail -n 1 || true; } | tee -a "$GB_REPORT"
  { grep -E 'Error|error|out of memory' "$LOG" | grep -v '^\s*File' | tail -n 3 | cut -c1-240 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"
done

cd "$GB_ROOT"
git add src reports/T015.txt runs/T015_K*/curve.jsonl 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T015: stage-1 training of new Oromo token rows (K4096, K16384)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "QUOTA: $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
if [ "$(grep -c '^SAVED' "$GB_REPORT" || true)" -eq 2 ]; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
