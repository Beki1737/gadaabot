#!/usr/bin/env bash
# =============================================================================
# T036b_encoders_fix.sh | GadaaBot 2.0 | Batch B29 | E7 rerun with crash guard
# -----------------------------------------------------------------------------
# T036 E7 died after BGE-M3 (gte-multilingual step). extra.py now skips a broken
# encoder instead of dying, and adds paired tests: BM25 analyzers vs prefix-4,
# and each encoder vs AfriE5 at dense / hybrid / full-pipeline stage.
# GPU:    YES. 1 card (>= 20 GiB). ~15 to 20 min, queued if busy. No vLLM.
# Disk:   +0 (LaBSE / gte already local or skipped)
# Needs:  extra.py (new) uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T036b_encoders_fix.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T036b.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T036b_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T036b.pid"; disown -a
  echo "T036b started in background, pid $(cat "$HOME/gadaabot/runs/T036b.pid")"
  echo "Result: cat ~/gadaabot/reports/T036b.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T036b gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
E="$GB_ROOT/src/gadaabot/eval"; OUTD="$GB_ROOT/runs/T036"; mkdir -p "$OUTD"
[ -f "$GB_ROOT/tasks/extra.py" ] || { gb_say "MISSING: upload extra.py"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/extra.py" "$E/"; python -m py_compile "$E/extra.py"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
RET=(--qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct"
     --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite")
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"

# --- 0) why T036 E7 died (NO GPU) ----------------------------------------------------------------------------
OLD="$(ls -t "$GB_ROOT"/logs/T036_e7_*.log 2>/dev/null | head -n 1 || true)"
[ -n "$OLD" ] && grep -vE '^E7 ' "$OLD" | grep -E 'Error|error|raise' | tail -n 4 | sed 's/^/T036-CAUSE /' | cut -c1-300 | tee -a "$GB_REPORT" || true
for m in Alibaba-NLP/gte-multilingual-base Sentence-Transformers/LaBSE; do
  f="no"; for r in "$GB_ROOT/models" "$SHARED_MODELS"; do [ -f "$r/$m/config.json" ] && f="$r/$m"; done
  gb_say "ENC-PATH $m -> $f"
done

# --- 1) E7 rerun (1 GPU) -------------------------------------------------------------------------------------
GPU="$(gb_wait_gpus 1 20 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
LOG="$GB_ROOT/logs/T036b_e7_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.extra encoders --root "$GB_ROOT" --out-dir "$OUTD" "${RET[@]}" \
  --shared-models "$SHARED_MODELS" > "$LOG" 2>&1 || gb_say "E7 exited non-zero (see $LOG)"
{ grep -E '^(E7 FAIL|E7 component (gte|labse)|E7-PAIR|E7 pre)' "$LOG" || grep -E 'Error' "$LOG" | tail -n 6; } | cut -c1-400 | tee -a "$GB_REPORT" || true
git add src reports/T036b.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T036b: E7 retrieval components with paired tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^E7-PAIR full' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
