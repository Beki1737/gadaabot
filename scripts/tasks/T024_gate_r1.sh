#!/usr/bin/env bash
# =============================================================================
# T024_gate_r1.sh | GadaaBot 2.0 | Batch B19 | Gate R1 retrieval comparison
# -----------------------------------------------------------------------------
# Runs the retrieval arms (BM25 normalized, BM25 4-letter prefixes, AfriE5
# dense, hybrid RRF, hybrid DBSF, reranked best hybrid) on the domain dev set.
# Before your review: PRELIMINARY on all 364 candidates. After your review
# (keep_y_n filled in data/eval/gadaabot_dev/devset_review.tsv): FINAL on the
# kept, edited questions. Rerun the same command after the review.
# GPU:    YES, 1 card among 6-9 (>= 10 GiB free), ~5 min, queued if busy
# Needs:  retrieval_eval.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T024_gate_r1.sh
# Result: cat ~/gadaabot/reports/T024.txt
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
source "$HOME/gadaabot/scripts/common.sh"
gb_init T024 gb-core
cd "$GB_ROOT"
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
[ -f "$GB_ROOT/tasks/retrieval_eval.py" ] || { gb_say "MISSING: upload retrieval_eval.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/retrieval_eval.py" "$GB_ROOT/src/gadaabot/eval/"; python -m py_compile "$GB_ROOT/src/gadaabot/eval/retrieval_eval.py"
# if you edited the review copy in reports/tables, use it
if [ -f "$GB_ROOT/reports/tables/T023_devset_review.tsv" ] && [ "$GB_ROOT/reports/tables/T023_devset_review.tsv" -nt "$GB_ROOT/data/eval/gadaabot_dev/devset_review.tsv" ]; then
  cp -f "$GB_ROOT/reports/tables/T023_devset_review.tsv" "$GB_ROOT/data/eval/gadaabot_dev/devset_review.tsv"
  gb_say "using your edited review file from reports/tables"
fi
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1 || true)"
GPU="$(gb_wait_gpus 1 10 60 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "GATE R1 on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" python -m gadaabot.eval.retrieval_eval --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  ${RERANK:+--reranker "$RERANK"} --out "$GB_ROOT/reports/tables/T024_gate_r1.json" 2>&1 \
  | grep -E '^(DEV|ARM|BREAKDOWN|OVERLAP|BEST)|Error' | tee -a "$GB_REPORT"
git add src reports/T024.txt reports/tables/T024_gate_r1.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T024: Gate R1 retrieval arms on the domain dev set

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^BEST' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
