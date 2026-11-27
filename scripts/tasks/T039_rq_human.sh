#!/usr/bin/env bash
# =============================================================================
# T039_rq_human.sh | GadaaBot 2.0 | Batch B32 | real questions: human scores vs judges + reranker gate
# -----------------------------------------------------------------------------
# Unblinds the two annotator sheets with the real key, re-tests H1 to H3, checks
# both LLM judges against the humans on the same 316 answers, and tests the E4
# reranker gate (tau 2.7, fixed in T034) on real questions.
# GPU:    NO. CPU only, < 1 min.
# Needs:  rq_human.py, GadaaBot_realq_blind_A1.xlsx, GadaaBot_realq_blind_A2.xlsx in ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T039_rq_human.sh
# Result: printed, also in ~/gadaabot/reports/T039.txt
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T039 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
T="$GB_ROOT/tasks"
for f in rq_human.py GadaaBot_realq_blind_A1.xlsx GadaaBot_realq_blind_A2.xlsx; do
  [ -f "$T/$f" ] || { gb_say "MISSING: upload $f"; gb_report_print; exit 1; }
done
cp -f "$T/rq_human.py" "$GB_ROOT/src/gadaabot/eval/"
mkdir -p "$GB_ROOT/data/eval/real_questions/human"
cp -f "$T/GadaaBot_realq_blind_A1.xlsx" "$T/GadaaBot_realq_blind_A2.xlsx" "$GB_ROOT/data/eval/real_questions/human/"
LOG="$GB_ROOT/logs/T039_${GB_TS}.log"
python -m gadaabot.eval.rq_human --root "$GB_ROOT" --a1 "$T/GadaaBot_realq_blind_A1.xlsx" --a2 "$T/GadaaBot_realq_blind_A2.xlsx" \
  --key "$GB_ROOT/runs/T038/realq_key_PRIVATE.tsv" --arms-dir "$GB_ROOT/runs/T038_realq" \
  --g27 "$GB_ROOT/runs/T038/rejudge_g27_judge.jsonl" --out-dir "$GB_ROOT/runs/T039" > "$LOG" 2>&1 || gb_say "rq_human exited non-zero (see $LOG)"
{ grep -E '^(RQH|DONE)' "$LOG" || tail -n 8 "$LOG"; } | cut -c1-700 | tee -a "$GB_REPORT"
git add src reports/T039.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T039: real-question human analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
grep -q '^DONE rq_human' "$GB_REPORT" && gb_say "STATUS: PASS" || gb_say "STATUS: FAIL"
gb_report_print
