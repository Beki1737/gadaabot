#!/usr/bin/env bash
# =============================================================================
# T037_safety_sheet.sh | GadaaBot 2.0 | Batch B30 | clinician safety-review sheet
# -----------------------------------------------------------------------------
# Builds the blind clinician workbook: 40 health test questions x 3 systems
# (native RAG, closed-book, NLLB cascade) = 120 rows, labels A/B/C shuffled,
# citations, source lists and disclaimer stripped. Plus danger-sign list,
# fixed messages and reviewer background tabs. Key kept in a private TSV.
# GPU:    NO. CPU only, < 1 min.
# Needs:  safety_sheet.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T037_safety_sheet.sh
# Result: download runs/T037/GadaaBot_clinician_safety_review.xlsx (MobaXterm SFTP)
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T037 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
[ -f "$GB_ROOT/tasks/safety_sheet.py" ] || { gb_say "MISSING: upload safety_sheet.py"; gb_report_print; exit 1; }
python -c "import openpyxl" 2>/dev/null || pip install -q openpyxl
cp -f "$GB_ROOT/tasks/safety_sheet.py" "$GB_ROOT/src/gadaabot/eval/"
python -m gadaabot.eval.safety_sheet --root "$GB_ROOT" --out-dir "$GB_ROOT/runs/T037" 2>&1 | tee -a "$GB_REPORT"
grep -q '^SAFETY sheet' "$GB_REPORT" && gb_say "STATUS: PASS" || gb_say "STATUS: FAIL"
gb_report_print
