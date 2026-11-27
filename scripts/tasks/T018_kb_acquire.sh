#!/usr/bin/env bash
# =============================================================================
# T018_kb_acquire.sh | GadaaBot 2.0 | Batch B13 | knowledge base, step 1
# -----------------------------------------------------------------------------
# What:   downloads the first Afaan Oromo health/education sources with
#         provenance and a license tier per document (A open, B non-commercial,
#         C research-index-only), runs an Oromo check per document, writes
#         data/kb/raw/*.jsonl.gz + manifest.json:
#           wikipedia (CC BY-SA), tico19 (CC0, COVID-19), storybook
#           (African Storybook etc., CC), ohkb (CC BY 4.0 health summaries),
#           mdh (Minnesota Dept of Health Oromo PDFs, tier C)
#         Web downloads are polite (1 request/s, identified user agent).
# GPU:    NONE                 Time: 5 to 15 min        Disk: < 0.5 GiB
# Needs:  kb_acquire.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T018_kb_acquire.sh
# Result: cat ~/gadaabot/reports/T018.txt
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T018 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

mkdir -p "$GB_ROOT/src/gadaabot/kb"; [ -f "$GB_ROOT/src/gadaabot/kb/__init__.py" ] || : > "$GB_ROOT/src/gadaabot/kb/__init__.py"
[ -f "$GB_ROOT/tasks/kb_acquire.py" ] || { gb_say "MISSING: upload kb_acquire.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/kb_acquire.py" "$GB_ROOT/src/gadaabot/kb/"
python -m py_compile "$GB_ROOT/src/gadaabot/kb/kb_acquire.py"

CONS="$GB_ROOT/envs/gb-core.constraints.full.txt"
python -m pip list --format=freeze | grep -viE '^(pip|setuptools|wheel)==' > "$CONS"
python -c "import fitz" 2>/dev/null || python -m pip install -q -c "$CONS" pymupdf 2>&1 | tail -n 2
python -c "import fitz; print('pymupdf', fitz.__doc__.split()[1] if fitz.__doc__ else 'ok')" | tee -a "$GB_REPORT"

python -m gadaabot.kb.kb_acquire --out "$GB_ROOT/data/kb/raw" 2>&1 | grep -vE 'Warning|warn' | tee -a "$GB_REPORT" || true

cd "$GB_ROOT"
git add src reports/T018.txt data/kb/raw/manifest.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T018: knowledge-base acquisition step 1 (Wikipedia, TICO-19, storybooks, OHKB, MDH) with license tiers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^KB RAW TOTAL' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
