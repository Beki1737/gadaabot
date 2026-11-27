#!/usr/bin/env bash
# =============================================================================
# T007_corpus.sh | GadaaBot 2.0 | Batch B03
# -----------------------------------------------------------------------------
# What:   1) installs the first project modules into ~/gadaabot/src/gadaabot/data/
#            (qubee.py = Qubee normalizer + language heuristics,
#             build_corpus.py = cleaning pipeline) and commits them to git
#         2) records the approved project quota (150 GiB) in gadaabot.env
#         3) builds the cleaned Oromo corpus v1 (train + held-out) from T005 data
# Needs:  qubee.py, build_corpus.py, tok_audit.py uploaded to ~/gadaabot/tasks/
# Tool:   MobaXterm terminal          GPU: NONE          Time: 2 to 10 minutes
# Disk:   about 0.3 GiB output
# Run:    bash ~/gadaabot/tasks/T007_corpus.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T007 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR

# --- 1) install modules -----------------------------------------------------
DST="$GB_ROOT/src/gadaabot/data"
for f in qubee.py build_corpus.py tok_audit.py; do
  if [ ! -f "$GB_ROOT/tasks/$f" ]; then
    gb_say "MISSING: upload $f to ~/gadaabot/tasks/ first"; gb_report_print; exit 1
  fi
  cp -f "$GB_ROOT/tasks/$f" "$DST/$f"
done
export PYTHONPATH="$GB_ROOT/src"
python -m py_compile "$DST"/qubee.py "$DST"/build_corpus.py "$DST"/tok_audit.py
gb_say "MODULES: qubee.py, build_corpus.py, tok_audit.py installed in src/gadaabot/data"

# --- 2) project quota -------------------------------------------------------
ENVF="$GB_ROOT/configs/gadaabot.env"
if grep -q '^GB_PROJECT_QUOTA_GIB=' "$ENVF"; then
  sed -i 's/^GB_PROJECT_QUOTA_GIB=.*/GB_PROJECT_QUOTA_GIB=150/' "$ENVF"
else
  echo "GB_PROJECT_QUOTA_GIB=150" >> "$ENVF"
fi
gb_say "QUOTA: project cap set to 150 GiB (lab head approval, 2026-10-27)"

# --- 3) build corpus --------------------------------------------------------
python -m gadaabot.data.build_corpus \
  --raw "$GB_ROOT/data/raw" --out "$GB_ROOT/data/interim" 2>&1 | tee -a "$GB_REPORT"

cd "$GB_ROOT"
git add src configs/gadaabot.env reports/T007.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T007: Qubee normalizer and corpus v1 builder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"

gb_say "STATUS: PASS"
gb_say "QUOTA: project at $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
gb_report_print
