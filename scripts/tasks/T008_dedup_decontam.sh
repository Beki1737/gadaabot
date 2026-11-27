#!/usr/bin/env bash
# =============================================================================
# T008_dedup_decontam.sh | GadaaBot 2.0 | Batch B04
# -----------------------------------------------------------------------------
# What:   1) installs dedup.py + audit_filter.py into src/gadaabot/data/ and
#            datasketch 2.0.0 into gb-core
#         2) corpus v1 -> v2: removes training/held-out documents that share a
#            13-word sequence with ANY Oromo eval set (FLORES+, Belebele, SIB-200,
#            AfriMMLU, AfriXNLI, AfriMGSM); removes training near-duplicates of
#            held-out docs; removes near-duplicates inside training (MinHash,
#            Jaccard >= 0.8, 5-word shingles)
#         3) writes an 80-row filter-audit sheet for native-speaker review:
#            ~/gadaabot/reports/tables/T008_filter_audit.tsv
# Needs:  dedup.py, audit_filter.py uploaded to ~/gadaabot/tasks/
# Tool:   MobaXterm terminal          GPU: NONE (uses 32 CPU cores)
# Time:   5 to 20 minutes             Disk: about 0.15 GiB
# Run:    bash ~/gadaabot/tasks/T008_dedup_decontam.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T008 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

DST="$GB_ROOT/src/gadaabot/data"
for f in dedup.py audit_filter.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$DST/$f"
done
python -m py_compile "$DST/dedup.py" "$DST/audit_filter.py"
python -m pip install -q -c "$GB_ROOT/envs/gb-core.constraints.txt" "datasketch==2.0.0"
gb_say "MODULES: dedup.py, audit_filter.py installed; datasketch $(python -c 'import datasketch;print(datasketch.__version__)')"

python -m gadaabot.data.dedup --interim "$GB_ROOT/data/interim" \
  --eval-root "$GB_ROOT/data/eval" --workers 32 2>&1 | tee -a "$GB_REPORT"

mkdir -p "$GB_ROOT/reports/tables"
python -m gadaabot.data.audit_filter --raw "$GB_ROOT/data/raw" \
  --out "$GB_ROOT/reports/tables/T008_filter_audit.tsv" 2>&1 | tee -a "$GB_REPORT"

python -m pip freeze > "$GB_ROOT/envs/gb-core.lock.txt"
cd "$GB_ROOT"
git add src envs reports/T008.txt reports/tables 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T008: decontamination and near-duplicate removal (corpus v2), filter audit sheet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"

gb_say "CLEANUP CANDIDATE: data/interim/oromo_corpus.v1.*.jsonl.gz (about 0.12 GiB) once v2 is accepted"
gb_say "STATUS: PASS"
gb_report_print
