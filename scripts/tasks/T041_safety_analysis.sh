#!/usr/bin/env bash
# T041: unblind the clinician safety review (CPU only, no GPU, about 10 seconds).
# Before running: upload your transcribed sheet to ~/gadaabot/runs/T037/GadaaBot_clinician_safety_review_RETURNED.xlsx
# and this .sh plus T041_safety_analysis.py to ~/gadaabot/tasks/ (MobaXterm SFTP). Run: bash ~/gadaabot/tasks/T041_safety_analysis.sh
set -euo pipefail
GB="${HOME}/gadaabot"
SHEET="${GB}/runs/T037/GadaaBot_clinician_safety_review_RETURNED.xlsx"
KEY="${GB}/runs/T037/safety_key_PRIVATE.tsv"
OUT="${GB}/reports/T041.txt"
mkdir -p "${GB}/reports" "${GB}/logs"
[ -f "${SHEET}" ] || { echo "STATUS: FAIL missing ${SHEET}"; exit 1; }
[ -f "${KEY}" ] || { echo "STATUS: FAIL missing ${KEY}"; exit 1; }
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate gb-core
python "${GB}/tasks/T041_safety_analysis.py" --sheet "${SHEET}" --key "${KEY}" --out "${OUT}" 2>&1 | tee "${GB}/logs/T041_$(date +%Y%m%d_%H%M%S).log"
echo "### T041 REPORT START ###"; cat "${OUT}"; echo "### T041 REPORT END ###"
