#!/usr/bin/env bash
# =============================================================================
# T011_corpus_v3.sh | GadaaBot 2.0 | Batch B05 | corpus filter v3 (D26)
# -----------------------------------------------------------------------------
# What:   1) installs qubee.py (adds v3 helpers, v1 functions unchanged),
#            clean_v3.py (new), dedup.py (adds --in-tag/--out-tag) into src
#         2) raw -> v3pre: line-level cleaning (caption/navigation lines,
#            Ge'ez/foreign-script lines, corpus-repeated boilerplate lines), then
#            Qubee-alphabet rule and foreign-dominance rule (replaces the v1
#            Oromo function-word minimum), exact dedup
#         3) v3pre -> v3: same decontamination + near-dup removal as T008
#         4) writes an 80-row audit sheet for you to label (like T008):
#            ~/gadaabot/reports/tables/T011_v3_audit.tsv
#         v1 and v2 files stay untouched (needed for the filter ablation).
# Needs:  qubee.py, clean_v3.py, dedup.py uploaded to ~/gadaabot/tasks/
# Tool:   MobaXterm terminal          GPU: NONE (CPU, 32 cores for MinHash)
# Time:   10 to 30 minutes            Disk: about 0.3 GiB
# Run:    bash ~/gadaabot/tasks/T011_corpus_v3.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T011 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

DST="$GB_ROOT/src/gadaabot/data"
for f in qubee.py clean_v3.py dedup.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$DST/$f"
done
python -m py_compile "$DST/qubee.py" "$DST/clean_v3.py" "$DST/dedup.py"
gb_say "MODULES: qubee.py (v3 helpers), clean_v3.py, dedup.py (tags) installed"

INT="$GB_ROOT/data/interim"
mkdir -p "$GB_ROOT/reports/tables"
python -m gadaabot.data.clean_v3 --raw "$GB_ROOT/data/raw" --out "$INT" \
  --audit "$GB_ROOT/reports/tables/T011_v3_audit.tsv" 2>&1 | tee -a "$GB_REPORT"

python -m gadaabot.data.dedup --interim "$INT" --eval-root "$GB_ROOT/data/eval" \
  --workers 32 --in-tag v3pre --out-tag v3 2>&1 | tee -a "$GB_REPORT"

python - "$INT" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
def words(tag):
    ps = json.loads((d / f"oromo_corpus.{tag}.stats.json").read_text())["per_source"]
    return {k: v.get("words", v.get("kept_words", 0)) for k, v in ps.items() if k.endswith("/train")}
v2, v3 = words("v2"), words("v3")
print("TRAIN WORDS v2 -> v3 per source:")
for k in sorted(set(v2) | set(v3)):
    a, b = v2.get(k, 0), v3.get(k, 0)
    print(f"   {k:<18} {a:>12,} -> {b:>12,}  ({(b-a)/a*100 if a else 0:+.1f}%)")
print(f"   {'TOTAL':<18} {sum(v2.values()):>12,} -> {sum(v3.values()):>12,}")
PY

rm -f "$INT"/oromo_corpus.v3pre.*.jsonl.gz
cd "$GB_ROOT"
git add src reports/T011.txt reports/tables/T011_v3_audit.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T011: corpus filter v3 (line-level script and boilerplate cleaning, alphabet and foreign-dominance rules)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "NEXT: label column 'label' in reports/tables/T011_v3_audit.tsv (y / mixed / n), send it back"
gb_say "STATUS: PASS"
gb_report_print
