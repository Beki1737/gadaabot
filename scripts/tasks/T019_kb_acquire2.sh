#!/usr/bin/env bash
# =============================================================================
# T019_kb_acquire2.sh | GadaaBot 2.0 | Batch B14 | knowledge base, step 2
# -----------------------------------------------------------------------------
# Fixes from T018:
#   * HF offline mode was on in this shell/env, so the 3 Hugging Face sources
#     failed; this run turns it off for the acquisition only
#   * OHKB is bilingual (English abstracts + Oromo summaries): whole-document
#     language check rejected it. Now every PDF source keeps only its Oromo
#     sentences (sentence-level check), MDH included (PDFs reused from cache)
# New sources (all tier C = local research index, not redistributed):
#   textbooks  Oromia/MoE school textbooks (Saayinsii, Fayyaa, ...): polite
#              crawler (robots.txt, same host, 1 request/s) from Camara
#              Ethiopia, the Oromia Education Bureau library and Temari
#   ethnomed   EthnoMed Oromo patient-education PDFs
#   webdomain  health/education documents filtered from corpus v3 by an
#              Oromo topic lexicon (heuristic; audited next)
# GPU:    NONE        Time: 20 to 60 min (crawl is rate-limited)
# Disk:   up to ~3 GiB of PDFs (textbooks are large)
# Needs:  kb_acquire.py (new version) uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T019_kb_acquire2.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T019.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/logs" "$HOME/gadaabot/runs"
  OUT="$HOME/gadaabot/logs/T019_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T019.pid"; disown -a
  echo "T019 started in background, pid $(cat "$HOME/gadaabot/runs/T019.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T019.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T019 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

[ -f "$GB_ROOT/tasks/kb_acquire.py" ] || { gb_say "MISSING: upload kb_acquire.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/kb_acquire.py" "$GB_ROOT/src/gadaabot/kb/"
python -m py_compile "$GB_ROOT/src/gadaabot/kb/kb_acquire.py"

WHERE="$(grep -ls 'HF_HUB_OFFLINE' "$GB_ROOT/configs/gadaabot.env" "$HOME/.bashrc" "$HOME/.profile" 2>/dev/null | tr '\n' ' ' || true)"
gb_say "HF offline flag was: HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} (set in: ${WHERE:-shell only}); turned off for this run"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0

python -m gadaabot.kb.kb_acquire --out "$GB_ROOT/data/kb/raw" \
  --sources wikipedia,tico19,storybook,ohkb,mdh,textbooks,ethnomed,webdomain 2>&1 \
  | grep -vE 'Warning|warn\(|FutureWarning' | tee -a "$GB_REPORT" || true

python - "$GB_ROOT/data/kb/raw/manifest.json" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import json, sys
m = json.load(open(sys.argv[1]))
print("KB MANIFEST                 docs   Oromo-ok      words  tiers")
for k, v in m.items():
    if v.get("status") == "ok":
        print(f"   {k:<14} {v['docs']:>10,} {v['oromo_ok_docs']:>10,} {v['oromo_ok_words']:>10,}  {v['tiers']}")
    else:
        print(f"   {k:<14} {v.get('status')}")
PY
du -sh "$GB_ROOT/data/kb" | tee -a "$GB_REPORT"
cd "$GB_ROOT"
git add src reports/T019.txt data/kb/raw/manifest.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T019: knowledge-base acquisition step 2 (sentence-level Oromo filter for PDFs, textbooks, EthnoMed, corpus health/education filter)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^KB RAW TOTAL' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
