#!/usr/bin/env bash
# =============================================================================
# T013_vocab_extend.sh | GadaaBot 2.0 | Batch B06 | GATE T1 (tokenizer)
# -----------------------------------------------------------------------------
# What:   0) Gate M1 head-to-head: paired bootstrap AfriqueQwen-14B vs the chosen
#            AfriqueQwen3.5-9B-it (re-reads T010c per-item logs, seconds)
#         1) installs vocab_extend.py into src/gadaabot/data/
#         2) continued BPE on corpus v3 (train split only) from the exact state of
#            the Qwen3.5 tokenizer used by AfriqueQwen3.5-9B-it; appends Oromo
#            merges for K = 1k, 2k, 4k, 8k, 16k, 32k new tokens
#         3) measures each K on FLORES+ devtest (Oromo and English) and on the
#            corpus held-out split; checks exactness and lossless round-trip
#         Gate T1 target: Oromo fertility <= 2.30 tokens per word (base 3.07).
#         Tokenizers only: no model weights are touched in this task.
# Needs:  vocab_extend.py uploaded to ~/gadaabot/tasks/
# Tool:   MobaXterm terminal          GPU: NONE (CPU; 32 cores for counting)
# Time:   30 to 90 minutes            Disk: about 0.3 GiB (6 tokenizers)
# Run:    bash ~/gadaabot/tasks/T013_vocab_extend.sh
#         (self-detaches; close MobaXterm if you like)
# Watch:  tail -f ~/gadaabot/logs/T013_nohup_*.log
# Result: cat ~/gadaabot/reports/T013.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T013_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T013.pid"; disown -a
  echo "T013 started in background, pid $(cat "$HOME/gadaabot/runs/T013.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T013.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T013 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 0) Gate M1 head-to-head (NO GPU) --------------------------------------------
gb_say "GATE M1 HEAD-TO-HEAD (reference = AfriqueQwen3.5-9B-it):"
python -m gadaabot.eval.m1_lmeval_summary --root "$GB_ROOT/runs/T010c_m1" --ref AfriqueQwen3.5-9B-it 2>&1 \
  | grep -A4 -E '^\s+AfriqueQwen-14B' | tee -a "$GB_REPORT" || true
mkdir -p "$GB_ROOT/reports/tables"
cp -f "$GB_ROOT/runs/T010c_m1/m1_summary.json" "$GB_ROOT/reports/tables/T010_m1_vs_9b_it.json" 2>/dev/null || true
python -m gadaabot.eval.m1_lmeval_summary --root "$GB_ROOT/runs/T010c_m1" --ref AfriqueQwen-8B > /dev/null 2>&1 || true

# --- 1) install module -------------------------------------------------------------
DST="$GB_ROOT/src/gadaabot/data"
[ -f "$GB_ROOT/tasks/vocab_extend.py" ] || { gb_say "MISSING: upload vocab_extend.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/vocab_extend.py" "$DST/"
python -m py_compile "$DST/vocab_extend.py"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)" | tee -a "$GB_REPORT"

# --- 2) + 3) continued BPE and evaluation ----------------------------------------------
BASE="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
if [ ! -f "$BASE/tokenizer.json" ]; then
  BASE="$(dirname "$(ls -d "$GB_ROOT"/models/McGill-NLP/AfriqueQwen3.5-9B*/tokenizer.json | head -n 1)")"
fi
[ -f "$BASE/tokenizer.json" ] || { gb_say "MISSING base tokenizer.json"; gb_report_print; exit 1; }
INT="$GB_ROOT/data/interim"
FL="$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest"
for f in "$INT/oromo_corpus.v3.train.jsonl.gz" "$INT/oromo_corpus.v3.heldout.jsonl.gz" "$FL/gaz_Latn.jsonl" "$FL/eng_Latn.jsonl"; do
  [ -f "$f" ] || { gb_say "MISSING: $f"; gb_report_print; exit 1; }
done
mkdir -p "$GB_ROOT/models/tokenizers" "$GB_ROOT/reports/tables"
python -m gadaabot.data.vocab_extend --base "$BASE" \
  --train "$INT/oromo_corpus.v3.train.jsonl.gz" --heldout "$INT/oromo_corpus.v3.heldout.jsonl.gz" \
  --flores-orm "$FL/gaz_Latn.jsonl" --flores-eng "$FL/eng_Latn.jsonl" \
  --out "$GB_ROOT/models/tokenizers" --table "$GB_ROOT/reports/tables/T013_vocab_ext.json" \
  --workers 32 2>&1 | tee -a "$GB_REPORT"

cd "$GB_ROOT"
git add src reports/T013.txt reports/tables/T013_vocab_ext.* reports/tables/T010_m1_vs_9b_it.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T013: continued-BPE Oromo vocabulary extension of the Qwen3.5 tokenizer (Gate T1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "STATUS: PASS"
gb_say "QUOTA: project at $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB"
gb_report_print
