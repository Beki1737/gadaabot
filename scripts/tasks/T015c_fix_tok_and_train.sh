#!/usr/bin/env bash
# =============================================================================
# T015c_fix_tok_and_train.sh | GadaaBot 2.0 | Batch B10
# -----------------------------------------------------------------------------
# Bug (found by D014 + the T015 assertion): our extended tokenizer.json files
# let the tokenizers library re-number the 33 special tokens onto the last 33
# new-token ids (<|endoftext|> lost id 248044; 129/31,499 held-out docs did not
# round-trip). Plain-text token ids were never affected, so T013 fertility and
# the T014 checkpoints/probes stay valid; this script proves that too.
# What:   1) installs fix_tokenizers.py and the fixed vocab_extend.py
#         2) repairs tokenizer.json in the 6 tokenizers and the 2 extended
#            checkpoints (NO GPU, 1 to 2 min); checks per file: specials keep
#            base ids, identical ids on plain text before/after, 0 round-trip
#            failures over all held-out docs
#         3) only if every check passes: starts T015 training (GPU: YES,
#            2 cards, as before; detaches itself)
# Needs:  fix_tokenizers.py, vocab_extend.py uploaded to ~/gadaabot/tasks/
#         (T015_embed_train.sh and embed_train.py from before stay in tasks/)
# Run:    bash ~/gadaabot/tasks/T015c_fix_tok_and_train.sh
# Result: cat ~/gadaabot/reports/T015c.txt now, then reports/T015.txt later
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T015c gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

if [ -f "$GB_ROOT/runs/T015.pid" ] && kill -0 "$(cat "$GB_ROOT/runs/T015.pid")" 2>/dev/null; then
  gb_say "STOP: the previous T015 (pid $(cat "$GB_ROOT/runs/T015.pid")) is still running; wait for it or kill it first"
  gb_report_print; exit 1
fi
for f in fix_tokenizers.py vocab_extend.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/data/$f"
  python -m py_compile "$GB_ROOT/src/gadaabot/data/$f"
done

DIRS=("$GB_ROOT"/models/tokenizers/qwen35-orm-ext-K* "$GB_ROOT"/models/local/AfriqueQwen3.5-9B-it-orm-K4096 "$GB_ROOT"/models/local/AfriqueQwen3.5-9B-it-orm-K16384)
if python -m gadaabot.data.fix_tokenizers --base "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" \
     --heldout "$GB_ROOT/data/interim/oromo_corpus.v3.heldout.jsonl.gz" "${DIRS[@]}" 2>&1 | tee -a "$GB_REPORT"; then
  gb_say "REPAIR PASS: starting T015 training (GPU: 2 cards)"
  rm -rf "$GB_ROOT"/runs/T015_K4096 "$GB_ROOT"/runs/T015_K16384
  cd "$GB_ROOT"
  git add src reports/T015c.txt 2>/dev/null || true
  git diff --cached --quiet || git commit -q -m "T015c: register special tokens in the extended BPE vocab (fixes special-id shift)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
  gb_say "STATUS: PASS"
  gb_report_print
  bash "$GB_ROOT/tasks/T015_embed_train.sh"
else
  gb_say "REPAIR FAILED: training NOT started; STATUS: FAIL"
  gb_report_print
  exit 1
fi
