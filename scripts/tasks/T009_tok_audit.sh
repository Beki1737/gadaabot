#!/usr/bin/env bash
# =============================================================================
# T009_tok_audit.sh | GadaaBot 2.0 | Batch B03 | RUN AFTER T007
# -----------------------------------------------------------------------------
# What:   tokenizer audit on FLORES+ devtest (1,012 parallel Oromo/English
#         sentences): fertility with 95% CI, English premium, chars per token,
#         single-token retention (STRR), and the cost of unnormalized hudhaa.
#         Covers every tokenizer we may use: generators, embedders, references.
# Needs:  T007 done (installs tok_audit.py); HF login done (FLORES+ English)
# Tool:   MobaXterm terminal          GPU: NONE          Time: 2 to 5 minutes
# Output: ~/gadaabot/reports/tables/T009_tok_audit.{json,csv}
# Run:    bash ~/gadaabot/tasks/T009_tok_audit.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T009 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

FL="$GB_ROOT/data/eval/openlanguagedata/flores_plus"
ORM="$FL/devtest/gaz_Latn.jsonl"
ENG="$FL/devtest/eng_Latn.jsonl"
if [ ! -f "$ENG" ]; then
  python - "$FL" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("openlanguagedata/flores_plus", "devtest/eng_Latn.jsonl",
                repo_type="dataset", local_dir=sys.argv[1])
PY
fi
for f in "$ORM" "$ENG"; do
  [ -f "$f" ] || { gb_say "MISSING: $f"; gb_report_print; exit 1; }
done

SM="$SHARED_MODELS"
TOKS="$GB_ROOT/runs/T009_tokenizers.json"
cat > "$TOKS" <<EOF
{
  "AfriqueQwen-8B":        "$GB_ROOT/models/McGill-NLP/AfriqueQwen-8B",
  "Qwen3-8B (base of it)": "$SM/Qwen/Qwen3-8B",
  "AfriqueQwen3.5-9B-it":  "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview",
  "Qwen3.5-9B":            "$SM/Qwen/Qwen3.5-9B",
  "Gemma-4-E4B":           "$SM/Google/gemma-4-E4B-it",
  "Gemma-3-12B":           "$SM/Google/gemma-3-12b-Instruct",
  "Llama-3.3 (128k)":      "$SM/MetaAI/Llama-3.3-70B-Instruct",
  "AfriE5 (embedder)":     "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct",
  "BGE-M3 (embedder)":     "$SM/BAAI/bge-m3",
  "XLM-R large":           "$SM/FacebookAI/xlm-roberta-large"
}
EOF

mkdir -p "$GB_ROOT/reports/tables"
python -m gadaabot.data.tok_audit --orm "$ORM" --eng "$ENG" --tokenizers "$TOKS" \
  --out "$GB_ROOT/reports/tables/T009_tok_audit.json" 2>&1 \
  | grep -v "PyTorch was not found" | tee -a "$GB_REPORT"

cd "$GB_ROOT"
git add reports/tables reports/T009.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T009: tokenizer audit on FLORES+ devtest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "STATUS: PASS"
gb_report_print
