#!/usr/bin/env bash
# =============================================================================
# T022_kb_index.sh | GadaaBot 2.0 | Batch B17 | knowledge base index v1
# -----------------------------------------------------------------------------
# T021 failure cause: the background job was started from a folder that was
# deleted afterwards (getcwd() fails), which broke every library that resolves
# a relative path (datasets, transformers). All scripts now cd to ~/gadaabot.
# What:   1) installs kb_acquire.py (adds 'oeb') and kb_index.py
#         2) oeb: Oromia Education Bureau school books for grades 1-8 (T019
#            found only grades 1, 3, 5): directory listings, else name-pattern
#            probing (polite, 1 request/s)                       NO GPU
#         3) builds the index: cleaning (admin/statistics PDFs and duplicates
#            removed), parents ~350 words, children ~80 words with 1-sentence
#            overlap, AfriE5 dense + BM25 (4-letter prefixes, and normalized
#            words) into Qdrant collection gadaabot_kb_v1, parents in SQLite;
#            smoke test: 8 Oromo questions -> hybrid DBSF -> rerank -> parents
#            GPU: YES, 1 card among 6-9 (>= 10 GiB free), ~10 to 20 min, queued
#         4) chat probe rerun for control and extended models (re-merges them
#            on CPU if you already deleted them)
#            GPU: YES, 1 card among 6-9 (>= 40 GiB free), ~10 min, queued
# Disk:   +~1 GiB (index) +~0.5 GiB (books); +34 GiB only if re-merge needed
# Needs:  kb_acquire.py, kb_index.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T022_kb_index.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T022.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p logs runs
  OUT="$HOME/gadaabot/logs/T022_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T022.pid"; disown -a
  echo "T022 started in background, pid $(cat "$HOME/gadaabot/runs/T022.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T022.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T022 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

for f in kb_acquire.py kb_index.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/kb/$f"; python -m py_compile "$GB_ROOT/src/gadaabot/kb/$f"
done

# --- 2) school books, grades 1-8 (NO GPU) --------------------------------------------------
python -m gadaabot.kb.kb_acquire --out "$GB_ROOT/data/kb/raw" --sources oeb 2>&1 \
  | grep -E '^(SOURCE|KB RAW|  oeb|  crawl)' | tee -a "$GB_REPORT" || true
python - "$GB_ROOT/data/kb/raw/oeb.jsonl.gz" <<'PY' 2>&1 | tee -a "$GB_REPORT" || true
import gzip, json, sys
rows = [json.loads(l) for l in gzip.open(sys.argv[1], "rt", encoding="utf-8")]
print(f"OEB books: {len(rows)}")
for r in sorted(rows, key=lambda r: r["title"]):
    print(f"   {r['n_words']:>7,} w ok={int(r['oromo_ok'])} {r['domain']:<9} {r['title'][:70]}")
PY

# --- 3) index (GPU: 1 card among 6-9, queued) ---------------------------------------------------------
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1 || true)"
GPU="$(gb_wait_gpus 1 10 60 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "INDEX on GPU $GPU ($(date +%H:%M))"
ILOG="$GB_ROOT/logs/T022_index_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" python -m gadaabot.kb.kb_index --raw "$GB_ROOT/data/kb/raw" --out "$GB_ROOT/data/kb/index" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  ${RERANK:+--reranker "$RERANK"} > "$ILOG" 2>&1 || gb_say "INDEX exited non-zero (see $ILOG)"
{ grep -E '^(DOCS|CHUNKS|EMBED|INDEX|SMOKE|   \[|DONE)' "$ILOG" || true; } | cut -c1-300 | tee -a "$GB_REPORT"
{ grep -E 'Error' "$ILOG" | grep -v '^\s*File' | tail -n 3 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"

# --- 4) chat probe rerun (GPU: 1 card among 6-9, queued) ------------------------------------------------
L="$GB_ROOT/models/local"
for spec in "control|$L/AfriqueQwen3.5-9B-it-K0" "ext|$L/AfriqueQwen3.5-9B-it-orm-K4096-s1"; do
  IFS='|' read -r a base <<< "$spec"
  if [ ! -f "$L/AfriqueQwen3.5-9B-it-cpt-$a/gadaabot_stage2_merged.json" ]; then
    gb_say "re-merging $a (CPU)"
    CUDA_VISIBLE_DEVICES="" python -m gadaabot.train.merge_lora --base "$base" --run "$GB_ROOT/runs/T016_$a" \
      --out "$L/AfriqueQwen3.5-9B-it-cpt-$a" 2>&1 | grep -E '^(ROWS|MERGED)|Error' | tee -a "$GB_REPORT" || true
  fi
done
conda activate gb-serve
cd "$GB_ROOT"
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
CHAT="$GB_ROOT/reports/tables/T020_chat.tsv"
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no 40 GiB GPU in 12 h; chat probe skipped"; GPU=""; }
if [ -n "$GPU" ]; then
  gb_say "CHAT PROBE on GPU $GPU ($(date +%H:%M))"
  for a in control ext; do
    grep -q "^$a	" "$CHAT" 2>/dev/null && { gb_say "skip chat $a (already in TSV)"; continue; }
    LOG="$GB_ROOT/logs/T022_chat_${a}_${GB_TS}.log"
    CUDA_VISIBLE_DEVICES="$GPU" setsid timeout 1500 python "$GB_ROOT/src/gadaabot/eval/chat_probe.py" \
      --model "$L/AfriqueQwen3.5-9B-it-cpt-$a" --label "$a" --out "$CHAT" > "$LOG" 2>&1 &
    CP=$!
    wait "$CP" || gb_say "chat probe $a exited non-zero (log $LOG)"
    kill -KILL -- "-$CP" 2>/dev/null || true
    sleep 10
    { grep -E '^CHAT' "$LOG" || grep -E 'Error' "$LOG" | tail -n 3; } | cut -c1-260 | tee -a "$GB_REPORT" || true
  done
fi
gb_say "CHAT TSV rows: $(($(wc -l < "$CHAT") - 1)) (expect 18)"

git add src reports/T022.txt data/kb/raw/manifest.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T022: knowledge-base index v1 (parent-child chunks, AfriE5 + BM25 prefix-4, Qdrant), OEB books, chat probe rerun

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^DONE' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
