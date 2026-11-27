#!/usr/bin/env bash
# =============================================================================
# T023_index_v11_devset.sh | GadaaBot 2.0 | Batch B18
# -----------------------------------------------------------------------------
# 0) your chat scores: per-model means (NO GPU)
# 1) index v1.1 (GPU: 1 card among 6-9, >= 10 GiB free, ~5 min, queued):
#      * MADLAD mojibake repaired (U+FFFD -> hudhaa), PDF bullet glyphs removed
#      * authoritative sources first (Wikipedia, TICO-19, OHKB, MDH > books >
#        web), so duplicates keep the better provenance
#      * near-duplicate parents removed (MinHash, Jaccard >= 0.7) across sources
#      * web documents titled by their headline instead of corpus ids
# 2) dev-set draft (GPU: 1 card among 6-9, >= 40 GiB free, ~10 min, queued):
#      420 stratified passages -> AfriqueQwen3.5-9B-it writes one user-style
#      question + short answer each -> automatic filters -> review TSV
# Needs:  kb_index.py, devset_gen.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T023_index_v11_devset.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T023.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T023_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T023.pid"; disown -a
  echo "T023 started in background, pid $(cat "$HOME/gadaabot/runs/T023.pid")"
  echo "Result: cat ~/gadaabot/reports/T023.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T023 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
for spec in "kb_index.py|kb" "devset_gen.py|eval"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"; python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"
done

# --- 0) chat scores (NO GPU) ------------------------------------------------------------
python - "$GB_ROOT/reports/tables/T020_chat.tsv" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
print("CHAT SCORES (1-5)      fluency  correctness   n   per prompt (fluency/correctness)")
for m in ("base", "control", "ext"):
    rs = [r for r in rows if r["model"] == m]
    f = [num(r["fluent_1to5"]) for r in rs if num(r["fluent_1to5"]) is not None]
    c = [num(r["correct_1to5"]) for r in rs if num(r["correct_1to5"]) is not None]
    per = " ".join(f"{r['fluent_1to5']}/{r['correct_1to5']}" for r in rs)
    if f:
        print(f"   {m:<10} {statistics.mean(f):9.2f} {statistics.mean(c) if c else float('nan'):11.2f} {len(f):4d}   {per}")
for r in rows:
    if r.get("notes", "").strip():
        print(f"   NOTE {r['model']}/{r['domain']} {r['prompt'][:40]}...: {r['notes'][:200]}")
PY

# --- 1) index v1.1 (1 GPU among 6-9) ------------------------------------------------------------
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1 || true)"
GPU="$(gb_wait_gpus 1 10 60 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "INDEX v1.1 on GPU $GPU ($(date +%H:%M))"
ILOG="$GB_ROOT/logs/T023_index_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" python -m gadaabot.kb.kb_index --raw "$GB_ROOT/data/kb/raw" --out "$GB_ROOT/data/kb/index" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  ${RERANK:+--reranker "$RERANK"} > "$ILOG" 2>&1 || gb_say "INDEX exited non-zero (see $ILOG)"
{ grep -E '^(DOCS|CHUNKS|EMBED|INDEX|SMOKE|   \[|DONE)' "$ILOG" || true; } | cut -c1-260 | tee -a "$GB_REPORT"

# --- 2) dev-set draft (1 GPU among 6-9, gb-serve) --------------------------------------------------
conda activate gb-serve
cd "$GB_ROOT"
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH="$GB_ROOT/src"
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no 40 GiB GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "DEVSET on GPU $GPU ($(date +%H:%M))"
DLOG="$GB_ROOT/logs/T023_devset_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" setsid timeout 3600 python -m gadaabot.eval.devset_gen \
  --parents "$GB_ROOT/data/kb/index/parents.sqlite" --model "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" \
  --out "$GB_ROOT/data/eval/gadaabot_dev" --n 420 > "$DLOG" 2>&1 &
CP=$!
wait "$CP" || gb_say "devset_gen exited non-zero (see $DLOG)"
kill -KILL -- "-$CP" 2>/dev/null || true
{ grep -E '^(STRATUM|GENERATED|FILTER|KEPT|SAMPLE)' "$DLOG" || grep -E 'Error' "$DLOG" | tail -n 3; } | cut -c1-300 | tee -a "$GB_REPORT" || true
cp -f "$GB_ROOT/data/eval/gadaabot_dev/devset_review.tsv" "$GB_ROOT/reports/tables/T023_devset_review.tsv" 2>/dev/null || true

git add src reports/T023.txt reports/tables/T023_devset_review.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T023: index v1.1 (cleaning, source priority, near-dup parents), dev-set draft for review

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^FILTER' "$GB_REPORT" && grep -q '^DONE' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
