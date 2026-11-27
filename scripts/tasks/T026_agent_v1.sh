#!/usr/bin/env bash
# =============================================================================
# T026_agent_v1.sh | GadaaBot 2.0 | Batch B21 | agent v1 + textbook permission
# -----------------------------------------------------------------------------
# v1 fixes from T025: IndexError on the regeneration path (6/54 crashed),
#   multi-query retrieval (original question + refined query), and a 3-level
#   evidence grade (yes / partly / no) so partial answers are given with the gap
#   stated instead of abstaining.
# Also:   textbooks + OEB books -> tier P (deployment permission granted by the
#   Oromia Education Bureau, 2026-11-01) in Qdrant payloads and parents.sqlite,
#   and docs/permissions/README.md records it (add the letter file there).
# What:   1) installs graph v1 (retriever, llm, run_batch from T025 stay),
#            updates tiers (NO GPU)
#         2) serves AfriqueQwen3.5-9B-it with vLLM (OpenAI API, 127.0.0.1:48000,
#            non-thinking) on ONE GPU among 6-9; the retriever's AfriE5 and
#            reranker share the same card
#         3) runs the cyclic agent (refine -> hybrid retrieve -> grade ->
#            reformulate loop -> generate with [n] citations -> claim verify ->
#            finalize / abstain; emergency screen at ingress) on 6 chat prompts +
#            48 dev questions, next to a closed-book answer from the same model
#         4) stops the server
# GPU:    YES, 1 card among 6-9 (>= 40 GiB free), ~30 to 60 min, queued if busy
# Output: reports/tables/T026_rag_vs_closed.tsv (for your scoring),
#         runs/T026_agent/answers.jsonl (full traces)
# Needs:  graph.py (v1) uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T026_agent_v1.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T026.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T026_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T026.pid"; disown -a
  echo "T026 started in background, pid $(cat "$HOME/gadaabot/runs/T026.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T026.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T026 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 1) install graph v1 (NO GPU) -----------------------------------------------------------
A="$GB_ROOT/src/gadaabot/agent"
[ -f "$GB_ROOT/tasks/graph.py" ] || { gb_say "MISSING: upload graph.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/graph.py" "$A/graph.py"; python -m py_compile "$A/graph.py"

# --- 1b) textbook permission -> tier P (NO GPU) ---------------------------------------------------
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
python - "$GB_ROOT/data/kb/index/parents.sqlite" "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" <<'PY2' 2>&1 | tee -a "$GB_REPORT"
import sqlite3, sys
from qdrant_client import QdrantClient, models
LIC = "(c) Biiroo Barnootaa Oromiyaa; deployment permission granted to GadaaBot 2026-11-01"
db = sqlite3.connect(sys.argv[1])
n = db.execute("UPDATE parents SET tier='P', license=? WHERE source IN ('textbooks','oeb')", (LIC,)).rowcount
db.commit()
qc = QdrantClient(url=sys.argv[2])
flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchAny(any=["textbooks", "oeb"]))])
qc.set_payload("gadaabot_kb_v1", payload={"tier": "P", "license": LIC}, points=flt, wait=True)
cnt = qc.count("gadaabot_kb_v1", count_filter=models.Filter(must=[models.FieldCondition(key="tier", match=models.MatchValue(value="P"))])).count
print(f"TIER P: {n} parents in SQLite, {cnt} children in Qdrant (textbooks + OEB books)")
PY2
mkdir -p "$GB_ROOT/docs/permissions"
[ -f "$GB_ROOT/docs/permissions/README.md" ] || cat > "$GB_ROOT/docs/permissions/README.md" <<'MD'
# Content permissions

| Source | Holder | Scope | Granted | Evidence file |
|---|---|---|---|---|
| Oromia school books (grades 1-12), sources `textbooks` and `oeb` | Biiroo Barnootaa Oromiyaa (Oromia Education Bureau) | Indexing, quoting short passages with citation in the free GadaaBot assistant; no redistribution of full books, no commercial use | 2026-11-01 | ADD the letter or email here (PDF) |

Notes: redistribution of the books' text in a released dataset is NOT covered unless the letter says so.
Check that the Afaan Oromoo grade 9-12 student/teacher guides are Bureau publications (copyright page).
MD
gb_say "PERMISSION record: docs/permissions/README.md (add the letter file)"

# --- 2) serve the generator (GPU) ----------------------------------------------------------
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "SERVE + AGENT on GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T026_vllm_${GB_TS}.log"
(
  conda activate gb-serve
  cd "$GB_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve \
    "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" --host 127.0.0.1 --port "$PORT" \
    --served-model-name gadaabot-gen --max-model-len 16384 --gpu-memory-utilization 0.72 --seed 7
) > "$SLOG" 2>&1 &
SP=$!
for i in $(seq 1 60); do
  curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SP" 2>/dev/null || { gb_say "vLLM server died (see $SLOG)"; tail -n 5 "$SLOG" | tee -a "$GB_REPORT"; gb_report_print; exit 1; }
  sleep 10
done
curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null || { gb_say "vLLM not ready after 10 min"; kill -KILL -- "-$SP" 2>/dev/null || true; gb_report_print; exit 1; }
gb_say "vLLM ready on 127.0.0.1:$PORT"

# --- 3) agent batch (same GPU) -------------------------------------------------------------------
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
BLOG="$GB_ROOT/logs/T026_batch_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.agent.run_batch --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --out-jsonl "$GB_ROOT/runs/T026_agent/answers.jsonl" --out-tsv "$GB_ROOT/reports/tables/T026_rag_vs_closed.tsv" \
  > "$BLOG" 2>&1 || gb_say "batch exited non-zero (see $BLOG)"
{ grep -E '^(BATCH|Q |SUMMARY|CHAT-RAG)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 5; } | cut -c1-400 | tee -a "$GB_REPORT" || true

# --- 4) stop server ------------------------------------------------------------------------------
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src docs reports/T026.txt reports/tables/T026_rag_vs_closed.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T026: agent v1 (regeneration fix, multi-query retrieval, partial answers), textbooks tier P

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^SUMMARY' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
