#!/usr/bin/env bash
# =============================================================================
# T025_agent_v0.sh | GadaaBot 2.0 | Batch B20 | Phase 3: agent v0 end to end
# -----------------------------------------------------------------------------
# What:   1) installs the agent (src/gadaabot/agent: retriever, llm, graph,
#            run_batch) and LangGraph into gb-core (core pins frozen)
#         2) serves AfriqueQwen3.5-9B-it with vLLM (OpenAI API, 127.0.0.1:48000,
#            non-thinking) on ONE GPU among 6-9; the retriever's AfriE5 and
#            reranker share the same card
#         3) runs the cyclic agent (refine -> hybrid retrieve -> grade ->
#            reformulate loop -> generate with [n] citations -> claim verify ->
#            finalize / abstain; emergency screen at ingress) on 6 chat prompts +
#            48 dev questions, next to a closed-book answer from the same model
#         4) stops the server
# GPU:    YES, 1 card among 6-9 (>= 40 GiB free), ~30 to 60 min, queued if busy
# Output: reports/tables/T025_rag_vs_closed.tsv (for your scoring),
#         runs/T025_agent/answers.jsonl (full traces)
# Needs:  retriever.py, llm.py, graph.py, run_batch.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T025_agent_v0.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T025.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T025_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T025.pid"; disown -a
  echo "T025 started in background, pid $(cat "$HOME/gadaabot/runs/T025.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T025.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T025 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 1) install (NO GPU) ---------------------------------------------------------------
A="$GB_ROOT/src/gadaabot/agent"; mkdir -p "$A"; [ -f "$A/__init__.py" ] || : > "$A/__init__.py"
for f in retriever.py llm.py graph.py run_batch.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$A/$f"; python -m py_compile "$A/$f"
done
CONS="$GB_ROOT/envs/gb-core.constraints.full.txt"
python -m pip list --format=freeze | grep -viE '^(pip|setuptools|wheel)==' > "$CONS"
python -c "import langgraph" 2>/dev/null || HF_HUB_OFFLINE=0 python -m pip install -q -c "$CONS" "langgraph>=1.0" 2>&1 | tail -n 3
python -c "import langgraph, httpx, importlib.metadata as m; print('langgraph', m.version('langgraph'), '| httpx', httpx.__version__)" | tee -a "$GB_REPORT"
python -m pip freeze > "$GB_ROOT/envs/gb-core.lock.txt"

# --- 2) serve the generator (GPU) ----------------------------------------------------------
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "SERVE + AGENT on GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T025_vllm_${GB_TS}.log"
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
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
BLOG="$GB_ROOT/logs/T025_batch_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.agent.run_batch --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --out-jsonl "$GB_ROOT/runs/T025_agent/answers.jsonl" --out-tsv "$GB_ROOT/reports/tables/T025_rag_vs_closed.tsv" \
  > "$BLOG" 2>&1 || gb_say "batch exited non-zero (see $BLOG)"
{ grep -E '^(BATCH|Q |SUMMARY|CHAT-RAG)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 5; } | cut -c1-400 | tee -a "$GB_REPORT" || true

# --- 4) stop server ------------------------------------------------------------------------------
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src reports/T025.txt reports/tables/T025_rag_vs_closed.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T025: GadaaBot agent v0 (LangGraph cyclic RAG) vs closed-book on chat prompts and dev sample

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^SUMMARY' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
