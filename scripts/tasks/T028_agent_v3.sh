#!/usr/bin/env bash
# =============================================================================
# T028_agent_v3.sh | GadaaBot 2.0 | Batch B23 | agent v3 + LLM-judge calibration
# -----------------------------------------------------------------------------
# v3 fixes from the T027 human scoring (RAG +0.63 correctness [0.20, 1.07] vs
#   closed book; remaining errors: question echo, answer = only "[1]", textbook
#   exercise debris, off-topic extractive fallback, 3 abstains with gold evidence):
#   - sentence splitter keeps trailing [n] with its sentence, drops fragments
#   - verify drops question-echo sentences; < 6 words left -> fallback
#   - grader "no" overridden to "partly" when reranker top score >= 2.5
#   - extractive fallback only from children with reranker score >= 2.0, no debris
#   - run_batch prints automatic failure FLAGS + stores context texts (RAGAS)
#   - NEW gadaabot.eval.judge: 9B-it judge (2 variants: ref, ref+passages),
#     3 samples each, calibrated on YOUR 54 T027 labels: QWK, Spearman, bias,
#     RAG-minus-closed difference; then scores T027 vs T028 with the same judge
# What:   1) installs graph v3, run_batch, judge (NO GPU)
#         2) serves AfriqueQwen3.5-9B-it with vLLM on ONE GPU among 6-9
#         3) reruns the same 54 questions (6 chat + 48 dev) with agent v3
#         4) judge calibration on T027 + automatic T027 vs T028 comparison
#         5) stops the server
# GPU:    YES, 1 card among 6-9 (>= 40 GiB free), ~45 to 75 min, queued if busy
# Output: reports/tables/T028_rag_vs_closed.tsv (NO scoring needed this time),
#         reports/tables/T028_judge_calibration.tsv, runs/T028_agent/answers.jsonl,
#         runs/T028_judge/{calib_ref,calib_ref_ctx}.jsonl + summary.json
# Needs:  graph.py, run_batch.py, judge.py uploaded to ~/gadaabot/tasks/
#         and your scored reports/tables/T027_rag_vs_closed.tsv (already there)
# Run:    bash ~/gadaabot/tasks/T028_agent_v3.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T028.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T028_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T028.pid"; disown -a
  echo "T028 started in background, pid $(cat "$HOME/gadaabot/runs/T028.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T028.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T028 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 0) check the human labels used for calibration (NO GPU) ---------------------------------------
python - "$GB_ROOT/reports/tables/T027_rag_vs_closed.tsv" <<'PY2' 2>&1 | tee -a "$GB_REPORT"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
ok = sum(1 for r in rows if r["rag_correct_1to5"].strip() and r["closed_correct_1to5"].strip())
print(f"T027 human labels: {ok}/{len(rows)} rows with both correctness scores")
PY2

# --- 1) install graph v3 + run_batch + judge (NO GPU) ---------------------------------------------------
A="$GB_ROOT/src/gadaabot/agent"; E="$GB_ROOT/src/gadaabot/eval"
mkdir -p "$E"; [ -f "$E/__init__.py" ] || touch "$E/__init__.py"
for f in graph.py run_batch.py judge.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
done
cp -f "$GB_ROOT/tasks/graph.py" "$GB_ROOT/tasks/run_batch.py" "$A/"; cp -f "$GB_ROOT/tasks/judge.py" "$E/"
for f in "$A/graph.py" "$A/run_batch.py" "$E/judge.py"; do python -m py_compile "$f"; done
grep -q 'RERANK_TRUST' "$A/graph.py" || { gb_say "graph.py is not v3; STATUS: FAIL"; gb_report_print; exit 1; }
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"

# --- 2) serve the generator (GPU) ----------------------------------------------------------
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "SERVE + AGENT on GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T028_vllm_${GB_TS}.log"
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
BLOG="$GB_ROOT/logs/T028_batch_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.agent.run_batch --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --out-jsonl "$GB_ROOT/runs/T028_agent/answers.jsonl" --out-tsv "$GB_ROOT/reports/tables/T028_rag_vs_closed.tsv" \
  > "$BLOG" 2>&1 || gb_say "batch exited non-zero (see $BLOG)"
{ grep -E '^(BATCH|Q |SUMMARY|ABSTAIN REASONS|FLAGS|CHAT-RAG)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 5; } | cut -c1-400 | tee -a "$GB_REPORT" || true

# --- 4) judge calibration on YOUR T027 labels + T027 vs T028 (same GPU, via the server) -----------------
JLOG="$GB_ROOT/logs/T028_judge_${GB_TS}.log"
HF_HUB_OFFLINE=1 python -m gadaabot.eval.judge --llm-url "http://127.0.0.1:$PORT" \
  --parents "$GB_ROOT/data/kb/index/parents.sqlite" \
  --calib-tsv "$GB_ROOT/reports/tables/T027_rag_vs_closed.tsv" --calib-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" \
  --runs "T027=$GB_ROOT/runs/T027_agent/answers.jsonl" "T028=$GB_ROOT/runs/T028_agent/answers.jsonl" \
  --samples 3 --workers 8 --out-dir "$GB_ROOT/runs/T028_judge" \
  --out-table "$GB_ROOT/reports/tables/T028_judge_calibration.tsv" > "$JLOG" 2>&1 || gb_say "judge exited non-zero (see $JLOG)"
{ grep -E '^(CALIB|JUDGE)' "$JLOG" || grep -E 'Error' "$JLOG" | tail -n 5; } | cut -c1-500 | tee -a "$GB_REPORT" || true

# --- 5) stop server ------------------------------------------------------------------------------
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src reports/T028.txt reports/tables/T028_rag_vs_closed.tsv reports/tables/T028_judge_calibration.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T028: agent v3 (echo/citation-only guards, rerank-trust override, guarded fallback) + LLM judge calibrated on T027 labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^SUMMARY' "$GB_REPORT" && grep -q '^JUDGE-PICK' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
