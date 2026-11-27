#!/usr/bin/env bash
# =============================================================================
# T034_robustness.sh | GadaaBot 2.0 | Batch B29 | council round 1: E1, E4, E5, E6
# -----------------------------------------------------------------------------
# Post hoc robustness runs (test data used as test only; rules fixed in advance):
#   0) (NO GPU) model inventory for choosing the 2nd backbone and 2nd judge
#   E4 (NO GPU) reranker-score abstention for native RAG; threshold calibrated on
#      the 132 review-dropped held-out items, applied unchanged to the 184 test items
#   E1 (GPU) judge-input protocol: ref / ctx / ref+ctx / ref+ctx without [n] markers
#      on the blind sheet (150 answers, 2 annotators) and dev (108 answers), plus the
#      citation-marker control on all 184 x 8 test answers
#   E5 (GPU) native RAG on the legal public index (tiers A + P only)
#   E6 (GPU) 2 extra sampling seeds for closed, native RAG, agent, full agent
# GPU:    YES, 1 card among 6-9 (>= 40 GiB free), ~2 to 2.5 h, queued if busy
# Disk:   < 50 MB
# Output: runs/T034/{E1_*,E1c_*,E4_*,E5_*,E6_*}, reports/T034.txt
# Needs:  extra.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T034_robustness.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T034.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T034_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T034.pid"; disown -a
  echo "T034 started in background, pid $(cat "$HOME/gadaabot/runs/T034.pid")"
  echo "Result: cat ~/gadaabot/reports/T034.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T034 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
OUTD="$GB_ROOT/runs/T034"; mkdir -p "$OUTD"
E="$GB_ROOT/src/gadaabot/eval"

# --- 0) model inventory (NO GPU) ------------------------------------------------------------------------
{
  echo "INV shared models (${SHARED_MODELS}):"
  for d in "$SHARED_MODELS"/*/ "$SHARED_MODELS"/*/*/; do
    [ -f "$d/config.json" ] || continue
    python - "$d" <<'PY2' 2>/dev/null || true
import json, os, sys
d = sys.argv[1].rstrip("/")
c = json.load(open(os.path.join(d, "config.json")))
arch = (c.get("architectures") or ["?"])[0]
size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs if f.endswith((".safetensors", ".bin"))) / 1e9
print(f"INV {os.path.relpath(d, os.path.dirname(os.path.dirname(d)))[:70]:<70} {arch:<38} {size:6.1f} GB")
PY2
  done
  echo "INV local: $(ls "$GB_ROOT/models" "$GB_ROOT/models/local" 2>/dev/null | tr '\n' ' ')"
  python -c "from huggingface_hub import whoami; print('INV HF login:', whoami().get('name'))" 2>/dev/null || echo "INV HF login: none"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/INV GPU /'
} 2>&1 | tee -a "$GB_REPORT"

# --- install ---------------------------------------------------------------------------------------------
[ -f "$GB_ROOT/tasks/extra.py" ] || { gb_say "MISSING: upload extra.py to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/extra.py" "$E/"; python -m py_compile "$E/extra.py"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
COMMON=(--root "$GB_ROOT" --out-dir "$OUTD" --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}"
        --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" --reranker "$RERANK"
        --parents "$GB_ROOT/data/kb/index/parents.sqlite")

# --- E4 (NO GPU) -----------------------------------------------------------------------------------------
python -m gadaabot.eval.extra abstain "${COMMON[@]}" 2>&1 | grep -E '^(E4|DONE)|Error' | cut -c1-400 | tee -a "$GB_REPORT" || true

# --- serve -----------------------------------------------------------------------------------------------
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T034_vllm_${GB_TS}.log"
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
gb_say "vLLM ready"

# --- E1, E5, E6 (GPU) -------------------------------------------------------------------------------------
for step in "judge-inputs --test-control" "tierfilter" "seeds"; do
  LOG="$GB_ROOT/logs/T034_${step%% *}_${GB_TS}.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.extra $step "${COMMON[@]}" \
    --llm-url "http://127.0.0.1:$PORT" > "$LOG" 2>&1 || gb_say "${step%% *} exited non-zero (see $LOG)"
  { grep -E '^(E1|E5|E6|DONE)' "$LOG" || grep -E 'Error' "$LOG" | tail -n 6; } | cut -c1-600 | tee -a "$GB_REPORT" || true
done
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src reports/T034.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T034: post hoc robustness (judge inputs, abstention, public index, seeds)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^DONE seeds' "$GB_REPORT" && grep -q '^DONE judge-inputs' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
