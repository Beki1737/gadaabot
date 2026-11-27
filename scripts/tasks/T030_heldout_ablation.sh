#!/usr/bin/env bash
# =============================================================================
# T030_heldout_ablation.sh | GadaaBot 2.0 | Batch B25 | paper experiment E1
# -----------------------------------------------------------------------------
# Answers the two claims the paper still has no evidence for, on the ~316 dev
# questions never used for agent tuning (T025-T028 items excluded):
#   (1) native processing vs translation cascade (the project thesis), same LLM
#   (2) does the cyclic agent beat plain RAG? (node ablation)
# Arms: closed | closed_tt_nllb | closed_tt_self | rag_naive | rag_tt_nllb |
#       agent_noverify | agent_full (graph v4)
# Scoring: calibrated judge (ref_ctx, D11 PASS), same background passages for
#   all arms, 3 samples; paired bootstrap CIs; abstain / fallback / latency /
#   English leakage; blind 40x3 sheet for a 2nd native-speaker annotator.
# Graph v4: source-list echo bug fixed (T029 '0 words'), raw drafts logged,
#   node switches for the ablation.
# What:   0) storage check + download NLLB-200-3.3B (~17.6 GB, NO GPU)
#         1) installs graph v4 + ablation.py (NO GPU)
#         2) serves AfriqueQwen3.5-9B-it with vLLM on ONE GPU among 6-9;
#            retriever + NLLB share the card
#         3) runs all 7 arms + judge
#         4) stops the server
# GPU:    YES, 1 card among 6-9 (>= 44 GiB free), ~2 to 3 h, queued if busy
# Disk:   +17.6 GB (models/facebook/nllb-200-3.3B); aborts if project > 130 GiB
# Output: reports/tables/T030_arms.tsv, T030_pairs.tsv, T030_blind_human.tsv,
#         runs/T030_ablation/{answers,judge}.jsonl, summary.json, blind_key.json
# Needs:  graph.py (v4), ablation.py uploaded to ~/gadaabot/tasks/
#         (judge.py from T028 is already installed)
# Run:    bash ~/gadaabot/tasks/T030_heldout_ablation.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T030.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T030_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T030.pid"; disown -a
  echo "T030 started in background, pid $(cat "$HOME/gadaabot/runs/T030.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T030.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T030 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 0) storage + NLLB-200-3.3B (NO GPU) ----------------------------------------------------------
USED=$(du -s --block-size=1G "$GB_ROOT" 2>/dev/null | cut -f1 || true); USED=${USED:-0}
gb_say "project storage used: ${USED} GiB (cap 150)"
NLLB="$GB_ROOT/models/facebook/nllb-200-3.3B"
if [ ! -f "$NLLB/config.json" ]; then
  [ "$USED" -lt 130 ] || { gb_say "storage too high for +18 GB; free space first; STATUS: FAIL"; gb_report_print; exit 1; }
  HF_HUB_OFFLINE=0 python - "$NLLB" <<'PY2' 2>&1 | tail -n 3 | tee -a "$GB_REPORT"
import sys
from huggingface_hub import HfApi, snapshot_download
repo = "facebook/nllb-200-3.3B"
files = HfApi().list_repo_files(repo)
weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
snapshot_download(repo, local_dir=sys.argv[1], allow_patterns=["*.json", "*.model", "*.txt"] + weights)
print("NLLB downloaded:", sys.argv[1], "weights", weights)
PY2
fi
[ -f "$NLLB/config.json" ] || { gb_say "NLLB download failed; STATUS: FAIL"; gb_report_print; exit 1; }

# --- 1) install graph v4 + ablation (NO GPU) -------------------------------------------------------
A="$GB_ROOT/src/gadaabot/agent"; E="$GB_ROOT/src/gadaabot/eval"
for f in graph.py ablation.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
done
[ -f "$E/judge.py" ] || { gb_say "MISSING: judge.py not installed (run T028 first)"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/graph.py" "$A/"; cp -f "$GB_ROOT/tasks/ablation.py" "$E/"
for f in "$A/graph.py" "$E/ablation.py"; do python -m py_compile "$f"; done
grep -q 'use_verify' "$A/graph.py" || { gb_say "graph.py is not v4; STATUS: FAIL"; gb_report_print; exit 1; }
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"

# --- 2) serve the generator (GPU) ----------------------------------------------------------
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 44 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "SERVE + AGENT on GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T030_vllm_${GB_TS}.log"
(
  conda activate gb-serve
  cd "$GB_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve \
    "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" --host 127.0.0.1 --port "$PORT" \
    --served-model-name gadaabot-gen --max-model-len 16384 --gpu-memory-utilization 0.60 --seed 7
) > "$SLOG" 2>&1 &
SP=$!
for i in $(seq 1 60); do
  curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$SP" 2>/dev/null || { gb_say "vLLM server died (see $SLOG)"; tail -n 5 "$SLOG" | tee -a "$GB_REPORT"; gb_report_print; exit 1; }
  sleep 10
done
curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null || { gb_say "vLLM not ready after 10 min"; kill -KILL -- "-$SP" 2>/dev/null || true; gb_report_print; exit 1; }
gb_say "vLLM ready on 127.0.0.1:$PORT"

# --- 3) held-out ablation + judge (same GPU) ---------------------------------------------------------
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
OUTD="$GB_ROOT/runs/T030_ablation"
BLOG="$GB_ROOT/logs/T030_ablation_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.ablation --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --exclude-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" --nllb "$NLLB" --workers 8 --judge-samples 3 \
  --out-dir "$OUTD" > "$BLOG" 2>&1 || gb_say "ablation exited non-zero (see $BLOG)"
{ grep -E '^(HELDOUT|RETRIEVAL|ARM|NLLB|JUDGE|PAIR|DONE)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 8; } | cut -c1-600 | tee -a "$GB_REPORT" || true
for f in table_arms table_pairs blind_human; do
  if [ -f "$OUTD/$f.tsv" ]; then cp -f "$OUTD/$f.tsv" "$GB_ROOT/reports/tables/T030_${f#table_}.tsv"; fi
done

# --- 5) stop server ------------------------------------------------------------------------------
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src reports/T030.txt reports/tables/T030_*.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T030: held-out ablation (native vs translation cascade, agent node ablation), graph v4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^DONE' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
