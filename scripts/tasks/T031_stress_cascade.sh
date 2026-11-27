#!/usr/bin/env bash
# =============================================================================
# T031_stress_cascade.sh | GadaaBot 2.0 | Batch B26 | paper experiments E1b + E1c
# -----------------------------------------------------------------------------
# T030c result: native RAG 4.18 >> translate-everything RAG 3.08 (+1.10) and
#   closed 2.85 (+1.33); agent_noverify = naive RAG (-0.04); verify hurts (-0.37).
# This run closes the remaining holes (resumes the 7 T030 arms, reuses judge scores):
#   rag_tt_self          cascade RAG with the 9B model as translator (stronger
#                        than NLLB; reviewers will ask)
#   closed/rag_naive/agent_noverify @noisy   informal Qubee spelling (no hudhaa,
#                        degemination, short vowels): does refine help? retrieval?
#   rag_naive/agent_noverify/agent_full @nogold   gold passage removed from the
#                        index: abstain or hallucinate? (where grading/verify should pay)
#   + Holm-adjusted p-values, confident-wrong %, blind human sheet v2 (30 x 5)
#   + diagnostic: why did naive RAG 'abstain' on 5.7% (empty drafts?)
# GPU:    YES, 1 card among 6-9 (>= 44 GiB free), ~2 to 2.5 h, queued if busy;
#         may borrow a 2nd card (6-9) for NLLB (only needed if an NLLB arm reruns)
# Output: reports/tables/T031_arms.tsv, T031_pairs.tsv, T031_blind_human_v2.tsv,
#         runs/T030_ablation/* (same dir, resumable)
# Needs:  ablation.py (new) uploaded to ~/gadaabot/tasks/; graph v4 installed (T030)
# Run:    bash ~/gadaabot/tasks/T031_stress_cascade.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T031.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T031_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T031.pid"; disown -a
  echo "T031 started in background, pid $(cat "$HOME/gadaabot/runs/T031.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T031.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T031 gb-core
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
for f in ablation.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
done
[ -f "$E/judge.py" ] || { gb_say "MISSING: judge.py not installed (run T028 first)"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/ablation.py" "$E/"
for f in "$A/graph.py" "$E/ablation.py"; do python -m py_compile "$f"; done
grep -q 'use_verify' "$A/graph.py" || { gb_say "graph.py is not v4 (run T030 first); STATUS: FAIL"; gb_report_print; exit 1; }
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"

# --- 2) serve the generator (GPU) ----------------------------------------------------------
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 44 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "SERVE + AGENT on GPU $GPU ($(date +%H:%M))"
SLOG="$GB_ROOT/logs/T031_vllm_${GB_TS}.log"
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
G2="$(GB_REPORT=/dev/null gb_pick_gpus 1 12 50 2>/dev/null || true)"
if [ -n "$G2" ] && [ "$G2" != "$GPU" ]; then VIS="$GPU,$G2"; NDEV="cuda:1"; gb_say "NLLB on second GPU $G2";
else VIS="$GPU"; NDEV="cuda:0"; gb_say "NLLB shares GPU $GPU"; fi
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
OUTD="$GB_ROOT/runs/T030_ablation"
BLOG="$GB_ROOT/logs/T031_ablation_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$VIS" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.ablation --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
  --exclude-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" --nllb "$NLLB" --nllb-device "$NDEV" --workers 8 --judge-samples 3 \
  --out-dir "$OUTD" > "$BLOG" 2>&1 || gb_say "ablation exited non-zero (see $BLOG)"
{ grep -E '^(HELDOUT|RETRIEVAL|RESUME|ARM|NLLB|JUDGE|PAIR|DONE)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 8; } | cut -c1-600 | tee -a "$GB_REPORT" || true
for f in table_arms table_pairs blind_human_v2; do
  if [ -f "$OUTD/$f.tsv" ]; then cp -f "$OUTD/$f.tsv" "$GB_ROOT/reports/tables/T031_${f#table_}.tsv"; fi
done
# diagnostic (NO GPU): naive-RAG abstains, noise examples, storage
python - "$OUTD" <<'PY2' 2>&1 | cut -c1-400 | tee -a "$GB_REPORT" || true
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "arm_rag_naive.jsonl", encoding="utf-8")]
ab = [r for r in rows if r["abstained"]]
print(f"DIAG rag_naive abstains {len(ab)}/{len(rows)}")
for r in ab[:8]:
    dr = (r.get("drafts") or [""])[0].replace("\n", " / ")
    print(f"DIAG  {r['id']} reason={r.get('abstain_reason')} draft={dr[:220]!r}")
for r in [json.loads(l) for l in open(d / "arm_rag_naive@noisy.jsonl", encoding="utf-8")][:3]:
    print(f"DIAG noisy asked: {r.get('asked', '')[:100]!r}")
PY2
du -sh "$GB_ROOT"/models/* "$GB_ROOT"/models/local/* "$GB_ROOT"/runs/* 2>/dev/null | sort -h | tail -n 12 | sed 's/^/DISK /' | tee -a "$GB_REPORT" || true

# --- 5) stop server ------------------------------------------------------------------------------
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"
git add src reports/T031.txt reports/tables/T031_*.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T031: self-translate RAG cascade, noisy-Qubee and missing-gold stress conditions, Holm p-values

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^DONE' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
