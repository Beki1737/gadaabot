#!/usr/bin/env bash
# =============================================================================
# T032_final_eval.sh | GadaaBot 2.0 | Batch B26 | final numbers on the REVIEWED set
# -----------------------------------------------------------------------------
# Your review kept 223/364 dev items (184 of them held out from agent tuning) and
# you scored the blind sheet (150 rows). This run produces the paper's final numbers:
#   A) (NO GPU) installs your review as the official dev-set filter; snapshots the
#      T030/T031 outputs (runs/T030_ablation_v4); computes on them: dataset keep
#      rates, human correctness per system (blind, 30 q x 5), human paired
#      differences, judge-vs-human agreement incl. translated outputs
#   B) (GPU) final Gate R1 retrieval on the reviewed items
#   C) (GPU) graph v5: the echo rule wrongly discarded answers that began with the
#      word 'Passages [1] akka jedhuutti' (18/316 naive-RAG 'abstains' in T031 DIAG).
#      Fixed; the 8 graph arms (rag_naive, agent_noverify, agent_full, + @noisy,
#      @nogold) are regenerated; closed and cascade arms are reused
#   D) (NO GPU) final arm table + Holm-adjusted pairs on the 184 reviewed held-out items
# GPU:    YES, 1 card among 6-9 (>= 44 GiB free), ~2 h, queued if busy
# Disk:   +53 MB (snapshot)
# Output: reports/tables/T032_v4_{devset_stats,human_arms,human_pairs,judge_human,final_arms,final_pairs}.tsv
#         reports/tables/T032_final_{arms,pairs}.tsv, reports/tables/T032_gate_r1_final.json
# Needs:  graph.py (v5), final_analysis.py uploaded to ~/gadaabot/tasks/
#         (your scored files stay where they are: reports/tables/T023_devset_review.tsv,
#          reports/tables/T031_blind_human_v2.tsv)
# Run:    bash ~/gadaabot/tasks/T032_final_eval.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T032.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T032_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T032.pid"; disown -a
  echo "T032 started in background, pid $(cat "$HOME/gadaabot/runs/T032.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T032.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T032 gb-core
cd "$GB_ROOT"
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "${SP:-}" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
T="$GB_ROOT/reports/tables"
A="$GB_ROOT/src/gadaabot/agent"; E="$GB_ROOT/src/gadaabot/eval"
DEVD="$GB_ROOT/data/eval/gadaabot_dev"
OUTD="$GB_ROOT/runs/T030_ablation"; SNAP="$GB_ROOT/runs/T030_ablation_v4"

# --- A) install + analysis of the T030/T031 outputs (NO GPU) -----------------------------------------
for f in graph.py final_analysis.py; do
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
done
for f in "$T/T023_devset_review.tsv" "$T/T031_blind_human_v2.tsv" "$OUTD/blind_key_v2.json"; do
  [ -f "$f" ] || { gb_say "MISSING: $f"; gb_report_print; exit 1; }
done
cp -f "$GB_ROOT/tasks/graph.py" "$A/"; cp -f "$GB_ROOT/tasks/final_analysis.py" "$E/"
for f in "$A/graph.py" "$E/final_analysis.py"; do python -m py_compile "$f"; done
grep -q '_PROMPT_ECHO' "$A/graph.py" || { gb_say "graph.py is not v5; STATUS: FAIL"; gb_report_print; exit 1; }
cp -f "$T/T023_devset_review.tsv" "$DEVD/devset_review.tsv"
gb_say "review installed: $(awk -F'\t' 'NR>1 && tolower($7) ~ /^y/' "$DEVD/devset_review.tsv" | wc -l) kept items"
[ -d "$SNAP" ] || cp -r "$OUTD" "$SNAP"
python -m gadaabot.eval.final_analysis --ablation-dir "$SNAP" --review "$DEVD/devset_review.tsv" \
  --blind "$T/T031_blind_human_v2.tsv" --dev "$DEVD/devset_candidates.jsonl" --out-dir "$GB_ROOT/runs/T032_v4" \
  2>&1 | grep -E '^(DEVSET|FINAL|HUMAN|JUDGE-HUMAN|DONE)|Error' | sed 's/^/V4 /' | cut -c1-400 | tee -a "$GB_REPORT" || true
for f in devset_stats human_arms human_pairs judge_human final_arms final_pairs; do
  if [ -f "$GB_ROOT/runs/T032_v4/$f.tsv" ]; then cp -f "$GB_ROOT/runs/T032_v4/$f.tsv" "$T/T032_v4_$f.tsv"; fi
done

# --- B) serve + final Gate R1 (GPU) ------------------------------------------------------------------
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
PORT=48000
ss -ltn 2>/dev/null | grep -q ":$PORT " && { gb_say "port $PORT busy; STATUS: FAIL"; gb_report_print; exit 1; }
GPU="$(gb_wait_gpus 1 44 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "GPU $GPU ($(date +%H:%M))"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 python -m gadaabot.eval.retrieval_eval --dev "$DEVD" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --out "$T/T032_gate_r1_final.json" 2>&1 \
  | grep -E '^(DEV|ARM|BREAKDOWN|OVERLAP|BEST)|Error' | sed 's/^/R1 /' | cut -c1-400 | tee -a "$GB_REPORT" || true

SLOG="$GB_ROOT/logs/T032_vllm_${GB_TS}.log"
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

# --- C) regenerate the 6 graph arms with graph v5 (closed + cascade arms resume) ----------------------
for a in rag_naive agent_noverify agent_full "rag_naive@noisy" "agent_noverify@noisy" "rag_naive@nogold" \
         "agent_noverify@nogold" "agent_full@nogold"; do
  rm -f "$OUTD/arm_$a.jsonl"
done
BLOG="$GB_ROOT/logs/T032_ablation_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.ablation \
  --llm-url "http://127.0.0.1:$PORT" \
  --qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" \
  --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite" --dev "$DEVD" \
  --exclude-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" --nllb "$GB_ROOT/models/facebook/nllb-200-3.3B" \
  --workers 8 --judge-samples 3 --out-dir "$OUTD" > "$BLOG" 2>&1 || gb_say "ablation exited non-zero (see $BLOG)"
{ grep -E '^(HELDOUT|RESUME|ARM [a-z]|JUDGE|DONE)' "$BLOG" || grep -E 'Error' "$BLOG" | tail -n 8; } | cut -c1-300 | tee -a "$GB_REPORT" || true
kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true
gb_say "vLLM stopped"

# --- D) final tables on the reviewed held-out set (NO GPU) ----------------------------------------------
python -m gadaabot.eval.final_analysis --ablation-dir "$OUTD" --review "$DEVD/devset_review.tsv" \
  --dev "$DEVD/devset_candidates.jsonl" --out-dir "$GB_ROOT/runs/T032_final" \
  2>&1 | grep -E '^(FINAL|DONE)|Error' | cut -c1-400 | tee -a "$GB_REPORT" || true
for f in final_arms final_pairs; do
  if [ -f "$GB_ROOT/runs/T032_final/$f.tsv" ]; then cp -f "$GB_ROOT/runs/T032_final/$f.tsv" "$T/T032_$f.tsv"; fi
done
python - "$OUTD" <<'PY2' 2>&1 | cut -c1-300 | tee -a "$GB_REPORT" || true
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/arm_rag_naive.jsonl", encoding="utf-8")]
ab = [r for r in rows if r["abstained"]]
print(f"DIAG v5 rag_naive abstains {len(ab)}/{len(rows)}; retried echoes "
      f"{sum(1 for r in rows if any('<<RETRY>>' in d for d in r.get('drafts', [])))}")
PY2
git add src reports/T032.txt reports/tables/T032_* data/eval/gadaabot_dev/devset_review.tsv 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T032: final evaluation on the native-speaker-reviewed set, human blind scores, graph v5 echo fix

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^FINAL-ARM' "$GB_REPORT" && grep -q '^V4 HUMAN' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
