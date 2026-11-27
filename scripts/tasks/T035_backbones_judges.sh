#!/usr/bin/env bash
# =============================================================================
# T035_backbones_judges.sh | GadaaBot 2.0 | Batch B29 | council round 2: E2, E3, E1b
# -----------------------------------------------------------------------------
# E2  second backbones on the 184 test questions, 6 systems each (closed, closed
#     self-translate, native RAG, NLLB RAG cascade, self-translate RAG cascade,
#     agent without verification):
#       Qwen3.5-9B          same architecture as our generator, WITHOUT African CPT
#                           (weak Oromo reader: the controlled test of "native wins
#                           only when the model can read the language")
#       Gemma-3-12B-it      different model family, mid-size
#     all answers judged by the calibrated AfriqueQwen judge (fixed protocol)
# E3  translation quality on TICO-19 en-om (300 pairs, chrF, both directions):
#     NLLB-200 3.3B vs self-translation by each of the 3 models
# E1b second judge from another family: Gemma-3-27B-it (2 GPUs) runs the judge-input
#     protocol on the blind + dev sets and re-judges all test answers (all backbones)
# Phases (one vLLM server at a time): Qwen3.5-9B -> Gemma-3-12B -> AfriqueQwen judge
#     -> Gemma-3-27B judge
# GPU:    YES, 1 card among 6-9 (>= 40 GiB) for phases 1-3, then 2 cards (>= 42 GiB
#         each) for phase 4; ~5 to 6 h total, queued if busy
# Disk:   < 100 MB (+ TICO-19 test set zip, 5 MB, if not already on disk)
# Output: runs/T035_{qwen35_9b,gemma3_12b}/ (answers), runs/T035/ (judging, MT)
# Needs:  extra.py (new) uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T035_backbones_judges.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T035.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
cd "$HOME/gadaabot"
if [ "${GB_DETACHED:-0}" != "1" ]; then
  OUT="$HOME/gadaabot/logs/T035_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T035.pid"; disown -a
  echo "T035 started in background, pid $(cat "$HOME/gadaabot/runs/T035.pid")"
  echo "Result: cat ~/gadaabot/reports/T035.txt   (finished when it shows STATUS)"
  exit 0
fi
source "$HOME/gadaabot/scripts/common.sh"
gb_init T035 gb-core
cd "$GB_ROOT"
SP=""
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; [ -n "$SP" ] && kill -KILL -- "-$SP" 2>/dev/null; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
E="$GB_ROOT/src/gadaabot/eval"; OUTD="$GB_ROOT/runs/T035"; mkdir -p "$OUTD"
PORT=48000
[ -f "$GB_ROOT/tasks/extra.py" ] || { gb_say "MISSING: upload extra.py"; gb_report_print; exit 1; }
cp -f "$GB_ROOT/tasks/extra.py" "$E/"; python -m py_compile "$E/extra.py"
RERANK="$(find "$SHARED_MODELS" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1)"
NLLB="$GB_ROOT/models/facebook/nllb-200-3.3B"
QWEN="$SHARED_MODELS/Qwen/Qwen3.5-9B"; G12="$SHARED_MODELS/Google/gemma-3-12b-Instruct"; G27="$SHARED_MODELS/Google/gemma-3-27b-Instruct"
AFQ="$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview"
for m in "$QWEN" "$G12" "$G27" "$AFQ"; do [ -f "$m/config.json" ] || { gb_say "MISSING model $m"; gb_report_print; exit 1; }; done
RET=(--qdrant-url "http://127.0.0.1:${GB_QDRANT_HTTP:-46333}" --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct"
     --reranker "$RERANK" --parents "$GB_ROOT/data/kb/index/parents.sqlite")
ARMS="closed,closed_tt_self,rag_naive,rag_tt_nllb,rag_tt_self,agent_noverify"
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"

serve() {  # serve <model_path> <gpus> <tp> <util> [extra vllm args...]
  local path="$1" gpus="$2" tp="$3" util="$4"; shift 4
  local log="$GB_ROOT/logs/T035_vllm_$(basename "$path")_${GB_TS}.log"
  ( conda activate gb-serve; cd "$GB_ROOT"
    CUDA_VISIBLE_DEVICES="$gpus" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn exec setsid vllm serve "$path" \
      --host 127.0.0.1 --port "$PORT" --served-model-name gadaabot-gen --max-model-len 16384 \
      --tensor-parallel-size "$tp" --gpu-memory-utilization "$util" --seed 7 "$@" ) > "$log" 2>&1 &
  SP=$!
  for i in $(seq 1 90); do
    curl -fs "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { gb_say "served $(basename "$path") on GPU $gpus"; return 0; }
    kill -0 "$SP" 2>/dev/null || { gb_say "vLLM died for $(basename "$path") (see $log)"; tail -n 4 "$log" | tee -a "$GB_REPORT"; SP=""; return 1; }
    sleep 10
  done
  gb_say "vLLM not ready for $(basename "$path")"; kill -KILL -- "-$SP" 2>/dev/null || true; SP=""; return 1
}
unserve() { [ -n "$SP" ] && { kill -TERM -- "-$SP" 2>/dev/null || true; sleep 15; kill -KILL -- "-$SP" 2>/dev/null || true; }; SP=""; sleep 5; }
show() { { grep -E "$2" "$1" || grep -E 'Error' "$1" | tail -n 6; } | cut -c1-500 | tee -a "$GB_REPORT" || true; }
extra() {  # extra <logname> <subcommand args...>
  local log="$GB_ROOT/logs/T035_$1_${GB_TS}.log"; shift
  CUDA_VISIBLE_DEVICES="$VIS" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.extra "$@" \
    --root "$GB_ROOT" "${RET[@]}" --llm-url "http://127.0.0.1:$PORT" > "$log" 2>&1 || gb_say "extra $1 exited non-zero (see $log)"
  show "$log" '^(E1|E2|E3|DONE)'
}

# --- phases 1 and 2: second backbones generate (E2) + self-translation quality (E3) ----------------------
for spec in "qwen35_9b|$QWEN|{\"image\":0,\"video\":0}" "gemma3_12b|$G12|{\"image\":0}"; do
  IFS='|' read -r TAG MP MM <<< "$spec"
  GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
  G2="$(GB_REPORT=/dev/null gb_pick_gpus 1 12 50 2>/dev/null || true)"
  if [ -n "$G2" ] && [ "$G2" != "$GPU" ]; then VIS="$GPU,$G2"; NDEV="cuda:1"; else VIS="$GPU"; NDEV="cuda:0"; fi
  serve "$MP" "$GPU" 1 0.60 --limit-mm-per-prompt "$MM" || continue
  AD="$GB_ROOT/runs/T035_$TAG"; ALOG="$GB_ROOT/logs/T035_ablation_${TAG}_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="$VIS" HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m gadaabot.eval.ablation \
    --llm-url "http://127.0.0.1:$PORT" "${RET[@]}" --dev "$GB_ROOT/data/eval/gadaabot_dev" \
    --exclude-jsonl "$GB_ROOT/runs/T027_agent/answers.jsonl" --nllb "$NLLB" --nllb-device "$NDEV" \
    --arms "$ARMS" --workers 8 --judge-samples 1 --out-dir "$AD" > "$ALOG" 2>&1 || gb_say "ablation $TAG exited non-zero (see $ALOG)"
  show "$ALOG" '^(HELDOUT|RESUME|ARM [a-z]|DONE)'
  gb_say "SELFJUDGE $TAG below: backbone judging its own answers, 1 sample, all 316 held-out (self-preference check only)"
  show "$ALOG" '^ARM-RESULT arm'
  extra "mt_$TAG" mt --out-dir "$OUTD" --mt-systems "self" --tag "$TAG" --nllb "$NLLB" --nllb-device "$NDEV"
  unserve
done

# --- phase 3: AfriqueQwen judge: re-judge both backbones (E2) + NLLB and AfriqueQwen self-translation (E3) ------
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
VIS="$GPU"
serve "$AFQ" "$GPU" 1 0.60 || { gb_report_print; exit 1; }
extra mt_afq mt --out-dir "$OUTD" --mt-systems nllb self --tag afriqueqwen9b --nllb "$NLLB" --nllb-device cuda:0
for TAG in qwen35_9b gemma3_12b; do
  extra "rejudge_afq_$TAG" rejudge --out-dir "$OUTD" --arms-dir "$GB_ROOT/runs/T035_$TAG" --tag afq_judge \
    --arm-list closed closed_tt_self rag_naive rag_tt_nllb rag_tt_self agent_noverify
done
unserve

# --- phase 4: Gemma-3-27B judge on 2 cards (E1b + cross-judging all backbones) -------------------------------------
GP="$(gb_wait_gpus 2 42 30 720)" || { gb_say "no 2 GPUs in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
VIS="$GP"
serve "$G27" "$GP" 2 0.80 --limit-mm-per-prompt '{"image":0}' || { gb_report_print; exit 1; }
extra e1b judge-inputs --out-dir "$OUTD" --judge-url "http://127.0.0.1:$PORT" --judge-model gadaabot-gen --tag gemma3_27b
extra rejudge_g27_afq rejudge --out-dir "$OUTD" --arms-dir "$GB_ROOT/runs/T030_ablation" --tag g27_judge \
  --judge-url "http://127.0.0.1:$PORT" --judge-model gadaabot-gen
for TAG in qwen35_9b gemma3_12b; do
  extra "rejudge_g27_$TAG" rejudge --out-dir "$OUTD" --arms-dir "$GB_ROOT/runs/T035_$TAG" --tag g27_judge \
    --judge-url "http://127.0.0.1:$PORT" --judge-model gadaabot-gen \
    --arm-list closed closed_tt_self rag_naive rag_tt_nllb rag_tt_self agent_noverify
done
unserve
gb_say "vLLM stopped"
git add src reports/T035.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T035: second backbones, MT quality, second judge family

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
if grep -q '^E2-PAIR \[g27_judge\]' "$GB_REPORT" && grep -q '^E2-PAIR \[afq_judge\]' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
