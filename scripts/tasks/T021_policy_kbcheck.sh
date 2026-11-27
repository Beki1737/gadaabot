#!/usr/bin/env bash
# =============================================================================
# T021_policy_kbcheck.sh | GadaaBot 2.0 | Batch B16
# -----------------------------------------------------------------------------
# 1) GPU POLICY (NO GPU): only GPUs 6,7,8,9 from now on; if busy, QUEUE.
#    Appends a new gb_pick_gpus (restricted to GB_ALLOWED_GPUS) and
#    gb_wait_gpus (waits in 5-min steps, default up to 12 h) to
#    scripts/common.sh, and sets GB_ALLOWED_GPUS=6,7,8,9 in gadaabot.env.
# 2) KB CHECK (NO GPU): manifest table, per-source domain/tier counts, the
#    textbook titles found, and why storybook/ethnomed came back empty.
# 3) CHAT PROBE RERUN (GPU: 1 card among 6-9, >= 40 GiB free, ~10 min,
#    queued if busy): control and extended models, in the gb-serve env, each
#    in its own process group so vLLM's engine process is fully killed
#    afterwards (T020 failure cause: the previous engine kept the GPU memory).
# Run:    bash ~/gadaabot/tasks/T021_policy_kbcheck.sh   (detaches itself)
# Result: cat ~/gadaabot/reports/T021.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail
if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/logs" "$HOME/gadaabot/runs"
  OUT="$HOME/gadaabot/logs/T021_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T021.pid"; disown -a
  echo "T021 started in background, pid $(cat "$HOME/gadaabot/runs/T021.pid")"
  echo "Result: cat ~/gadaabot/reports/T021.txt   (finished when it shows STATUS)"
  exit 0
fi
G="$HOME/gadaabot"

# --- 1) GPU policy ---------------------------------------------------------------------
if ! grep -q 'GB_GPU_POLICY_V2' "$G/scripts/common.sh"; then
cat >> "$G/scripts/common.sh" <<'SH'

# --- GB_GPU_POLICY_V2 (2026-10-30): only GB_ALLOWED_GPUS (default 6,7,8,9); queue when busy ---
gb_pick_gpus() {
  local need="$1" min_gib="$2" max_util="${3:-30}" allowed="${GB_ALLOWED_GPUS:-6,7,8,9}" picks got
  picks="$(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', *' -v m="$min_gib" -v u="$max_util" -v a=",$allowed," \
        '{free=($2-$3)/1024; if (index(a, "," $1 ",") > 0 && free >= m && $4 <= u) printf "%.1f %s\n", free, $1}' \
    | sort -rn | head -n "$need" | awk '{print $2}' | paste -sd, -)"
  got="$(echo "$picks" | awk -F, 'NF{print NF} !NF{print 0}')"
  if [ -z "$picks" ] || [ "$got" -lt "$need" ]; then
    gb_say "GPU CHECK: need $need of [$allowed] with >= ${min_gib} GiB free, <= ${max_util}% util; qualifying: ${picks:-none}"
    return 3
  fi
  gb_say "GPU CHECK: OK, using GPU(s) $picks"
  echo "$picks"
}
# gb_wait_gpus <count> <min_free_GiB> [max_util=30] [max_wait_min=720]: queue until free
gb_wait_gpus() {
  local waited=0 max="${4:-720}" g
  while :; do
    if g="$(GB_REPORT=/dev/null gb_pick_gpus "$1" "$2" "${3:-30}" 2>/dev/null)"; then echo "$g"; return 0; fi
    [ "$waited" -ge "$max" ] && return 3
    echo "$(date +%H:%M) queued: waiting for $1 GPU(s) among ${GB_ALLOWED_GPUS:-6,7,8,9} (>= $2 GiB free)" >&2
    sleep 300; waited=$((waited + 5))
  done
}
SH
fi
ENVF="$G/configs/gadaabot.env"
grep -q '^GB_ALLOWED_GPUS=' "$ENVF" && sed -i 's/^GB_ALLOWED_GPUS=.*/GB_ALLOWED_GPUS=6,7,8,9/' "$ENVF" || echo "GB_ALLOWED_GPUS=6,7,8,9" >> "$ENVF"

source "$G/scripts/common.sh"
gb_init T021 gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"
gb_say "GPU POLICY: allowed GPUs ${GB_ALLOWED_GPUS}; tasks queue when they are busy"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  now: /' | tee -a "$GB_REPORT"

# --- 2) KB check (NO GPU) ----------------------------------------------------------------------
python - "$GB_ROOT/data/kb/raw" <<'PY' 2>&1 | tee -a "$GB_REPORT"
import gzip, json, sys
from collections import Counter
from pathlib import Path
raw = Path(sys.argv[1])
m = json.loads((raw / "manifest.json").read_text())
print("KB MANIFEST        docs   Oromo-ok      words(ok)  tiers / status")
for k, v in m.items():
    if v.get("status") == "ok":
        print(f"   {k:<12} {v['docs']:>7,} {v['oromo_ok_docs']:>9,} {v['oromo_ok_words']:>13,}  {v['tiers']}")
    else:
        print(f"   {k:<12} {v.get('status')}")
dom = Counter(); words = Counter()
for f in sorted(raw.glob("*.jsonl.gz")):
    for line in gzip.open(f, "rt", encoding="utf-8"):
        d = json.loads(line)
        if d["oromo_ok"]:
            dom[(d["source"], d["domain"])] += 1; words[(d["source"], d["domain"])] += d["n_words"]
print("KB BY SOURCE x DOMAIN (Oromo-ok): " + "; ".join(f"{s}/{d} {n} docs {words[(s, d)]:,} w" for (s, d), n in sorted(dom.items())))
tb = raw / "textbooks.jsonl.gz"
if tb.exists():
    rows = [json.loads(l) for l in gzip.open(tb, "rt", encoding="utf-8")]
    rows.sort(key=lambda r: -r["n_words"])
    print(f"TEXTBOOKS found: {len(rows)} PDFs (Oromo-ok {sum(r['oromo_ok'] for r in rows)})")
    for r in rows[:45]:
        print(f"   {r['n_words']:>7,} w  ok={int(r['oromo_ok'])}  {r['domain']:<9} {r['title'][:70]}  <- {r['url'][-70:]}")
PY
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python - <<'PY' 2>&1 | grep -vE 'Warning|warn' | tee -a "$GB_REPORT" || true
from collections import Counter
from datasets import load_dataset
ds = load_dataset("cis-lmu/GlotStoryBook", "default", split="train", streaming=True)
c, cols = Counter(), None
for i, r in enumerate(ds):
    cols = list(r)
    for k, v in r.items():
        if isinstance(v, str) and len(v) < 20 and ("or" in v.lower() or "gaz" in v.lower() or "om" == v.lower()):
            c[(k, v)] += 1
    if i > 60000:
        break
print("STORYBOOK columns:", cols)
print("STORYBOOK short values containing or/gaz/om:", c.most_common(15))
PY

# --- 3) chat probe rerun (1 GPU among 6-9, queued) ----------------------------------------------
conda activate gb-serve
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
CHAT="$GB_ROOT/reports/tables/T020_chat.tsv"
GPU="$(gb_wait_gpus 1 40 30 720)" || { gb_say "no GPU in 12 h; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "CHAT PROBE on GPU $GPU ($(date +%H:%M))"
L="$GB_ROOT/models/local"
for spec in "control|$L/AfriqueQwen3.5-9B-it-cpt-control" "ext|$L/AfriqueQwen3.5-9B-it-cpt-ext"; do
  IFS='|' read -r lab mp <<< "$spec"
  grep -q "^$lab	" "$CHAT" 2>/dev/null && { gb_say "skip chat $lab (already in TSV)"; continue; }
  LOG="$GB_ROOT/logs/T021_chat_${lab}_${GB_TS}.log"
  CUDA_VISIBLE_DEVICES="$GPU" setsid timeout 1500 python "$GB_ROOT/src/gadaabot/eval/chat_probe.py" \
    --model "$mp" --label "$lab" --out "$CHAT" > "$LOG" 2>&1 &
  CP=$!
  wait "$CP" || gb_say "chat probe $lab exited non-zero (log $LOG)"
  kill -KILL -- "-$CP" 2>/dev/null || true   # kill the whole process group (vLLM engine included)
  sleep 10
  { grep -E '^CHAT' "$LOG" || grep -E 'Error' "$LOG" | tail -n 3; } | cut -c1-260 | tee -a "$GB_REPORT" || true
done
gb_say "CHAT TSV rows: $(($(wc -l < "$CHAT") - 1)) (expect 18 = 3 models x 6 prompts)"
if [ "$(($(wc -l < "$CHAT") - 1))" -ge 18 ]; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
