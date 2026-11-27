#!/usr/bin/env bash
# =============================================================================
# T017_qdrant_bench.sh | GadaaBot 2.0 | Batch B12 | Phase 2 start (retrieval)
# -----------------------------------------------------------------------------
# Safe to run while T016 trains (different GPUs or CPU only).
# What:   1) installs the Qdrant server as a standalone binary (D18) in
#            ~/gadaabot/opt/qdrant, storage in ~/gadaabot/qdrant, listening on
#            127.0.0.1 only, non-default ports (46333 http / 46334 grpc) so it
#            cannot clash with other lab services; telemetry off
#         2) writes scripts/qdrant_start.sh and scripts/qdrant_stop.sh
#         3) installs qdrant-client into gb-core (core pins frozen)
#         4) first retrieval arms on Oromo Belebele (900 questions -> their own
#            passage): BM25 with 6 Oromo analyzers, dense AfriE5 and BGE-M3,
#            hybrid AfriE5+BM25 with RRF and DBSF fusion, then the
#            bge-reranker-v2-m3 cross-encoder on the best hybrid
# GPU:    OPTIONAL, small: uses 1 card with >= 8 GiB free if one exists
#         (encoders + reranker, ~5 min); otherwise runs on CPU (~30 min)
# Disk:   ~0.2 GiB
# Needs:  oromo_analyzer.py, retrieval_bench.py uploaded to ~/gadaabot/tasks/
# Run:    bash ~/gadaabot/tasks/T017_qdrant_bench.sh
# Result: cat ~/gadaabot/reports/T017.txt
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T017 gb-core
exec > >(tee -a "$GB_LOG") 2>&1
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- modules ---------------------------------------------------------------------
mkdir -p "$GB_ROOT/src/gadaabot/retrieval"; [ -f "$GB_ROOT/src/gadaabot/retrieval/__init__.py" ] || : > "$GB_ROOT/src/gadaabot/retrieval/__init__.py"
for spec in "oromo_analyzer.py|retrieval" "retrieval_bench.py|eval"; do
  IFS='|' read -r f sub <<< "$spec"
  [ -f "$GB_ROOT/tasks/$f" ] || { gb_say "MISSING: upload $f to ~/gadaabot/tasks/"; gb_report_print; exit 1; }
  cp -f "$GB_ROOT/tasks/$f" "$GB_ROOT/src/gadaabot/$sub/$f"; python -m py_compile "$GB_ROOT/src/gadaabot/$sub/$f"
done

# --- 1) Qdrant binary (NO GPU) ---------------------------------------------------------
QD="$GB_ROOT/opt/qdrant"; mkdir -p "$QD" "$GB_ROOT/qdrant/storage" "$GB_ROOT/qdrant/snapshots"
if [ ! -x "$QD/qdrant" ]; then
  TAG="$(python - <<'PY'
import json, urllib.request
print(json.load(urllib.request.urlopen("https://api.github.com/repos/qdrant/qdrant/releases/latest", timeout=30))["tag_name"])
PY
)"
  URL="https://github.com/qdrant/qdrant/releases/download/$TAG/qdrant-x86_64-unknown-linux-musl.tar.gz"
  curl -fsSL "$URL" -o "$QD/qdrant.tgz"
  tar -xzf "$QD/qdrant.tgz" -C "$QD" && rm -f "$QD/qdrant.tgz"
  echo "$TAG" > "$QD/VERSION"
fi
gb_say "QDRANT binary: $("$QD/qdrant" --version 2>/dev/null | head -n 1 || cat "$QD/VERSION")"

# --- 2) start/stop helpers ---------------------------------------------------------------
cat > "$GB_ROOT/scripts/qdrant_start.sh" <<'SH'
#!/usr/bin/env bash
# Start the GadaaBot Qdrant server (localhost only). Idempotent.
set -euo pipefail
G="$HOME/gadaabot"; PIDF="$G/qdrant/qdrant.pid"
HTTP="${GB_QDRANT_HTTP:-46333}"; GRPC="${GB_QDRANT_GRPC:-46334}"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "qdrant already running (pid $(cat "$PIDF")) on $HTTP"; exit 0; fi
if ss -ltn 2>/dev/null | grep -q ":$HTTP "; then echo "port $HTTP busy (someone else); set GB_QDRANT_HTTP/GRPC"; exit 1; fi
cd "$G/qdrant"
QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__SERVICE__HTTP_PORT="$HTTP" QDRANT__SERVICE__GRPC_PORT="$GRPC" \
QDRANT__STORAGE__STORAGE_PATH="$G/qdrant/storage" QDRANT__STORAGE__SNAPSHOTS_PATH="$G/qdrant/snapshots" \
QDRANT__TELEMETRY_DISABLED=true QDRANT__LOG_LEVEL=WARN \
  nohup "$G/opt/qdrant/qdrant" > "$G/logs/qdrant_server.log" 2>&1 &
echo $! > "$PIDF"
for i in $(seq 1 30); do curl -fs "http://127.0.0.1:$HTTP/readyz" >/dev/null 2>&1 && { echo "qdrant ready on 127.0.0.1:$HTTP (pid $(cat "$PIDF"))"; exit 0; }; sleep 1; done
echo "qdrant did not become ready; see $G/logs/qdrant_server.log"; exit 1
SH
cat > "$GB_ROOT/scripts/qdrant_stop.sh" <<'SH'
#!/usr/bin/env bash
PIDF="$HOME/gadaabot/qdrant/qdrant.pid"
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && echo "qdrant stopped" || echo "qdrant not running"
rm -f "$PIDF"
SH
chmod +x "$GB_ROOT/scripts/qdrant_start.sh" "$GB_ROOT/scripts/qdrant_stop.sh"
bash "$GB_ROOT/scripts/qdrant_start.sh" | tee -a "$GB_REPORT"
QURL="http://127.0.0.1:${GB_QDRANT_HTTP:-46333}"

# --- 3) client (NO GPU) ---------------------------------------------------------------------
CONS="$GB_ROOT/envs/gb-core.constraints.full.txt"
python -m pip list --format=freeze | grep -viE '^(pip|setuptools|wheel)==' > "$CONS"
python -c "import qdrant_client" 2>/dev/null || python -m pip install -q -c "$CONS" "qdrant-client>=1.12" 2>&1 | tail -n 3
python -c "import qdrant_client, importlib.metadata as m; print('qdrant-client', m.version('qdrant-client'))" | tee -a "$GB_REPORT"
python -m pip freeze > "$GB_ROOT/envs/gb-core.lock.txt"

# --- 4) benchmark (optional small GPU) -----------------------------------------------------------
SM="$SHARED_MODELS"
BGEM3="$(find "$SM" -maxdepth 2 -iname 'bge-m3' -type d | head -n 1 || true)"
RERANK="$(find "$SM" -maxdepth 2 -iname 'bge-reranker-v2-m3' -type d | head -n 1 || true)"
GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 8 50 2>/dev/null || true)"
if [ -n "$GPU" ]; then gb_say "ENCODERS on GPU $GPU (small, ~3 GiB)"; export CUDA_VISIBLE_DEVICES="$GPU"
else gb_say "ENCODERS on CPU (no card with 8 GiB free)"; export CUDA_VISIBLE_DEVICES=""; fi
python -m gadaabot.eval.retrieval_bench \
  --belebele "$GB_ROOT/data/eval/facebook/belebele/data/gaz_Latn.jsonl" --qdrant-url "$QURL" \
  --afrie5 "$GB_ROOT/models/McGill-NLP/AfriE5-Large-instruct" ${BGEM3:+--bgem3 "$BGEM3"} ${RERANK:+--reranker "$RERANK"} \
  --out "$GB_ROOT/reports/tables/T017_retrieval_belebele.json" > "$GB_ROOT/logs/T017_bench_${GB_TS}.log" 2>&1 \
  || gb_say "BENCH exited non-zero (see logs/T017_bench_${GB_TS}.log)"
{ grep -E '^(DATA|ENCODE|INDEX|ARM|BEST)' "$GB_ROOT/logs/T017_bench_${GB_TS}.log" || true; } | tee -a "$GB_REPORT"
{ grep -E 'Error' "$GB_ROOT/logs/T017_bench_${GB_TS}.log" | grep -v '^\s*File' | tail -n 3 | cut -c1-240 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"

cd "$GB_ROOT"
git add src scripts/qdrant_start.sh scripts/qdrant_stop.sh reports/T017.txt reports/tables/T017_retrieval_belebele.json 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T017: Qdrant standalone server, Oromo BM25 analyzers, first retrieval arms on Belebele

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
gb_say "NOTE: Qdrant keeps running (bash ~/gadaabot/scripts/qdrant_stop.sh to stop)"
if grep -q '^BEST:' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
