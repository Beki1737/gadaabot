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
