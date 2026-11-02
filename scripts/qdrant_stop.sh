#!/usr/bin/env bash
PIDF="$HOME/gadaabot/qdrant/qdrant.pid"
[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null && echo "qdrant stopped" || echo "qdrant not running"
rm -f "$PIDF"
