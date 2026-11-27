#!/usr/bin/env bash
# =============================================================================
# T003b_fix_serve.sh | GadaaBot 2.0 | Batch B02 (fix for T003)
# -----------------------------------------------------------------------------
# Root cause (from T003 log): vLLM 0.32 imports torchcodec at startup; pip took
#   torchcodec 0.17.0 from PyPI, which is a CUDA 13 build (needs libnvrtc.so.13).
#   Driver 575 / our cu129 stack only has CUDA 12 libraries, so `from vllm import
#   LLM` crashed before any model loaded. vLLM and torch themselves are fine.
# v2:     the cu129 index has no torchcodec 0.17 (newest is 0.16), and vLLM 0.32
#         guards the torchcodec import against ImportError but not OSError.
#         So the fix is to REMOVE torchcodec: vLLM then skips its audio backend,
#         which GadaaBot (text only) never uses. v1 also exited silently at the
#         library scan (a failing ldd under 'set -o pipefail'); fixed.
# What:   1) make sure torchcodec is absent; record that in the env notes
#         2) scan gb-serve for any other compiled library that wants CUDA 13
#         3) import check of vLLM (no GPU)
#         4) re-run the T003 smoke test (both models, one after the other)
# Tool:   MobaXterm terminal, inside tmux        Time: about 10 minutes
# GPU:    YES at step 4: 1 card, >= 40 GiB free, <= 30% util (auto-picked;
#         stops cleanly if none). Steps 1 to 3 use NO GPU.
# Disk:   no change (swaps one package)
# Run:    tmux new -s gb ; bash ~/gadaabot/tasks/T003b_fix_serve.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T003b gb-serve
exec > >(tee -a "$GB_LOG") 2>&1
# never exit silently again: any unexpected failure prints where it stopped
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR

TC_VER="$(python -c 'from importlib.metadata import version; print(version("torchcodec"))' 2>/dev/null || echo none)"
gb_say "TORCHCODEC before: $TC_VER"

# ---------------------------------------------------------------------------
# 1) remove torchcodec (NO GPU)
# ---------------------------------------------------------------------------
if [ "$TC_VER" != "none" ]; then
  python -m pip uninstall -y torchcodec
  gb_say "TORCHCODEC: removed $TC_VER (CUDA 13 build; no cu129 build of 0.17 exists)"
else
  gb_say "TORCHCODEC: absent, as required"
fi
cat > "$GB_ROOT/envs/gb-serve.NOTES.txt" <<'EOF2'
gb-serve rules
- torchcodec must NOT be installed: the PyPI build needs CUDA 13 (libnvrtc.so.13),
  driver 575 provides CUDA <= 12.9, and vLLM 0.32 only skips the import on ImportError.
- Never 'pip install vllm' from PyPI here: the PyPI wheel is CUDA 13. Use the
  +cu129 wheel from the vLLM GitHub release (see T003).
EOF2

# ---------------------------------------------------------------------------
# 2) scan for compiled libraries that still need CUDA 13 (NO GPU)
# ---------------------------------------------------------------------------
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
BAD="$( { find "$SITE" -name '*.so*' -type f 2>/dev/null \
  | xargs -r -n 50 ldd 2>/dev/null \
  | awk '/^\//{lib=$1} /\.so\.13 => not found/{print lib" -> "$1}' | sort -u | head -n 15; } || true )"
if [ -z "$BAD" ]; then
  gb_say "CUDA13 SCAN: clean, no library asks for CUDA 13 runtime pieces"
else
  gb_say "CUDA13 SCAN: libraries still wanting CUDA 13 (only a problem if imported at startup):"
  echo "$BAD" | tee -a "$GB_REPORT" >&2
fi

# ---------------------------------------------------------------------------
# 3) import check (NO GPU)
# ---------------------------------------------------------------------------
import_ok() { CUDA_VISIBLE_DEVICES="" python -c "from vllm import LLM, SamplingParams; import vllm; print('VLLM IMPORT OK', vllm.__version__)" 2>&1 | tail -n 3 || true; }
OUT="$(import_ok)"
if echo "$OUT" | grep -q "VLLM IMPORT OK"; then
  gb_say "$(echo "$OUT" | grep 'VLLM IMPORT OK')"
else
  gb_say "VLLM IMPORT: FAILED without torchcodec: $(echo "$OUT" | tail -n 1 | cut -c1-200)"
  gb_say "STATUS: FAIL"; gb_report_print; exit 1
fi
python -m pip freeze > "$GB_ROOT/envs/gb-serve.lock.txt"

# ---------------------------------------------------------------------------
# 4) smoke test (GPU: YES, 1 card)
# ---------------------------------------------------------------------------
if ! GPU="$(gb_pick_gpus 1 40 30)"; then
  gb_say "SMOKE: skipped, no free GPU right now. Re-run this script later; steps 1 to 3 are quick."
  gb_say "STATUS: PARTIAL"; gb_report_print; exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for pair in "McGill-NLP/AfriqueQwen-8B|AfriqueQwen-8B" \
            "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1|AfriqueQwen3.5-9B-Instruct"; do
  path="$GB_ROOT/models/${pair%%|*}"; label="${pair##*|}"
  echo ">>> smoke test $label on GPU $GPU"
  RUN_LOG="$GB_ROOT/logs/T003b_${label}_${GB_TS}.log"
  if python "$GB_ROOT/runs/T003_smoke.py" "$path" "$label" > "$RUN_LOG" 2>&1; then
    grep -E '^(SMOKE|SAMPLE)' "$RUN_LOG" | tee -a "$GB_REPORT" >&2
    grep -E 'KV cache|Maximum concurrency|Available KV' "$RUN_LOG" | tail -n 2 | sed 's/^/  vllm: /' | tee -a "$GB_REPORT" >&2
  else
    gb_say "SMOKE FAIL $label (full log: $RUN_LOG); last error:"
    grep -E 'Error|error' "$RUN_LOG" | tail -n 3 | cut -c1-220 | tee -a "$GB_REPORT" >&2
  fi
done

if grep -q "SMOKE FAIL" "$GB_REPORT"; then gb_say "STATUS: PARTIAL"; else gb_say "STATUS: PASS"; fi
gb_report_print
