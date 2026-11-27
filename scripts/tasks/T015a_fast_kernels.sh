#!/usr/bin/env bash
# =============================================================================
# T015a_fast_kernels.sh | GadaaBot 2.0 | Batch B08 | training speed fix
# -----------------------------------------------------------------------------
# Why:    T014 measured 1,110 tok/s for embedding training: Qwen3.5's linear-
#         attention layers (Gated DeltaNet) ran on the slow pure-PyTorch path
#         because flash-linear-attention (fla) and causal-conv1d are missing.
#         At that speed 20M training tokens = 5 h per model.
# What:   1) installs flash-linear-attention (Triton, no compile) into gb-core
#            without changing torch / transformers (constraints file)
#         2) tries causal-conv1d (CUDA build, up to 30 min; optional: if it
#            fails, PyTorch conv is used, a small cost)
#         3) checks that transformers' Qwen3.5 code now finds the fast kernels
#         4) reruns the T014 probe for K4096: bits-per-word must equal T014
#            (7.7776 / 17.1705 / 5.321 / 5.3458, same docs) = kernels correct;
#            training speed should rise several-fold
# GPU:    YES in step 4 only: 1 card, >= 40 GiB free, about 10 min.
#         Steps 1 to 3: NO GPU (step 2 compiles on CPU).
# Time:   15 to 45 min; detaches itself (nohup)
# Run:    bash ~/gadaabot/tasks/T015a_fast_kernels.sh
# Result: cat ~/gadaabot/reports/T015a.txt   (finished when it shows STATUS)
# =============================================================================
set -euo pipefail

if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T015a_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T015a.pid"; disown -a
  echo "T015a started in background, pid $(cat "$HOME/gadaabot/runs/T015a.pid")"
  echo "Watch:  tail -f $OUT"
  echo "Result: cat ~/gadaabot/reports/T015a.txt   (finished when it shows STATUS)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T015a gb-core
trap 'rc=$?; gb_say "ABORTED: line $LINENO exited with code $rc"; gb_report_print' ERR
export PYTHONPATH="$GB_ROOT/src"

# --- 1) fla (NO GPU) ---------------------------------------------------------------
CONS="$GB_ROOT/envs/gb-core.constraints.full.txt"
python -m pip freeze > "$GB_ROOT/envs/gb-core.lock.before_T015a.txt"
python -m pip list --format=freeze | grep -viE '^(pip|setuptools|wheel)==' > "$CONS"
python -m pip install -q -c "$CONS" einops 2>&1 | tail -n 2 || true
if ! python -c "import fla" 2>/dev/null; then
  python -m pip install -q -c "$CONS" flash-linear-attention 2>&1 | tail -n 3 \
    || python -m pip install -q --no-deps flash-linear-attention fla-core 2>&1 | tail -n 3 || true
fi
python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
from importlib.metadata import version
for p in ("torch", "transformers", "triton", "flash-linear-attention", "fla-core"):
    try:
        print(f"VERSION {p}=={version(p)}")
    except Exception:
        print(f"VERSION {p}: not installed")
PY

# --- 2) causal-conv1d (NO GPU, optional) -------------------------------------------------
if ! python -c "import causal_conv1d" 2>/dev/null; then
  gb_say "causal-conv1d: building (up to 30 min, optional)"
  export CUDA_HOME MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="8.9"
  timeout 1800 python -m pip install -q -c "$CONS" --no-build-isolation causal-conv1d > "$GB_ROOT/logs/T015a_cconv_${GB_TS}.log" 2>&1 \
    && gb_say "causal-conv1d: installed" || gb_say "causal-conv1d: FAILED (optional; PyTorch conv fallback stays). log: logs/T015a_cconv_${GB_TS}.log"
fi
python -c "import torchcodec" 2>/dev/null && { gb_say "WARNING: torchcodec appeared; removing"; python -m pip uninstall -y torchcodec; } || true

# --- 3) does transformers see the fast path? (NO GPU) -----------------------------------------
python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
import importlib, inspect
mod = None
for name in ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_next.modeling_qwen3_next"):
    try:
        mod = importlib.import_module(name); break
    except Exception as exc:
        print(f"IMPORT {name}: {type(exc).__name__}: {str(exc)[:160]}")
if mod:
    names = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update",
             "FusedRMSNormGated", "is_fast_path_available")
    found = {n: getattr(mod, n, None) is not None for n in names}
    print("FAST PATH in", mod.__name__, found)
    fast = getattr(mod, "is_fast_path_available", None)
    print("is_fast_path_available =", fast)
PY
python -m pip freeze > "$GB_ROOT/envs/gb-core.lock.txt"
diff <(grep -iE '^(torch|transformers|triton|tokenizers|accelerate|peft)==' "$GB_ROOT/envs/gb-core.lock.before_T015a.txt") \
     <(grep -iE '^(torch|transformers|triton|tokenizers|accelerate|peft)==' "$GB_ROOT/envs/gb-core.lock.txt") \
  && gb_say "CORE PINS unchanged (torch, transformers, triton, tokenizers, accelerate, peft)" \
  || gb_say "WARNING: a core pin changed (see diff above)"

# --- 4) probe again on K4096 (GPU: 1 card) ---------------------------------------------------------
GPU=""
for attempt in $(seq 1 13); do
  if GPU="$(GB_REPORT=/dev/null gb_pick_gpus 1 40 30 2>/dev/null)"; then break; fi
  GPU=""; echo "$(date +%H:%M) no free GPU; waiting 5 min"; sleep 300
done
[ -n "$GPU" ] || { gb_say "SKIP probe: no GPU free for 60 min; STATUS: PARTIAL"; gb_report_print; exit 0; }
gb_say "PROBE K4096 on GPU $GPU"
PL="$GB_ROOT/logs/T015a_probe_${GB_TS}.log"
CUDA_VISIBLE_DEVICES="$GPU" timeout 3600 python -m gadaabot.eval.probe_extended \
  --models "K4096=$GB_ROOT/models/local/AfriqueQwen3.5-9B-it-orm-K4096" \
  --base-tok "$GB_ROOT/models/local/AfriqueQwen3.5-9B-Instruct-textview" \
  --heldout "$GB_ROOT/data/interim/oromo_corpus.v3.heldout.jsonl.gz" \
  --flores-eng "$GB_ROOT/data/eval/openlanguagedata/flores_plus/devtest/eng_Latn.jsonl" \
  --out "$GB_ROOT/reports/tables/T015a_probe.json" > "$PL" 2>&1 || gb_say "PROBE exited non-zero (see $PL)"
{ grep -E '^(KERNELS|LOAD|BPW|TRAIN-PROBE)' "$PL" || true; } | tee -a "$GB_REPORT"
{ grep -E 'Error|error|fast path|fall back|fallback' "$PL" | grep -v '^\s*File' | sort | uniq -c | sort -rn | head -n 5 | cut -c1-240 || true; } | sed 's/^/  log: /' | tee -a "$GB_REPORT"
gb_say "REFERENCE T014 K4096: BPW Oromo 7.7776 -> 17.1705 | English 5.321 -> 5.3458 | 1,110 tok/s, 29.4 GiB"
if grep -q '^TRAIN-PROBE K4096' "$GB_REPORT"; then gb_say "STATUS: PASS"; else gb_say "STATUS: PARTIAL"; fi
gb_report_print
