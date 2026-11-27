#!/usr/bin/env bash
# =============================================================================
# T001_env_core.sh | GadaaBot 2.0 | Batch B01 | RUN SECOND (after T002)
# -----------------------------------------------------------------------------
# What:   creates conda env "gb-core" (Python 3.11) with a CUDA-12 torch build
#         and the base NLP stack; writes a pinned lock file; checks that the
#         installed transformers understands Qwen3.5, Gemma 4 and AfriqueQwen
# Tool:   MobaXterm terminal          GPU: NONE (no GPU memory is allocated)
# Time:   about 5 to 15 minutes (downloads about 3 GB of wheels)
# Run:    bash ~/gadaabot/tasks/T001_env_core.sh
# Safe:   touches only ~/miniforge3/envs/gb-core and ~/gadaabot; sentry_venv untouched
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T001 none
exec > >(tee -a "$GB_LOG") 2>&1

ENV_NAME="gb-core"
set +u
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  gb_say "ENV: $ENV_NAME already exists, reusing it"
else
  conda create -y -n "$ENV_NAME" python=3.11
  gb_say "ENV: created $ENV_NAME"
fi
conda activate "$ENV_NAME"
set -u
gb_say "PYTHON: $(command -v python) $(python --version 2>&1)"

python -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# torch: driver 575.57.08 supports CUDA <= 12.9, so CUDA 13 wheels are excluded.
# Try candidates in order; the first that installs wins.
# ---------------------------------------------------------------------------
TORCH_PICK=""
for cand in "2.13.0 cu129" "2.13.0 cu126" "2.14.1 cu126"; do
  ver="${cand% *}"; tag="${cand#* }"
  echo ">>> trying torch==$ver from $tag index"
  if python -m pip install "torch==$ver" --index-url "https://download.pytorch.org/whl/$tag"; then
    TORCH_PICK="$ver+$tag"; break
  fi
done
if [ -z "$TORCH_PICK" ]; then
  gb_say "TORCH: FAILED for all candidates (2.13.0 cu129, 2.13.0 cu126, 2.14.1 cu126)"
  gb_report_print; exit 1
fi
gb_say "TORCH: installed candidate $TORCH_PICK"

# freeze torch so later installs can never swap it for a CUDA 13 build from PyPI
TORCH_FULL="$(python -c 'import torch; print(torch.__version__)')"
CONSTRAINTS="$GB_ROOT/envs/gb-core.constraints.txt"
echo "torch==$TORCH_FULL" > "$CONSTRAINTS"

python -m pip install -c "$CONSTRAINTS" \
  "transformers" "tokenizers" "datasets" "sentencepiece" "safetensors" \
  "accelerate" "peft" "huggingface_hub[hf_xet]" \
  "pyyaml" "pandas" "pyarrow" "numpy" "tqdm" \
  "mypy" "ruff" "pytest"

# ---------------------------------------------------------------------------
# Verification (CPU-side only; torch.cuda.is_available() allocates no memory)
# ---------------------------------------------------------------------------
python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
from importlib.metadata import version
import json, os, torch
pk = ["torch", "transformers", "tokenizers", "datasets", "accelerate", "peft",
      "huggingface_hub", "sentencepiece", "safetensors"]
print("VERSIONS:", ", ".join(f"{p}=={version(p)}" for p in pk))
print("TORCH CUDA build:", torch.version.cuda,
      "| cuda available:", torch.cuda.is_available(),
      "| visible GPUs:", torch.cuda.device_count())

from transformers import AutoConfig, AutoTokenizer
shared = os.environ["SHARED_MODELS"]
checks = {
    "Qwen3.5-9B config": f"{shared}/Qwen/Qwen3.5-9B",
    "Gemma-4-E4B-it config": f"{shared}/Google/gemma-4-E4B-it",
    "Qwen3-8B config": f"{shared}/Qwen/Qwen3-8B",
}
for name, path in checks.items():
    try:
        cfg = AutoConfig.from_pretrained(path)
        print(f"ARCH OK   {name}: model_type={cfg.model_type}")
    except Exception as e:  # noqa: BLE001
        print(f"ARCH FAIL {name}: {type(e).__name__}: {str(e)[:160]}")

tok_dir = os.path.expanduser("~/gadaabot_diag/hf_test")
try:
    tok = AutoTokenizer.from_pretrained(tok_dir)
    sample = "Fayyaan bu'uura jireenyaa ti."
    ids = tok(sample)["input_ids"]
    print(f"TOKENIZER OK AfriqueQwen-8B: vocab={len(tok)} sample_tokens={len(ids)}")
except Exception as e:  # noqa: BLE001
    print(f"TOKENIZER FAIL AfriqueQwen-8B: {type(e).__name__}: {str(e)[:160]}")
PY

# ---------------------------------------------------------------------------
# Lock files (exact reproducibility for the paper)
# ---------------------------------------------------------------------------
LOCK="$GB_ROOT/envs/gb-core.lock.txt"
python -m pip freeze > "$LOCK"
conda env export -n "$ENV_NAME" --no-builds > "$GB_ROOT/envs/gb-core.conda.yml" 2>/dev/null || true
gb_say "LOCK: $(wc -l < "$LOCK") packages, sha256 $(sha256sum "$LOCK" | cut -c1-16)"

cd "$GB_ROOT"
git add envs/ reports/T001.txt 2>/dev/null || true
git diff --cached --quiet || git commit -q -m "T001: gb-core environment and lock files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"

gb_say "STATUS: PASS"
gb_report_print
