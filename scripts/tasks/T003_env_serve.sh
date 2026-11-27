#!/usr/bin/env bash
# =============================================================================
# T003_env_serve.sh | GadaaBot 2.0 | Batch B02
# -----------------------------------------------------------------------------
# What:   1) creates conda env "gb-serve" with vLLM 0.32.0 built for CUDA 12.9
#            (the default PyPI vLLM wheel is CUDA 13 and cannot run on driver 575)
#         2) GPU smoke test: loads AfriqueQwen-8B, then AfriqueQwen3.5-9B-Instruct,
#            one after the other, generates short Oromo continuations, measures
#            load time and tokens per second
# Tool:   MobaXterm terminal, inside tmux (takes 15 to 30 minutes)
# GPU:    YES. 1 card with >= 40 GiB free and <= 30% utilization, picked
#         automatically at the smoke-test step; refuses to run if none qualifies.
#         The env install part uses NO GPU.
# Disk:   about 8 to 10 GiB (env). Refuses if the project would pass 90 GiB.
# Run:    tmux new -s gb ; bash ~/gadaabot/tasks/T003_env_serve.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T003 none
exec > >(tee -a "$GB_LOG") 2>&1

ENV_NAME="gb-serve"
VLLM_VER="0.32.0"
TORCH_VER="2.13.0"
CU_TAG="cu129"

FOOT="$(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}')"
gb_say "QUOTA: project at ${FOOT} GiB before T003"
if awk -v f="$FOOT" 'BEGIN{exit !(f > 80)}'; then
  gb_say "QUOTA: FAILED, project above 80 GiB; env not created"; gb_report_print; exit 1
fi

# ---------------------------------------------------------------------------
# 1) environment (NO GPU)
# ---------------------------------------------------------------------------
set +u
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  gb_say "ENV: $ENV_NAME exists, reusing"
else
  conda create -y -n "$ENV_NAME" python=3.11
  gb_say "ENV: created $ENV_NAME"
fi
conda activate "$ENV_NAME"
set -u
python -m pip install --upgrade pip wheel

python -m pip install "torch==${TORCH_VER}" --index-url "https://download.pytorch.org/whl/${CU_TAG}"
echo "torch==${TORCH_VER}+${CU_TAG}" > "$GB_ROOT/envs/gb-serve.constraints.txt"

# find the cu129 x86_64 vLLM wheel among the GitHub release assets
WHEEL_URL="$(curl -sS --max-time 30 "https://api.github.com/repos/vllm-project/vllm/releases/tags/v${VLLM_VER}" \
  | python -c "import json,sys; d=json.load(sys.stdin); \
print(next((a['browser_download_url'] for a in d.get('assets', []) \
if '${CU_TAG}' in a['name'] and 'x86_64' in a['name'] and a['name'].endswith('.whl')), ''))")"
if [ -n "$WHEEL_URL" ]; then
  gb_say "VLLM: wheel from GitHub release: ${WHEEL_URL##*/}"
  python -m pip install -c "$GB_ROOT/envs/gb-serve.constraints.txt" "$WHEEL_URL" \
    --extra-index-url "https://download.pytorch.org/whl/${CU_TAG}"
else
  gb_say "VLLM: no ${CU_TAG} asset found on the release page; trying wheels.vllm.ai index"
  python -m pip install -c "$GB_ROOT/envs/gb-serve.constraints.txt" "vllm==${VLLM_VER}" \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_VER}/${CU_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${CU_TAG}"
fi

python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
from importlib.metadata import version
import torch
print("VERSIONS: vllm==%s torch==%s transformers==%s" % (version("vllm"), torch.__version__, version("transformers")))
print("TORCH CUDA build:", torch.version.cuda)
PY
python -m pip freeze > "$GB_ROOT/envs/gb-serve.lock.txt"
gb_say "LOCK: $(wc -l < "$GB_ROOT/envs/gb-serve.lock.txt") packages"

# ---------------------------------------------------------------------------
# 2) smoke test (GPU: YES, 1 card)
# ---------------------------------------------------------------------------
if ! GPU="$(gb_pick_gpus 1 40 30)"; then
  gb_say "SMOKE: skipped, no free GPU. Env is ready; re-run this script later (install steps are skipped)."
  gb_say "STATUS: PARTIAL"
  gb_report_print; exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SMOKE_PY="$GB_ROOT/runs/T003_smoke.py"
cat > "$SMOKE_PY" <<'PY'
"""T003 vLLM smoke test: load one model, generate, report speed."""
import sys
import time

from vllm import LLM, SamplingParams

model_path, label = sys.argv[1], sys.argv[2]
prompts = [
    "Busaan dhukkuba",
    "Fotosinteesiin adeemsa",
    "Sirni Gadaa",
    "Barnoonni saayinsii",
] * 8
t0 = time.time()
llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=0.85, trust_remote_code=False)
load_s = time.time() - t0
params = SamplingParams(temperature=0.0, max_tokens=128)
t1 = time.time()
outs = llm.generate(prompts, params)
gen_s = time.time() - t1
n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
sample = outs[0].outputs[0].text.replace("\n", " ")[:160]
print(f"SMOKE OK {label}: load {load_s:.0f}s, {len(prompts)} prompts, "
      f"{n_tok} tokens in {gen_s:.1f}s = {n_tok / gen_s:.0f} tok/s")
print(f"SAMPLE {label}: Busaan dhukkuba -> {sample}")
PY

for pair in "McGill-NLP/AfriqueQwen-8B|AfriqueQwen-8B" \
            "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1|AfriqueQwen3.5-9B-Instruct"; do
  path="$GB_ROOT/models/${pair%%|*}"; label="${pair##*|}"
  echo ">>> smoke test $label on GPU $GPU"
  if python "$SMOKE_PY" "$path" "$label" 2>&1 | tee -a "$GB_LOG" | grep -E '^(SMOKE|SAMPLE)' >> "$GB_REPORT"; then
    :
  else
    gb_say "SMOKE FAIL $label: see full log (last error lines below)"
    tail -n 5 "$GB_LOG" >> "$GB_REPORT"
  fi
done

if grep -q "SMOKE FAIL" "$GB_REPORT"; then gb_say "STATUS: PARTIAL"; else gb_say "STATUS: PASS"; fi
gb_say "QUOTA: project at $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB after T003"
gb_report_print
