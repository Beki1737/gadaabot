#!/usr/bin/env bash
# =============================================================================
# T002_skeleton.sh | GadaaBot 2.0 | Batch B01 | RUN THIS FIRST
# -----------------------------------------------------------------------------
# What:   creates ~/gadaabot (folders, env file, shared shell helpers, git repo)
#         and moves every T0*.sh you uploaded to ~ into ~/gadaabot/tasks/
# Tool:   MobaXterm terminal          GPU: NONE          Time: < 1 minute
# Run:    bash ~/T002_skeleton.sh
# Safe:   idempotent; never deletes anything; never overwrites gadaabot.env
# =============================================================================
set -euo pipefail

GB_ROOT="$HOME/gadaabot"

mkdir -p "$GB_ROOT"/{tasks,scripts,configs,envs,models,hf_cache,runs,logs,reports,paper,tests} \
         "$GB_ROOT"/data/{raw,interim,processed} \
         "$GB_ROOT"/src/gadaabot/{data,tokenization,training,retrieval,agent,eval}

for d in "$GB_ROOT/src/gadaabot" "$GB_ROOT"/src/gadaabot/*/; do
  [ -f "${d%/}/__init__.py" ] || : > "${d%/}/__init__.py"
done

# ---------------------------------------------------------------------------
# Environment file (sourced by every task). Written once; edit freely later.
# ---------------------------------------------------------------------------
if [ ! -f "$GB_ROOT/configs/gadaabot.env" ]; then
cat > "$GB_ROOT/configs/gadaabot.env" <<EOF
# GadaaBot 2.0 runtime environment, sourced by scripts/common.sh
GB_ROOT=$GB_ROOT
HF_HOME=$GB_ROOT/hf_cache
HF_HUB_DISABLE_TELEMETRY=1
HF_XET_HIGH_PERFORMANCE=1
TOKENIZERS_PARALLELISM=false
PYTHONNOUSERSITE=1
CUDA_HOME=/usr/local/cuda-12.8
SHARED_MODELS=$SHARED_MODELS
SHARED_DATASETS=$SHARED_DATASETS
EOF
fi

# ---------------------------------------------------------------------------
# Shared helpers for all task scripts (always refreshed; tracked in git)
# ---------------------------------------------------------------------------
cat > "$GB_ROOT/scripts/common.sh" <<'EOF'
#!/usr/bin/env bash
# GadaaBot 2.0 shared helpers. Usage inside a task script:
#   source "$HOME/gadaabot/scripts/common.sh"
#   gb_init T00X <conda-env-or-none>
GB_ROOT="$HOME/gadaabot"

gb_init() {
  GB_TASK="$1"; GB_ENV="${2:-none}"
  # leave sentry_venv (auto-activated at login) without touching it
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "^${VIRTUAL_ENV}/bin$" | paste -sd: -)"
    export PATH; unset VIRTUAL_ENV PYTHONHOME
  fi
  set +u
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  if [ "$GB_ENV" != "none" ]; then conda activate "$GB_ENV"; fi
  set -a
  # shellcheck disable=SC1091
  source "$GB_ROOT/configs/gadaabot.env"
  set +a
  set -u
  ulimit -n 65536 2>/dev/null || true
  GB_TS="$(date +%Y%m%d_%H%M%S)"
  GB_LOG="$GB_ROOT/logs/${GB_TASK}_${GB_TS}.log"
  GB_REPORT="$GB_ROOT/reports/${GB_TASK}.txt"
  mkdir -p "$GB_ROOT/logs" "$GB_ROOT/reports"
  : > "$GB_REPORT"
  export GB_TASK GB_ENV GB_TS GB_LOG GB_REPORT
}

# one line into the report (and onto the screen via stderr)
gb_say() { echo "$*" | tee -a "$GB_REPORT" >&2; }

gb_report_print() {
  {
    echo
    echo "### ${GB_TASK} REPORT START ###"
    cat "$GB_REPORT"
    echo "### ${GB_TASK} REPORT END ###"
    echo "Full log: $GB_LOG"
  } >&2
}

# gb_pick_gpus <count> <min_free_GiB> [max_util_percent=30]
# Prints a comma list of the emptiest qualifying GPUs on stdout, or returns 3.
gb_pick_gpus() {
  local need="$1" min_gib="$2" max_util="${3:-30}" picks got
  picks="$(nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader,nounits \
    | awk -F', *' -v m="$min_gib" -v u="$max_util" \
        '{free=($2-$3)/1024; if (free>=m && $4<=u) printf "%.1f %s\n", free, $1}' \
    | sort -rn | head -n "$need" | awk '{print $2}' | paste -sd, -)"
  got="$(echo "$picks" | awk -F, 'NF{print NF} !NF{print 0}')"
  if [ -z "$picks" ] || [ "$got" -lt "$need" ]; then
    gb_say "GPU CHECK: FAILED. Need $need GPU(s) with >= ${min_gib} GiB free and <= ${max_util}% util; qualifying: ${picks:-none}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
      | tee -a "$GB_REPORT" >&2
    return 3
  fi
  gb_say "GPU CHECK: OK, using GPU(s) $picks"
  echo "$picks"
}
EOF
chmod +x "$GB_ROOT/scripts/common.sh"

# ---------------------------------------------------------------------------
# Repo files
# ---------------------------------------------------------------------------
cat > "$GB_ROOT/.gitignore" <<'EOF'
models/
hf_cache/
data/
runs/
logs/
envs/*.tmp
__pycache__/
*.pyc
*.safetensors
*.bin
*.pt
.mypy_cache/
.ruff_cache/
.pytest_cache/
EOF

[ -f "$GB_ROOT/README.md" ] || cat > "$GB_ROOT/README.md" <<'EOF'
# GadaaBot 2.0

Native Afaan Oromo RAG and agent system. Task scripts live in tasks/, one per
tracker ID (T###). Every task writes logs/T###_<time>.log and reports/T###.txt.
EOF

# move uploaded task scripts from home into tasks/
moved=0
for f in "$HOME"/T0*.sh; do
  [ -e "$f" ] || continue
  mv -f "$f" "$GB_ROOT/tasks/"; moved=$((moved + 1))
done

# git repository
cd "$GB_ROOT"
[ -d .git ] || git init -q
git config user.name  >/dev/null 2>&1 || git config user.name  "user"
git config user.email >/dev/null 2>&1 || git config user.email "user@andlab-server"
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "T002: project skeleton, env file, shared helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GAECzsedVFXUF6DMLD9JSi"
fi

# ---------------------------------------------------------------------------
echo
echo "### T002 REPORT START ###"
echo "root: $GB_ROOT"
echo "task scripts moved into tasks/: $moved"
ls -1 "$GB_ROOT/tasks"
echo "--- folders ---"
find "$GB_ROOT" -maxdepth 2 -type d -not -path '*/.git*' | sed "s#$GB_ROOT#~/gadaabot#" | sort
echo "--- git ---"
git log --oneline -n 1
echo "--- free space on home volume ---"
df -h "$GB_ROOT" | tail -n 1
echo "### T002 REPORT END ###"
