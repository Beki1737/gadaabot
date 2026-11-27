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
