#!/usr/bin/env bash
# =============================================================================
# T004_stage_models.sh | GadaaBot 2.0 | Batch B01 | RUN THIRD (after T001 PASS)
# -----------------------------------------------------------------------------
# What:   downloads the missing models into ~/gadaabot/models/<org>/<name>
#           McGill-NLP/AfriE5-Large-instruct                 (~2 GB)
#           McGill-NLP/AfriqueQwen-8B                        (~16 GB)
#           McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1 (~18 GB)
#         (v4: quota read from GB_PROJECT_QUOTA_GIB in gadaabot.env; extra repos via GB_EXTRA_REPOS)
#         Models already on nvme01 (Qwen3-8B, Qwen3.5-9B, Gemma 4 E4B, BGE-M3,
#         bge-reranker-v2-m3) are NOT downloaded again.
# Tool:   MobaXterm terminal          GPU: NONE
# Time:   long (hours at ~4 MB/s); it detaches itself with nohup, so you can
#         close MobaXterm. Re-running resumes; finished files are skipped.
# Run:    bash ~/gadaabot/tasks/T004_stage_models.sh
#         extra:  GB_EXTRA_REPOS="McGill-NLP/AfriqueQwen-14B" bash ~/gadaabot/tasks/T004_stage_models.sh
# Check:  tail -f ~/gadaabot/logs/T004_*.log      or   cat ~/gadaabot/reports/T004.txt
# =============================================================================
set -euo pipefail

# --- self-detach: relaunch under nohup unless already detached --------------
if [ "${GB_DETACHED:-0}" != "1" ]; then
  mkdir -p "$HOME/gadaabot/runs" "$HOME/gadaabot/logs"
  OUT="$HOME/gadaabot/logs/T004_nohup_$(date +%Y%m%d_%H%M%S).log"
  GB_DETACHED=1 GB_EXTRA_REPOS="${GB_EXTRA_REPOS:-}" nohup bash "$0" > "$OUT" 2>&1 &
  echo $! > "$HOME/gadaabot/runs/T004.pid"
  disown -a
  echo "T004 started in background, pid $(cat "$HOME/gadaabot/runs/T004.pid")"
  echo "Follow:  tail -f $OUT"
  echo "Result:  cat ~/gadaabot/reports/T004.txt   (finished when it shows a STATUS line)"
  exit 0
fi

source "$HOME/gadaabot/scripts/common.sh"
gb_init T004 gb-core

# --- project cap: 100 GiB per project (~/gadaabot + gb-* conda envs) --------
export GB_QUOTA_GIB="${GB_QUOTA_GIB:-${GB_PROJECT_QUOTA_GIB:-100}}"
export GB_HEADROOM_GIB="${GB_HEADROOM_GIB:-10}"
gb_footprint_gib() { du -scb "$HOME/gadaabot" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}'; }
gb_say "QUOTA: GadaaBot project uses $(gb_footprint_gib) GiB of ${GB_QUOTA_GIB} GiB (headroom kept: ${GB_HEADROOM_GIB} GiB)"

python - <<'PY'
import os, sys, time, json, subprocess
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ["GB_ROOT"]) / "models"
report = Path(os.environ["GB_REPORT"])
repos = [
    "McGill-NLP/AfriE5-Large-instruct",
    "McGill-NLP/AfriqueQwen-8B",
    "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1",
]
# v4: extra repos can be added at run time, e.g. GB_EXTRA_REPOS="McGill-NLP/AfriqueQwen-14B"
repos += [r for r in os.environ.get("GB_EXTRA_REPOS", "").split() if r and r not in repos]
# Deferred under the 100 GiB cap (add back only if storage allows):
#   McGill-NLP/AfriqueQwen3.5-9B-50Langs (base, ~18 GB)
#   McGill-NLP/AfriqueQwen-14B (~30 GB)
# weights + configs + tokenizer only; skip GGUF, ONNX, legacy .bin duplicates
allow = ["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "*.py", "*.tiktoken", "*.jinja"]

def say(line: str) -> None:
    print(line, flush=True)
    with report.open("a") as fh:
        fh.write(line + "\n")

api = HfApi()
summary = []
GIB = 1024 ** 3
budget = (int(os.environ["GB_QUOTA_GIB"]) - int(os.environ["GB_HEADROOM_GIB"])) * GIB

def home_used() -> int:
    """Bytes used by the GadaaBot project: ~/gadaabot plus gb-* conda envs."""
    import glob
    paths = [os.path.expanduser("~/gadaabot")] + glob.glob(os.path.expanduser("~/miniforge3/envs/gb-*"))
    out = subprocess.run(["du", "-scb", *paths], capture_output=True, text=True, check=False).stdout
    last = out.strip().splitlines()[-1] if out.strip() else "0"
    return int(last.split()[0])

for repo in repos:
    target = root / repo
    t0 = time.time()
    try:
        info = api.model_info(repo, files_metadata=True)
        expected = sum((s.size or 0) for s in info.siblings
                       if s.rfilename.endswith(".safetensors"))
        lic = getattr(info.card_data, "license", None) if info.card_data else None
        have = sum(p.stat().st_size for p in target.rglob("*.safetensors")) if target.exists() else 0
        need = max(expected - have, 0)
        used = home_used()
        if used + need > budget:
            say(f"SKIP {repo}: needs {need/GIB:.1f} GiB more, project at {used/GIB:.1f} GiB, "
                f"budget {budget/GIB:.0f} GiB")
            summary.append({"repo": repo, "ok": False, "error": "quota"})
            continue
        say(f"START {repo}: expected weights {expected/1e9:.1f} GB, license={lic}, sha={info.sha[:10]}")
        snapshot_download(repo_id=repo, local_dir=str(target),
                          allow_patterns=allow, max_workers=8)
        got = sum(p.stat().st_size for p in target.rglob("*.safetensors"))
        ok = expected > 0 and got >= expected
        say(f"{'DONE' if ok else 'CHECK'} {repo}: {got/1e9:.1f} GB weights in "
            f"{(time.time()-t0)/60:.1f} min -> {target}")
        summary.append({"repo": repo, "ok": ok, "gb": round(got/1e9, 2),
                        "license": lic, "sha": info.sha})
    except Exception as e:  # noqa: BLE001  (one failed repo must not stop the others)
        say(f"FAILED {repo}: {type(e).__name__}: {str(e)[:200]}")
        summary.append({"repo": repo, "ok": False, "error": type(e).__name__})

(root / "T004_manifest.json").write_text(json.dumps(summary, indent=2))
n_ok = sum(1 for s in summary if s["ok"])
say(f"STATUS: {'PASS' if n_ok == len(repos) else 'PARTIAL'} ({n_ok}/{len(repos)} repos complete)")
PY

gb_say "QUOTA: GadaaBot project uses $(gb_footprint_gib) GiB after download"
gb_report_print
