#!/usr/bin/env bash
# =============================================================================
# T006_stage_evalsets.sh | GadaaBot 2.0 | Batch B02
# -----------------------------------------------------------------------------
# What:   downloads ONLY the Oromo slices of the evaluation sets into
#         ~/gadaabot/data/eval/<org>/<name>/ (never used for training):
#           facebook/belebele              (gaz_Latn reading comprehension)
#           Davlan/sib200                  (gaz_Latn topic classification)
#           openlanguagedata/flores_plus   (gaz_Latn; GATED, may fail without HF token)
#           masakhane/afrimmlu, afrixnli, afrimgsm  (IrokoBench, Oromo "orm")
#         AfriMTEB-Lite is pulled later through the mteb package (T020).
# Tool:   MobaXterm terminal          GPU: NONE          Time: 2 to 15 minutes
# Disk:   under 100 MiB expected
# Run:    bash ~/gadaabot/tasks/T006_stage_evalsets.sh
# =============================================================================
set -euo pipefail
source "$HOME/gadaabot/scripts/common.sh"
gb_init T006 gb-core
exec > >(tee -a "$GB_LOG") 2>&1

export GB_FETCH_SPEC='[
  ["facebook/belebele",            "gaz_Latn",                "" ],
  ["Davlan/sib200",                "gaz_Latn",                "" ],
  ["openlanguagedata/flores_plus", "gaz_Latn",                "" ],
  ["masakhane/afrimmlu",           "(^|/|_)orm([/_.-]|$)",    "" ],
  ["masakhane/afrixnli",           "(^|/|_)orm([/_.-]|$)",    "" ],
  ["masakhane/afrimgsm",           "(^|/|_)orm([/_.-]|$)",    "" ]
]'
export GB_FETCH_DEST="$GB_ROOT/data/eval"

python - <<'PY' 2>&1 | tee -a "$GB_REPORT"
"""Download matching files from HF dataset repos and summarize them."""
import gzip, json, os, re, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

spec = json.loads(os.environ["GB_FETCH_SPEC"])
dest = Path(os.environ["GB_FETCH_DEST"])
api = HfApi()
ok = 0

def count_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    if path.name.endswith((".jsonl.gz", ".json.gz", ".txt.gz")):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    if path.suffix in {".jsonl", ".txt", ".tsv", ".csv"}:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    return -1

for repo, pat, excl in spec:
    t0 = time.time()
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {repo}: cannot list files ({type(e).__name__}: {str(e)[:150]})")
        continue
    sel = [f for f in files if re.search(pat, f) and not (excl and re.search(excl, f))]
    if not sel:
        tops = sorted({f.split("/")[0] for f in files})[:15]
        lang_like = sorted({f for f in files if re.search(r"(?i)(orm|oromo|gaz|/om[_/.])", f)})[:10]
        print(f"NO MATCH {repo} (pattern {pat!r}); top-level: {tops}; oromo-like paths: {lang_like}")
        continue
    target = dest / repo
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target),
                          allow_patterns=sel, max_workers=8)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED {repo}: download ({type(e).__name__}: {str(e)[:150]})")
        continue
    paths = [target / f for f in sel]
    size = sum(p.stat().st_size for p in paths if p.exists())
    rows = [count_rows(p) for p in paths if p.exists()]
    n_rows = sum(r for r in rows if r >= 0)
    print(f"DONE {repo}: {len(sel)} files, {size/1e6:.1f} MB, rows {n_rows:,}, "
          f"{time.time()-t0:.0f}s; first file: {sel[0]}")
    ok += 1

print(f"STATUS: {'PASS' if ok == len(spec) else 'PARTIAL'} ({ok}/{len(spec)} sources)")
PY

gb_say "QUOTA: project at $(du -scb "$GB_ROOT" "$HOME"/miniforge3/envs/gb-* 2>/dev/null | tail -n 1 | awk '{printf "%.1f", $1/1024^3}') GiB after T006"
gb_report_print
