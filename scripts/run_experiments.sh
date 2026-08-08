#!/usr/bin/env bash
# Guarded sequential launcher for training runs.
#
#   scripts/run_experiments.sh [--no-export] EXPERIMENT [EXPERIMENT ...]
#
# Refuses to start outside tmux (ALLOW_NO_TMUX=1 overrides), then validates
# everything that would otherwise fail mid-run or at run end: CUDA presence,
# every experiment name, every experiment's dataset splits on disk, and B2
# credentials when exporting. Runs experiments in order with full console
# capture under logs/, aborting the sequence on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPORT_FLAG="--export-b2"
if [[ "${1:-}" == "--no-export" ]]; then EXPORT_FLAG=""; shift; fi
[[ $# -ge 1 ]] || { echo "usage: $0 [--no-export] EXPERIMENT [EXPERIMENT ...]" >&2; exit 2; }

# -- an ssh drop kills a bare foreground run hours in; tmux is the cheap insurance
if [[ -z "${TMUX:-}" && -z "${ALLOW_NO_TMUX:-}" ]]; then
    echo "not inside tmux; run under a tmux session or set ALLOW_NO_TMUX=1" >&2
    exit 2
fi

python - "$EXPORT_FLAG" "$@" <<'EOF'
import json, sys
from pathlib import Path

import torch

from fire_fusion.config.path_config import MODEL_DIR, PROCESSED_DATA_DIR

export, experiments = bool(sys.argv[1]), sys.argv[2:]
errors = []

# -- get_device_config falls back to CPU silently; on a rented GPU node that
#    burns money on a run nobody wants
if not torch.cuda.is_available():
    errors.append("CUDA unavailable: this would train on CPU")

params = json.load(open(MODEL_DIR / "params.json"))
for exp in experiments:
    if exp not in params:
        errors.append(f"unknown experiment '{exp}' (options: {sorted(params)})")
        continue
    ds = params[exp]["dataset"]
    missing = [m for m in ("train.zarr", "eval.zarr", "manifest.json")
               if not (PROCESSED_DATA_DIR / ds / m).exists()]
    if missing:
        errors.append(f"{exp}: dataset '{ds}' missing {missing} "
                      f"(pull with: python -m fire_fusion.dataset.fetch_cloud pull "
                      f"--kind processed --datasets {ds})")

# -- export failures otherwise surface only after the last epoch; validating the
#    credentials and bucket now costs one HEAD request
if export:
    try:
        from fire_fusion.dataset.fetch_cloud import B2Store
        B2Store()
    except Exception as e:
        errors.append(f"B2 export check failed ({type(e).__name__}): {e}")

for e in errors:
    print(f"[preflight] {e}", file=sys.stderr)
sys.exit(1 if errors else 0)
EOF

mkdir -p logs
for exp in "$@"; do
    log="logs/${exp}_$(date +%m%d-%H%M%S).log"
    echo "=== ${exp} >> ${log}"
    python -m fire_fusion.train --experiment "$exp" ${EXPORT_FLAG:+$EXPORT_FLAG} 2>&1 | tee "$log"
done
