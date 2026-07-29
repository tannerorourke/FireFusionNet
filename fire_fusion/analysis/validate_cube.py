"""
Integrity checks for a built cube, run per split. Report only PASS/FAIL table 
and never mutates the cube. Streams each split in time chunks (bounded memory) 
and asserts the invariants a downstream loader and loss assume hold:

  - X carries no non-finite values.
  - ign_next is binary {0, 1}.
  - ign_next_cause stays in [-1, n_cause_classes-1]; -1 marks 'no cause'.
  - valid_cause_mask is a superset of (cause >= 0): every labelled cause is masked in.
  - land_mask fraction is reported (WA land is ~0.93 of the grid).

  python -m fire_fusion.analysis.validate_cube --dataset wa1000
"""
import argparse
import json

import numpy as np
import xarray as xr

from ..config.path_config import PROCESSED_DATA_DIR


def _manifest(dataset: str) -> dict:
    return json.loads((PROCESSED_DATA_DIR / dataset / "manifest.json").read_text())


def validate_split(dataset: str, split: str, n_cause: int, time_chunk: int):
    ds = xr.open_zarr(PROCESSED_DATA_DIR / dataset / f"{split}.zarr")
    T = ds["X"].sizes["time"]

    nonfinite = 0
    ign_vals = set()
    cause_min, cause_max = np.inf, -np.inf
    cause_mask_violations = 0
    land_sum, land_n = 0.0, 0

    for t0 in range(0, T, time_chunk):
        sl = slice(t0, min(t0 + time_chunk, T))
        X = ds["X"].isel(time=sl).values
        nonfinite += int((~np.isfinite(X)).sum())

        ign = ds["ign_next"].isel(time=sl).values
        ign_vals |= set(np.unique(ign).tolist())

        cause = ds["ign_next_cause"].isel(time=sl).values
        cause_min = min(cause_min, float(cause.min()))
        cause_max = max(cause_max, float(cause.max()))

        vcm = ds["valid_cause_mask"].isel(time=sl).values.astype(bool)
        cause_mask_violations += int(((cause >= 0) & (~vcm)).sum())

        land = ds["land_mask"].isel(time=sl).values
        land_sum += float(land.sum()); land_n += land.size

    land_frac = land_sum / land_n
    checks = {
        "X_all_finite": (nonfinite == 0, f"{nonfinite} non-finite"),
        "ign_binary": (ign_vals <= {0, 1}, f"values={sorted(ign_vals)}"),
        "cause_range": (cause_min >= -1 and cause_max <= n_cause - 1,
                        f"[{cause_min:.0f},{cause_max:.0f}] vs [-1,{n_cause-1}]"),
        "cause_mask_superset": (cause_mask_violations == 0,
                                f"{cause_mask_violations} labelled-but-unmasked"),
        "land_frac": (0.80 <= land_frac <= 0.99, f"{land_frac:.3f}"),
    }
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wa1000")
    ap.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    ap.add_argument("--time-chunk", type=int, default=64)
    args = ap.parse_args()

    m = _manifest(args.dataset)
    n_cause = int(m["n_cause_classes"])
    print(f"[validate] {args.dataset}  grid={m['grid']}  in_channels={m['in_channels']}  "
          f"n_cause={n_cause}")

    all_ok = True
    for split in args.splits:
        checks = validate_split(args.dataset, split, n_cause, args.time_chunk)
        print(f"\n  {split}:")
        for name, (ok, detail) in checks.items():
            all_ok &= ok
            print(f"    {'PASS' if ok else 'FAIL'}  {name:<22} {detail}")

    print(f"\n[validate] {args.dataset} >> {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")


if __name__ == "__main__":
    main()
