"""Restrict a metric to one tier's ground footprint on any other tier's grid.

Cross-resolution comparison is only meaningful over identical ground, and the
tiers do not share any. `cascades500` covers a corridor whose ignition rate is
roughly three times the state average, so a statewide model scored statewide and
a corridor model scored on the corridor are not answering the same question, and
the corridor will look better for a reason unrelated to resolution.

Every claim that compares resolutions should pass the reference tier's footprint
into scoring. Claims about a single tier on its own extent should not.

  python -m fire_fusion.analysis.footprint --reference cascades500
"""
import argparse
from typing import Tuple

import numpy as np
import xarray as xr
from pyproj import Transformer

from fire_fusion.config.dataset_config import DATASET_CONFIGS, get_dataset_config
from fire_fusion.dataset.grid import GRID_CRS


def reference_envelope(reference: str) -> Tuple[float, float, float, float]:
    """ Projected (x0, x1, y0, y1) envelope of a tier's configured bounds. """
    cfg = get_dataset_config(reference)
    tf = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    lat0, lat1 = min(cfg.lat_bounds), max(cfg.lat_bounds)
    lon0, lon1 = min(cfg.lon_bounds), max(cfg.lon_bounds)
    corners = [tf.transform(lo, la) for lo in (lon0, lon1) for la in (lat0, lat1)]
    xs, ys = zip(*corners)
    return min(xs), max(xs), min(ys), max(ys)


def footprint_mask(ds: xr.Dataset, reference: str) -> np.ndarray:
    # -- every tier shares a CRS, so this is a coordinate window, not a
    # -- reprojection. A cell is kept when its centre falls inside the envelope,
    # -- so the same footprint covers slightly different area at each resolution.
    x0, x1, y0, y1 = reference_envelope(reference)
    x = np.asarray(ds["x"].values)
    y = np.asarray(ds["y"].values)
    return ((y >= y0) & (y <= y1))[:, None] & ((x >= x0) & (x <= x1))[None, :]


def _land(ds: xr.Dataset) -> np.ndarray:
    land = ds["land_mask"].values.astype(bool)
    return land[0] if land.ndim == 3 else land


def summarize(tiers, reference: str, store: str) -> None:
    from fire_fusion.config.path_config import PROCESSED_DATA_DIR

    print(f"\nfootprint = {reference} bounds, measured on {store}\n")
    print(f"{'tier':14s} {'cells in':>9s} {'km2':>10s} {'ign cell-days':>14s} {'per 1000 km2':>13s}")
    print("-" * 66)
    for t in tiers:
        path = PROCESSED_DATA_DIR / t / store
        if not path.exists():
            print(f"{t:14s} {'not built':>9s}")
            continue
        ds = xr.open_zarr(path)
        m = footprint_mask(ds, reference) & _land(ds)
        cell_km2 = (get_dataset_config(t).resolution / 1000.0) ** 2
        area = m.sum() * cell_km2
        ign = float(ds["ign_next"].sum("time").values[m].sum())
        print(f"{t:14s} {m.sum():9,d} {area:10,.0f} {ign:14,.0f} {ign / area * 1000:13.1f}")


def main():
    ap = argparse.ArgumentParser(description="Common-footprint accounting across resolution tiers")
    ap.add_argument("--reference", default="cascades500", choices=sorted(DATASET_CONFIGS),
                    help="tier whose bounds define the shared footprint")
    ap.add_argument("--tiers", nargs="+", default=sorted(DATASET_CONFIGS))
    ap.add_argument("--store", default="dataset.zarr",
                    help="store to measure; use train.zarr to read a split instead")
    args = ap.parse_args()
    summarize(args.tiers, args.reference, args.store)


if __name__ == "__main__":
    main()
