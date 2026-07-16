#!/usr/bin/env python3
# Two-phase datacube builder.
#
# Phase 1 (extract): every processor feature is written straight into a
# staging zarr as it is produced, so peak memory is one feature layer rather
# than the full cube.
#
# Phase 2 (finalize): the staging cube is reopened lazily (dask), derived
# features / normalization / NaN-fill are applied out-of-core, and the
# train/eval/test splits are written with all feature channels stacked into a
# single (time, channel, y, x) float32 array "X" that the training loader can
# slice directly. A manifest.json records channel order, normalization
# statistics, grid shape, and the ignition class balance.
import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import dask
import numpy as np
import pandas as pd
import xarray as xr

from .grid import create_coordinate_grid
from fire_fusion.config.dataset_config import (
    DATASET_CONFIGS, DatasetConfig, get_dataset_config
)
from fire_fusion.config.feature_config import (
    Feature, base_feat_config, drv_feat_config,
    get_labels, get_masks
)

from .processors.processor import Processor
from .processors.proc_derived_feats import DerivedProcessor
from .processors.proc_gpw import GPW
from .processors.proc_gridmet import GridMet
from .processors.proc_landfire import Landfire
from .processors.proc_modis import Modis
from .processors.proc_nlcd import NLCD
from .processors.proc_usfs import UsfsFire
from .processors.proc_croads import CensusRoads
from .processors.proc_usda import UsdaWui

PROC_CLASSES = {
    "CENSUSROADS": CensusRoads,
    "USDA_WUI": UsdaWui,
    "FIRE_USFS": UsfsFire,
    "GPW": GPW,
    "GRIDMET": GridMet,
    "LANDFIRE": Landfire,
    "MODIS": Modis,
    "NLCD": NLCD,
}

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class FeatureGrid:
    """ Builds one named dataset (see config/dataset_config.py):
        raw sources -> staging cube.zarr -> {train,eval,test}.zarr + manifest.json
    """
    def __init__(self, ds_cfg: DatasetConfig):
        self.cfg = ds_cfg
        self.fconfig = base_feat_config()
        self.drv_config = drv_feat_config()
        self.label_names = [l.name for l in get_labels()]
        self.mask_names = [m.name for m in get_masks()]
        print(f"[FeatureGrid] dataset: {ds_cfg.name} @ {ds_cfg.resolution:.0f}m")
        print("labels: ", self.label_names)
        print("masks: ", self.mask_names)

        self.time_index = pd.date_range(ds_cfg.start_date, ds_cfg.end_date, freq="D")
        self.grid = create_coordinate_grid(
            self.time_index,
            ds_cfg.resolution,
            ds_cfg.lat_bounds, ds_cfg.lon_bounds
        )
        self._staging_initialized = False

    def build(self) -> None:
        self._extract_to_staging()
        self._finalize_splits()

    # --- Phase 1: extract ---------------------------------------------------------

    def _extract_to_staging(self) -> None:
        print("Warming up GPU using low-emission wildfire simulations...")
        self.cfg.root.mkdir(parents=True, exist_ok=True)
        if self.cfg.staging_path.exists():
            shutil.rmtree(self.cfg.staging_path)
        self._staging_initialized = False

        for pname, features in self.fconfig.items():
            processor: Processor = PROC_CLASSES[pname](features, self.grid)
            processor.sink = self._write_layer

            for config in features:
                try:
                    layer = processor.build_feature(config)
                except Exception as e:
                    print(f"Oh no! feature extraction failed for {config.name}: ", e)
                    raise

                if isinstance(layer, xr.DataArray):
                    layer = layer.to_dataset(name=layer.name or config.name)

                # An empty Dataset means the processor already streamed its
                # parts through the sink
                if len(layer.data_vars) > 0:
                    self._write_layer(layer)

                del layer
                gc.collect()

            del processor
            gc.collect()

        print(f"[FeatureGrid] staging cube written to {self.cfg.staging_path}")

    def _write_layer(self, layer: xr.Dataset) -> None:
        """ Append a layer's variables to the staging cube and release them.
            Every variable must already sit on the master grid/time index.
        """
        if isinstance(layer, xr.DataArray):
            layer = layer.to_dataset(name=layer.name)

        layer = layer.drop_vars("spatial_ref", errors="ignore")

        for name, da in layer.items():
            self._check_grid_alignment(str(name), da)
            self._print_layer_stats(str(name), da)

        # zarr stores attrs as JSON; rio transform/CRS objects don't serialize
        layer.attrs.clear()
        for v in layer.variables.values():
            v.attrs.clear()
            v.encoding.clear()

        layer = layer.chunk({
            d: (self.cfg.stage_time_chunk if d == "time" else -1)
            for d in layer.dims
        })
        layer.to_zarr(self.cfg.staging_path, mode=("a" if self._staging_initialized else "w"))
        self._staging_initialized = True

    def _check_grid_alignment(self, name: str, da: xr.DataArray) -> None:
        """ The staging cube replaces the old outer-merge; misaligned coords
            must fail loudly instead of silently expanding the axes
        """
        ny, nx = self.grid.sizes["y"], self.grid.sizes["x"]
        if "y" not in da.dims or "x" not in da.dims:
            raise ValueError(f"[FeatureGrid] '{name}' missing spatial dims: {da.dims}")
        if da.sizes["y"] != ny or da.sizes["x"] != nx:
            raise ValueError(
                f"[FeatureGrid] '{name}' shape ({da.sizes['y']}, {da.sizes['x']}) "
                f"does not match grid ({ny}, {nx})"
            )
        if not np.allclose(da["y"].values, self.grid["y"].values) or \
           not np.allclose(da["x"].values, self.grid["x"].values):
            raise ValueError(f"[FeatureGrid] '{name}' y/x coordinates diverge from the master grid")
        if "time" in da.dims and not da.indexes["time"].equals(self.time_index):
            raise ValueError(
                f"[FeatureGrid] '{name}' time axis ({da.sizes['time']} steps) "
                f"does not match the master index ({len(self.time_index)} steps)"
            )

    def _print_layer_stats(self, name: str, da: xr.DataArray) -> None:
        try:
            ff = da.where(np.isfinite(da))
            f_min = float(ff.min(skipna=True))
            f_max = float(ff.max(skipna=True))
            f_mean = float(ff.mean(skipna=True))
            f_std = float(ff.std(skipna=True))
            total = da.size
            finite = int(np.isfinite(da).sum())
            frac_finite = finite / float(total) if total > 0 else 0.0
            print(
                f"  + {name:25s} "
                f"min={f_min:10.4f} max={f_max:10.4f} "
                f"mean={f_mean:10.4f} std={f_std:10.4f} "
                f"finite={finite:,}/{total:,} ({frac_finite:6.2%})"
            )
        except Exception as e:
            print(f"  + {name} (stats print failed: {e})")

    # --- Phase 2: finalize ----------------------------------------------------------

    def _finalize_splits(self) -> None:
        ds = xr.open_zarr(self.cfg.staging_path)

        ds = self._apply_derived(ds)
        # Normalize while missing cells are still NaN, so statistics only see
        # valid observations; the zero-fill afterwards lands on the post-norm mean
        ds, norm_stats = self._apply_normalize(ds)
        ds = self._fill_missing(ds)
        pos_weight = self._compute_pos_weight(ds)
        self._save_splits(ds, norm_stats, pos_weight)

    def _apply_derived(self, ds: xr.Dataset) -> xr.Dataset:
        print(f"[FeatureGrid] Deriving anti-arson techniques through feature derivation..")

        drv_processor = DerivedProcessor()
        for cfg in self.drv_config:
            func        = cfg.func
            inputs      = cfg.inputs

            new_fname = cfg.expand_names if cfg.expand_names else cfg.name

            if func:
                drv_fn = getattr(drv_processor, func)

                if func == "build_doy_sin":
                    out = drv_fn(ds, new_fname, self.grid)
                else:
                    subds = ds[inputs]
                    out = drv_fn(subds, new_fname)

                if isinstance(out, xr.DataArray):
                    ds[out.name] = out
                elif isinstance(out, xr.Dataset):
                    ds = ds.merge(out)

            if cfg.drop_inputs is not None:
                try:
                    ds = ds.drop_vars(cfg.drop_inputs)
                except Exception as e:
                    print(f"Failed to drop inputs for {new_fname}. Continuing")
                    pass

        # the burn_cause dimension coordinate outlives its dropped variable
        ds = ds.drop_vars("burn_cause", errors="ignore")

        print(f"[FeatureGrid] Finished deriving features!")
        print(f"- dims: {ds.dims}")
        return ds

    def _apply_normalize(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        # Statistics come from finite train-split cells only, and are re-computed
        # after each step in the norm chain so stacked transforms compose correctly
        train_slice = slice(
            f"{self.cfg.train_yrs[0]}-01-01", f"{self.cfg.train_yrs[1]}-12-31"
        )

        def _train_stats(da: xr.DataArray):
            src = da.sel(time=train_slice) if "time" in da.dims else da
            ff = src.where(np.isfinite(src))
            mean, std, vmin, vmax = dask.compute(
                ff.mean(skipna=True), ff.std(skipna=True),
                ff.min(skipna=True), ff.max(skipna=True),
            )
            return float(mean), float(std), float(vmin), float(vmax)

        all_configs = (
            [c for fl in base_feat_config().values() for c in fl] +
            [c for c in drv_feat_config()]
        )
        norm_stats: Dict[str, List[Dict]] = {}

        for f in list(ds.data_vars):
            if f in self.mask_names or f in self.label_names:
                continue

            feature = ds[f]
            f_config = next((
                cfg for cfg in all_configs
                if (cfg.name == f or f in (cfg.expand_names or []))
            ), None)

            if f_config is None:
                print(f"can't find feature config for '{f}'")
                continue

            print(f"[FeatureGrid] normalizing {f}")
            steps: List[Dict] = []

            clip = getattr(f_config, "ds_clip", None)
            if clip is not None:
                feature = feature.clip(clip[0], clip[1])
                steps.append({"step": "clip", "min": float(clip[0]), "max": float(clip[1])})

            norms = getattr(f_config, "ds_norms", None)
            if norms is not None:
                for ntype in norms:
                    if ntype == "log1p":
                        feature = xr.apply_ufunc(np.log1p, feature, dask="allowed")
                        steps.append({"step": "log1p"})
                    elif ntype == "to_sin":
                        feature = xr.apply_ufunc(np.sin, feature, dask="allowed")
                        steps.append({"step": "to_sin"})
                    elif ntype == "z_score":
                        f_mean, f_std, _, _ = _train_stats(feature)
                        f_std = f_std if f_std > 0 else 1.0
                        feature = (feature - f_mean) / f_std
                        steps.append({"step": "z_score", "mean": f_mean, "std": f_std})
                    elif ntype == "minmax":
                        _, _, f_min, f_max = _train_stats(feature)
                        denom = abs(f_max - f_min)
                        denom = denom if denom > 0.0 else 1.0
                        feature = (feature - f_min) / denom
                        steps.append({"step": "minmax", "min": f_min, "max": f_max})
                    elif ntype == "scale_max":
                        _, _, _, f_max = _train_stats(feature)
                        f_max = f_max if f_max != 0 else 1.0
                        feature = feature / f_max
                        steps.append({"step": "scale_max", "max": f_max})

            ds[f] = feature
            norm_stats[f] = steps

        return ds, norm_stats

    def _fill_missing(self, ds: xr.Dataset) -> xr.Dataset:
        excluded = set(self.label_names) | set(self.mask_names)
        for name in list(ds.data_vars):
            if name in excluded:
                continue
            if np.issubdtype(ds[name].dtype, np.floating):
                ds[name] = ds[name].fillna(0.0)
        return ds

    def _compute_pos_weight(self, ds: xr.Dataset) -> float:
        train_slice = slice(
            f"{self.cfg.train_yrs[0]}-01-01", f"{self.cfg.train_yrs[1]}-12-31"
        )
        ign = ds["ign_next"].sel(time=train_slice)
        fire_mask = ds["act_fire_mask"].sel(time=train_slice)
        water_mask = ds["water_mask"].sel(time=train_slice)

        # land pixels (water_mask: 1 = water) that are not already burning
        ign_valid = ign.where((water_mask == 0) & (fire_mask == 1))
        n_ign_pos, n_ign_neg = dask.compute(
            (ign_valid == 1).sum(), (ign_valid == 0).sum()
        )
        n_ign_pos, n_ign_neg = int(n_ign_pos), int(n_ign_neg)

        ign_pos_weight = n_ign_neg / float(max(n_ign_pos, 1))
        print(
            f"[FeatureGrid] Class imbalance (train split):",
            f"- positive ignitions  = {n_ign_pos:,}",
            f"- negative ignitions  = {n_ign_neg:,}",
            f"- pos_weight = {ign_pos_weight:.2f}"
        )
        return ign_pos_weight

    def _save_splits(self, ds: xr.Dataset, norm_stats: Dict, pos_weight: float) -> None:
        print("Spraying neutrino stabilization goo in sub-basement level 7...")
        excluded = set(self.label_names) | set(self.mask_names)

        feature_names = sorted(
            str(n) for n in ds.data_vars
            if n not in excluded and ds[n].dims == ("time", "y", "x")
        )
        skipped = [
            str(n) for n in ds.data_vars
            if n not in excluded and str(n) not in feature_names
        ]
        if skipped:
            print(f"[Warning] excluded from X (unexpected dims): {skipped}")

        channel_ix = pd.Index(feature_names, name="channel")
        X = xr.concat(
            [ds[n].astype("float32") for n in feature_names], dim=channel_ix
        ).transpose("time", "channel", "y", "x")

        out = xr.Dataset({"X": X})
        for lname in self.label_names:
            out[lname] = ds[lname].astype("int8")
        for mname in self.mask_names:
            out[mname] = ds[mname].astype("uint8")

        ny, nx = self.grid.sizes["y"], self.grid.sizes["x"]
        chunks = {
            "time": self.cfg.x_time_chunk,
            "channel": -1,
            "y": int(np.ceil(ny / self.cfg.spatial_splits)),
            "x": int(np.ceil(nx / self.cfg.spatial_splits)),
        }

        for split in ("train", "eval", "test"):
            y0, y1 = self.cfg.split_years(split)
            sub = out.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
            sub = sub.chunk({d: c for d, c in chunks.items() if d in sub.dims})

            # zarr saves attrs as JSON; stale read-encodings clash with new chunks
            sub.attrs.clear()
            for v in sub.variables.values():
                v.attrs.clear()
                v.encoding.clear()

            path = self.cfg.split_path(split)
            if path.exists():
                shutil.rmtree(path)
            print(f"[FeatureGrid] writing {split}: {sub.sizes['time']} days -> {path}")
            sub.to_zarr(path, mode="w")

        manifest = {
            "dataset": self.cfg.name,
            "resolution_m": self.cfg.resolution,
            "lat_bounds": list(self.cfg.lat_bounds),
            "lon_bounds": list(self.cfg.lon_bounds),
            "grid": {"height": ny, "width": nx},
            "time": {"start": self.cfg.start_date, "end": self.cfg.end_date},
            "splits": {s: list(self.cfg.split_years(s)) for s in ("train", "eval", "test")},
            "channels": feature_names,
            "in_channels": len(feature_names),
            "labels": self.label_names,
            "masks": self.mask_names,
            "ign_pos_weight": pos_weight,
            "norm_stats": norm_stats,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cfg.manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"Saved splits to .zarrs <3")
        print("--- MANIFEST ---")
        print(f"- grid: {ny} x {nx}, channels: {len(feature_names)}")
        print(f"- pos_weight: {pos_weight:.2f}")
        for c in feature_names:
            print(f"  channel: {c}")

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a named FireFusion dataset")
    parser.add_argument(
        "--dataset", default="wa2000",
        help=f"one of {sorted(DATASET_CONFIGS)} or 'all'",
    )
    args = parser.parse_args()

    names = sorted(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for name in names:
        FeatureGrid(get_dataset_config(name)).build()
