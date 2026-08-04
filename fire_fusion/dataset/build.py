#!/usr/bin/env python3
"""
Three-stage datacube builder. extract streams processor features into a
staging zarr. publish derives them and applies deterministic normalization,
giving a split-agnostic dataset.zarr. compile redoes train-dependent
derivations, fits statistics on train, fills, and writes the splits.
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

from .grid import create_coordinate_grid, season_time_index, supervised_mask
from .build_utils import release_memory
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
from .processors.proc_prism import Prism
from .processors.proc_aorc import Aorc
from .processors.proc_landfire import Landfire
from .processors.proc_lightning import Lightning
from .processors.proc_modis import Modis
from .processors.proc_nlcd import NLCD
from .processors.proc_usfs import UsfsFire
from .processors.proc_croads import CensusRoads
from .processors.proc_usda import UsdaWui

# Upper bound on dask threads while writing the split stores. Every in-flight
# chunk carries all channels, so peak memory scales with the worker count rather
# than the store size; wa2000 measures ~10.5 GB at 4.
SPLIT_WRITE_WORKERS = 4

# zstd trades a bit of throughput for ~25% smaller stores than lz4
SPLIT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

# Labels and masks are int8/uint8 and overwhelmingly zero, so they compress to
# almost nothing and gain no locality from the spatial split that X needs. Full
# spatial extent per chunk keeps the file count down.
LABEL_TIME_CHUNK = 64


def _rss_gb() -> float:
    # Resident set size of this process, for extraction memory tracing
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0

PROC_CLASSES = {
    "CENSUSROADS": CensusRoads,
    "USDA_WUI": UsdaWui,
    "FIRE_USFS": UsfsFire,
    "GPW": GPW,
    "PRISM": Prism,
    "AORC": Aorc,
    "LANDFIRE": Landfire,
    "LIGHTNING": Lightning,
    "MODIS": Modis,
    "NLCD": NLCD,
}


class FeatureGrid:
    """ Builds one named dataset (see config/dataset_config.py):
        raw sources -> cube.zarr -> dataset.zarr -> {train,eval,test}.zarr + manifest.json
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

        self.time_index = season_time_index(
            ds_cfg.start_date, ds_cfg.end_date,
            ds_cfg.season_months, ds_cfg.halo_lead_days, ds_cfg.halo_trail_days,
        )
        if ds_cfg.season_months is not None:
            n_sup = int(supervised_mask(self.time_index, ds_cfg.season_months).sum())
            print(
                f"season: months {ds_cfg.season_months[0]}-{ds_cfg.season_months[1]}, "
                f"{len(self.time_index)} days extracted, {n_sup} supervised"
            )
        self.grid = create_coordinate_grid(
            self.time_index,
            ds_cfg.resolution,
            ds_cfg.lat_bounds, ds_cfg.lon_bounds
        )
        self._staging_initialized = False

    def build(self) -> None:
        self.extract()
        self.publish()
        self.compile()


    def extract(self) -> None:
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

                # An empty Dataset means the processor already streamed its parts through the sink
                if len(layer.data_vars) > 0:
                    self._write_layer(layer)

                del layer
                release_memory()
                print(f"[mem] after {pname}/{config.name}: RSS={_rss_gb():.2f} GB", flush=True)

            del processor
            release_memory()

        print(f"[FeatureGrid] staging cube written to {self.cfg.staging_path}")

    def _write_layer(self, layer: xr.Dataset) -> None:
        """ Append a layer's variables to the staging cube and release them.
            Every variable must already sit on the master grid/time index.
        """
        if isinstance(layer, xr.DataArray):
            layer = layer.to_dataset(name=layer.name)

        layer = layer.drop_vars("spatial_ref", errors="ignore")

        # Several processors return float64 only because xarray's .interp()
        # promotes; X is assembled as float32, so halving here cuts the memory
        # block size down without loss of precision.
        for name, da in layer.items():
            if da.dtype == np.float64:
                layer[name] = da.astype("float32")

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
        # -- misaligned coords must fail loudly instead of silently expanding the axes
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
            # stream the reductions over time chunks; a full-array mean/std on a
            # large cube materializes a float64 deviation copy (~8x) that OOMs
            if da.chunks is None and "time" in da.dims:
                da = da.chunk({"time": self.cfg.stage_time_chunk})
            total = da.size
            is_int = np.issubdtype(da.dtype, np.integer)
            if is_int:
                # integers carry no NaN/inf, so every cell is finite
                finite = total
                f_min = float(da.min())
                f_max = float(da.max())
                f_mean = float(da.mean())
                f_std = float(da.std())
            else:
                finite = int(np.isfinite(da).sum())
                f_min = float(da.min(skipna=True))
                f_max = float(da.max(skipna=True))
                f_mean = float(da.mean(skipna=True))
                f_std = float(da.std(skipna=True))
            frac_finite = finite / float(total) if total > 0 else 0.0
            print(
                f"  + {name:25s} "
                f"min={f_min:10.4f} max={f_max:10.4f} "
                f"mean={f_mean:10.4f} std={f_std:10.4f} "
                f"finite={finite:,}/{total:,} ({frac_finite:6.2%})"
            )
        except Exception as e:
            print(f"  + {name} (stats print failed: {e})")


    def publish(self) -> None:
        ds = xr.open_zarr(self.cfg.staging_path)
        ds = self._apply_derived(ds, train_yrs=None)
        # Halo days have served their purpose once the temporal derivations have
        # run. Dropping them here rather than at write time keeps normalization
        # statistics and the class balance describing exactly the days that ship.
        ds = self._drop_halo(ds)
        ds, det_stats = self._apply_deterministic(ds)
        self._save_published(ds, det_stats)

    def _save_published(self, ds: xr.Dataset, det_stats: Dict) -> None:
        print("Publishing the split-agnostic cube...")
        excluded = set(self.label_names) | set(self.mask_names)
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        channels = sorted(str(n) for n in ds.data_vars if n not in excluded)
        n_cause_classes = int(ds.sizes["burn_cause"])

        for lname in self.label_names:
            ds[lname] = ds[lname].astype("int8")
        for mname in self.mask_names:
            ds[mname] = ds[mname].astype("uint8")

        ds = ds.chunk({
            d: (self.cfg.stage_time_chunk if d == "time" else -1)
            for d in ds.dims
        })
        ds.attrs.clear()
        for v in ds.variables.values():
            v.attrs.clear()
            v.encoding.clear()

        encoding = {str(n): {"compressor": SPLIT_COMPRESSOR} for n in ds.data_vars}
        write_workers = min(SPLIT_WRITE_WORKERS, os.cpu_count() or SPLIT_WRITE_WORKERS)

        if self.cfg.published_path.exists():
            shutil.rmtree(self.cfg.published_path)
        print(f"[FeatureGrid] writing published cube -> {self.cfg.published_path}")
        with dask.config.set(scheduler="threads", num_workers=write_workers):
            ds.to_zarr(self.cfg.published_path, mode="w", encoding=encoding)

        manifest = {
            "dataset": self.cfg.name,
            "resolution_m": self.cfg.resolution,
            "lat_bounds": list(self.cfg.lat_bounds),
            "lon_bounds": list(self.cfg.lon_bounds),
            "grid": {"height": ny, "width": nx},
            "time": {
                "start": self.cfg.start_date,
                "end": self.cfg.end_date,
                "season_months": (
                    list(self.cfg.season_months) if self.cfg.season_months else None
                ),
                "contiguous": self.cfg.season_months is None,
            },
            "channels": channels,
            "labels": self.label_names,
            "masks": self.mask_names,
            "n_cause_classes": n_cause_classes,
            "norm_stats": det_stats,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cfg.published_manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved published cube. channels: {len(channels)}")

    def compile(self) -> None:
        ds = xr.open_zarr(self.cfg.published_path)
        # read before drop_inputs consumes the one-hot cause grid
        n_cause_classes = int(ds.sizes["burn_cause"])

        ds, redone_stats = self._recompute_train_dependent(ds)
        ds = self._apply_drop_inputs(ds)
        # Normalize while missing cells are still NaN, so statistics only see
        # valid observations; the zero-fill afterwards lands on the post-norm mean
        ds, stat_stats = self._apply_statistical(ds)
        ds = self._fill_missing(ds)
        pos_weight = self._compute_pos_weight(ds)
        cause_counts = self._compute_cause_counts(ds, n_cause_classes)

        published = json.loads(self.cfg.published_manifest_path.read_text())
        det_stats = published["norm_stats"]
        excluded = set(self.label_names) | set(self.mask_names)
        norm_stats: Dict[str, List[Dict]] = {}
        for f in ds.data_vars:
            if f in excluded:
                continue
            f = str(f)
            det = redone_stats.get(f, det_stats.get(f, []))
            norm_stats[f] = det + stat_stats.get(f, [])

        self._save_splits(ds, norm_stats, pos_weight, n_cause_classes, cause_counts)

    # -- keeps only supervised (in-season) days. A staging cube built before seasonal
    # -- windowing holds every day of the record, a superset of any halo range, so
    # -- this selects out of either layout.
    def _drop_halo(self, ds: xr.Dataset) -> xr.Dataset:
        if self.cfg.season_months is None:
            return ds

        keep = supervised_mask(ds.indexes["time"], self.cfg.season_months)
        n_before = ds.sizes["time"]
        ds = ds.isel(time=np.flatnonzero(keep))
        print(
            f"[FeatureGrid] season window: {n_before} -> {ds.sizes['time']} days "
            f"({ds.sizes['time'] / n_before:.1%} kept)"
        )
        return ds

    def _apply_derived(self, ds: xr.Dataset, train_yrs: Optional[Tuple[int, int]]) -> xr.Dataset:
        print(f"[FeatureGrid] Deriving anti-arson techniques through feature derivation..")

        drv_processor = DerivedProcessor(train_yrs=train_yrs)
        for cfg in self.drv_config:
            func   = cfg.func
            inputs = cfg.inputs
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

        print(f"[FeatureGrid] Finished deriving features!")
        print(f"- dims: {ds.dims}")
        return ds

    # -- shared by the publish pass and by recomputing a single train-dependent
    # -- feature; sound only because no feature's ds_norms mixes a deterministic
    # -- step after a statistical one, checked below rather than assumed
    def _deterministic_chain(
        self, name: str, feature: xr.DataArray, f_config: Feature
    ) -> Tuple[xr.DataArray, List[Dict]]:
        norms = getattr(f_config, "ds_norms", None) or []
        stat_types = {"z_score", "minmax", "scale_max"}
        det_types = {"log1p", "to_sin"}
        det_ix = [i for i, n in enumerate(norms) if n in det_types]
        stat_ix = [i for i, n in enumerate(norms) if n in stat_types]
        if det_ix and stat_ix and max(det_ix) > min(stat_ix):
            raise ValueError(f"deterministic norm ordered after a statistical norm for '{name}'")

        steps: List[Dict] = []
        clip = getattr(f_config, "ds_clip", None)
        if clip is not None:
            feature = feature.clip(clip[0], clip[1])
            steps.append({"step": "clip", "min": float(clip[0]), "max": float(clip[1])})

        for ntype in norms:
            if ntype == "log1p":
                feature = xr.apply_ufunc(np.log1p, feature, dask="allowed")
                steps.append({"step": "log1p"})
            elif ntype == "to_sin":
                feature = xr.apply_ufunc(np.sin, feature, dask="allowed")
                steps.append({"step": "to_sin"})

        return feature, steps

    def _apply_deterministic(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
        all_configs = (
            [c for fl in base_feat_config().values() for c in fl] +
            [c for c in drv_feat_config()]
        )
        det_stats: Dict[str, List[Dict]] = {}

        for f in list(ds.data_vars):
            if f in self.mask_names or f in self.label_names:
                continue

            f_config = next((
                cfg for cfg in all_configs
                if (cfg.name == f or f in (cfg.expand_names or []))
            ), None)
            if f_config is None:
                print(f"can't find feature config for '{f}'")
                continue

            print(f"[FeatureGrid] deterministic norm {f}")
            feature, steps = self._deterministic_chain(f, ds[f], f_config)
            ds[f] = feature
            det_stats[f] = steps

        return ds, det_stats

    def _apply_statistical(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
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
        stat_stats: Dict[str, List[Dict]] = {}

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

            print(f"[FeatureGrid] statistical norm {f}")
            steps: List[Dict] = []
            norms = getattr(f_config, "ds_norms", None) or []
            for ntype in norms:
                if ntype == "z_score":
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
            stat_stats[f] = steps

        return ds, stat_stats

    # -- the published copy of a train-dependent feature was derived over the
    # -- whole record, which would let held-out years inform a training input
    def _recompute_train_dependent(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
        drv_processor = DerivedProcessor(train_yrs=self.cfg.train_yrs)
        redone_stats: Dict[str, List[Dict]] = {}

        for cfg in self.drv_config:
            if not cfg.train_dependent:
                continue
            drv_fn = getattr(drv_processor, cfg.func)
            out = drv_fn(ds[cfg.inputs], cfg.name)
            ds[out.name] = out

            feature, steps = self._deterministic_chain(cfg.name, ds[cfg.name], cfg)
            ds[cfg.name] = feature
            redone_stats[cfg.name] = steps

        return ds, redone_stats

    def _apply_drop_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        for cfg in self.drv_config:
            if cfg.drop_inputs is not None:
                ds = ds.drop_vars(cfg.drop_inputs, errors="ignore")
        # the burn_cause dimension coordinate outlives its dropped variable
        ds = ds.drop_vars("burn_cause", errors="ignore")
        return ds

    def _fill_missing(self, ds: xr.Dataset) -> xr.Dataset:
        excluded = set(self.label_names) | set(self.mask_names)
        for name in list(ds.data_vars):
            if name in excluded:
                continue
            if np.issubdtype(ds[name].dtype, np.floating):
                # -- catches +/-inf as well as NaN, so an overflow upstream cannot survive into X
                ds[name] = ds[name].where(np.isfinite(ds[name]), 0.0)
        return ds

    def _compute_pos_weight(self, ds: xr.Dataset) -> float:
        train_slice = slice(
            f"{self.cfg.train_yrs[0]}-01-01", f"{self.cfg.train_yrs[1]}-12-31"
        )
        ign = ds["ign_next"].sel(time=train_slice)
        no_act_fire_mask = ds["no_act_fire_mask"].sel(time=train_slice)
        land_mask = ds["land_mask"].sel(time=train_slice)

        # the population the ignition head is supervised on
        ign_valid = ign.where((land_mask == 1) & (no_act_fire_mask == 1))
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

    # -- counted on the same cells the cause head is supervised on, so a loss weight
    # -- derived from these matches the population it is applied to
    def _compute_cause_counts(self, ds: xr.Dataset, n_cause_classes: int) -> List[int]:
        train_slice = slice(
            f"{self.cfg.train_yrs[0]}-01-01", f"{self.cfg.train_yrs[1]}-12-31"
        )
        ign = ds["ign_next"].sel(time=train_slice)
        cause = ds["ign_next_cause"].sel(time=train_slice)
        no_act_fire_mask = ds["no_act_fire_mask"].sel(time=train_slice)
        land_mask = ds["land_mask"].sel(time=train_slice)

        supervised = (land_mask == 1) & (no_act_fire_mask == 1) & (ign == 1) & (cause != -1)
        counts = dask.compute(*[
            ((cause == c) & supervised).sum() for c in range(n_cause_classes)
        ])
        counts = [int(c) for c in counts]
        print(
            f"[FeatureGrid] Cause classes (train split): {counts}",
            f"- imbalance = {max(counts) / max(min(counts), 1):.1f}x"
        )
        return counts

    def _save_splits(
        self, ds: xr.Dataset, norm_stats: Dict, pos_weight: float,
        n_cause_classes: int, cause_counts: List[int]
    ) -> None:
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

        ny, nx = ds.sizes["y"], ds.sizes["x"]
        # `spatial_splits` rises with resolution to hold the per-chunk byte count
        # near wa2000's measured ~92 MB, which is what makes SPLIT_WRITE_WORKERS
        # a resolution-independent memory bound.
        x_chunks = {
            "time": self.cfg.x_time_chunk,
            "channel": -1,
            "y": int(np.ceil(ny / self.cfg.spatial_splits)),
            "x": int(np.ceil(nx / self.cfg.spatial_splits)),
        }
        label_chunks = {"time": LABEL_TIME_CHUNK, "y": -1, "x": -1}
        flat_names = list(self.label_names) + list(self.mask_names)
        split_days: Dict[str, int] = {}

        # Every in-flight chunk carries all channels, so peak memory scales with
        # the worker count rather than the store size. Dask's default of one
        # thread per core overruns a 16-core box well before the write finishes.
        write_workers = min(SPLIT_WRITE_WORKERS, os.cpu_count() or SPLIT_WRITE_WORKERS)

        for split in ("train", "eval", "test"):
            y0, y1 = self.cfg.split_years(split)
            sub = out.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
            sub["X"] = sub["X"].chunk(x_chunks)
            for n in flat_names:
                sub[n] = sub[n].chunk(label_chunks)

            # zarr saves attrs as JSON; stale read-encodings clash with new chunks
            sub.attrs.clear()
            for v in sub.variables.values():
                v.attrs.clear()
                v.encoding.clear()

            encoding = {
                str(n): {"compressor": SPLIT_COMPRESSOR} for n in sub.data_vars
            }

            path = self.cfg.split_path(split)
            if path.exists():
                shutil.rmtree(path)
            print(f"[FeatureGrid] writing {split}: {sub.sizes['time']} days -> {path}")
            with dask.config.set(scheduler="threads", num_workers=write_workers):
                sub.to_zarr(path, mode="w", encoding=encoding)
            split_days[split] = int(sub.sizes["time"])

        manifest = {
            "dataset": self.cfg.name,
            "resolution_m": self.cfg.resolution,
            "lat_bounds": list(self.cfg.lat_bounds),
            "lon_bounds": list(self.cfg.lon_bounds),
            "grid": {"height": ny, "width": nx},
            "time": {
                "start": self.cfg.start_date,
                "end": self.cfg.end_date,
                # Days are contiguous within a season block but jump at the
                # year boundary; a loader must not build a window across the gap
                "season_months": (
                    list(self.cfg.season_months) if self.cfg.season_months else None
                ),
                "contiguous": self.cfg.season_months is None,
            },
            "splits": {s: list(self.cfg.split_years(s)) for s in ("train", "eval", "test")},
            "split_days": split_days,
            "channels": feature_names,
            "in_channels": len(feature_names),
            "labels": self.label_names,
            "masks": self.mask_names,
            # the cause head's width: read from the built cube so the decoder and
            # the metrics cannot drift from the classes the labels actually carry
            "n_cause_classes": n_cause_classes,
            "ign_pos_weight": pos_weight,
            "cause_counts": cause_counts,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a named FireFusion dataset")
    parser.add_argument(
        "--dataset", default="wa2000",
        help=f"one of {sorted(DATASET_CONFIGS)} or 'all'",
    )
    parser.add_argument(
        "--stage", default="all", choices=["extract", "publish", "compile", "all"],
        help="pipeline stage to run",
    )
    args = parser.parse_args()

    names = sorted(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for name in names:
        grid = FeatureGrid(get_dataset_config(name))
        if args.stage == "all":
            grid.build()
        else:
            getattr(grid, args.stage)()
