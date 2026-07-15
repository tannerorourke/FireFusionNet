#!/usr/bin/env python3
import gc
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
# from numcodecs import Blosc
# from zarr.codecs import BloscCodec

from .data_loader import FireDataset
from .grid import create_coordinate_grid
from fire_fusion.config.path_config import TRAIN_DATA_DIR, EVAL_DATA_DIR, TEST_DATA_DIR
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

# xr.set_options(use_new_combine_kwarg_defaults=True)

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
    """ Builds the master feature dataset, resulting in a (T, H, W, C) tensor
        - Saves train, test, and eval data to .zarr files
        - Calls Processors to extract data
        - Concatenates features and projects their data onto a shared grid
    """
    def __init__(self,
        start_date="2009-01-01",
        end_date="2020-12-31",
        resolution: float = 4000,
        lat_bounds = (45.4, 49.1),
        lon_bounds = (-124.8, -117.0),
    ):
        self.fconfig = base_feat_config()
        self.drv_config = drv_feat_config()
        self.label_names = [l.name for l in get_labels()]
        self.mask_names = [m.name for m in get_masks()]
        print("labels: ", self.label_names)
        print("masks: ", self.mask_names)

        # Split boundaries; normalization statistics are computed on the train years only
        self.train_yrs = (2009, 2016)
        self.eval_yrs = (2017, 2018)
        self.test_yrs = (2019, 2020)

        self.time_index = pd.date_range(start_date, end_date, freq="D")
        self.grid = create_coordinate_grid(
            self.time_index,
            resolution,
            lat_bounds, lon_bounds
        )
        
        self.processors: Dict[str, Processor] = {
            pname: PROC_CLASSES[pname](features, self.grid)
            for pname, features in self.fconfig.items()
        }
        self.drv_processor = DerivedProcessor()
        self.build_features()

    # ------------------------------------------------------------------------------------
    
    def _apply_mask_nan(self) -> None:
        print(f"[FeatureGrid] Reversing polarity of the of the anti-polarity reverser...")
        print(f"[FeatureGrid] Baking some muffins...")
        
        excluded = set(self.label_names) | set(self.mask_names) | {"nan_mask"}

        # keeps cells where ALL channels are finite, everywhere else 0
        # convert back to dataset
        for name in list(self.master_ds.data_vars):
            if name in excluded:
                continue

            var = self.master_ds[name]
            valid = self.master_ds[name].notnull()

            if valid.sum() == 0:
                print(f"[Warning] Feature '{name}' has no finite values; zeroing it out.")
                self.master_ds[name] = xr.zeros_like(var)
                continue

            masked_var = var.where(valid, 0.0).fillna(0.0)
            self.master_ds[name] = masked_var
    

    def _apply_normalize(self) -> None:
        # Statistics come from finite train-split cells only, and are re-computed
        # after each step in the norm chain so stacked transforms compose correctly
        train_slice = slice(f"{self.train_yrs[0]}-01-01", f"{self.train_yrs[1]}-12-31")

        def _train_stats(da: xr.DataArray):
            src = da.sel(time=train_slice) if "time" in da.dims else da
            ff = src.where(np.isfinite(src))
            return (
                float(ff.mean(skipna=True)),
                float(ff.std(skipna=True)),
                float(ff.min(skipna=True)),
                float(ff.max(skipna=True)),
            )

        for f in self.master_ds.data_vars:
            if f in self.mask_names or f in self.label_names:
                continue

            # find config
            feature = self.master_ds[f]
            f_config = next((cfg for
                cfg in (
                    [c for fl in base_feat_config().values()
                        for c in fl
                            if (c.name == f or f in (c.expand_names or []))
                    ] + [
                        c for c in drv_feat_config()
                            if (c.name == f or f in (c.expand_names or []))
                    ]
                )
            ), None)

            if f_config is None:
                print(f"can't find feature config for '{f}'")
                continue

            print(f"[FeatureGrid] normalizing {f_config.name if f_config.name else f_config.expand_names}")

            clip = getattr(f_config, "ds_clip", None)
            if clip is not None:
                feature = feature.clip(clip[0], clip[1])

            norms = getattr(f_config, "ds_norms", None)
            if norms is not None:
                for ntype in norms:
                    if ntype == "log1p":
                        feature = xr.apply_ufunc(np.log1p, feature)
                    elif ntype == "to_sin":
                        feature = xr.apply_ufunc(np.sin, feature)
                    elif ntype == "z_score":
                        f_mean, f_std, _, _ = _train_stats(feature)
                        feature = (feature - f_mean) / (f_std if f_std > 0 else 1.0)
                    elif ntype == "minmax":
                        _, _, f_min, f_max = _train_stats(feature)
                        denom = abs(f_max - f_min)
                        feature = (feature - f_min) / (denom if denom > 0.0 else 1.0)
                    elif ntype == "scale_max":
                        _, _, _, f_max = _train_stats(feature)
                        feature = feature / (f_max if f_max != 0 else 1.0)

            self.master_ds[f] = feature


    def _apply_derived(self) -> None:
        print(f"[FeatureGrid] Deriving anti-arson techniques through feature derivation..")

        for cfg in self.drv_config:
            func        = cfg.func
            inputs      = cfg.inputs
            
            new_fname = cfg.expand_names if cfg.expand_names else cfg.name

            if func:
                drv_fn = getattr(self.drv_processor, func)

                if func == "build_doy_sin":
                    subds = self.master_ds
                    out = drv_fn(subds, new_fname, self.grid)
                else:
                    subds = self.master_ds[inputs]
                    out = drv_fn(subds, new_fname)

                if isinstance(out, xr.DataArray):
                    self.master_ds[out.name] = out
                elif isinstance(out, xr.Dataset):
                    self.master_ds = self.master_ds.merge(out)

            if cfg.drop_inputs is not None:
                try:
                    self.master_ds = self.master_ds.drop_vars(cfg.drop_inputs)
                except Exception as e:
                    print(f"Failed to drop inputs for {new_fname}. Continuing")
                    pass

        print(f"[FeatureGrid] Finished deriving features!")
        print(f"- dims: {self.master_ds.dims}")

    
    def _save_splits_to_zarr(self) -> None:
        print("Spraying neutrino stabilization goo in sub-basement level 7...")
        train_yrs, eval_yrs, test_yrs = self.train_yrs, self.eval_yrs, self.test_yrs
 
        t = self.master_ds.time
        years = t.dt.year.values
        times = t.values

        train_mask = (years >= train_yrs[0]) & (years <= train_yrs[1])
        eval_mask  = (years >= eval_yrs[0]) & (years <= eval_yrs[1])
        test_mask  = (years >= test_yrs[0]) & (years <= test_yrs[1])

        train_times = times[train_mask]
        eval_times  = times[eval_mask]
        test_times  = times[test_mask]

        def clear_attrs(ds):
            """ zarr saves as JSON, doesn't allow complex datatypes """
            ds.attrs.clear()
            for var in ds.values(): var.attrs.clear()

        def save_split(split_times, OUT_DATA_DIR, fname: str):
            split_ds = self.master_ds.sel(time=split_times)
            # ALLOWS BATCHING
            split_ds = split_ds.chunk({ "time": 64, "y": -1, "x": -1 }) # -1 = full dim length    
            clear_attrs(split_ds)
            split_ds.to_zarr(
                os.path.join(OUT_DATA_DIR, fname),
                mode="w"
            )
            return split_ds
        
        print("Detaching sub-basement level 7 from core modules")
        train_ds = save_split(train_times, TRAIN_DATA_DIR, "train.zarr")
        _ = save_split(eval_times,  EVAL_DATA_DIR,  "eval.zarr")
        _ = save_split(test_times,  TEST_DATA_DIR,  "test.zarr")
        print(f"Saved splits to .zarrs <3")
        print("--- TRAIN DATASET ---")
        for d in train_ds.dims:
            print(f"dim: {d}")
        for f in train_ds.data_vars:
            print(f"Channels: {f}, dims: {train_ds[f].dims}, shape:{np.array(train_ds[f].data).shape}")


    def _print_class_imbalance(self):
        ign  = self.master_ds["ign_next"]
        fire_mask = self.master_ds["act_fire_mask"]
        water_mask = self.master_ds["water_mask"]

        # land pixels (water_mask: 1 = water) that are not already burning
        ign_valid = ign.where((water_mask == 0) & (fire_mask == 1))
        n_ign_pos = (ign_valid == 1).sum().compute().item()
        n_ign_neg = (ign_valid == 0).sum().compute().item()

        ign_pos_weight = n_ign_neg / float(n_ign_pos)

        print(
            f"[FeatureGrid] Class imbalance:",
            f"- positive ignitions  = {n_ign_pos:,}",
            f"- negative ignitions  = {n_ign_neg:,}",
            f"- pos_weight = {ign_pos_weight:.2f}"
        )

        
    def build_features(self) -> None:
        print("Warming up GPU using low-emission wildfire simulations...")
        layers: List[xr.DataArray] = []

        for src, processor in self.processors.items():
            features: list[Feature] = processor.cfg
            for config in features:
                try:
                    layer = processor.build_feature(config)
                except Exception as e:
                    print(f"Oh no! feature extraction failed for {config.name}: ", e)
                    raise

                if isinstance(layer, xr.DataArray):
                    layer = layer.to_dataset(name=layer.name or config.name)

                for name, f in layer.items():
                    layers.append(f)
                    try:
                        ff = f.where(np.isfinite(f))
                        print(f"Adding {f.name}...")
                        f_min = float(ff.min(skipna=True))
                        f_max = float(ff.max(skipna=True))
                        f_mean = float(ff.mean(skipna=True))
                        f_std = float(ff.std(skipna=True))
                        total = f.size
                        finite = int(np.isfinite(f).sum())
                        frac_finite = finite / float(total) if total > 0 else 0.0
                        print(
                            f"  {f.name:25s} "
                            f"min={f_min:10.4f} max={f_max:10.4f} "
                            f"mean={f_mean:10.4f} std={f_std:10.4f} "
                            f"finite={finite:,}/{total:,} ({frac_finite:6.2%})"
                        )
                    except Exception as e:
                        print(f"(stats print failed for {name}: {e})")

        self._print_debug_stats("pre_merge")

        del self.processors
        try:
            self.master_ds: xr.Dataset = xr.merge(layers, join="outer")
        except Exception as e:
            print(f"Oh no! merging failed: ", e)
            raise
        del layers

        self._print_debug_stats("after_merge")
        self._apply_derived()
        self._print_debug_stats("after_derived_features")
        # Normalize while missing cells are still NaN, so statistics only see
        # valid observations; the zero-fill afterwards lands on the post-norm mean
        self._apply_normalize()
        self._print_debug_stats("after_normalization")
        self._apply_mask_nan()
        self._print_debug_stats("after_nan mask")
        self._print_class_imbalance()
        # self._save_splits_to_pt()
        self._save_splits_to_zarr()
        

    def _print_debug_stats(self, tag: str):
        if not hasattr(self, "master_ds"):
            print(f"[Debug:{tag}] master_ds not set yet")
            return

        excluded = set(self.label_names) | set(self.mask_names) | {"nan_mask"}

        print(f"\n[Debug:{tag}] Feature statistics")
        for name, da in self.master_ds.data_vars.items():
            if name in excluded:
                continue

            ff = da.where(np.isfinite(da))

            total = da.size
            finite = int(np.isfinite(da).sum())
            frac_finite = finite / float(total) if total > 0 else 0.0

            try:
                f_min = float(ff.min(dim=ff.dims, skipna=True))
                f_max = float(ff.max(dim=ff.dims, skipna=True))
                f_mean = float(ff.mean(dim=ff.dims, skipna=True))
                f_std = float(ff.std(dim=ff.dims, skipna=True))
            except Exception as e:
                print(f"  {name}: <error computing stats: {e}>")
                continue

            print(
                f"  {name:25s} "
                f"min={f_min:10.4f} max={f_max:10.4f} "
                f"mean={f_mean:10.4f} std={f_std:10.4f} "
                f"finite={finite:,}/{total:,} ({frac_finite:6.2%})"
            )
# -----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------- 

if __name__ == "__main__":
    feature_dataset = FeatureGrid(
        start_date="2009-01-01", end_date="2020-12-31",
        resolution = 2000,
        lat_bounds = (45.5, 49.0),
        lon_bounds = (-122.5, -117.0),
        # lat_bounds = (46., 48.5),
        # lon_bounds = (-122.0, -118.0),
    )