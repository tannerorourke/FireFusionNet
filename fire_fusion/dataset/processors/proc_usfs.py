from pathlib import Path
import xarray as xr, rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter

from .processor import Processor
from fire_fusion.config.feature_config import CAUSAL_CLASSES, CAUSE_RAW_MAP, Feature
from fire_fusion.config.path_config import USFS_DIR


class UsfsFire(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        self.grx_min, self.grx_max = self.gridref.attrs['x_min'], self.gridref.attrs['x_max']
        self.gry_min, self.gry_max = self.gridref.attrs['y_min'], self.gridref.attrs['y_max']
        self.mt_ix = self.gridref.attrs['time_index']
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        if f_cfg.key == "Fire_Perimeter":
            print(f"\n[USFS] computing fire perimeter")

            file = USFS_DIR / "National_USFS_Fire_Perimeter_(Feature_Layer).shp"
            layer = self._build_perim_layer(file, f_cfg)

            layer = layer.to_dataset(name=f_cfg.name).sortby("time").transpose("time", "y", "x", ...)
            return layer
        

        elif f_cfg.key == "Fire_Occurence":
            print(f"\n[USFS] computing fire occurence layer")
            self.occ_cause_layer = None

            file = USFS_DIR / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).shp"
            self.occ_cause_layer = layer = self._build_occ_cause_layers(file, f_cfg)

            layer = layer.sortby("time").transpose("time", "y", "x", ...)
            return layer


        elif f_cfg.key == "Fire_KDE":
            print(f"\n[USFS] computing fire KDE")

            # Apply a cum sum at each timestep T + gaussian filter to smooth Y/X
            layer = self._build_kde_layers(f_cfg)
            print({ 
                name: {
                    "shape": da.shape,
                    "finite": int(np.isfinite(da).sum()),
                    "min": float(da.where(np.isfinite(da)).min(skipna=True)),
                    "max": float(da.where(np.isfinite(da)).max(skipna=True)),
                    "mean": float(da.where(np.isfinite(da)).mean(skipna=True)),
                    "unique": len(np.unique(da.values))
                } 
                for name, da in layer.data_vars.items()
            })
            layer = layer.sortby("time").transpose("time", "y", "x", ...)
            return layer
            
        return xr.Dataset()
    

    def get_clipped(self, fp: Path):
        layer = gpd.read_file(fp).to_crs(self.mCRS)
        return gpd.clip(layer, 
            box(
                self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
                self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
            )
        )
    
    def normalize_occ_statcause(self, raw):
        if raw is None:
            return np.nan
        val = str(raw).strip().lower()
        if val == "":
            return np.nan

        # Values arrive as bare codes ("5"), bare text ("debris burning"), or
        # "code - text" combos; match whole tokens only, numeric code first
        candidates = [val]
        if "-" in val:
            code, _, text = val.partition("-")
            candidates += [code.strip(), text.strip()]

        for cand in candidates:
            for kls, keywords in CAUSE_RAW_MAP.items():
                if cand in keywords:
                    return kls
        return np.nan
    
    def _build_occ_cause_layers(self, fp: Path, f_cfg: Feature) -> xr.Dataset:
        fires_usfs = self.get_clipped(fp)

        discovery_dates = pd.to_datetime(
            fires_usfs["DISCOVERYD"], errors="coerce"
        ).dt.floor("D")

        # remove rows with missing discovery date
        missing = discovery_dates.isna()
        discovery_dates = discovery_dates.loc[~missing]
        fires_usfs = fires_usfs.loc[~missing].copy()

        # clip to date bounds and cols by bounds
        clip_date = (discovery_dates >= self.mt_ix[0]) & (discovery_dates <= self.mt_ix[-1])
        discovery_dates = discovery_dates.loc[clip_date]
        fires_usfs = fires_usfs.loc[clip_date].copy()

        # --- create new index with discovery dates, align to the grid index
        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)
        fires_usfs["t_idx"] = time2index.reindex(discovery_dates).to_numpy()
        
        # ADDT'L: drop rows where CAUSE is empty/unknown per normalization
        fires_usfs["burn_cause_class"] = fires_usfs["STATCAUSE"].apply(self.normalize_occ_statcause)
        
        fires_usfs = fires_usfs[
            fires_usfs["burn_cause_class"].notna() &
            fires_usfs["t_idx"].notna()
        ].copy()
        fires_usfs["t_idx"] = fires_usfs["t_idx"].astype("int32")

        # === Fire Occurences
        occ_grid = np.zeros((len(self.mt_ix), len(self.gridref.y), len(self.gridref.x)), dtype="uint8")
        for t_idx in np.unique(fires_usfs["t_idx"].to_numpy()):
            sub = fires_usfs[fires_usfs["t_idx"] == t_idx]
            shapes = [(geom, 1) for geom in sub.geometry]
            occ_grid[int(t_idx)] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0,
                dtype="uint8"
            )

        # === Fire Cause
        cause_labels = pd.Index(CAUSAL_CLASSES, name="burn_cause")
        cause_grid = np.zeros((
            len(self.mt_ix), len(cause_labels), 
            len(self.gridref.y), len(self.gridref.x)
        ), dtype="uint8")
        for (t_idx, cause), fires_group in fires_usfs.groupby(["t_idx", "burn_cause_class"]):
            if cause not in cause_labels:
                continue
            cause_idx = cause_labels.get_loc(cause)
            shapes = [(geom, 1) for geom in fires_group.geometry]
            cause_grid[int(t_idx), cause_idx] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)), 
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8",
            )

        assert f_cfg.expand_names is not None, "burn occ/cause expects expand names"

        fire_occ_cause_tyx = xr.Dataset({
            f_cfg.expand_names[0]: xr.DataArray(
                occ_grid,
                name=f_cfg.expand_names[0],
                coords={
                    "time":self.mt_ix,
                    "y":self.gridref.coords['y'].values,
                    "x":self.gridref.coords['x'].values 
                },
                dims=("time", "y", "x")
            ),
            f_cfg.expand_names[1]: xr.DataArray(
                cause_grid,
                name=f_cfg.expand_names[1],
                coords={ 
                    "time": self.mt_ix, 
                    "burn_cause": cause_labels, 
                    "y":self.gridref.coords['y'].values,
                    "x":self.gridref.coords['x'].values 
                },
                dims=("time", "burn_cause", "y", "x")
            )
        })

        fire_occ_cause_tyx = fire_occ_cause_tyx.rio.write_crs(self.gridref.rio.crs)
        fire_occ_cause_tyx = fire_occ_cause_tyx.rio.write_transform(self.gridref.rio.transform())
        return fire_occ_cause_tyx
    

    def _build_kde_layers(self, f_cfg: Feature) -> xr.Dataset:
    
        assert self.occ_cause_layer is not None, "Fire-KDE expected burn data/occurence layer pre-computed"

        fire_occurences = self.occ_cause_layer["usfs_burn_cause"]

        # Sigma = how wide the bell curve is IN 2d PIXELS = equals average of X/Y pixel
        # Radius = max radius of filter influence in meters (coordinates)
        px_size_xkm = float(abs(self.gridref.rio.transform().a) / 1000)
        px_size_ykm = float(abs(self.gridref.rio.transform().e) / 1000)
        pixel_size_km = (px_size_xkm + px_size_ykm) / 2.0

        kde_radius = f_cfg.kde_smooth_radius_km if f_cfg.kde_smooth_radius_km is not None else 10
        sigma_pixels = (
            kde_radius / pixel_size_km if pixel_size_km > 0 else 0.0
        )
        
        print(f"sigma pixels = {kde_radius} / {pixel_size_km} = {sigma_pixels}")
        
        # Loop over burn causes, computing gaussian filter for each class.
        # Each timestep's map only accumulates fires observed up to that day:
        # smoothing is linear, so smoothing daily occurrences and cumsum-ing over
        # time is identical to smoothing the running cumulative map at every day
        kde_by_class = {}
        for cause in fire_occurences.coords["burn_cause"].values:
            occ_txy = fire_occurences.sel(burn_cause=cause).values.astype("float32")

            if occ_txy.sum() == 0:
                print(f"WARNING: Sum of All X/Y across time is 0 for {cause}")
                kde_txy = occ_txy
            else:
                smoothed = np.zeros_like(occ_txy)
                fire_days = np.flatnonzero(occ_txy.reshape(occ_txy.shape[0], -1).sum(axis=1) > 0)
                for t in fire_days:
                    smoothed[t] = gaussian_filter(occ_txy[t], sigma=sigma_pixels, mode="constant")
                kde_txy = np.cumsum(smoothed, axis=0, dtype="float32")

            da_kde = xr.DataArray(
                kde_txy,
                coords={
                    "time": fire_occurences.coords["time"],
                    "y": fire_occurences.coords["y"],
                    "x": fire_occurences.coords["x"],
                },
                dims=("time", "y", "x"),
                name=f"kde_{str(cause).lower()}"
            )
            kde_by_class[f"kde_{str(cause).lower()}"] = da_kde

        kde_ds = xr.Dataset(kde_by_class)
        kde_ds = kde_ds.rio.write_crs(self.gridref.rio.crs)
        kde_ds = kde_ds.rio.write_transform(self.gridref.rio.transform())
        return kde_ds


    def _build_perim_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_usfs = self.get_clipped(fp)

        # -- Fire Start/End Time --
        start_date = pd.to_datetime(fires_usfs["DISCOVERYD"], errors="coerce")
        final_date = pd.to_datetime(fires_usfs["PERIMETERD"], errors="coerce")

        # -- drop rows with missing disco date --
        valid_dfull = start_date.notna() & final_date.notna()
        fires_usfs = fires_usfs.loc[valid_dfull].copy()
        start_date = start_date.loc[valid_dfull]
        final_date   = final_date.loc[valid_dfull]

        # -- convert to datetime, move days ending in 00:00:00 back one day
        start_dates = start_date.dt.floor("D")
        end_dates = final_date.dt.floor("D")
        
        is_EOD = ((final_date.dt.hour == 0) & 
                  (final_date.dt.minute == 0) & 
                  (final_date.dt.second == 0))
        end_dates = end_dates - pd.to_timedelta(is_EOD.astype("int64"), unit="D")

        # -- crop by start/end time index --
        clip_dates = (
            start_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
            & end_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
        )
        fires_usfs = fires_usfs.loc[clip_dates].copy()
        start_dates = start_dates.loc[clip_dates]
        end_dates   = end_dates.loc[clip_dates]

        if (end_dates < start_dates).any():
            end_dates[end_dates < start_dates] = start_dates[end_dates < start_dates]

        # -- align discovery dates to the grid index --
        start_dates = start_dates.clip(self.mt_ix[0], self.mt_ix[-1])
        end_dates   = end_dates.clip(self.mt_ix[0], self.mt_ix[-1])

        start_idx = self.mt_ix.searchsorted(start_dates.values, side="left")
        end_idx   = self.mt_ix.searchsorted(end_dates.values,   side="right") - 1
        valid_idx = (end_idx >= 0) & (start_idx < len(self.mt_ix))
        fires_usfs = fires_usfs.iloc[valid_idx].copy()

        fires_usfs["start_idx"] = start_idx[valid_idx]
        fires_usfs["end_idx"] = end_idx[valid_idx]

        # -- rasterize each day --
        time_grid = np.zeros((
            len(self.mt_ix), 
            len(self.gridref.y), 
            len(self.gridref.x)
        ), dtype="uint8")
        for t_idx in range(len(self.mt_ix)):
            active = (fires_usfs["start_idx"] <= t_idx) & (fires_usfs["end_idx"] >= t_idx)
            if not active.any():
                continue
            
            time_grid[t_idx] = rasterize(
                shapes=[(geom, 1) for geom in fires_usfs.loc[active].geometry],
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8"
            )

        perim_txy = xr.DataArray(
            time_grid,
            name=f_cfg.name,
            coords={
                "time": self.mt_ix,
                "y":    self.gridref.coords['y'].values,
                "x":    self.gridref.coords['x'].values 
            },
            dims=("time", "y", "x")
        )
        perim_txy = perim_txy.rio.write_crs(self.gridref.rio.crs)
        perim_txy = perim_txy.rio.write_transform(self.gridref.rio.transform())
        return perim_txy
