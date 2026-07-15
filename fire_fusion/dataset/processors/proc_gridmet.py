from collections import defaultdict
from typing import List
import numpy as np
import xarray as xr
import pandas as pd

from .processor import Processor
from ..build_utils import K_to_F, load_as_xarr
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import GRIDMET_DIR

class GridMet(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def group_by_year(self, key: str):
        yr_groups = defaultdict(list)
        for f in [f for f in GRIDMET_DIR.glob("*.nc") if key in f.stem]:
            year = f.stem.split("_")[-1]
            yr_groups[year].append(f)
        return yr_groups
    
    def _ensure_time_dim(self, arr: xr.DataArray, year:str):
        if "day" in arr.dims:
            n_days = arr.sizes["day"]

            # rename existing if
            if "day" in arr.coords and np.issubdtype(arr.coords["day"].dtype, np.datetime64):
                arr = arr.rename({"day": "time"})
            else:
                time_coords = pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
                arr = arr.assign_coords(day=time_coords).rename({"day": "time"})
            # print("[gridMET] new dimensions: ", arr.dims)
            return arr
        return arr
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        feature_by_yrs: List[xr.DataArray] = []

        if f_cfg.key in ["tmm", "rm"]: # read in pairs of two
            yr_groups = self.group_by_year(f_cfg.key)

            def _pick(files, prefix: str):
                fp = next((f for f in files if f.stem.startswith(prefix)), None)
                if fp is None:
                    raise FileNotFoundError(
                        f"[gridMET] Missing '{prefix}*' file among {[f.name for f in files]}"
                    )
                return fp

            for (year, files) in sorted(yr_groups.items()):
                if f_cfg.key == "tmm":
                    print(f"[gridMET] Creating temperature gradient with unorthodox methods for {year}..")
                    vmin = load_as_xarr(_pick(files, "tmmn"), name=f_cfg.name, variable='air_temperature')
                    vmax = load_as_xarr(_pick(files, "tmmx"), name=f_cfg.name, variable='air_temperature')
                elif f_cfg.key == "rm":
                    print(f"[gridMET] Retrieving humidity data for {year}; atmosphere refusing to disclose exact moisture..")
                    vmin = load_as_xarr(_pick(files, "rmin"), name=f_cfg.name, variable='relative_humidity')
                    vmax = load_as_xarr(_pick(files, "rmax"), name=f_cfg.name, variable='relative_humidity')

                arr_min = self._preclip_native_arr(vmin)
                arr_max = self._preclip_native_arr(vmax)
                arr_min = self._reproject_arr_to_mgrid(arr_min, f_cfg.resampling)
                arr_max = self._reproject_arr_to_mgrid(arr_max, f_cfg.resampling)
                print("post reproj bounds:", arr_min.rio.bounds())
                print("post reproj finite:", int(np.isfinite(arr_min).sum()))
                print("post reproj min/max:", arr_min.min().item(), arr_min.max().item())

                if f_cfg.key == "tmm":
                    arr = self._build_temp(arr_min, arr_max, f_cfg)

                elif f_cfg.key == "rm":
                    arr = self._build_rel_humidity(arr_min, arr_max, f_cfg)
                
                arr = self._ensure_time_dim(arr, year)
                feature_by_yrs.append(arr)


        elif f_cfg.key in ["th", "vs", "pr", "fm100"]:
            files = GRIDMET_DIR.glob(f"{f_cfg.key}*.nc")

            for i, fp in enumerate(sorted(files)):
                year = fp.stem.split("_")[-1]

                if f_cfg.key == "th":
                    print(f"[gridMET] {year} Just broke wind. {' AGAIN' if i > 2 else ''}")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable='wind_from_direction')
                    v = self._preclip_native_arr(raw)
                    vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling)
                    arr = self._build_wind_dir(vals, f_cfg)
                    print("post reproj bounds:", arr.rio.bounds())
                    print("post reproj finite:", int(np.isfinite(arr).sum()))
                    print("post reproj min/max:", arr.min().item(), arr.max().item())

                elif f_cfg.key == "vs":
                    print(f"[gridMET] Cranking {year} backyard wind tunnel to {i*36 + (i*8) % 3}mph")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="wind_speed")

                    v = self._preclip_native_arr(raw)
                    vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling)
                    arr = self._build_wind_spd(vals, f_cfg)
                    print("post reproj bounds:", arr.rio.bounds())
                    print("post reproj finite:", int(np.isfinite(arr).sum()))
                    print("post reproj min/max:", arr.min().item(), arr.max().item())

                elif f_cfg.key == "pr":
                    print(f"[gridMET] {year} Negotiating with the rainman")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="precipitation_amount")
                    v = self._preclip_native_arr(raw)
                    vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling)
                    arr = self._build_precip_mm(vals, f_cfg)
                    print("post reproj bounds:", arr.rio.bounds())
                    print("post reproj finite:", int(np.isfinite(arr).sum()))
                    print("post reproj min/max:", arr.min().item(), arr.max().item())
                
                elif f_cfg.key == "fm100":
                    print(f"[gridMET] {year} Collecting fuel moisture from the dead veggies")
                    raw = load_as_xarr(fp, name=f_cfg.name, variable="dead_fuel_moisture_100hr")
                    v = self._preclip_native_arr(raw)
                    vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling)
                    arr = self._build_dead_fuel_moisture_pct(vals, f_cfg)
                    print("post reproj bounds:", arr.rio.bounds())
                    print("post reproj finite:", int(np.isfinite(arr).sum()))
                    print("post reproj min/max:", arr.min().item(), arr.max().item())
                
                arr = self._ensure_time_dim(arr, year)
                feature_by_yrs.append(arr)
                
        
        feature: xr.Dataset = xr.concat(feature_by_yrs, dim="time").to_dataset(name=f_cfg.name)

        print(f"[GridMet] Finished building {f_cfg.name}.. dims -> {feature.dims}")

        feature = feature.sortby("time")
        feature = self._time_interpolate(feature, f_cfg.time_interp)
        feature = feature.transpose("time", "y", "x", ...)
        return feature
        

    def _build_temp(self, vmin: xr.DataArray, vmax: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: min/max near-surface temprature (K)
            Out: clip -> avg near-surface temperature (F)
        """
        vmin = xr.apply_ufunc(K_to_F, vmin).astype("float32")
        vmax = xr.apply_ufunc(K_to_F, vmax).astype("float32")

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            vmin = vmin.clip(low, high)
            vmax = vmax.clip(low, high)

        data = ((vmax + vmin) / 2)
        data.name = f_cfg.name
        return data

    def _build_rel_humidity(self, vmin: xr.DataArray, vmax: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: min relative humidity (%), max relative humidity (%)
            Out: clipped, averagerd
        """
        vmin = vmin.clip(0, 100)
        vmax = vmax.clip(0, 100)

        data = ((vmax + vmin) / 2).astype("float32")
        data.name = f_cfg.name
        return data

    def _build_wind_dir(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: wind (coming from) direction
            Out: clipped
        """
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val

    def _build_wind_spd(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: wind speeed (m/s)
            Out: wind speed (m/h) + clip
        """
        data = (val * 2.23693629) # scale factor

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            data = data.clip(low, high)
        
        data.name = f_cfg.name
        return data.astype("float32")
    
    def _build_precip_mm(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: precipitation (mm)
            Out: precipitation (mm)
        """
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val.astype("float32")
    
    def _build_dead_fuel_moisture_pct(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        """ In: 100-hr dead fuel moisture (%)
            Out: 100-hr dead fuel moisture (%), clipped
        """
        data = val.clip(0, 100).astype("float32")
        data.name = f_cfg.name
        return data
        



