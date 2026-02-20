# https://earthaccess.readthedocs.io/en/stable/
from multiprocessing import AuthenticationError
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

import earthaccess
from earthaccess import DataGranule

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import MODIS_DIR
from ..build_utils import load_as_xdataset
from .processor import Processor

class Modis(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

        self.auth = earthaccess.login(strategy="netrc", persist=True)
        if self.auth.username:
            print(f"Logged into EARTH DATA for {self.auth.username}")
        else:
            raise AuthenticationError("EARTH DATA credentials via root _netrc file not found")
        
        self.latlon_tup = (
            self.gridref.attrs['lon_min'], self.gridref.attrs['lat_min'],
            self.gridref.attrs['lon_max'], self.gridref.attrs['lat_max']
        )
        MODIS_DIR.mkdir(exist_ok=True, parents=True)

        # Granule HDF file names are formatted as "<poduct>/<year>/<day-of-year>/<granule-files>.hdf
        # ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
        self.tiles = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]
        self.version = "061"
        self.max_parallel_req = 4
        
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        feature_by_yr = xr.Dataset()
    
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            if f_cfg.key == "MOD13Q1":
                print("[LAADS] Purchasing satellite to collect more vegetation data...")
                requests = {
                    executor.submit(self._fetch_ndvi, f_cfg, yr): yr 
                    for yr in self.gridref.attrs['years']
                }
            elif f_cfg.key == "MCD15A2H":
                print(f"[LAADS] Checking how big the leaves are")
                requests = {
                    executor.submit(self._fetch_lai, f_cfg, yr): yr 
                    for yr in self.gridref.attrs['years']
                }
            elif f_cfg.key == "MCD64A1":
                print(f"[LAADS] Staring at the shiny objects")
                requests = {
                    executor.submit(self._fetch_burns, f_cfg, yr): yr 
                    for yr in list(range(2000, 2020 + 1))
                }
            
            for req in as_completed(requests):
                yr = requests[req]
                try:
                    yr_ds = req.result()
                    if yr_ds is None:
                        continue
                    feature_by_yr = xr.merge([ feature_by_yr, yr_ds ], join="outer")

                except Exception as e:
                    print(f"[LAADS] Satellite tried to load data for {yr}, epic fail! --> {e}\n\n")


        ### -- compute months since last burn -- ###
        if f_cfg.key == "MCD64A1" and len(feature_by_yr.data_vars) > 0:
            feature_by_yr = self._aggregate_burns(feature_by_yr, f_cfg)

        # -----------------------------------------------------------------------
        return self._time_interpolate(
            feature_by_yr.sortby("time"), 
            f_cfg.time_interp
        ).transpose("time", "y", "x", ...)
    

    def _parse_date(self, filename):
        """ 
        Parse filename to retrieve year and date
        - ex: MOD13Q1.A2000049.h09v05.006.2015136104623.hdf = MOD13Q1, year 2000, day 49, h09/v05, hash
        """
        year, doy = None, None
        for p in filename.split("."):
            if (p[0] == "A" and p[1:8].isdigit()):
                year = int(p[1:5])
                doy = int(p[5:8])
                break
        if year is None or doy is None:
            raise ValueError(f"[MODIS] Couldn't parse filename {filename}")
        return datetime(year, 1, 1) + timedelta(days=doy - 1)
    

    ### ------------------------------------------------------------
    ### MCD152AH LAI
    def _fetch_lai(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None

        year_data: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            # print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["Lai_500m", "FparLai_QC", "FparExtra_QC"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue
                
                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                lai  = arr["Lai_500m"].fillna(0).astype("float32")
                qc   = arr["FparLai_QC"].fillna(0).astype("uint8")
                qcx  = arr["FparExtra_QC"].fillna(0).astype("uint8")

                # Primary QC (FParLai_QC)
                modland_gq = (qc & 0b1) == 0
                clouds_npres = ((qc >> 3) & 0b11).isin([0, 3])
                confident       = ((qc >> 5) & 0b111) < 4

                # ExtraWC (FparExtra_QC)
                island   = (qcx & 0b11).isin([0, 1])
                not_snow  = ((qcx >> 2) & 0b1) == 0
                
                aerosol   = (qcx >> 3) & 0b1
                cirrus    = (qcx >> 4) & 0b1
                int_cloud = (qcx >> 5) & 0b1
                shadow    = (qcx >> 6) & 0b1
                atm_good  = (aerosol == 0) & (cirrus == 0)
                no_clouds = (int_cloud == 0) & (shadow == 0)

                lai: xr.DataArray = lai.where(
                    modland_gq & clouds_npres & confident & 
                    island & not_snow & no_clouds & atm_good
                )

                # Drop fill cal 255 and non-terrestrial 249
                lai = lai.where((lai >= 0) & (lai <= 100))

            lai = lai.expand_dims(time=[ts])
            year_data.append(lai)

        stacked = xr.concat(year_data, dim="time").sortby("time")
        stacked = stacked.groupby("time").max("time")
        stacked = stacked.to_dataset(name=f_cfg.name)

        for name, da in stacked.data_vars.items():
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

        return stacked
    

    ### ------------------------------------------------------------
    ### MOD13Q1 NDVI
    def _fetch_ndvi(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(short_name=f_cfg.key or "", year=year)
        if len(nc_files) == 0:
            return None
        
        year_data_ndvi: List[xr.DataArray] = []
        year_data_water: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            # print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["250m 16 days NDVI", "250m 16 days VI Quality"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                ndvi = arr["250m 16 days NDVI"].astype('float32')
                qa = arr["250m 16 days VI Quality"].fillna(0).astype("uint16")

                fill_val = ndvi.rio.nodata
                if fill_val is None:
                    fill_val = ndvi.attrs.get("_FillValue", -3000.0)
                ndvi = ndvi.where(ndvi != fill_val)

                # --- QA decoding ---
                quality      = (qa & 0b11) <= 1
                vi_useful    = ((qa >> 2) & 0b1111) < 13
                no_adj_cloud = ((qa >> 8) & 0b1) == 0
                no_mixed_cloud = ((qa >> 10) & 0b1) == 0
                valid_viq = quality & vi_useful & no_adj_cloud & no_mixed_cloud

                is_land = ((qa >> 11) & 0b111) == 1
                is_deep_water   = ((qa >> 11) & 0b111).isin([0, 2, 5, 6, 7])

                ndvi = ndvi.where(valid_viq & is_land) * 1.0e-4
                water_mask = xr.where(valid_viq & is_deep_water, 1, 0).astype("uint8")

            ndvi = ndvi.expand_dims(time=[ts])
            year_data_ndvi.append(ndvi)

            water_mask = water_mask.expand_dims(time=[ts])
            year_data_water.append(water_mask)

        # stack tiles for each day returned
        stacked_ndvi = xr.concat(year_data_ndvi, dim="time").sortby("time").groupby("time").max("time")
        stacked_wmask = xr.concat(year_data_water, dim="time").sortby("time").groupby("time").max("time")

        ys, ye = f"{year}-01-01", f"{year}-12-31"
        stacked_ndvi = stacked_ndvi.resample(time="1D").ffill().sel(time=slice(ys, ye))
        stacked_wmask = stacked_wmask.resample(time="1D").ffill().sel(time=slice(ys, ye))

        assert f_cfg.expand_names is not None, "expected f_cfg.expand_names"

        stacked = xr.Dataset(data_vars={
            f_cfg.expand_names[0]: stacked_ndvi,
            f_cfg.expand_names[1]: stacked_wmask
        })

        for name, da in stacked.data_vars.items():
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

        return stacked
    
    
    
    ### ------------------------------------------------------------
    ### MCD64A1 burns
    def _fetch_burns(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None
        
        monthly_burn_flags: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(str(fp.name)))
            month_start = pd.Timestamp(ts.year, ts.month, 1)
            # print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["Burn Date", "QA"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                burn = arr["Burn Date"].fillna(-1).astype("int16")
                qa   = arr["QA"].fillna(0).astype("uint8")

                # QA
                land_valid = ((qa & 0b11) == 0b11) # bits 1/2 = 1
                burn = burn.where(land_valid)
                burn = burn.where((burn >= 1) & (burn <= 366)) # -1/-2 = bad val
                
                burned_flag = burn.notnull().astype("uint8")

            burned_flag = burned_flag.expand_dims(time=[month_start])
            monthly_burn_flags.append(burned_flag)

        if len(monthly_burn_flags) == 0:
            return None
        
        stacked_burns = xr.concat(monthly_burn_flags, dim="time").sortby("time")
        stacked_burns = stacked_burns.groupby("time").max("time").astype("uint8")
        stacked_burns.name = f_cfg.name
        stacked_burns = stacked_burns.to_dataset(name=f_cfg.name)

        return stacked_burns
    

    def _aggregate_burns(self, burn_da: xr.Dataset, f_cfg: Feature) -> xr.Dataset:
        burn_flag = burn_da[f_cfg.name].sortby("time").fillna(0).astype("uint8")

        # Create a grid of months, from first to last month
        # Missing months == 0 "no burn"
        time_min = pd.to_datetime(burn_flag.time.min().values)
        time_max = pd.to_datetime(burn_flag.time.max().values)
        
        full_time = pd.date_range(
            start=pd.Timestamp(time_min.year, time_min.month, 1), 
            end  =pd.Timestamp(time_max.year, time_max.month, 1), 
            freq="MS"
        )

        # reindex burn flags over monthly index
        monthly_burn_flag = burn_flag.reindex(time=full_time, fill_value=0) # type: ignore

        # -1 : never burned yet, up to the month
        # 0  : burned this month
        # 1  : burned last month
        # 2+ : burned further in the past
        def _months_since_last_burn_1d(burn_1d: np.ndarray) -> np.ndarray:
            # last_burns = [0, 0, 0, ..., 1, 2, 3, 4, 5, ..., 0]
            out = np.empty_like(burn_1d, dtype=np.int16)
            last_seen = -1

            for i, val in enumerate(burn_1d):
                if val >= 1: # reset counter
                    last_seen = 0
                    out[i] = 0
                else:
                    if last_seen == -1: # NOT burned before, and not this timestep
                        out[i] = -1
                    else: # burned before, but not this timestep
                        last_seen += 1
                        out[i] = last_seen
            return out

        months_since = xr.apply_ufunc(
            _months_since_last_burn_1d,
            monthly_burn_flag,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True, # chatGPT help <-- runs per (x, y)
            dask="parallelized",
            output_dtypes=[np.int16],
        )
        # print(months_since.shape)

        # replace "never seen" values (-1) to "total months in the record"
        # (conservative estimate of months since last burn)
        total_months = months_since.sizes["time"]
        sentinel_value = np.int16(total_months)

        months_since = months_since.where(months_since >= 0, other=sentinel_value)
        
        # Clip to actual feature grid's selected years
        clip_start = pd.Timestamp(min(self.gridref.attrs['years']), 1, 1)
        clip_end   = pd.Timestamp(max(self.gridref.attrs['years']), 12, 31)
        months_since = months_since.sel(time=slice(clip_start, clip_end))

        # Convert monthly to daily with forward fill
        months_since = months_since.resample(time="1D").ffill()
        months_since.name = f_cfg.name
        months_since = months_since.to_dataset(name=f_cfg.name)

        for name, da in months_since.data_vars.items():
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

        return months_since

    # ----------------------------------------------------------------------
    # Fetching
    # ----------------------------------------------------------------------
    def _get_hdf_links(self, granules: List[DataGranule]) -> List[str]:
        """ Get unique HDF download link for a list of granules """
        links: list[str] = []
        seen = set()
        for g in granules:
            for u in g["umm"].get("RelatedUrls", []):
                t = u.get("Type","")
                url = str(u.get("URL",""))
                if (url not in seen and
                    t == "GET DATA" and 
                    url.startswith("https") and url.lower().endswith(".hdf")
                ):
                    seen.add(url)
                    links.append(url)
        return links

    def _granule_pattern(self, short_name, tile):
        return f"{short_name}.*.{tile}.*"

    def _fetch_year(self, short_name: str, year: int) -> List[Path]:
        """ fetch data for a given product and year, return Path to existing/fetched .nc file """

        if not short_name or not year:
            return []

        product_folder = MODIS_DIR / f"{short_name}.{year}"
        ex_files = list(product_folder.glob("*.hdf"))

        if ex_files and all([True for f in ex_files if f.name.startswith(f"{short_name}")]):
            # print(f"[LAADS] {short_name} Found my yearbook from {year}, DAMN I look good..")
            return ex_files
        
        comb_results = []
        for tile in self.tiles:
            wildcard = self._granule_pattern(short_name, tile)
            granules = earthaccess.search_data(
                short_name=short_name, 
                version=self.version,
                granule_name=wildcard, 
                temporal=(f"{str(year)}-01-01", f"{str(year)}-12-31"),
                day_night_flag="day",
                bounding_box=self.latlon_tup,
                downloadable=True, 
                count=-1,
            )
            if len(granules) > 0:
                comb_results.extend(granules)
        
        if not comb_results:
            print(f"[LAADS] {short_name}: 'We aint found %$#@ for {year}'")
            return list()
        
        unq_links = self._get_hdf_links(comb_results)
        print(f"[LAADS] downloading using {len(unq_links)} links (from {len(granules)} granules) for {year}")

        dl_file_paths = earthaccess.download(
            granules=unq_links, 
            local_path=product_folder, 
            show_progress=True
        )
        return dl_file_paths

    
