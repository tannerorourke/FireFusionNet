""" NOAA AORC v1.1 800m weather, reduced to daily.

Processor (import):
- Reads daily NetCDFs staged under AORC_DIR onto the master grid.
- Preclips, reprojects and clips to the extent of the master grid.

Fetch (running):
- AORC is served as one anonymous-access consolidated Zarr per year on S3.
- Lazily subsets the Washington extent and the handful of fields AORC uniquely
    supplies over PRISM (humidity + wind), derives the daily fire-weather fields at
    native resolution, and stages one NetCDF per year under AORC_DIR.
- Relative humidity recovered from specific humidity, temperature and pressure
    (Bolton 1980 saturation vapour pressure); wind is decomposed to a daily speed
    and a daily vector-mean wind-from direction.

  python -m fire_fusion.dataset.processors.proc_aorc --years 2000 2020
"""

import argparse
from typing import List

import numpy as np
import xarray as xr

from .processor import Processor
from ..build_utils import load_as_xarr, release_memory
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import AORC_DIR


class Aorc(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        valid_years = {int(y) for y in self.gridref.attrs["years"]}
        feature_by_yrs: List[xr.DataArray] = []

        for fp in sorted(AORC_DIR.glob("aorc_daily_*.nc")):
            year = fp.stem.split("_")[-1]
            if not year.isdigit() or int(year) not in valid_years:
                continue

            print(f"[AORC] {f_cfg.name} <- {fp.name}")
            raw = load_as_xarr(fp, name=f_cfg.name, variable=f_cfg.key).astype("float32")
            raw = raw.rio.write_crs("EPSG:4326")

            v = self._preclip_native_arr(raw)
            vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling).astype("float32")
            arr = self._transform(vals, f_cfg)
            feature_by_yrs.append(arr)

            del raw, v, vals
            release_memory()

        feature: xr.Dataset = xr.concat(feature_by_yrs, dim="time").to_dataset(name=f_cfg.name)
        feature_by_yrs.clear()
        release_memory()

        feature = feature.sortby("time")
        feature = self._time_interpolate(feature, f_cfg.time_interp)
        feature = feature.transpose("time", "y", "x", ...)
        return feature

    def _transform(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val.astype("float32")


# ----------------------------------------------------------------------
# Fetch (module entrypoint)
# ----------------------------------------------------------------------
BUCKET = "noaa-nws-aorc-v1-1-1km"
SRC_VARS = [
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "PRES_surface",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

# Washington extent (latitude is ascending in the store -> slice low to high).
LON_MIN, LON_MAX = -125.0, -116.5
LAT_MIN, LAT_MAX = 45.5, 49.5

EPS = 0.622            # R_d / R_v
MS_TO_MPH = 2.23693629


def _rh_from_q(q, T_k, p_pa):
    """ Relative humidity (%) from specific humidity (kg/kg), temperature (K)
        and surface pressure (Pa). Saturation vapour pressure over liquid water.
    """
    e = q * p_pa / (EPS + (1.0 - EPS) * q)
    Tc = T_k - 273.15
    es = 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
    return (100.0 * e / es).clip(0.0, 100.0)


def fetch_year(fs, year: int) -> None:
    import s3fs

    out = AORC_DIR / f"aorc_daily_{year}.nc"
    if out.exists():
        print(f"[AORC] {out.name} exists, skipping")
        return

    store = s3fs.S3Map(f"{BUCKET}/{year}.zarr", s3=fs)
    ds = xr.open_zarr(store, consolidated=True)[SRC_VARS]
    ds = ds.sel(latitude=slice(LAT_MIN, LAT_MAX), longitude=slice(LON_MIN, LON_MAX))

    rh = _rh_from_q(ds["SPFH_2maboveground"], ds["TMP_2maboveground"], ds["PRES_surface"])
    u, v = ds["UGRD_10maboveground"], ds["VGRD_10maboveground"]
    speed = np.sqrt(u * u + v * v)

    daily = xr.Dataset({
        "rel_humidity": rh.resample(time="1D").min(),
        "rh_max": rh.resample(time="1D").max(),
        "wind_mph": (speed.resample(time="1D").mean() * MS_TO_MPH),
    })
    # Vector-mean wind-from direction from the daily-mean components.
    u_bar = u.resample(time="1D").mean()
    v_bar = v.resample(time="1D").mean()
    daily["wind_dir"] = (np.degrees(np.arctan2(-u_bar, -v_bar)) % 360.0)

    daily = daily.rename({"latitude": "y", "longitude": "x"}).astype("float32")
    daily = daily.rio.write_crs("EPSG:4326")

    AORC_DIR.mkdir(parents=True, exist_ok=True)
    # zlib keeps the smooth daily fields ~3-4x smaller on disk and over the wire to B2
    daily.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in daily.data_vars})
    print(f"[AORC] wrote {out.name}  {dict(daily.sizes)}")


def main() -> None:
    import s3fs

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2020], metavar=("START", "END"))
    args = ap.parse_args()

    fs = s3fs.S3FileSystem(anon=True)
    for year in range(args.years[0], args.years[1] + 1):
        fetch_year(fs, year)


if __name__ == "__main__":
    main()
