import ctypes
import gc
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import xarray as xr, rioxarray
import rasterio
from rasterio.errors import NotGeoreferencedWarning

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def rss_gb() -> float:
    """ Resident set size of this process in GB, for memory tracing. """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def release_memory() -> None:
    """ Collect Python garbage and hand freed arenas back to the OS.

        The reproject/warp step allocates large short-lived native arrays; glibc
        keeps those arenas after free(), so resident memory climbs across a
        per-year extraction loop even though little is live. malloc_trim returns
        them, keeping peak RSS flat.
    """
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def print_layer_stats(name: str, da: xr.DataArray) -> None:
    """ Print min/max/mean/std/finite for a layer without copying it.

        `da.where(np.isfinite(da))` materializes a float64 copy of the whole
        array (~8x an int8 layer); at 4-D cause-grid or daily-MODIS scale that
        copy alone exceeds the memory guard. skipna reductions ignore NaN in
        place, and integer layers carry no NaN at all.
    """
    try:
        total = int(da.size)
        if np.issubdtype(da.dtype, np.integer):
            finite = total
            f_min, f_max = float(da.min()), float(da.max())
            f_mean, f_std = float(da.mean()), float(da.std())
        else:
            finite = int(np.isfinite(da).sum())
            f_min, f_max = float(da.min(skipna=True)), float(da.max(skipna=True))
            f_mean, f_std = float(da.mean(skipna=True)), float(da.std(skipna=True))
        frac = finite / float(total) if total else 0.0
        print(
            f"  {name:25s} min={f_min:10.4f} max={f_max:10.4f} "
            f"mean={f_mean:10.4f} std={f_std:10.4f} "
            f"finite={finite:,}/{total:,} ({frac:6.2%})"
        )
    except Exception as e:
        print(f"  {name} (stats print failed: {e})")


def K_to_F(k):
    return (k * (9/5)) - 459.67


def F_to_K(f):
    return (f + 459.67) * 5/9


def read_hdf_file(subdataset: str, var = None):
    data = rioxarray.open_rasterio(subdataset, variable=var, masked=True)

    if "band" in data.dims and data.sizes.get("band", 1) == 1:  # type: ignore
        data = data.squeeze("band")  # type: ignore
    data.name = var # type: ignore
    return data


def load_as_xdataset(
    file: Path,
    variables: List[str] = [],
) -> xr.Dataset:
    suffix = file.suffix.lower()
    try:
        if suffix == ".hdf":
            if not variables:
                raise ValueError("[LOAD_AS_XDATASET] For .hdf need variables list")

            v_dict: Dict[str, xr.DataArray] = {}

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(file) as src:
                    sds_list = src.subdatasets

            if not sds_list:
                raise ValueError(f"[LOAD_AS_XDATASET] No subdatasets for {file.name}. run 'conda install libgdal-hdf4' and try again?")

            for var in variables:
                try:
                    sub = next(sd for sd in sds_list if sd.endswith(f":{var}"))
                except StopIteration:
                    raise ValueError(f"No subdataset ending with ':{var}' in {file.name}. Available:\n{sds_list}")

                da = rioxarray.open_rasterio(sub, masked=True)
                if "band" in da.dims and da.sizes.get("band", 1) == 1: # type: ignore
                    da = da.squeeze("band") # type: ignore
                da.name = var # type: ignore
                v_dict[var] = da # type: ignore

            return xr.Dataset(v_dict)
        
        print("[LOAD_AS_XDATASET] Dont know this file type")
        return xr.Dataset()

    except Exception as e:
        print(f"[LOAD_AS_XDATASET] Failed to load file '{file.stem}':", e)
        return xr.Dataset()



def load_as_xarr(
    file: Path,
    name: str,
    variable = None, # 
    grid = None,
    no_data_val = None,
) -> xr.DataArray:
    """
    load file with a specific variable. To ensure DataArray return type, must provide variable
    """
    suffix = file.suffix.lower()

    # Landfire, GPWv4, NLCD
    if suffix in {".tif", ".tiff"}:
        try:
            darr = rioxarray.open_rasterio(file, masked=True)

            if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
                darr = darr.squeeze("band") # type: ignore
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load tif '{file.stem}': ", e)
            return xr.DataArray()

    # gridMET
    elif suffix == ".nc":
        try:
            ds = xr.open_dataset(file, engine="netcdf4")

            if variable is not None and variable in ds.data_vars:
                darr = ds[variable]
            elif len(ds.data_vars) == 1:
                darr = ds[list(ds.data_vars)[0]]
            else:
                raise ValueError(
                    f"[LOAD_AS_XARR] '{file.stem}' needs a variable. Options are: {list(ds.data_vars.keys())}"
                )
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load nc '{file.stem}': ", e)
            return xr.DataArray()

    # LAADS
    elif suffix == ".hdf":
        try:
            if grid is None or variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable and grid for .hdf files")

            with rasterio.open(file) as src:
                sds = src.subdatasets
                if not sds: raise ValueError("No hdf5 subdatasets")

            darr = None
            for sd in sds:
                if sd.endswith(f":{variable}"):
                    darr = read_hdf_file(sd, variable)
                    break
            if darr is None:
                raise ValueError(f"No subdataset ending with ':{variable}' in {file.name}")

        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load hdf '{file.stem}': ", e)
            return xr.DataArray()

    # None yet?
    elif suffix in {".h5", ".hdf5"}:
        try:
            ds = xr.open_dataset(file, 
                engine="h5netcdf", 
                decode_coords="all", 
                decode_times=True
            )

            if variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable for .h5/.hdf files")
            if variable == "all":
                return ds[:, :] # return
            if variable not in ds:
                raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
            
            darr = ds[variable]
            del ds
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load hdf5 '{file.stem}': ", e)
            return xr.DataArray()
    else:
        raise ValueError(f"[LOAD_AS_XARR] Unsupported file type '{suffix}' for {file.name}")

    if no_data_val is not None:
        darr = darr.where(darr != no_data_val, other=np.nan) # type: ignore

    darr.name = name # type: ignore
    return darr # type: ignore

    
