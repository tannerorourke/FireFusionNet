from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import xarray as xr, rioxarray
import rasterio
from rasterio.errors import NotGeoreferencedWarning


def K_to_F(k):
    return (k * (9/5)) - 459.67


def F_to_K(f):
    return (f + 459.67) * 5/9


def read_hdf_file(subdataset: str, var = None):
    data = rioxarray.open_rasterio(subdataset, variable=var, masked=True)

    if "band" in data.dims and data.sizes.get("band", 1) == 1:  # type: ignore
        data = data.squeeze("band")  # type: ignore
    data.name = np.var # type: ignore
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

    # Landfire, GPWv4
    if suffix == ".tif":
        try:
            darr = rioxarray.open_rasterio(file, masked=True)

            if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
                darr = darr.squeeze("band") # type: ignore
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load tif '{file.stem}': ", e)
            return xr.DataArray()

    # NLCD
    if suffix == ".tiff":
        try:
            darr = rioxarray.open_rasterio(file, masked=True)   
            if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
                darr = darr.squeeze("band") # type: ignore
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load tiff '{file.stem}': ", e)
            return xr.DataArray()

        
    # gridMET
    elif suffix == ".nc":
        try:
            ds = xr.open_dataarray(file, 
                engine="netcdf4", 
                # decode_coords="all",
                # decode_times=True
            )
            
            print("raw encoding:", ds.encoding)
            if variable is None:
                raise ValueError(f"[LOAD_AS_XARR] expects variable. Options are: {list(ds.data_vars.keys())}")
            # if variable not in ds:
            #     raise KeyError(f"{variable} not in dataset. Available: {list(ds.data_vars.keys())}")
            darr = ds
            del ds
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

            for sd in sds:
                if sd.endswith(f":{variable}"):
                    darr = read_hdf_file(sd, variable)

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
        print("why am I here")

    if no_data_val is not None:
        darr = darr.where(darr != no_data_val, other=np.nan) # type: ignore

    darr.name = name # type: ignore
    return darr # type: ignore

    
