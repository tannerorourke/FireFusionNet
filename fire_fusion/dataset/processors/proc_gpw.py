import xarray as xr, rioxarray
import numpy as np
import pandas as pd

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import GPW_DIR
from ..build_utils import load_as_xarr
from .processor import Processor

class GPW(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def build_feature(self, f_cfg: Feature) -> xr.DataArray:
        pop_5yr = xr.Dataset()

        print(f"[GPWv4] Sorting people by perceived radness")
        for i, fp in enumerate(sorted(GPW_DIR.glob("*.tif"))):
            fstem = fp.stem or None
            year = fstem.split("_")[-1] if fstem else str(2000 + (i * 5))
            print(f"[GPWv4] Counting them one by one...")
            
            with load_as_xarr(fp, name=f_cfg.name) as raw:
                p_grid = self._preclip_native_arr(raw)
                p_grid = self._reproject_arr_to_mgrid(self.gridref, f_cfg.resampling)

                p_grid = p_grid.where(p_grid >= 0)

                if "time" not in p_grid.dims:
                    ts = pd.Timestamp(f"{year}-07-01")
                    p_grid = p_grid.expand_dims(time=[ts])
                pop_5yr[f_cfg.name] = p_grid
        
        
        pop_ot = pop_5yr.sortby("time")
        pop_ot = self._time_interpolate(pop_ot, f_cfg.time_interp)
        pop_ot = pop_ot.transpose("time", "y", "x")

        # log1p + z score
        pop_ot = xr.apply_ufunc(np.log1p, pop_ot)
        pop_ot = (pop_ot - pop_ot.mean()) / (pop_ot.std() + 1e-6)
        return pop_ot
                

