from pathlib import Path
from typing import Dict, List, Literal, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr

from ..config.path_config import EVAL_DATA_DIR, TEST_DATA_DIR, TRAIN_DATA_DIR
from fire_fusion.config.feature_config import drv_feat_config

from dask import config as daskconfig
# speed up the data loading if num_workers > 0
daskconfig.set(scheduler='synchronous')

class FireDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        window_size: int = 8,
        window_stride: int = 2
    ):
        """
        yield SINGLE spatiotemporal windows

        data: xarray Dataset with dims ("time", "y", "x") for all variables.
        device (torch).
        batch_size: number of temporal windows per batch (B).
        shuffle: whether to shuffle the order of windows each epoch.
        window_size: length of the temporal window T (# time steps).
        window_stride: stride between start of consecutive windows.
        """
        super().__init__()
        print(f"opening >> {dataset_path}")
        self.ds = xr.open_zarr(dataset_path)

        self.label_names= [l.name for l in drv_feat_config() if l.is_label==True]
        self.mask_names = [m.name for m in drv_feat_config() if m.is_mask==True]

        self.window_size = window_size
        self.window_stride = window_stride
        
        excluded = set(self.label_names) | set(self.mask_names) | {"spatial_ref"}
        self.feature_names = []
        for n in self.ds.data_vars:
            if n in excluded: 
                continue
            
            dims = self.ds[n].dims
            if dims == ("time", "y", "x"):
                self.feature_names.append(n)
            else:
                print(f"[FireDataset] dropping {n} from features due to dims={dims}")

        self.n_timesteps = self.ds.sizes["time"]
        self.window_starts = np.arange(
            0, max(self.ds.sizes["time"] - window_size + 1, 0),
            window_stride,
            dtype=int,
        )

    def __len__(self) -> int:
        # return math.ceil(self.n_windows / self.batch_size)
        return len(self.window_starts)
    
    def __getitem__(self, idx: int) -> int:
        return idx
    
    def collate_batch(self, start_idxs: Sequence[int]):
        """
            start_idxs: list of indices
        """
        sidxs = np.asarray(start_idxs)
        batch_size = sidxs.shape[0]

        t0s = torch.from_numpy(self.window_starts)[sidxs] # (B)
        seq = torch.arange(self.window_size, dtype=torch.long) # [0, win size]

        # broadcast offset positions to index
        lookup = t0s.unsqueeze(1) + seq.unsqueeze(0) # (B, T)
        # lookup flattened list of indices
        flat_times = lookup.view(-1).numpy() # (B*T)
        # select full sliding window of times from dataset (B*T)
        sub = self.ds.isel(time=flat_times)

        x_feats: List[np.ndarray] = []
        for name in self.feature_names:
            da = sub[name] # <- ("time", "y", "x")
            arr = da.values.astype(np.float32) # <-- TRIGGERS UNLOAD   # shape <- (B*T, H, W)
            BTHW = arr.reshape(batch_size, self.window_size, *arr.shape[1:]) # <- (B, T, H, W)
            x_feats.append(BTHW)

        # Stack along channel axis -> (B, T, C, H, W)
        X_np = np.stack(x_feats, axis=2)
        X = torch.from_numpy(X_np)

        # 7) Labels & masks: same pattern
        label_tensors: Dict[str, torch.Tensor] = {}
        mask_tensors: Dict[str, torch.Tensor] = {}

        for lname in self.label_names:
            da = sub[lname]
            arr = da.values # <-- TRIGGERS UNLOAD
            BTHW = arr.reshape(batch_size, self.window_size, *arr.shape[1:])
            label_tensors[lname] = torch.as_tensor(BTHW)

        for mname in self.mask_names:
            da = sub[mname]
            arr = da.values # <-- TRIGGERS UNLOAD
            BTHW = arr.reshape(batch_size, self.window_size, *arr.shape[1:])
            mask_tensors[mname] = torch.as_tensor(BTHW)

        return X, label_tensors, mask_tensors


def init_data_loader(
    dataset: Literal['train', 'eval', 'test'],
    num_workers = 0,
    batch_size = 1
):
    pin_memory = True
    
    dataset_path = TRAIN_DATA_DIR / f"{dataset}.zarr"
    if dataset == 'eval':
        dataset_path = EVAL_DATA_DIR / f"{dataset}.zarr"
    if dataset == 'test':
        dataset_path = TEST_DATA_DIR / f"{dataset}.zarr"

    ds = FireDataset(
        dataset_path,
        window_size   = 10,
        window_stride = 2,
    )

    shuffle = False
    if dataset == 'train':
        shuffle = True

    isGPU = torch.cuda.is_available()

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle = shuffle,
        num_workers=0,
        collate_fn=ds.collate_batch,
        # pin_memory=isGPU,
        # persistent_workers=isGPU
    )
