# Loads a built dataset split (see dataset/build.py). Splits store one
# pre-stacked (time, channel, y, x) float32 array "X" plus per-day label and
# mask arrays; a window sample is a single contiguous time-slice read, and all
# channel bookkeeping comes from the dataset's manifest.json.
import json
from typing import Dict, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from ..config.dataset_config import DatasetConfig, get_dataset_config

from dask import config as daskconfig
# zarr reads happen inside DataLoader workers; nested dask threads only add overhead
daskconfig.set(scheduler='synchronous')


class FireDataset(Dataset):
    """ Yields spatiotemporal windows:
        - X: (T, C, H, W) float32, the model input window
        - labels/masks: (H, W) at the window's final day (the prediction target
          is the transition into the following day)
    """
    def __init__(
        self,
        ds_config: DatasetConfig,
        split: Literal['train', 'eval', 'test'],
        window_size: int = 10,
        window_stride: int = 2,
    ):
        super().__init__()
        self.manifest = json.loads(ds_config.manifest_path.read_text())

        path = ds_config.split_path(split)
        print(f"opening >> {path}")
        self.ds = xr.open_zarr(path)
        self.X = self.ds["X"]

        self.feature_names = list(self.manifest["channels"])
        self.label_names = list(self.manifest["labels"])
        self.mask_names = list(self.manifest["masks"])
        self.in_channels = int(self.manifest["in_channels"])
        self.out_size = (
            int(self.manifest["grid"]["height"]),
            int(self.manifest["grid"]["width"]),
        )
        self.n_cause_classes = int(self.manifest["n_cause_classes"])
        self.ign_pos_weight = float(self.manifest["ign_pos_weight"])

        if self.X.sizes["channel"] != self.in_channels:
            raise ValueError(
                f"{path} has {self.X.sizes['channel']} channels; "
                f"manifest says {self.in_channels}"
            )

        self.window_size = window_size
        self.window_stride = window_stride
        self.n_timesteps = self.ds.sizes["time"]
        self.window_starts = np.arange(
            0, max(self.n_timesteps - window_size + 1, 0),
            window_stride,
            dtype=int,
        )

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Dict]:
        t0 = int(self.window_starts[idx])
        t1 = t0 + self.window_size

        x = torch.from_numpy(
            np.ascontiguousarray(self.X.isel(time=slice(t0, t1)).values)
        )  # (T, C, H, W) float32

        last = t1 - 1
        labels = {
            name: torch.as_tensor(self.ds[name].isel(time=last).values)
            for name in self.label_names
        }
        masks = {
            name: torch.as_tensor(self.ds[name].isel(time=last).values)
            for name in self.mask_names
        }
        return x, labels, masks


def init_data_loader(
    split: Literal['train', 'eval', 'test'],
    dataset_name: str = "wa2000",
    num_workers: int = 0,
    batch_size: int = 1,
    window_size: int = 10,
    window_stride: int = 2,
):
    ds = FireDataset(
        get_dataset_config(dataset_name),
        split,
        window_size=window_size,
        window_stride=window_stride,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
