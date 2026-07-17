"""
Splits store one pre-stacked (time, channel, y, x) float32 array "X" 
plus per-day label and mask arrays; a window sample is a single contiguous 
time-slice read, and all channel bookkeeping comes from the dataset's 
manifest.json.
"""
import json
import random
from typing import Dict, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import xarray as xr

from ..config.dataset_config import DatasetConfig, get_dataset_config

"""
An output cell's receptive field reaches 16 cells in each direction, so a crop
supervised out to its own border would be trained on cells whose context is
partly zero padding -- padding that only ever occurs at the true domain edge
during full-grid inference. Crops therefore carry a halo of this width that
supplies real context and is excluded from the loss.
"""
CROP_HALO = 16

"""
The encoder's stride (2) and the attention window (2) compose, so a crop whose
origin is not a multiple of 4 shifts the window partition relative to a
full-grid pass and changes the prediction for the same cell.
"""
CROP_ALIGN = 4

from dask import config as daskconfig
# zarr reads happen inside DataLoader workers; nested dask threads only add overhead
daskconfig.set(scheduler='synchronous')


class FireDataset(Dataset):
    """ Yields spatiotemporal windows:
        - X: (T, C, H, W) float32, the model input window
        - labels/masks: (H, W) at the window's final day (the prediction target
          is a fresh ignition within the forward horizon of that day)
    """
    def __init__(
        self,
        ds_config: DatasetConfig,
        split: Literal['train', 'eval', 'test'],
        window_size: int = 10,
        window_stride: int = 2,
        crop_size: int | None = None,
        crop_seed: int | None = None,
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

        # crop_size is the supervised extent; the sample read is that plus a halo
        # on every side, so a crop_size of 96 reads 128x128 and supervises the
        # middle 96x96
        self.crop_size = crop_size
        self._rng = np.random.default_rng(crop_seed)
        if crop_size is not None:
            if crop_size % CROP_ALIGN:
                raise ValueError(f"crop_size {crop_size} must be a multiple of {CROP_ALIGN}")
            self.read_size = crop_size + 2 * CROP_HALO
            H, W = self.out_size
            if self.read_size > H or self.read_size > W:
                raise ValueError(
                    f"crop_size {crop_size} needs a {self.read_size}px read, "
                    f"larger than the {H}x{W} grid"
                )

    def __len__(self) -> int:
        return len(self.window_starts)

    def _crop_origin(self) -> Tuple[int, int]:
        """ Aligned top-left origin for a halo-padded read. """
        H, W = self.out_size
        y = self._rng.integers(0, (H - self.read_size) // CROP_ALIGN + 1) * CROP_ALIGN
        x = self._rng.integers(0, (W - self.read_size) // CROP_ALIGN + 1) * CROP_ALIGN
        return int(y), int(x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict, Dict]:
        t0 = int(self.window_starts[idx])
        t1 = t0 + self.window_size
        last = t1 - 1

        if self.crop_size is None:
            ysel = xsel = slice(None)
            keep = None
        else:
            y0, x0 = self._crop_origin()
            ysel = slice(y0, y0 + self.read_size)
            xsel = slice(x0, x0 + self.read_size)
            keep = slice(CROP_HALO, CROP_HALO + self.crop_size)

        x = torch.from_numpy(
            np.ascontiguousarray(self.X.isel(time=slice(t0, t1), y=ysel, x=xsel).values)
        )  # (T, C, H, W) float32

        labels = {
            name: torch.as_tensor(self.ds[name].isel(time=last, y=ysel, x=xsel).values)
            for name in self.label_names
        }
        masks = {
            name: torch.as_tensor(self.ds[name].isel(time=last, y=ysel, x=xsel).values)
            for name in self.mask_names
        }

        if keep is not None:
            # the halo stays in the features so the supervised cells keep their
            # true context, and is dropped from every mask so no loss is taken on
            # cells whose own context runs off the edge of the read
            masks = {
                name: self._halo_masked(m, keep) for name, m in masks.items()
            }
        return x, labels, masks

    @staticmethod
    def _halo_masked(mask: torch.Tensor, keep: slice) -> torch.Tensor:
        out = torch.zeros_like(mask)
        out[keep, keep] = mask[keep, keep]
        return out


def _seed_worker(worker_id: int) -> None:
    """ Give each worker process its own reproducible RNG state.

    A worker receives a pickled copy of the dataset, so every worker would
    otherwise inherit one crop RNG at an identical state and draw the identical
    sequence of crop origins. torch derives each worker's initial seed from the
    loader's generator, which makes the reseed below both per-worker distinct
    and a function of the run seed.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

    info = get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(seed)


def init_data_loader(
    split: Literal['train', 'eval', 'test'],
    dataset_name: str = "wa2000",
    num_workers: int = 0,
    batch_size: int = 1,
    window_size: int = 10,
    window_stride: int = 2,
    crop_size: int | None = None,
    seed: int | None = None,
):
    # cropping is a training-time device for grids that do not fit whole; eval
    # and test read the full extent so their metrics stay comparable across runs
    ds = FireDataset(
        get_dataset_config(dataset_name),
        split,
        window_size=window_size,
        window_stride=window_stride,
        crop_size=crop_size if split == "train" else None,
        crop_seed=seed,
    )

    # the shuffle order is drawn from the loader's own generator rather than the
    # global RNG, so it stays fixed regardless of how much other work consumed
    # global draws before the loader was built
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        generator=generator,
        worker_init_fn=_seed_worker if seed is not None else None,
    )
