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

# -- the encoder's total stride and the attention window compose, so a crop whose
# -- origin is not a multiple of their product shifts the window partition relative
# -- to a full-grid pass and changes the prediction for the same cell
def crop_align(encoder_depth: int, attn_window: int) -> int:
    return (2 ** encoder_depth) * attn_window


# -- A crop supervised out to its own border would be trained on cells whose context
# -- is partly zero padding, which only ever occurs at the true domain edge during
# -- full-grid inference. Crops therefore carry a halo of real context, excluded
# -- from the loss, as wide as the receptive field radius of one output cell:
# -- 5 through the stem and the full-resolution residual stage, 7 * (2^d - 1) across
# -- the stride-2 stages, 2^d * (attn_window - 1) across the attention window, and
# -- 2^(d+1) - 1 back through the decoder's per-level 3x3.
def crop_halo(encoder_depth: int, attn_window: int) -> int:
    d, align = encoder_depth, crop_align(encoder_depth, attn_window)
    radius = 5 + 7 * (2 ** d - 1) + (2 ** d) * (attn_window - 1) + (2 ** (d + 1) - 1)
    return -(-radius // align) * align

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
        encoder_depth: int = 1,
        attn_window: int = 2,
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
        self.cause_counts = [int(c) for c in self.manifest["cause_counts"]]

        if self.X.sizes["channel"] != self.in_channels:
            raise ValueError(
                f"{path} has {self.X.sizes['channel']} channels; "
                f"manifest says {self.in_channels}"
            )

        self.window_size = window_size
        self.window_stride = window_stride
        self.n_timesteps = self.ds.sizes["time"]
        starts = np.arange(
            0, max(self.n_timesteps - window_size + 1, 0),
            window_stride,
            dtype=int,
        )
        # A seasonally windowed dataset jumps from one October to the next May,
        # so consecutive positions are not always consecutive days. Windows are
        # kept only where every step inside them is a single day; a window
        # straddling the gap would present two fire seasons as one sequence.
        if len(starts) and self.n_timesteps > 1:
            days = np.asarray(self.ds.indexes["time"], dtype="datetime64[D]")
            step = np.diff(days).astype(int)
            block = np.concatenate([[0], np.cumsum(step != 1)])
            keep = block[starts] == block[starts + window_size - 1]
            n_dropped = int((~keep).sum())
            if n_dropped:
                print(f"dropped {n_dropped} window(s) spanning a season gap")
            starts = starts[keep]
        self.window_starts = starts

        # crop_size is the supervised extent; the sample read is that plus a halo
        # on every side, so a crop_size of 96 reads 128x128 and supervises the
        # middle 96x96
        self.crop_size = crop_size
        self.crop_align = crop_align(encoder_depth, attn_window)
        self.crop_halo = crop_halo(encoder_depth, attn_window)
        self._rng = np.random.default_rng(crop_seed)
        if crop_size is not None:
            if crop_size % self.crop_align:
                raise ValueError(f"crop_size {crop_size} must be a multiple of {self.crop_align}")
            self.read_size = crop_size + 2 * self.crop_halo
            H, W = self.out_size
            if self.read_size > H or self.read_size > W:
                raise ValueError(
                    f"crop_size {crop_size} needs a {self.read_size}px read "
                    f"({self.crop_halo}px halo at encoder depth {encoder_depth}), "
                    f"larger than the {H}x{W} grid"
                )
            # An extent that is not a multiple of the alignment leaves the far edge
            # short of the last legal origin, so a strip of it is never read during
            # training while evaluation still scores it. Grids built since the
            # alignment was enforced have no such strip.
            strip = [f"{n}{ax}" for ax, n in (("y", H % self.crop_align), ("x", W % self.crop_align)) if n]
            if strip:
                print(f"crop training cannot reach {', '.join(strip)} at the far edge "
                      f"of the {H}x{W} grid (alignment {self.crop_align})")

    def __len__(self) -> int:
        return len(self.window_starts)

    def _crop_origin(self) -> Tuple[int, int]:
        H, W = self.out_size
        align = self.crop_align
        y = self._rng.integers(0, (H - self.read_size) // align + 1) * align
        x = self._rng.integers(0, (W - self.read_size) // align + 1) * align
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
            H, W = self.out_size
            ysel = slice(y0, y0 + self.read_size)
            xsel = slice(x0, x0 + self.read_size)
            keep = (self._keep_span(y0, H), self._keep_span(x0, W))

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

    # -- a crop side that sits on the domain edge loses no context there: its padding
    # -- is the padding full-grid inference sees anyway, so those cells stay supervised.
    # -- Holding them out instead would leave a halo-wide band of the domain that
    # -- training never scores and evaluation always does.
    def _keep_span(self, origin: int, extent: int) -> slice:
        lo = 0 if origin == 0 else self.crop_halo
        hi = self.read_size if origin + self.read_size == extent else self.read_size - self.crop_halo
        return slice(lo, hi)

    @staticmethod
    def _halo_masked(mask: torch.Tensor, keep: Tuple[slice, slice]) -> torch.Tensor:
        out = torch.zeros_like(mask)
        out[keep[0], keep[1]] = mask[keep[0], keep[1]]
        return out


def _seed_worker(worker_id: int) -> None:
    # -- a worker gets a pickled copy of the dataset, so without this reseed every
    # -- worker would inherit one crop RNG and draw the identical crop origins.
    # -- torch derives each worker's initial seed from the loader's generator.
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
    encoder_depth: int = 1,
    attn_window: int = 2,
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
        encoder_depth=encoder_depth,
        attn_window=attn_window,
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
