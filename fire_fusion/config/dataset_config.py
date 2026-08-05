# Dataset configurations, one per directory under data/processed/<name>/.
#
#   extract  raw sources  -> cube.zarr
#   publish  cube.zarr    -> dataset.zarr
#   compile  dataset.zarr -> {train,eval,test}.zarr + manifest.json
#
# dataset.zarr is the redistributable artifact: deterministic functions of the
# raw sources only. Data-estimated transforms live in compile, where the split
# years below apply.
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .path_config import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    resolution: float                      # meters per pixel
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    # 2003 is the earliest fully clean season; MODIS LAI MCD15A2H begins mid-2002
    start_date: str = "2003-01-01"
    end_date: str = "2020-12-31"

    # Split boundaries; normalization statistics come from the train years only
    train_yrs: Tuple[int, int] = (2003, 2016)
    eval_yrs: Tuple[int, int] = (2017, 2018)
    test_yrs: Tuple[int, int] = (2019, 2020)

    # Inclusive month bounds of the supervised fire season; None keeps every day.
    # Extraction carries a halo either side so temporal derivations run on real
    # history, sized by the longest operator in the derived stage: the
    # lightning-load IIR decays below 0.1% in ~40 days.
    season_months: Optional[Tuple[int, int]] = None
    halo_lead_days: int = 40
    halo_trail_days: int = 10

    # staging chunk: full spatial extent per chunk so spatial-kernel ops stay local
    stage_time_chunk: int = 16
    x_time_chunk: int = 8
    # >1 lets a patch-based loader read spatial crops without full-frame decompression
    spatial_splits: int = 1

    @property
    def root(self) -> Path:
        return PROCESSED_DATA_DIR / self.name

    @property
    def staging_path(self) -> Path:
        return self.root / "cube.zarr"

    @property
    def published_path(self) -> Path:
        return self.root / "dataset.zarr"

    @property
    def published_manifest_path(self) -> Path:
        return self.root / "dataset_manifest.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def split_path(self, split: str) -> Path:
        return self.root / f"{split}.zarr"

    def split_years(self, split: str) -> Tuple[int, int]:
        return {"train": self.train_yrs, "eval": self.eval_yrs, "test": self.test_yrs}[split]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    cfg.name: cfg
    for cfg in [
        # Washington state (grid 102x109)
        DatasetConfig(
            "wa4000", 4000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, season_months=(5, 10),
        ),
        # Washington state (grid 204x217)
        DatasetConfig(
            "wa2000", 2000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, season_months=(5, 10),
        ),
        # Washington state (grid 407x433)
        DatasetConfig(
            "wa1000", 1000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, spatial_splits=2, season_months=(5, 10),
        ),
        # Eastern Cascades, crest through Okanogan Highlands (544x544, a 272km square).
        # Bounds are picked so the natural extent is already a multiple of the
        # depth-3 stride-window product: it partitions evenly and needs no crop.
        # North clamps to the 49th parallel; US sources stop at the border.
        DatasetConfig(
            "cascades500", 500.0, (46.642, 49.0), (-121.85, -118.349),
            x_time_chunk=16, spatial_splits=3, season_months=(5, 10),
        ),
    ]
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {sorted(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[name]
