# Named dataset configurations: grid geometry, split years, output paths, and
# zarr chunking. A "dataset" is one fully built datacube (staging cube +
# train/eval/test splits + manifest) under data/processed/<name>/.
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from .path_config import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    resolution: float                      # meters per pixel
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    start_date: str = "2009-01-01"
    end_date: str = "2020-12-31"

    # Split boundaries; normalization statistics come from the train years only
    train_yrs: Tuple[int, int] = (2009, 2016)
    eval_yrs: Tuple[int, int] = (2017, 2018)
    test_yrs: Tuple[int, int] = (2019, 2020)

    # Chunking along time for the staging cube (per-variable, full spatial
    # extent per chunk so spatial-kernel ops stay chunk-local)
    stage_time_chunk: int = 16
    # Chunking of the final stacked X array in the split zarrs
    x_time_chunk: int = 8
    # Split y/x into this many chunks per axis in the split zarrs; >1 lets a
    # patch-based loader read spatial crops without full-frame decompression
    spatial_splits: int = 1

    @property
    def root(self) -> Path:
        return PROCESSED_DATA_DIR / self.name

    @property
    def staging_path(self) -> Path:
        return self.root / "cube.zarr"

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
        # Washington state (grid 204x220)
        DatasetConfig("wa2000", 2000.0, (45.5, 49.0), (-122.5, -117.0), x_time_chunk=16),
        # Washington state (grid 408x436)
        DatasetConfig("wa1000", 1000.0, (45.5, 49.0), (-122.5, -117.0)),
        # Eastern Cascades corridor (grid 696x856); the north edge is clamped
        # to the 49th parallel -- US sources carry no coverage into Canada
        DatasetConfig("cascades250", 250.0, (47.5, 49.0), (-121.8, -119.0), spatial_splits=2),
    ]
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {sorted(DATASET_CONFIGS)}")
    return DATASET_CONFIGS[name]
