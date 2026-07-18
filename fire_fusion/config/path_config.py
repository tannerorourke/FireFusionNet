# fire_fusion/config/path_config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT     = PROJECT_ROOT / "data"
FF_ROOT = PROJECT_ROOT / "fire_fusion"

RAW_DATA_DIR  = DATA_ROOT / "raw"
# Built datasets live under data/processed/<dataset-name>/; see dataset_config.py
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

LANDFIRE_DIR    = RAW_DATA_DIR / "landfire"
NLCD_DIR        = RAW_DATA_DIR / "nlcd"
GPW_DIR         = RAW_DATA_DIR / "nasa_gpw"
CROADS_DIR      = RAW_DATA_DIR / "census"
USFS_DIR        = RAW_DATA_DIR / "usfs"
GRIDMET_DIR     = RAW_DATA_DIR / "gridmet"
MODIS_DIR       = RAW_DATA_DIR / "modis"
USDA_DIR        = RAW_DATA_DIR / "usda.gdb"
NCEI_SWDI_DIR   = RAW_DATA_DIR / "ncei_swdi"

MODEL_DIR = FF_ROOT / "model"
MODEL_SAVE_DIR = MODEL_DIR / "saved"
PLOTS_DIR = FF_ROOT / "analysis" / "plots"



