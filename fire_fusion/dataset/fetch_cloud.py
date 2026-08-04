"""
Backblaze B2 transit for raw source data and run artifacts.

Processed transfers are selected by build step, not by directory, so a training
node pulls the splits without dragging the staging and published cubes with them.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..config.dataset_config import DATASET_CONFIGS
from ..config.path_config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LANDFIRE_DIR, NLCD_DIR, GPW_DIR, CROADS_DIR,
    USFS_DIR, PRISM_DIR, AORC_DIR, MODIS_DIR, USDA_DIR, NCEI_SWDI_DIR,
)

# B2 key namespaces. Local layout round-trips to the same paths: raw sources under
# data/raw/<source>, built cubes under data/processed/<dataset>, runs under runs/.
RAW_PREFIX = "raw"
PROCESSED_PREFIX = "processed"

# Source name -> local directory. The name doubles as the B2 key segment, and each
# dir sits directly under RAW_DATA_DIR, so a push/pull round-trips to the same path.
RAW_SOURCES: Dict[str, Path] = {
    p.name: p for p in [
        LANDFIRE_DIR, NLCD_DIR, GPW_DIR, CROADS_DIR, USFS_DIR,
        PRISM_DIR, AORC_DIR, MODIS_DIR, USDA_DIR, NCEI_SWDI_DIR,
    ]
}

# One build step writes each of these and a consumer wants exactly one: a training
# node needs the splits, a release needs the published cube. Naming members
# explicitly also keeps stray directories out of a push.
PROCESSED_STEPS: Dict[str, Tuple[str, ...]] = {
    "staging": ("cube.zarr",),
    "published": ("dataset.zarr", "dataset_manifest.json"),
    "splits": ("train.zarr", "eval.zarr", "test.zarr", "manifest.json"),
}


class B2Store:
    """ Minimal keyed object store over a Backblaze B2 bucket (S3 API). """

    def __init__(self, bucket: Optional[str] = None, endpoint: Optional[str] = None):
        import boto3

        endpoint = endpoint or os.environ["B2_ENDPOINT"]
        self.bucket = bucket or os.environ["B2_BUCKET"]
        # -- B2 signs against the region embedded in the endpoint host
        m = re.search(r"s3\.([^.]+)\.backblazeb2\.com", endpoint)
        region = m.group(1) if m else os.environ.get("B2_REGION", "us-east-005")

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=os.environ["B2_KEY_ID"],
            aws_secret_access_key=os.environ["B2_APP_KEY"],
        )
        self.client.head_bucket(Bucket=self.bucket)

    def _remote_size(self, key: str) -> Optional[int]:
        from botocore.exceptions import ClientError
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        except ClientError:
            return None

    def put_file(self, local: Path, key: str, overwrite: bool = False) -> None:
        size = self._remote_size(key)
        if size is not None and not overwrite and size == local.stat().st_size:
            print(f"[B2] skip {key} (present, same size)")
            return
        self.client.upload_file(str(local), self.bucket, key)
        print(f"[B2] put  {key}")

    def get_file(self, key: str, local: Path, overwrite: bool = False) -> None:
        if local.exists() and not overwrite:
            print(f"[B2] skip {local} (present)")
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local))
        print(f"[B2] get  {key}")

    def put_tree(self, root: Path, key_prefix: str, overwrite: bool = False) -> None:
        if not root.exists():
            print(f"[B2] {root} absent, nothing to push")
            return
        for f in sorted(root.rglob("*")):
            if f.is_file():
                self.put_file(f, f"{key_prefix}/{f.relative_to(root).as_posix()}", overwrite)

    def get_tree(self, key_prefix: str, root: Path, overwrite: bool = False) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{key_prefix}/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                self.get_file(key, root / key[len(key_prefix) + 1:], overwrite)


def _resolve_sources(names: Optional[Iterable[str]]) -> Dict[str, Path]:
    if not names:
        return RAW_SOURCES
    return {n: RAW_SOURCES[n] for n in names}


def _processed_datasets(names: Optional[Iterable[str]]) -> List[str]:
    if names:
        return list(names)
    # push with no explicit list -> every built cube on local disk; a pull needs
    # explicit names since the local processed dir may not exist yet on a fresh node
    if not PROCESSED_DATA_DIR.exists():
        return []
    return sorted(p.name for p in PROCESSED_DATA_DIR.iterdir() if p.is_dir())


def _step_members(steps: Iterable[str]) -> List[str]:
    seen = {m: None for s in steps for m in PROCESSED_STEPS[s]}
    return list(seen)


def _sync_processed(
    store: "B2Store", action: str, ds: str, 
    members: Iterable[str], overwrite: bool
) -> None:
    for member in members:
        local = PROCESSED_DATA_DIR / ds / member
        key = f"{PROCESSED_PREFIX}/{ds}/{member}"
        if action == "push":
            if local.is_dir():
                store.put_tree(local, key, overwrite)
            elif local.is_file():
                store.put_file(local, key, overwrite)
            else:
                print(f"[B2] {local} absent, skipping")
        elif member.endswith(".json"):
            store.get_file(key, local, overwrite)
        else:
            store.get_tree(key, local, overwrite)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync source data and built cubes between local disk and B2.")
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--kind", choices=["raw", "processed"], default="raw",
                    help="raw sources (data/raw) or built cubes (data/processed)")
    ap.add_argument("--sources", nargs="+", choices=sorted(RAW_SOURCES), default=None,
                    help="raw only: subset of sources; default all")
    ap.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=None,
                    help="processed only: dataset names; push defaults to all built locally")
    ap.add_argument("--step", nargs="+", choices=sorted(PROCESSED_STEPS), default=["splits"],
                    help="processed only: which build steps' outputs to move")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    store = B2Store()
    if args.kind == "raw":
        for name, path in _resolve_sources(args.sources).items():
            prefix = f"{RAW_PREFIX}/{name}"
            if args.action == "push":
                store.put_tree(path, prefix, args.overwrite)
            else:
                store.get_tree(prefix, RAW_DATA_DIR / name, args.overwrite)
    else:
        members = _step_members(args.step)
        for ds in _processed_datasets(args.datasets):
            _sync_processed(store, args.action, ds, members, args.overwrite)


if __name__ == "__main__":
    main()
