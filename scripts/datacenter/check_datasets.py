"""Lightweight validation for the two full NF-V2 data-center datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

REQUIRED = {
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    "Attack",
}

DATASETS = {
    "cicids": ("EIFFEL_CICIDS_PATH", "NF-CSE-CIC-IDS2018-v2"),
    "nb15": ("EIFFEL_NB15_PATH", "NF-UNSW-NB15-v2"),
}


def check(name: str, sample_rows: int) -> None:
    env_name, canonical = DATASETS[name]
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise SystemExit(f"{env_name} is not set ({canonical}).")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{env_name} points to a missing file: {path}")

    frame = pd.read_csv(path, nrows=sample_rows)
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise SystemExit(
            f"{canonical}: incompatible schema; missing required columns: {missing}"
        )

    labels = sorted(frame["Label"].dropna().astype(str).unique().tolist())[:10]
    attacks = sorted(frame["Attack"].dropna().astype(str).unique().tolist())[:20]
    mib = path.stat().st_size / (1024 * 1024)
    print(f"{name}: OK")
    print(f"  file: {path}")
    print(f"  size: {mib:.1f} MiB")
    print(f"  columns: {len(frame.columns)}")
    print(f"  sampled labels: {labels}")
    print(f"  sampled attack families: {attacks}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("cicids", "nb15", "all"), nargs="?", default="all")
    parser.add_argument("--sample-rows", type=int, default=4096)
    args = parser.parse_args()

    names = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        check(name, max(1, args.sample_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
