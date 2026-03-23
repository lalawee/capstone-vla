#!/usr/bin/env python3
"""
generate_episodes_stats.py
Generate meta/episodes_stats.jsonl required by LeRobot v2.1.

LeRobot v2.1 requires per-episode statistics that external pipelines
(Isaac Lab, custom data collection) don't generate automatically.

Usage:
    python generate_episodes_stats.py --dataset_root /path/to/dataset
    python generate_episodes_stats.py --dataset_root /path/to/dataset --keys observation.state action

Format produced (one JSON line per episode):
    {"episode_index": 0, "stats": {"observation.state": {"count": [N], "mean": [...], ...}}}
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_KEYS = [
    "observation.state",
    "action",
    "timestamp",
    "next.reward",
]


def compute_stats_for_column(series: pd.Series) -> dict:
    """Compute statistics for a single column (may be scalar or list values)."""
    # Expand list/array values into a 2D numpy array
    first = series.iloc[0]
    if hasattr(first, "__len__"):
        arr = np.stack(series.values)  # (T, D)
    else:
        arr = series.values.reshape(-1, 1)  # (T, 1)

    T, D = arr.shape
    return {
        "count": [T],           # always shape (1,) regardless of D
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "q01":   np.percentile(arr, 1,  axis=0).tolist(),
        "q99":   np.percentile(arr, 99, axis=0).tolist(),
    }


def generate_episodes_stats(dataset_root: str, keys: list[str]) -> None:
    root = Path(dataset_root)
    meta_dir = root / "meta"
    output_path = meta_dir / "episodes_stats.jsonl"

    # Find all episode parquet files
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No episode parquet files found under {data_dir}. "
            "Check your dataset structure."
        )

    print(f"Found {len(parquet_files)} episodes.")
    print(f"Keys to compute stats for: {keys}")
    print(f"Output: {output_path}")
    print("")

    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for parquet_path in tqdm(parquet_files, desc="Computing stats"):
            # Extract episode index from filename (episode_000000.parquet)
            stem = parquet_path.stem  # episode_000000
            ep_idx = int(stem.split("_")[-1])

            df = pd.read_parquet(parquet_path)

            stats = {}
            for key in keys:
                if key not in df.columns:
                    continue
                try:
                    stats[key] = compute_stats_for_column(df[key])
                except Exception as e:
                    print(f"\nWARNING: Could not compute stats for '{key}' in {parquet_path.name}: {e}")

            record = {"episode_index": ep_idx, "stats": stats}
            f.write(json.dumps(record) + "\n")

    line_count = sum(1 for _ in open(output_path))
    print(f"\nDone. Wrote {line_count} episode records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate episodes_stats.jsonl for LeRobot v2.1")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root (contains meta/, data/ dirs)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help=f"Feature keys to compute stats for. Default: {DEFAULT_KEYS}",
    )
    args = parser.parse_args()

    generate_episodes_stats(args.dataset_root, args.keys)


if __name__ == "__main__":
    main()