#!/usr/bin/env python3
"""
Convert LeRobot parquet files to CSV with fully expanded state/action columns.

Each parquet file -> one CSV file, with observation.state and action vectors
broken down into individual named columns based on the 44D modality layout:

State/Action column naming:
    state_left_arm_0 ... state_left_arm_6          (7 cols)
    state_left_hand_thumbCMC                        (6 cols)
    state_left_hand_thumbMCP
    state_left_hand_indexMCP
    state_left_hand_middleMCP
    state_left_hand_ringMCP
    state_left_hand_littleMCP
    state_left_leg_0 ... state_left_leg_5           (6 cols, zeros)
    state_neck_0 ... state_neck_2                   (3 cols, zeros)
    state_right_arm_0 ... state_right_arm_6         (7 cols)
    state_right_hand_thumbCMC ... state_right_hand_littleMCP  (6 cols)
    state_right_leg_0 ... state_right_leg_5         (6 cols, zeros)
    state_waist_0 ... state_waist_2                 (3 cols, zeros)

    action_* same pattern

Usage:
    # Convert all parquets under a dataset root
    python parquet_to_csv.py --input ./sdg_dataset_trial --output ./sdg_dataset_trial_csv

    # Convert a single parquet file
    python parquet_to_csv.py --input ./sdg_dataset_trial/data/chunk-000/episode_000000.parquet --output ./csv_out

    # Convert in-place (CSV alongside each parquet)
    python parquet_to_csv.py --input ./sdg_dataset_trial --inplace
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Column name definitions (mirrors modality.json 44D layout)
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_column_names(prefix: str) -> list[str]:
    """
    Build the 44 column names for a state or action vector.
    prefix is either 'state' or 'action'.
    """
    cols = []

    # left_arm: 7 DOF  [0:7]
    cols += [f"{prefix}.left_arm.joint_{i+1}" for i in range(7)]

    # left_hand: 6 DOF  [7:13]
    cols += [
        f"{prefix}.left_hand.thumbCMC",
        f"{prefix}.left_hand.thumbMCP",
        f"{prefix}.left_hand.indexMCP",
        f"{prefix}.left_hand.middleMCP",
        f"{prefix}.left_hand.ringMCP",
        f"{prefix}.left_hand.littleMCP",
    ]

    # left_leg: 6 DOF  [13:19]  — zeros in sim
    cols += [f"{prefix}.left_leg.{i}" for i in range(6)]

    # neck: 3 DOF  [19:22]  — zeros in sim
    cols += [f"{prefix}.neck.{i}" for i in range(3)]

    # right_arm: 7 DOF  [22:29]
    cols += [f"{prefix}.right_arm.joint_{i+1}" for i in range(7)]

    # right_hand: 6 DOF  [29:35]
    cols += [
        f"{prefix}.right_hand.thumbCMC",
        f"{prefix}.right_hand.thumbMCP",
        f"{prefix}.right_hand.indexMCP",
        f"{prefix}.right_hand.middleMCP",
        f"{prefix}.right_hand.ringMCP",
        f"{prefix}.right_hand.littleMCP",
    ]

    # right_leg: 6 DOF  [35:41]  — zeros in sim
    cols += [f"{prefix}.right_leg.{i}" for i in range(6)]

    # waist: 3 DOF  [41:44]  — zeros in sim
    cols += [f"{prefix}.waist.{i}" for i in range(3)]

    return cols  # 44 column names total


STATE_COLS  = build_vector_column_names("state")
ACTION_COLS = build_vector_column_names("action")

# Metadata columns that pass through unchanged
PASSTHROUGH_COLS = [
    "episode_chunk",
    "episode_id",
    "episode_index",
    "index",
    "timestamp",
    "annotation.human.action.task_description",
    "task_index",
    "annotation.human.validity",
    "next.reward",
    "next.done",
    "is_last_row",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────

def expand_vector_column(df: pd.DataFrame, col_name: str, new_col_names: list[str]) -> pd.DataFrame:
    """
    Expand a column containing lists/arrays into individual scalar columns.
    The original column is dropped.
    """
    if col_name not in df.columns:
        print(f"  [WARN] Column '{col_name}' not found, skipping expansion")
        return df

    # Convert list/array column to a proper 2D array
    arr = np.array(df[col_name].tolist(), dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != len(new_col_names):
        raise ValueError(
            f"Column '{col_name}' has shape {arr.shape}, "
            f"expected (T, {len(new_col_names)})"
        )

    expanded = pd.DataFrame(arr, columns=new_col_names, index=df.index)
    df = df.drop(columns=[col_name])
    df = pd.concat([df, expanded], axis=1)
    return df


def parquet_to_csv(parquet_path: Path, csv_path: Path) -> None:
    """Convert a single parquet file to an expanded CSV."""
    df = pd.read_parquet(parquet_path)

    # Expand observation.state -> 44 named state columns
    df = expand_vector_column(df, "observation.state", STATE_COLS)

    # Expand action -> 44 named action columns
    df = expand_vector_column(df, "action", ACTION_COLS)

    # Reorder: metadata first, then state, then action
    meta_cols_present = [c for c in PASSTHROUGH_COLS if c in df.columns]
    state_cols_present = [c for c in STATE_COLS if c in df.columns]
    action_cols_present = [c for c in ACTION_COLS if c in df.columns]

    # Any leftover columns not in our expected sets
    known = set(meta_cols_present + state_cols_present + action_cols_present)
    extra_cols = [c for c in df.columns if c not in known]

    final_order = meta_cols_present + state_cols_present + action_cols_present + extra_cols
    df = df[final_order]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"  {parquet_path.name} -> {csv_path.name}  ({len(df)} rows, {len(df.columns)} cols)")


# ─────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_parquets(root: Path) -> list[Path]:
    """Recursively find all .parquet files under root."""
    return sorted(root.rglob("*.parquet"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Expand LeRobot parquet files into CSVs with individual state/action columns."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a LeRobot dataset root directory, or a single .parquet file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for CSV files. Mirrors the parquet folder structure. "
             "Not needed if using --inplace.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write each CSV alongside its parquet file (ignores --output).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # ── Collect parquet files ────────────────────────────────────────────────
    if input_path.is_file() and input_path.suffix == ".parquet":
        parquet_files = [input_path]
        root = input_path.parent
    elif input_path.is_dir():
        parquet_files = find_parquets(input_path)
        root = input_path
    else:
        raise ValueError(f"--input must be a .parquet file or a directory, got: {input_path}")

    if not parquet_files:
        print(f"No .parquet files found under {input_path}")
        return

    print(f"\nFound {len(parquet_files)} parquet file(s)")
    print(f"State columns:  {len(STATE_COLS)} (44D expanded)")
    print(f"Action columns: {len(ACTION_COLS)} (44D expanded)")
    print(f"Total CSV columns per file: {len(PASSTHROUGH_COLS) + len(STATE_COLS) + len(ACTION_COLS)}\n")

    # ── Convert each file ────────────────────────────────────────────────────
    for pq_path in parquet_files:
        if args.inplace:
            csv_path = pq_path.with_suffix(".csv")
        else:
            if args.output is None:
                raise ValueError("Provide --output or use --inplace")
            # Mirror directory structure relative to root
            rel = pq_path.relative_to(root)
            csv_path = Path(args.output) / rel.with_suffix(".csv")

        parquet_to_csv(pq_path, csv_path)

    print(f"\nDone. {len(parquet_files)} CSV(s) written.")


if __name__ == "__main__":
    main()
