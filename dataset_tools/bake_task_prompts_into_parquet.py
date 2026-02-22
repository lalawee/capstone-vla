#!/usr/bin/env python3
"""
This is for Pi0!!!!!!!!!!!!!!!!!!!!!!!!


Bake task prompts (string) into LeRobot-style parquet files by mapping task_index -> task text
using a tasks.jsonl file.

It will:
- Read tasks.jsonl (JSONL) into a dict {task_index: task_string}
- Walk root/data/**.parquet
- Add a new column (default: "task") containing the string prompt
- Write parquet back (in place by default) with backups

Typical usage:
  python bake_task_prompts_into_parquet.py --root /path/to/dataset --tasks /path/to/tasks.jsonl

Notes:
- Your parquet likely stores the task id in one of these columns:
    "annotation.human.action.task_description"  (you described this)
    "task_index"
  The script auto-detects. You can force with --task-id-col.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd


def load_tasks_jsonl(tasks_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {tasks_path}: {e}") from e
            if "task_index" not in obj or "task" not in obj:
                raise ValueError(
                    f"Line {lineno} in {tasks_path} must contain keys 'task_index' and 'task'. Got: {obj}"
                )
            mapping[int(obj["task_index"])] = str(obj["task"])
    if not mapping:
        raise ValueError(f"No tasks loaded from {tasks_path}. Is the file empty?")
    return mapping


def detect_task_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "annotation.human.action.task_description",
        "annotation/human/action/task_description",
        "task_index",
        "task",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    # Avoid overwrite collisions
    if dst.exists():
        i = 1
        while True:
            cand = backup_dir / f"{src.stem}.bak{i}{src.suffix}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)
    return dst


def bake_one_parquet(
    parquet_path: Path,
    tasks_map: Dict[int, str],
    task_id_col: Optional[str],
    out_col: str,
    unknown_value: str,
    backup_dir: Optional[Path],
    dry_run: bool,
) -> Tuple[bool, str]:
    df = pd.read_parquet(parquet_path)

    col = task_id_col or detect_task_id_column(df)
    if col is None:
        return False, f"SKIP: {parquet_path} (could not detect task id column)"

    # Build prompt strings
    # Convert to int safely (handles numpy/pandas types)
    def map_task_id(x):
        try:
            k = int(x)
        except Exception:
            return unknown_value
        return tasks_map.get(k, unknown_value)

    prompts = df[col].map(map_task_id)

    # Add/overwrite output column
    df[out_col] = prompts.astype("string")

    if dry_run:
        # don't write anything
        return True, f"DRYRUN OK: {parquet_path} (task_id_col='{col}' -> out_col='{out_col}')"

    if backup_dir is not None:
        backup_file(parquet_path, backup_dir)

    # Write in place
    df.to_parquet(parquet_path, index=False)
    return True, f"OK: {parquet_path} (task_id_col='{col}' -> out_col='{out_col}')"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing data/ meta/ videos/")
    ap.add_argument("--tasks", type=str, required=True, help="Path to tasks.jsonl")
    ap.add_argument(
        "--task-id-col",
        type=str,
        default=None,
        help="Column in parquet that stores task index (optional; auto-detects if omitted)",
    )
    ap.add_argument(
        "--out-col",
        type=str,
        default="task",
        help="New string column to write (e.g. 'task' or 'prompt'). Default: task",
    )
    ap.add_argument(
        "--unknown",
        type=str,
        default="unknown_task",
        help="Value to use when task index is missing from tasks.jsonl",
    )
    ap.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Directory to store backups of original parquet files (recommended).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write, just print what would change.")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    tasks_path = Path(args.tasks).expanduser().resolve()
    backup_dir = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else None

    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {tasks_path}")

    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected {data_dir} to exist (root/data).")

    tasks_map = load_tasks_jsonl(tasks_path)

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    print(f"Found {len(parquet_files)} parquet files under {data_dir}")
    print(f"Loaded {len(tasks_map)} tasks from {tasks_path}")
    if backup_dir:
        print(f"Backups will be saved to: {backup_dir}")
    if args.dry_run:
        print("DRY RUN: no files will be modified.")

    ok = 0
    skipped = 0
    for p in parquet_files:
        success, msg = bake_one_parquet(
            parquet_path=p,
            tasks_map=tasks_map,
            task_id_col=args.task_id_col,
            out_col=args.out_col,
            unknown_value=args.unknown,
            backup_dir=backup_dir,
            dry_run=args.dry_run,
        )
        print(msg)
        if success:
            ok += 1
        else:
            skipped += 1

    print(f"\nDone. Updated={ok}, Skipped={skipped}.")


if __name__ == "__main__":
    main()