#!/usr/bin/env python3
"""
LeRobot dataset mixer + validator (A + x% of B -> C)

Folder structure (LeRobot-like):
C/
  meta/
    episodes.jsonl
    tasks.jsonl
    info.json
    modality.json
  data/chunk-XXX/episode_000000.parquet
  videos/chunk-XXX/<camera_key>/episode_000000.mp4

Mixing rules:
- Select floor(x% * |B|) episodes from B.
- If x% > 0 but floor(...) == 0, round up to 1 with warning.
- Refuse to run if modality.json differs between A and B.
- Output C is a "fresh slate" dataset:
  - modality.json is written (copied from A)
  - tasks.jsonl is rebuilt (A tasks + new B tasks, dedup by string; "valid" is dedup)
  - episodes.jsonl is rebuilt (A + appended B entries, with updated episode_index)
  - info.json is rebuilt/updated

Index rules:
- Global frame index column `index` must be continuous across C from 0..N-1.
- Appended episodes' `episode_index` are shifted to continue after A.
- If `task_index` exists in parquet, it is remapped based on merged tasks (by task string).
- We only care about: episode_index, (optional episode_id untouched), global index, (optional task_index mapping).

Validation checks (your requirements):
1) total_frames / num_cam_keys == total parquet rows == global_index+1 (0-indexed)
   - We check BOTH:
     A) total_frames == total_rows * num_cam_keys
     B) (total_frames / num_cam_keys) == total_rows
2) actual %B in C (episode-based): appended_b / total_c * 100
   - also prints: appended_b / total_b * 100 (how much of B was used)
3) file names exist for every episode index in [0..total_episodes-1]:
   - parquet exists and has episode_index matching filename
   - for each camera key, mp4 exists with the same episode number
4) index integrity:
   - each parquet has `index` monotonic increasing by +1
   - across dataset, all indices form exactly {0..total_rows-1} (contiguous, no gaps/dupes)

Usage:
  Mix:
    python mix_lerobot.py mix \
      --dataset-a /path/A \
      --dataset-b /path/B \
      --out /path/C \
      --percent-b 10 \
      --seed 0 \
      --force

  Check:
    python mix_lerobot.py check --dataset /path/C

Notes:
- Requires pandas + pyarrow (for parquet).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Basic I/O
# ----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)

def ensure_empty_dir(path: Path, force: bool = False) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# ----------------------------
# Dataset paths & structure
# ----------------------------

@dataclass
class DatasetPaths:
    root: Path
    meta: Path
    data: Path
    videos: Path

def ds_paths(root: Path) -> DatasetPaths:
    return DatasetPaths(
        root=root,
        meta=root / "meta",
        data=root / "data",
        videos=root / "videos",
    )

def list_chunk_dirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    chunks = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("chunk-")]
    return sorted(chunks)

def find_episode_parquet(ds: DatasetPaths, episode_index: int) -> Path:
    name = f"episode_{episode_index:06d}.parquet"
    for chunk in list_chunk_dirs(ds.data):
        cand = chunk / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index} under {ds.data}")

def find_episode_video(ds: DatasetPaths, camera_key: str, episode_index: int) -> Path:
    name = f"episode_{episode_index:06d}.mp4"
    for chunk in list_chunk_dirs(ds.videos):
        cand = chunk / camera_key / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find video for episode {episode_index}, camera {camera_key} under {ds.videos}")

def extract_camera_keys_from_modality(modality: Dict[str, Any]) -> List[str]:
    """
    For your modality format:
      modality["video"] = { "ego_view": {...}, "left_wrist_view": {...}, ... }
    """
    video = modality.get("video", {})
    if isinstance(video, dict):
        keys = [k for k in video.keys() if isinstance(k, str)]
        return sorted(keys)
    return []

def compute_episodes_per_chunk(chunks_size_videos: Optional[int], num_cameras: int) -> int:
    """
    Interpret chunks_size as "max videos per chunk" (your example).
    episodes_per_chunk = floor(chunks_size / num_cameras), at least 1.
    If missing/invalid -> effectively no chunking.
    """
    if chunks_size_videos is None:
        return 10**12
    try:
        cs = int(chunks_size_videos)
    except Exception:
        return 10**12
    if cs <= 0 or num_cameras <= 0:
        return 10**12
    return max(1, cs // max(1, num_cameras))

def chunk_name(chunk_idx: int) -> str:
    return f"chunk-{chunk_idx:03d}"

def episode_to_chunk(ep_index: int, episodes_per_chunk: int) -> int:
    return ep_index // episodes_per_chunk


# ----------------------------
# Tasks merging / mapping
# ----------------------------

def load_tasks(tasks_jsonl: Path) -> List[Tuple[int, str]]:
    rows = read_jsonl(tasks_jsonl)
    out: List[Tuple[int, str]] = []
    for r in rows:
        if "task_index" in r and "task" in r:
            out.append((int(r["task_index"]), str(r["task"])))
    out.sort(key=lambda x: x[0])
    return out

def tasks_to_maps(tasks: List[Tuple[int, str]]) -> Tuple[Dict[int, str], Dict[str, int]]:
    idx2task = {i: t for i, t in tasks}
    task2idx = {t: i for i, t in tasks}
    return idx2task, task2idx

def merge_tasks(a_tasks: List[Tuple[int, str]], b_tasks: List[Tuple[int, str]]) -> Tuple[List[Tuple[int, str]], Dict[int, int]]:
    """
    Returns:
      - merged list [(new_idx, task_str), ...]
      - mapping old_b_idx -> new_idx
    Dedup by task string (so "valid" won't duplicate).
    """
    _, a_task2idx = tasks_to_maps(a_tasks)
    merged_task2idx = dict(a_task2idx)

    merged: List[Tuple[int, str]] = sorted([(i, t) for t, i in merged_task2idx.items()], key=lambda x: x[0])

    def next_index() -> int:
        return 0 if not merged else (max(i for i, _ in merged) + 1)

    b_old2new: Dict[int, int] = {}
    for old_i, task_str in b_tasks:
        if task_str in merged_task2idx:
            b_old2new[old_i] = merged_task2idx[task_str]
        else:
            ni = next_index()
            merged_task2idx[task_str] = ni
            merged.append((ni, task_str))
            b_old2new[old_i] = ni

    merged.sort(key=lambda x: x[0])
    return merged, b_old2new


# ----------------------------
# Mixing
# ----------------------------

def pick_b_episodes(total_b: int, percent: float, seed: int) -> Tuple[List[int], bool]:
    raw = (percent / 100.0) * total_b
    k = int(math.floor(raw))
    rounded_up = False
    if k == 0 and percent > 0:
        k = 1
        rounded_up = True
    k = min(k, total_b)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(total_b), k))
    return chosen, rounded_up

def ensure_chunk_dirs(dst: DatasetPaths, chunk_idx: int, camera_keys: List[str]) -> None:
    (dst.data / chunk_name(chunk_idx)).mkdir(parents=True, exist_ok=True)
    for cam in camera_keys:
        (dst.videos / chunk_name(chunk_idx) / cam).mkdir(parents=True, exist_ok=True)

def copy_episode_assets(
    src_ds: DatasetPaths,
    dst_ds: DatasetPaths,
    src_episode: int,
    dst_episode: int,
    camera_keys: List[str],
    dst_chunk_idx: int,
) -> Path:
    """Copies parquet + all camera videos. Returns dst parquet path."""
    # parquet
    src_parquet = find_episode_parquet(src_ds, src_episode)
    dst_parquet = dst_ds.data / chunk_name(dst_chunk_idx) / f"episode_{dst_episode:06d}.parquet"
    copy_file(src_parquet, dst_parquet)

    # videos
    for cam in camera_keys:
        src_vid = find_episode_video(src_ds, cam, src_episode)
        dst_vid = dst_ds.videos / chunk_name(dst_chunk_idx) / cam / f"episode_{dst_episode:06d}.mp4"
        copy_file(src_vid, dst_vid)

    return dst_parquet

def rewrite_parquet_indices(
    parquet_path: Path,
    new_episode_index: int,
    start_global_index: int,
    b_task_old2new: Optional[Dict[int, int]] = None,
) -> int:
    """
    Updates in-place:
      - episode_index
      - index (global, continuous)
      - task_index (if present and mapping provided)
    Returns number of rows (frames).
    """
    df = pd.read_parquet(parquet_path)
    n = len(df)

    # episode_index
    if "episode_index" in df.columns:
        df["episode_index"] = int(new_episode_index)
    else:
        df["episode_index"] = int(new_episode_index)

    # global index
    df["index"] = range(int(start_global_index), int(start_global_index) + n)

    # task_index remap
    if b_task_old2new is not None and "task_index" in df.columns:
        if pd.api.types.is_numeric_dtype(df["task_index"]):
            df["task_index"] = df["task_index"].map(lambda x: b_task_old2new.get(int(x), int(x)))

    df.to_parquet(parquet_path, index=False)
    return n

def mix_datasets(dataset_a: Path, dataset_b: Path, out: Path, percent_b: float, seed: int, force: bool) -> None:
    A = ds_paths(dataset_a)
    B = ds_paths(dataset_b)
    C = ds_paths(out)

    # Load modality and enforce equality
    a_mod = read_json(A.meta / "modality.json")
    b_mod = read_json(B.meta / "modality.json")
    if canonical_json(a_mod) != canonical_json(b_mod):
        raise SystemExit("Refusing to mix: modality.json differs between A and B.")

    camera_keys = extract_camera_keys_from_modality(a_mod)
    if not camera_keys:
        raise SystemExit("Could not extract camera keys from modality.json (expected modality['video']).")
    num_cams = len(camera_keys)

    # Load meta
    a_info = read_json(A.meta / "info.json")
    b_info = read_json(B.meta / "info.json")
    a_eps = read_jsonl(A.meta / "episodes.jsonl")
    b_eps = read_jsonl(B.meta / "episodes.jsonl")
    a_tasks = load_tasks(A.meta / "tasks.jsonl")
    b_tasks = load_tasks(B.meta / "tasks.jsonl")

    total_a = len(a_eps) if a_eps else int(a_info.get("total_episodes", 0))
    total_b = len(b_eps) if b_eps else int(b_info.get("total_episodes", 0))
    if total_a <= 0 or total_b <= 0:
        raise SystemExit("Invalid total episodes (check episodes.jsonl / info.json).")

    chosen_b, rounded_up = pick_b_episodes(total_b, percent_b, seed)
    if rounded_up:
        print(f"[WARN] percent-b={percent_b}% floors to 0 episodes; rounding up to 1.")

    print(f"[INFO] A episodes={total_a}, B episodes={total_b}, chosen from B={len(chosen_b)}")
    print(f"[INFO] Camera keys ({num_cams}): {camera_keys}")

    # Determine chunking (using A's chunks_size)
    episodes_per_chunk = compute_episodes_per_chunk(a_info.get("chunks_size", None), num_cams)

    # Create fresh output
    ensure_empty_dir(C.root, force=force)
    (C.meta).mkdir(parents=True, exist_ok=True)

    # Write modality.json (fresh, but based on A)
    write_json(C.meta / "modality.json", a_mod)

    # Merge tasks
    merged_tasks, b_task_old2new = merge_tasks(a_tasks, b_tasks)
    out_tasks_rows = [{"task_index": i, "task": t} for i, t in merged_tasks]
    write_jsonl(C.meta / "tasks.jsonl", out_tasks_rows)

    # Copy A episodes (assets) and rewrite parquet indices to ensure global index starts at 0
    # We will rewrite ALL parquets in C to enforce a single clean contiguous global index from 0..N-1.
    global_cursor = 0
    out_episodes: List[Dict[str, Any]] = []

    for ep in range(total_a):
        ch = episode_to_chunk(ep, episodes_per_chunk)
        ensure_chunk_dirs(C, ch, camera_keys)
        dst_parquet = copy_episode_assets(A, C, src_episode=ep, dst_episode=ep, camera_keys=camera_keys, dst_chunk_idx=ch)

        # rewrite parquet (episode_index unchanged; global index rewritten)
        n = rewrite_parquet_indices(dst_parquet, new_episode_index=ep, start_global_index=global_cursor, b_task_old2new=None)
        global_cursor += n

        # episode meta row
        row = a_eps[ep] if ep < len(a_eps) else {"episode_index": ep, "tasks": [], "length": n}
        row = dict(row)
        row["episode_index"] = ep
        row["length"] = int(row.get("length", n))
        out_episodes.append(row)

    # Append selected B episodes
    appended_b_frames = 0
    appended_b = 0

    for src_ep in chosen_b:
        dst_ep = total_a + appended_b
        ch = episode_to_chunk(dst_ep, episodes_per_chunk)
        ensure_chunk_dirs(C, ch, camera_keys)
        dst_parquet = copy_episode_assets(B, C, src_episode=src_ep, dst_episode=dst_ep, camera_keys=camera_keys, dst_chunk_idx=ch)

        n = rewrite_parquet_indices(dst_parquet, new_episode_index=dst_ep, start_global_index=global_cursor, b_task_old2new=b_task_old2new)
        global_cursor += n
        appended_b_frames += n
        appended_b += 1

        row = b_eps[src_ep] if src_ep < len(b_eps) else {"episode_index": dst_ep, "tasks": [], "length": n}
        row = dict(row)
        row["episode_index"] = dst_ep
        row["length"] = int(row.get("length", n))
        out_episodes.append(row)

    write_jsonl(C.meta / "episodes.jsonl", out_episodes)

    # Build info.json fresh-ish using A as base for stable fields
    out_total_episodes = total_a + appended_b
    out_total_rows = global_cursor  # parquet rows == global index + 1
    out_total_frames = out_total_rows * num_cams  # per your convention
    out_total_videos = out_total_episodes * num_cams

    out_info = dict(a_info)
    out_info["total_episodes"] = out_total_episodes
    out_info["total_tasks"] = len(merged_tasks)
    out_info["total_videos"] = out_total_videos
    out_info["total_frames"] = out_total_frames
    out_info["total_chunks"] = int(math.ceil(out_total_episodes / episodes_per_chunk)) if episodes_per_chunk < 10**11 else 0
    out_info["chunks_size"] = int(a_info.get("chunks_size", out_total_videos))
    write_json(C.meta / "info.json", out_info)

    print("[DONE] Mix complete:", C.root)
    print(f"       Episodes: {out_total_episodes} (A={total_a}, appended B={appended_b})")
    print(f"       Rows:     {out_total_rows} (global index 0..{out_total_rows-1})")
    print(f"       Frames:   {out_total_frames} (= rows * num_cams)")
    print(f"       Videos:   {out_total_videos}")
    print(f"       %B in C (episodes): {appended_b/out_total_episodes*100:.3f}%")
    print(f"       %B used from B:     {appended_b/total_b*100:.3f}%")
    if appended_b:
        print(f"       Picked B episodes: {chosen_b}")

    # Auto-run checks
    print("\n[INFO] Running validation on output...")
    check_dataset(C.root)


# ----------------------------
# Validation (your requested checks)
# ----------------------------

def iter_all_parquets(ds: DatasetPaths) -> List[Path]:
    files: List[Path] = []
    for ch in list_chunk_dirs(ds.data):
        files.extend(sorted(ch.glob("episode_*.parquet")))
    return sorted(files)

def parse_episode_idx_from_name(p: Path) -> int:
    # episode_000123.parquet
    stem = p.stem  # episode_000123
    return int(stem.split("_")[1])

def check_dataset(dataset: Path) -> None:
    D = ds_paths(dataset)
    info = read_json(D.meta / "info.json")
    modality = read_json(D.meta / "modality.json")
    camera_keys = extract_camera_keys_from_modality(modality)
    num_cams = len(camera_keys)

    eps_rows = read_jsonl(D.meta / "episodes.jsonl")
    total_episodes = int(info.get("total_episodes", len(eps_rows)))
    total_frames = int(info.get("total_frames", -1))

    # 0) Basic file existence / episode sequence
    expected_eps = list(range(total_episodes))
    missing_parquets: List[int] = []
    missing_videos: List[Tuple[int, str]] = []

    # For checking "numerical order": ensure all expected exist; chunking can be arbitrary, so we search.
    for ep in expected_eps:
        try:
            pq = find_episode_parquet(D, ep)
        except FileNotFoundError:
            missing_parquets.append(ep)
            pq = None

        for cam in camera_keys:
            try:
                _ = find_episode_video(D, cam, ep)
            except FileNotFoundError:
                missing_videos.append((ep, cam))

        # If parquet exists, check episode_index matches filename
        if pq is not None:
            df = pd.read_parquet(pq, columns=["episode_index", "index"])
            if len(df) > 0:
                ep_col = int(df["episode_index"].iloc[0])
                if ep_col != ep:
                    raise SystemExit(f"[FAIL] episode_index mismatch: file episode_{ep:06d}.parquet has episode_index={ep_col}")

    if missing_parquets:
        raise SystemExit(f"[FAIL] Missing parquet episodes: {missing_parquets[:20]}{'...' if len(missing_parquets)>20 else ''}")
    if missing_videos:
        head = missing_videos[:20]
        raise SystemExit(f"[FAIL] Missing videos (episode,camera): {head}{'...' if len(missing_videos)>20 else ''}")

    # 1) Total rows and global index contiguity
    parquets = iter_all_parquets(D)
    if len(parquets) != total_episodes:
        print(f"[WARN] Found {len(parquets)} parquet files but info.json total_episodes={total_episodes}.")

    total_rows = 0
    min_idx = None
    max_idx = None
    seen_count = 0

    # To avoid huge memory, we do range checks + accumulate duplicates/gaps via arithmetic checks:
    # We confirm:
    #  - indices within each parquet are consecutive (+1)
    #  - global min==0
    #  - global max==total_rows-1
    #  - sum of lengths matches (max-min+1) and no overlaps by checking that each episode index range
    #    exactly matches expected cursor progression (stronger and cheaper).
    cursor = 0
    for pq in sorted(parquets, key=lambda p: parse_episode_idx_from_name(p)):
        df = pd.read_parquet(pq, columns=["index"])
        n = len(df)
        total_rows += n
        if n == 0:
            print(f"[WARN] Empty parquet: {pq}")
            continue

        idx0 = int(df["index"].iloc[0])
        idx1 = int(df["index"].iloc[-1])

        # per-parquet consecutive
        expected_last = idx0 + (n - 1)
        if idx1 != expected_last:
            raise SystemExit(f"[FAIL] Non-consecutive index within {pq.name}: first={idx0}, last={idx1}, n={n}")

        # global cursor must match
        if idx0 != cursor:
            raise SystemExit(f"[FAIL] Global index gap/overlap at {pq.name}: expected first index {cursor}, got {idx0}")

        cursor = idx1 + 1

        min_idx = idx0 if min_idx is None else min(min_idx, idx0)
        max_idx = idx1 if max_idx is None else max(max_idx, idx1)
        seen_count += n

    if min_idx != 0:
        raise SystemExit(f"[FAIL] Global index should start at 0, got min index = {min_idx}")
    if max_idx != total_rows - 1:
        raise SystemExit(f"[FAIL] Global index max mismatch: max_idx={max_idx}, but total_rows={total_rows} so expected {total_rows-1}")

    # 2) Your total_frames / num_cam == total_rows check
    if num_cams <= 0:
        raise SystemExit("[FAIL] num_cam_keys is 0 (could not extract camera keys from modality.json).")

    # A) total_frames == total_rows * num_cams
    expected_total_frames = total_rows * num_cams
    if total_frames != expected_total_frames:
        raise SystemExit(
            "[FAIL] total_frames mismatch:\n"
            f"       info.json total_frames={total_frames}\n"
            f"       expected total_frames=total_rows*num_cams={total_rows}*{num_cams}={expected_total_frames}"
        )

    # B) total_frames / num_cams == total_rows
    if (total_frames // num_cams) != total_rows or (total_frames % num_cams) != 0:
        raise SystemExit(
            "[FAIL] total_frames/num_cam_keys mismatch:\n"
            f"       total_frames={total_frames}, num_cams={num_cams}, total_frames/num_cams={total_frames/num_cams}\n"
            f"       total_rows={total_rows}"
        )

    # 3) %B in A mixture check (episode-based) — needs provenance fields
    # We support optional provenance in info.json written by you, but if missing, we just print N/A.
    # (You can extend: store {"mixing": {"a_episodes":..., "b_total":..., "b_selected":...}} )
    mixing = info.get("mixing", None)
    if isinstance(mixing, dict):
        a_eps = int(mixing.get("a_episodes", -1))
        b_total = int(mixing.get("b_total_episodes", -1))
        b_sel = int(mixing.get("b_selected_episodes", -1))
        if a_eps >= 0 and b_sel >= 0 and (a_eps + b_sel) > 0:
            pct_b_in_c = 100.0 * b_sel / (a_eps + b_sel)
            pct_b_used = 100.0 * b_sel / b_total if b_total > 0 else float("nan")
            print(f"[OK] %B in C (episodes): {pct_b_in_c:.3f}%")
            print(f"[OK] %B used from B:     {pct_b_used:.3f}%")
        else:
            print("[WARN] info.json has mixing{}, but fields are incomplete.")
    else:
        print("[INFO] %B checks: N/A (no provenance stored in info.json).")

    # 4) Filename numeric order (already ensured existence) — also ensure zero-padded naming is consistent
    # We check that for each episode index, we can locate the correct filename exactly:
    for ep in expected_eps:
        expected_pq_name = f"episode_{ep:06d}.parquet"
        pq = find_episode_parquet(D, ep)
        if pq.name != expected_pq_name:
            raise SystemExit(f"[FAIL] Parquet filename mismatch for episode {ep}: found {pq.name}, expected {expected_pq_name}")
        for cam in camera_keys:
            expected_mp4 = f"episode_{ep:06d}.mp4"
            mp4 = find_episode_video(D, cam, ep)
            if mp4.name != expected_mp4:
                raise SystemExit(f"[FAIL] Video filename mismatch ep={ep}, cam={cam}: found {mp4.name}, expected {expected_mp4}")

    print("[OK] Dataset validation PASSED")
    print(f"     total_episodes={total_episodes}, num_cams={num_cams}, total_rows={total_rows}, total_frames={total_frames}")


# ----------------------------
# CLI
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    mixp = sub.add_parser("mix", help="Mix A + x% of B -> C and validate")
    mixp.add_argument("--dataset-a", type=Path, required=True)
    mixp.add_argument("--dataset-b", type=Path, required=True)
    mixp.add_argument("--out", type=Path, required=True)
    mixp.add_argument("--percent-b", type=float, required=True, help="0 < x <= 100")
    mixp.add_argument("--seed", type=int, default=0)
    mixp.add_argument("--force", action="store_true")

    chk = sub.add_parser("check", help="Validate an existing dataset")
    chk.add_argument("--dataset", type=Path, required=True)

    return p

def main():
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "mix":
        if not (0 < args.percent_b <= 100.0):
            raise SystemExit("--percent-b must be in (0, 100].")
        mix_datasets(
            dataset_a=args.dataset_a,
            dataset_b=args.dataset_b,
            out=args.out,
            percent_b=args.percent_b,
            seed=args.seed,
            force=args.force,
        )
    elif args.cmd == "check":
        check_dataset(args.dataset)

if __name__ == "__main__":
    main()