#!/usr/bin/env python3
"""
LeRobot dataset mixer + validator (A + x% of B -> C) — CAPSTONE-READY

Key updates vs earlier version:
- Chunking is EPISODE-BASED per LeRobot v2 convention:
    episode_chunk = floor(episode_index / chunks_size)
  so chunks_size is interpreted as "episodes per chunk" (e.g. 900).
- total_chunks is ALWAYS >= 1 if total_episodes > 0:
    total_chunks = max(1, ceil(total_episodes / chunks_size))
- Writes provenance ("mixing") into info.json so %B checks always work later.
- Validator warns if info.json total_chunks mismatches expected.

Folder structure:
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
- Output C is fresh slate: only the required meta files above are created.

Index rules:
- Global frame index column `index` is rewritten to be contiguous 0..N-1 across C.
- episode_index is rewritten to match file numbering.
- If task_index exists in parquet, it is remapped based on merged tasks (dedup by string).

Validation checks:
1) total_frames == total_rows * num_cam_keys AND total_frames/num_cam_keys == total_rows
2) %B in C (episodes) and %B used from B (episodes), using provenance
3) File existence and naming consistency for parquet + all camera mp4s
4) index integrity: per-episode consecutive (+1), and across dataset no gaps/overlaps, starts at 0
5) total_chunks sanity: warns if mismatch vs expected

Usage:
  Mix:
    python data_mixer_capstone.py mix \
      --dataset-a /path/A \
      --dataset-b /path/B \
      --out /path/C \
      --percent-b 10 \
      --seed 0 \
      --force

  Check:
    python data_mixer_capstone.py check --dataset /path/C
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

def resolve_video_dirnames(ds: DatasetPaths, camera_keys: List[str]) -> Dict[str, str]:
    """
    Map logical camera keys (e.g. 'ego_view') to actual directory names on disk.

    Preference order (important to avoid double-prefix bugs):
    1) exact: 'observation.images.{key}'
    2) exact: '{key}'
    3) single suffix match: endswith('.{key}')
    4) single contains match
    """
    chunks = list_chunk_dirs(ds.videos)
    if not chunks:
        raise FileNotFoundError(f"No chunk-* directories found under {ds.videos}")

    # use first chunk that exists (usually chunk-000)
    chunk0 = chunks[0]
    subdirs = [p.name for p in chunk0.iterdir() if p.is_dir()]

    mapping: Dict[str, str] = {}
    for key in camera_keys:
        preferred = f"observation.images.{key}"
        if preferred in subdirs:
            mapping[key] = preferred
            continue
        if key in subdirs:
            mapping[key] = key
            continue

        suffix_matches = [d for d in subdirs if d.endswith(f".{key}")]
        if len(suffix_matches) == 1:
            mapping[key] = suffix_matches[0]
            continue
        if len(suffix_matches) > 1:
            # still ambiguous — show user candidates
            raise SystemExit(
                f"Ambiguous video dirs for camera key '{key}': {suffix_matches}. "
                f"Preferred '{preferred}'. Please delete/rename the extra folder(s)."
            )

        contains_matches = [d for d in subdirs if key in d]
        if len(contains_matches) == 1:
            mapping[key] = contains_matches[0]
            continue
        if len(contains_matches) > 1:
            raise SystemExit(
                f"Ambiguous video dirs for camera key '{key}': {contains_matches}. "
                f"Preferred '{preferred}'. Please delete/rename the extra folder(s)."
            )

        raise FileNotFoundError(
            f"Could not resolve a video directory for camera key '{key}'. "
            f"Found subdirs in {chunk0}: {subdirs}"
        )

    return mapping



def list_chunk_dirs(parent: Path) -> List[Path]:
    if not parent.exists():
        return []
    chunks = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("chunk-")]
    return sorted(chunks)

def chunk_name(chunk_idx: int) -> str:
    return f"chunk-{chunk_idx:03d}"

def episodes_per_chunk_from_info(info: Dict[str, Any], default_chunks_size: int = 900) -> int:
    """
    LeRobot v2 convention: chunks_size = EPISODES PER CHUNK.
    If missing/invalid, fall back to default (900).
    """
    cs = info.get("chunks_size", default_chunks_size)
    try:
        cs = int(cs)
    except Exception:
        cs = default_chunks_size
    return max(1, cs)

def episode_to_chunk(ep_index: int, episodes_per_chunk: int) -> int:
    # episode_chunk = floor(episode_index / chunks_size)
    return ep_index // episodes_per_chunk

def ensure_chunk_dirs(dst: DatasetPaths, chunk_idx: int, camera_keys: List[str]) -> None:
    """Create chunk directories for data and videos.
    Accepts either raw keys ('ego_view') or pre-prefixed ('observation.images.ego_view').
    Always strips any existing prefix before re-applying it to avoid double-prefix dirs.
    """
    (dst.data / chunk_name(chunk_idx)).mkdir(parents=True, exist_ok=True)
    for cam in camera_keys:
        raw_cam = cam[len("observation.images."):] if cam.startswith("observation.images.") else cam
        (dst.videos / chunk_name(chunk_idx) / f"observation.images.{raw_cam}").mkdir(parents=True, exist_ok=True)

def find_episode_parquet(ds: DatasetPaths, episode_index: int) -> Path:
    name = f"episode_{episode_index:06d}.parquet"
    for chunk in list_chunk_dirs(ds.data):
        cand = chunk / name
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not find parquet for episode {episode_index} under {ds.data}")

def find_episode_video(ds: DatasetPaths, camera_key: str, episode_index: int) -> Path:
    name = f"episode_{episode_index:06d}.mp4"
    raw_key = camera_key[len("observation.images."):] if camera_key.startswith("observation.images.") else camera_key
    cam_dir = f"observation.images.{raw_key}"
    for chunk in list_chunk_dirs(ds.videos):
        cand = chunk / cam_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find video for episode {episode_index}, camera {camera_key} under {ds.videos} "
        f"(expected folder '{cam_dir}')"
    )

def extract_camera_keys_from_modality(modality: Dict[str, Any]) -> List[str]:
    """
    Your modality format:
      modality["video"] = { "ego_view": {...}, "left_wrist_view": {...}, ... }
    """
    video = modality.get("video", {})
    if isinstance(video, dict):
        keys = [k for k in video.keys() if isinstance(k, str)]
        return sorted(keys)
    return []


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
# Mixing core
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

def copy_episode_assets(
    src_ds: DatasetPaths,
    dst_ds: DatasetPaths,
    src_episode: int,
    dst_episode: int,
    camera_keys: List[str],
    video_dir_map: Dict[str, str],
    dst_chunk_idx: int,
) -> Path:
    """Copies parquet + all camera videos. Returns dst parquet path."""

    # Copy parquet
    src_parquet = find_episode_parquet(src_ds, src_episode)
    dst_parquet = dst_ds.data / chunk_name(dst_chunk_idx) / f"episode_{dst_episode:06d}.parquet"
    copy_file(src_parquet, dst_parquet)

    # Copy videos (ALL cameras)
    for cam in camera_keys:
        src_dir = video_dir_map[cam]  # e.g. "observation.images.ego_view"
        src_vid = find_episode_video(src_ds, cam, src_episode)

        dst_vid = (
            dst_ds.videos
            / chunk_name(dst_chunk_idx)
            / src_dir
            / f"episode_{dst_episode:06d}.mp4"
        )
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

    df["episode_index"] = int(new_episode_index)
    df["index"] = range(int(start_global_index), int(start_global_index) + n)

    if b_task_old2new is not None and "task_index" in df.columns and pd.api.types.is_numeric_dtype(df["task_index"]):
        df["task_index"] = df["task_index"].map(lambda x: b_task_old2new.get(int(x), int(x)))

    df.to_parquet(parquet_path, index=False)
    return n

def compute_total_chunks(total_episodes: int, episodes_per_chunk: int) -> int:
    if total_episodes <= 0:
        return 0
    return max(1, int(math.ceil(total_episodes / episodes_per_chunk)))

def mix_datasets(dataset_a: Path, dataset_b: Path, out: Path, percent_b: float, seed: int, force: bool) -> None:
    A = ds_paths(dataset_a)
    B = ds_paths(dataset_b)
    C = ds_paths(out)

    a_mod = read_json(A.meta / "modality.json")
    b_mod = read_json(B.meta / "modality.json")
    if canonical_json(a_mod) != canonical_json(b_mod):
        raise SystemExit("Refusing to mix: modality.json differs between A and B.")

    camera_keys = extract_camera_keys_from_modality(a_mod)

    if not camera_keys:
        raise SystemExit("Could not extract camera keys from modality.json (expected modality['video']).")
    video_dir_map = resolve_video_dirnames(A, camera_keys)
    print("[INFO] Video directory mapping:")
    for k in camera_keys:
        print(f"       {k} -> {video_dir_map[k]}")
    num_cams = len(camera_keys)

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

    # EPISODE-BASED chunking
    episodes_per_chunk = episodes_per_chunk_from_info(a_info, default_chunks_size=900)
    print(f"[INFO] chunks_size (episodes per chunk) = {episodes_per_chunk}")

    # Fresh output
    ensure_empty_dir(C.root, force=force)
    C.meta.mkdir(parents=True, exist_ok=True)

    # modality.json
    write_json(C.meta / "modality.json", a_mod)

    # tasks.jsonl (merged)
    merged_tasks, b_task_old2new = merge_tasks(a_tasks, b_tasks)
    write_jsonl(C.meta / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in merged_tasks])

    global_cursor = 0
    out_episodes: List[Dict[str, Any]] = []

    # Copy + rewrite A (rewrite global index too, to guarantee clean 0..N-1)
    for ep in range(total_a):
        ch = episode_to_chunk(ep, episodes_per_chunk)
        video_dirnames = [video_dir_map[k] for k in camera_keys]
        ensure_chunk_dirs(C, ch, video_dirnames)

        dst_parquet = copy_episode_assets(
            A, C,
            src_episode=ep, dst_episode=ep,
            camera_keys=camera_keys,
            video_dir_map=video_dir_map,
            dst_chunk_idx=ch
        )
        n = rewrite_parquet_indices(dst_parquet, new_episode_index=ep, start_global_index=global_cursor, b_task_old2new=None)
        global_cursor += n

        row = a_eps[ep] if ep < len(a_eps) else {"episode_index": ep, "tasks": [], "length": n}
        row = dict(row)
        row["episode_index"] = ep
        row["length"] = int(row.get("length", n))
        out_episodes.append(row)

    # Resolve B's video directory mapping (may differ from A's if B has different naming)
    video_dir_map_b = resolve_video_dirnames(B, camera_keys)

    # Append selected B episodes
    appended_b = 0
    appended_b_frames = 0
    for src_ep in chosen_b:
        dst_ep = total_a + appended_b
        ch = episode_to_chunk(dst_ep, episodes_per_chunk)
        # Only call ensure_chunk_dirs ONCE with the already-prefixed dirnames
        video_dirnames = [video_dir_map[k] for k in camera_keys]
        ensure_chunk_dirs(C, ch, video_dirnames)

        dst_parquet = copy_episode_assets(
            B, C,                          # FIX: copy FROM B, not A
            src_episode=src_ep,            # FIX: use src_ep (B episode index), not stale ep
            dst_episode=dst_ep,
            camera_keys=camera_keys,
            video_dir_map=video_dir_map_b, # FIX: use B's video dir mapping
            dst_chunk_idx=ch
        )
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

    # info.json (fresh-ish)
    out_total_episodes = total_a + appended_b
    out_total_rows = global_cursor
    out_total_frames = out_total_rows * num_cams
    out_total_videos = out_total_episodes * num_cams

    out_info = dict(a_info)  # keep stable fields like robot_type, fps, codebase_version, etc.
    out_info["total_episodes"] = out_total_episodes
    out_info["total_tasks"] = len(merged_tasks)
    out_info["total_videos"] = out_total_videos
    out_info["total_frames"] = out_total_frames
    out_info["chunks_size"] = episodes_per_chunk

    out_info["total_chunks"] = compute_total_chunks(out_total_episodes, episodes_per_chunk)

    # Provenance (so checks always work later)
    out_info["mixing"] = {
        "a_episodes": total_a,
        "b_total_episodes": total_b,
        "b_selected_episodes": appended_b,
        "percent_b_requested": float(percent_b),
        "seed": int(seed),
        "chosen_b_episodes": chosen_b,
    }

    write_json(C.meta / "info.json", out_info)

    print("[DONE] Mix complete:", C.root)
    print(f"       Episodes: {out_total_episodes} (A={total_a}, appended B={appended_b})")
    print(f"       Rows:     {out_total_rows} (global index 0..{out_total_rows-1})")
    print(f"       Frames:   {out_total_frames} (= rows * num_cams)")
    print(f"       Videos:   {out_total_videos}")
    print(f"       total_chunks: {out_info['total_chunks']}  (chunks_size={episodes_per_chunk})")
    print(f"       %B in C (episodes): {appended_b/out_total_episodes*100:.3f}%")
    print(f"       %B used from B:     {appended_b/total_b*100:.3f}%")

    print("\n[INFO] Running validation on output...")
    check_dataset(C.root)


# ----------------------------
# Validation
# ----------------------------

def iter_all_parquets(ds: DatasetPaths) -> List[Path]:
    files: List[Path] = []
    for ch in list_chunk_dirs(ds.data):
        files.extend(sorted(ch.glob("episode_*.parquet")))
    return sorted(files)

def parse_episode_idx_from_name(p: Path) -> int:
    # episode_000123.parquet
    return int(p.stem.split("_")[1])

def check_dataset(dataset: Path) -> None:
    D = ds_paths(dataset)

    info = read_json(D.meta / "info.json")
    modality = read_json(D.meta / "modality.json")
    camera_keys = extract_camera_keys_from_modality(modality)
    num_cams = len(camera_keys)
    video_dir_map = resolve_video_dirnames(D, camera_keys)
    eps_rows = read_jsonl(D.meta / "episodes.jsonl")
    total_episodes = int(info.get("total_episodes", len(eps_rows)))
    total_frames = int(info.get("total_frames", -1))

    if total_episodes <= 0:
        raise SystemExit("[FAIL] total_episodes <= 0")
    
    # chunks_size semantics: episodes per chunk
    chunks_size = episodes_per_chunk_from_info(info, default_chunks_size=900)
    expected_total_chunks = compute_total_chunks(total_episodes, chunks_size)
    info_total_chunks = int(info.get("total_chunks", -1))
    if info_total_chunks != expected_total_chunks:
        print(f"[WARN] total_chunks mismatch: info.json={info_total_chunks}, expected={expected_total_chunks} (total_episodes={total_episodes}, chunks_size={chunks_size})")

    # 0) File existence + episode_index matches filename
    expected_eps = list(range(total_episodes))
    missing_parquets: List[int] = []
    missing_videos: List[Tuple[int, str]] = []

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

        if pq is not None:
            df = pd.read_parquet(pq, columns=["episode_index", "index"])
            if len(df) > 0:
                ep_col = int(df["episode_index"].iloc[0])
                if ep_col != ep:
                    raise SystemExit(f"[FAIL] episode_index mismatch: {pq.name} has episode_index={ep_col}, expected {ep}")

    if missing_parquets:
        raise SystemExit(f"[FAIL] Missing parquet episodes: {missing_parquets[:20]}{'...' if len(missing_parquets)>20 else ''}")
    if missing_videos:
        head = missing_videos[:20]
        raise SystemExit(f"[FAIL] Missing videos (episode,camera): {head}{'...' if len(missing_videos)>20 else ''}")

    # 1) Global index integrity: contiguous 0..N-1 with no gaps/overlaps
    parquets = iter_all_parquets(D)
    if len(parquets) != total_episodes:
        print(f"[WARN] Found {len(parquets)} parquet files but info.json total_episodes={total_episodes}.")

    total_rows = 0
    cursor = 0
    min_idx = None
    max_idx = None

    for pq in sorted(parquets, key=parse_episode_idx_from_name):
        df = pd.read_parquet(pq, columns=["index"])
        n = len(df)
        total_rows += n
        if n == 0:
            print(f"[WARN] Empty parquet: {pq}")
            continue

        idx0 = int(df["index"].iloc[0])
        idx1 = int(df["index"].iloc[-1])

        # per-parquet consecutive
        if idx1 != idx0 + (n - 1):
            raise SystemExit(f"[FAIL] Non-consecutive index within {pq.name}: first={idx0}, last={idx1}, n={n}")

        # cursor continuity
        if idx0 != cursor:
            raise SystemExit(f"[FAIL] Global index gap/overlap at {pq.name}: expected first index {cursor}, got {idx0}")

        cursor = idx1 + 1
        min_idx = idx0 if min_idx is None else min(min_idx, idx0)
        max_idx = idx1 if max_idx is None else max(max_idx, idx1)

    if min_idx != 0:
        raise SystemExit(f"[FAIL] Global index should start at 0, got min index={min_idx}")
    if max_idx != total_rows - 1:
        raise SystemExit(f"[FAIL] Global index max mismatch: max_idx={max_idx}, expected={total_rows-1}")

    # 2) total_frames checks
    if num_cams <= 0:
        raise SystemExit("[FAIL] num_cam_keys is 0 (could not extract camera keys from modality.json).")

    expected_total_frames = total_rows * num_cams
    if total_frames != expected_total_frames:
        raise SystemExit(
            "[FAIL] total_frames mismatch:\n"
            f"       info.json total_frames={total_frames}\n"
            f"       expected total_frames=total_rows*num_cams={total_rows}*{num_cams}={expected_total_frames}"
        )

    if (total_frames // num_cams) != total_rows or (total_frames % num_cams) != 0:
        raise SystemExit(
            "[FAIL] total_frames/num_cam_keys mismatch:\n"
            f"       total_frames={total_frames}, num_cams={num_cams}, total_frames/num_cams={total_frames/num_cams}\n"
            f"       total_rows={total_rows}"
        )

    # 3) %B checks from provenance
    mixing = info.get("mixing", {})
    if isinstance(mixing, dict) and "a_episodes" in mixing and "b_selected_episodes" in mixing and "b_total_episodes" in mixing:
        a_eps = int(mixing["a_episodes"])
        b_sel = int(mixing["b_selected_episodes"])
        b_total = int(mixing["b_total_episodes"])
        pct_b_in_c = 100.0 * b_sel / max(1, (a_eps + b_sel))
        pct_b_used = 100.0 * b_sel / max(1, b_total)
        print(f"[OK] %B in C (episodes): {pct_b_in_c:.3f}%")
        print(f"[OK] %B used from B:     {pct_b_used:.3f}%")
    else:
        print("[WARN] Provenance missing: cannot compute %B reliably (mixing{} not found in info.json).")

    # 4) Filename exactness (zero-pad check)
    for ep in expected_eps:
        expected_pq_name = f"episode_{ep:06d}.parquet"
        pq = find_episode_parquet(D, ep)
        if pq.name != expected_pq_name:
            raise SystemExit(f"[FAIL] Parquet filename mismatch ep={ep}: found {pq.name}, expected {expected_pq_name}")

        for cam in camera_keys:
            expected_mp4 = f"episode_{ep:06d}.mp4"
            mp4 = find_episode_video(D, video_dir_map[cam], ep)
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