#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 output to LeRobot dataset format for KuavoV4Pro pouring task.

HDF5 structure (per demo):
    data/demo_X/
        actions                  # (T, 34) commanded joint positions
        obs/
            actions              # (T, 34) last action = state proxy
            cam_egoview_rgb      # (T, H, W, 4) RGBA
            cam_leftwrist_rgb    # (T, H, W, 4) RGBA
            cam_rightwrist_rgb   # (T, H, W, 4) RGBA
            hand_joint_state     # (T, 20) finger joints
            left_eef_pos         # (T, 3)
            left_eef_quat        # (T, 4)
            right_eef_pos        # (T, 3)
            right_eef_quat       # (T, 4)
            robot_joint_pos      # (T, N_all_joints) full articulation

LeRobot 44D vector layout (from modality.json):
    [0:7]   left_arm    -> HDF5 actions[0:7]   (zarm_l1-l7)
    [7:13]  left_hand   -> HDF5 actions[14:24] extract [0,1,2,4,6,8] (10->6D)
    [13:19] left_leg    -> ZEROS
    [19:22] neck        -> ZEROS
    [22:29] right_arm   -> HDF5 actions[7:14]  (zarm_r1-r7)
    [29:35] right_hand  -> HDF5 actions[24:34] extract [0,1,2,4,6,8] (10->6D)
    [35:41] right_leg   -> ZEROS
    [41:44] waist       -> ZEROS

Hand 10-sim -> 6-real mapping (extract MCP joints, drop PIP duplicates):
    sim[0] thumbCMC  -> real[0] thumb_rot
    sim[1] thumbMCP  -> real[1] thumb_bend
    sim[2] indexMCP  -> real[2] index     (drop sim[3] indexPIP)
    sim[4] middleMCP -> real[3] middle    (drop sim[5] middlePIP)
    sim[6] ringMCP   -> real[4] ring      (drop sim[7] ringPIP)
    sim[8] littleMCP -> real[5] little    (drop sim[9] littlePIP)

observation.state uses obs/actions (last_action, time t)
action uses data/demo_X/actions (commanded, time t)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Indices into the 10D sim hand array to select the 6 real DOF
HAND_SIM_TO_REAL_IDX = [0, 1, 2, 4, 6, 8]

# 44D LeRobot state/action vector layout
STATE_DIM = 44
# Segment slices into the 44D vector
SEG_LEFT_ARM   = slice(0, 7)    # 7 DOF
SEG_LEFT_HAND  = slice(7, 13)   # 6 DOF
SEG_LEFT_LEG   = slice(13, 19)  # 6 DOF - zeros
SEG_NECK       = slice(19, 22)  # 3 DOF - zeros
SEG_RIGHT_ARM  = slice(22, 29)  # 7 DOF
SEG_RIGHT_HAND = slice(29, 35)  # 6 DOF
SEG_RIGHT_LEG  = slice(35, 41)  # 6 DOF - zeros
SEG_WAIST      = slice(41, 44)  # 3 DOF - zeros

# Camera key mapping: HDF5 key -> LeRobot video folder name
CAMERA_MAP = {
    "cam_egoview_rgb":    "observation.images.ego_view",
    "cam_leftwrist_rgb":  "observation.images.left_wrist_view",
    "cam_rightwrist_rgb": "observation.images.right_wrist_view",
}

# Chunk size (episodes per chunk folder)
CHUNK_SIZE = 900

# Simulation dt and decimation (from env cfg: dt=1/100, decimation=5)
# So each recorded step = decimation * dt = 5 * 0.01 = 0.05s
SIM_DT = 1.0 / 100.0
DECIMATION = 5
STEP_DT = SIM_DT * DECIMATION  # 0.05s per step


# ─────────────────────────────────────────────
# Core conversion helpers
# ─────────────────────────────────────────────

def sim_34d_to_lerobot_44d(sim_actions: np.ndarray) -> np.ndarray:
    """
    Convert (T, 34) Isaac Lab action array to (T, 44) LeRobot vector.

    Input layout (34D):
        [0:7]   left arm  (zarm_l1-l7)
        [7:14]  right arm (zarm_r1-r7)
        [14:24] left hand (10 sim DOF)
        [24:34] right hand (10 sim DOF)

    Output layout (44D): see modality.json
    """
    T = sim_actions.shape[0]
    out = np.zeros((T, STATE_DIM), dtype=np.float32)

    # Arms - direct copy
    out[:, SEG_LEFT_ARM]  = sim_actions[:, 0:7]
    out[:, SEG_RIGHT_ARM] = sim_actions[:, 7:14]

    # Hands - 10D sim -> 6D real via MCP index selection
    left_hand_sim  = sim_actions[:, 14:24]   # (T, 10)
    right_hand_sim = sim_actions[:, 24:34]   # (T, 10)

    out[:, SEG_LEFT_HAND]  = left_hand_sim[:, HAND_SIM_TO_REAL_IDX]   # (T, 6)
    out[:, SEG_RIGHT_HAND] = right_hand_sim[:, HAND_SIM_TO_REAL_IDX]  # (T, 6)

    # Leg, neck, waist remain zero (already initialised to 0)
    return out


def extract_frames_to_mp4(
    rgb_data: np.ndarray,
    out_path: Path,
    fps: float = 20.0,
) -> None:
    """
    Write (T, H, W, C) uint8 RGB(A) array to mp4 using imageio.
    Alpha channel is stripped if present (RGBA -> RGB).
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio required: pip install imageio[ffmpeg]")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip alpha if RGBA
    if rgb_data.shape[-1] == 4:
        rgb_data = rgb_data[..., :3]

    # Ensure uint8
    if rgb_data.dtype != np.uint8:
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for frame in rgb_data:
        writer.append_data(frame)
    writer.close()


# ─────────────────────────────────────────────
# Per-episode processing
# ─────────────────────────────────────────────

def process_episode(
    hdf5_file: h5py.File,
    demo_key: str,
    episode_index: int,
    global_index_offset: int,
    output_root: Path,
    task_description: str,
    fps: float,
    export_video: bool,
) -> dict:
    """
    Process a single demo from the HDF5 file.

    Returns episode metadata dict:
        {episode_index, length, tasks}
    """
    demo = hdf5_file[f"data/{demo_key}"]

    # ── Load raw arrays ──────────────────────────────────────────────
    # Commanded actions (T, 34) — this is "action" in LeRobot
    actions_34d = np.array(demo["actions"], dtype=np.float32)           # (T, 34)

    # State proxy: obs/actions = last_action = what was commanded at t
    # This is our best approximation of observed state in 34D space
    obs_actions_34d = np.array(demo["obs"]["actions"], dtype=np.float32)  # (T, 34)

    T = actions_34d.shape[0]

    # ── Convert to 44D ───────────────────────────────────────────────
    state_44d  = sim_34d_to_lerobot_44d(obs_actions_34d)   # (T, 44) - state at t
    action_44d = sim_34d_to_lerobot_44d(actions_34d)       # (T, 44) - command at t

    # ── Build timestamps ─────────────────────────────────────────────
    # t=0 corresponds to the first recorded step
    timestamps = np.arange(T, dtype=np.float32) * STEP_DT  # seconds

    # ── Determine chunk ──────────────────────────────────────────────
    chunk_index = episode_index // CHUNK_SIZE
    chunk_name  = f"chunk-{chunk_index:03d}"

    # ── Write video files ────────────────────────────────────────────
    if export_video:
        for hdf5_key, lerobot_key in CAMERA_MAP.items():
            if hdf5_key not in demo["obs"]:
                print(f"  [WARN] {hdf5_key} not found in {demo_key}, skipping")
                continue
            rgb_data = np.array(demo["obs"][hdf5_key], dtype=np.uint8)  # (T, H, W, C)
            video_path = output_root / "videos" / chunk_name / lerobot_key / f"episode_{episode_index:06d}.mp4"
            extract_frames_to_mp4(rgb_data, video_path, fps=fps)
            print(f"  Written: {video_path.relative_to(output_root)}")

    # ── Build parquet rows ───────────────────────────────────────────
    rows = []
    for t in range(T):
        is_last = (t == T - 1)

        row = {
            "episode_chunk": chunk_index,
            "episode_id":    episode_index,
            "episode_index": episode_index,
            "index":         global_index_offset + t,
            "observation.state": state_44d[t].tolist(),
            "action":            action_44d[t].tolist(),
            "timestamp":         float(timestamps[t]),
            "annotation.human.action.task_description": task_description,
            "task_index":   0,   # task 0 = main task (see tasks.jsonl)
            "annotation.human.validity": "valid",
            "next.reward":  float(is_last),  # 1.0 on last step, 0 otherwise
            "next.done":    bool(is_last),
            "is_last_row":  bool(is_last),
        }
        rows.append(row)

    # ── Write parquet ────────────────────────────────────────────────
    parquet_dir = output_root / "data" / chunk_name
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"  Written: {parquet_path.relative_to(output_root)}  ({T} rows)")

    return {
        "episode_index": episode_index,
        "tasks": [task_description, "valid"],
        "length": T,
    }


# ─────────────────────────────────────────────
# Meta file writers
# ─────────────────────────────────────────────

def write_tasks_jsonl(output_root: Path, task_description: str) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        # Index 0 = main task, index 1 = "valid" (reserved per your convention)
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "valid"}) + "\n")


def write_episodes_jsonl(output_root: Path, episode_metas: list) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_metas:
            f.write(json.dumps(ep) + "\n")


def write_info_json(output_root: Path, episode_metas: list, fps: float) -> None:
    total_rows = sum(ep["length"] for ep in episode_metas)
    total_episodes = len(episode_metas)
    num_video_keys = len(CAMERA_MAP)
    # LeRobot total_frames = sum of frames across ALL video streams
    # i.e. total_rows * num_cameras (each timestep has one frame per camera)
    total_frames = total_rows * num_video_keys

    info = {
        "codebase_version": "v2.0",
        "robot_type": "kuavo_v4pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * num_video_keys,
        "total_chunks": max(1, (total_episodes - 1) // CHUNK_SIZE + 1),
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": _build_state_names(),
            },
            "action": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": _build_state_names(),
            },
            **{
                v: {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "video_info": {
                        "fps": fps,
                        "codec": "av1",
                        "pix_fmt": "yuv420p",
                    },
                }
                for v in CAMERA_MAP.values()
            },
        },
    }

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def write_modality_json(output_root: Path) -> None:
    """Copy the modality.json structure matching your real dataset."""
    modality = {
        "state": {
            "left_arm":   {"start": 0,  "end": 7},
            "left_hand":  {"start": 7,  "end": 13},
            "left_leg":   {"start": 13, "end": 19},
            "neck":       {"start": 19, "end": 22},
            "right_arm":  {"start": 22, "end": 29},
            "right_hand": {"start": 29, "end": 35},
            "right_leg":  {"start": 35, "end": 41},
            "waist":      {"start": 41, "end": 44},
        },
        "action": {
            "left_arm":   {"start": 0,  "end": 7},
            "left_hand":  {"start": 7,  "end": 13},
            "left_leg":   {"start": 13, "end": 19},
            "neck":       {"start": 19, "end": 22},
            "right_arm":  {"start": 22, "end": 29},
            "right_hand": {"start": 29, "end": 35},
            "right_leg":  {"start": 35, "end": 41},
            "waist":      {"start": 41, "end": 44},
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            },
            "left_wrist_view": {
                "original_key": "observation.images.left_wrist_view"
            },
            "right_wrist_view": {
                "original_key": "observation.images.right_wrist_view"
            },
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)


def _build_state_names() -> list:
    """Human-readable names for each of the 44 state/action dimensions."""
    names = []
    # left arm
    names += [f"left_arm_joint_{i+1}" for i in range(7)]
    # left hand (6 real DOF: thumbCMC, thumbMCP, indexMCP, middleMCP, ringMCP, littleMCP)
    names += ["left_thumbCMC", "left_thumbMCP", "left_indexMCP", "left_middleMCP", "left_ringMCP", "left_littleMCP"]
    # left leg (zeros)
    names += [f"left_leg_{i}" for i in range(6)]
    # neck (zeros)
    names += [f"neck_{i}" for i in range(3)]
    # right arm
    names += [f"right_arm_joint_{i+1}" for i in range(7)]
    # right hand
    names += ["right_thumbCMC", "right_thumbMCP", "right_indexMCP", "right_middleMCP", "right_ringMCP", "right_littleMCP"]
    # right leg (zeros)
    names += [f"right_leg_{i}" for i in range(6)]
    # waist (zeros)
    names += [f"waist_{i}" for i in range(3)]
    return names


# ─────────────────────────────────────────────
# Integrity validation
# ─────────────────────────────────────────────

def validate_dataset(output_root: Path, episode_metas: list) -> None:
    """Run basic integrity checks on the generated dataset."""
    print("\n── Integrity Checks ──────────────────────────────────────")
    errors = []
    total_frames_expected = sum(ep["length"] for ep in episode_metas)

    # 1. Episode count vs parquet files
    all_parquets = sorted((output_root / "data").rglob("*.parquet"))
    if len(all_parquets) != len(episode_metas):
        errors.append(f"Episode count mismatch: {len(episode_metas)} meta entries vs {len(all_parquets)} parquet files")
    else:
        print(f"  ✓ Episode count: {len(episode_metas)}")

    # 2. Global index continuity
    all_indices = []
    for pq in all_parquets:
        df = pd.read_parquet(pq)
        all_indices.extend(df["index"].tolist())
    all_indices_sorted = sorted(all_indices)
    expected_indices = list(range(total_frames_expected))
    if all_indices_sorted != expected_indices:
        errors.append(f"Global index NOT continuous! min={min(all_indices)}, max={max(all_indices)}, total={len(all_indices)}")
    else:
        print(f"  ✓ Global index: 0 → {max(all_indices)} (no gaps, no duplicates)")

    # 3. Episode length consistency (meta vs parquet row count)
    for ep in episode_metas:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // CHUNK_SIZE
        pq_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            errors.append(f"Missing parquet: {pq_path}")
            continue
        df = pd.read_parquet(pq_path)
        if len(df) != ep["length"]:
            errors.append(f"Episode {ep_idx}: meta length={ep['length']} but parquet rows={len(df)}")

    if not errors:
        print(f"  ✓ All episode lengths match meta")

    # 4. is_last_row check (exactly one per episode)
    for ep in episode_metas:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // CHUNK_SIZE
        pq_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not pq_path.exists():
            continue
        df = pd.read_parquet(pq_path)
        last_count = df["is_last_row"].sum()
        if last_count != 1:
            errors.append(f"Episode {ep_idx}: expected 1 is_last_row=True, got {last_count}")
    if not errors:
        print(f"  ✓ is_last_row flags: exactly 1 per episode")

    # 5. State/action dimension check
    first_pq = pd.read_parquet(all_parquets[0])
    state_dim = len(first_pq["observation.state"].iloc[0])
    action_dim = len(first_pq["action"].iloc[0])
    if state_dim != STATE_DIM:
        errors.append(f"state_dim={state_dim}, expected {STATE_DIM}")
    else:
        print(f"  ✓ observation.state dim: {state_dim}")
    if action_dim != STATE_DIM:
        errors.append(f"action_dim={action_dim}, expected {STATE_DIM}")
    else:
        print(f"  ✓ action dim: {action_dim}")

    # Summary
    if errors:
        print(f"\n  ✗ {len(errors)} error(s) found:")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  ✓ All checks passed!")
    print("──────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 dataset to LeRobot format for KuavoV4Pro pouring task."
    )
    parser.add_argument(
        "--input",
        required=True,
        dest="input",
        help="Path to Isaac Lab output HDF5 file (e.g. output_dataset.hdf5)",
    )
    parser.add_argument(
        "--output",
        required=True,
        dest="output",
        help="Output directory for the LeRobot dataset",
    )
    parser.add_argument(
        "--task",
        default="Use both hands to pour contents from the cup into the bowl",
        help="Task description string for tasks.jsonl and parquet annotation",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate for exported videos (default: 20 Hz = 1/STEP_DT)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video export (useful for quick parquet-only testing)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes to convert (for testing)",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_root = Path(args.output)
    export_video = not args.no_video

    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_path}")

    print(f"\n{'='*60}")
    print(f"  Isaac Lab HDF5 → LeRobot Converter")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_root}")
    print(f"  FPS:    {args.fps}")
    print(f"  Video:  {'yes' if export_video else 'no (--no-video)'}")
    print(f"{'='*60}\n")

    # Write static meta files first
    write_tasks_jsonl(output_root, args.task)
    write_modality_json(output_root)

    episode_metas = []
    global_index = 0

    with h5py.File(input_path, "r") as f:
        # Collect demo keys in sorted order (demo_0, demo_1, ...)
        all_demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

        if args.max_episodes is not None:
            all_demo_keys = all_demo_keys[:args.max_episodes]

        print(f"Found {len(all_demo_keys)} demos to convert\n")

        for episode_index, demo_key in enumerate(all_demo_keys):
            print(f"[{episode_index+1}/{len(all_demo_keys)}] Processing {demo_key} → episode_{episode_index:06d}")

            meta = process_episode(
                hdf5_file=f,
                demo_key=demo_key,
                episode_index=episode_index,
                global_index_offset=global_index,
                output_root=output_root,
                task_description=args.task,
                fps=args.fps,
                export_video=export_video,
            )
            episode_metas.append(meta)
            global_index += meta["length"]

    # Write remaining meta files
    write_episodes_jsonl(output_root, episode_metas)
    write_info_json(output_root, episode_metas, args.fps)

    print(f"\nConversion complete!")
    print(f"  Total episodes: {len(episode_metas)}")
    print(f"  Total frames:   {global_index}")

    # Run integrity validation
    validate_dataset(output_root, episode_metas)

    print(f"Dataset written to: {output_root}")


if __name__ == "__main__":
    main()
