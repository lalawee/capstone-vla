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

python hdf5_to_lerobot.py --input "/home/sensethreat/lab_mimic/VLA_IL/capstone-vla/dataset_tools/data/Hf_data/Lusmse/output_2_dataset.hdf5" --output "/home/sensethreat/lab_mimic/VLA_IL/capstone-vla/dataset_tools/data/Hf_data/Lusmse/sdg_trial" --trim-start 0.5

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

# Indices into the 10D sim hand array to select the 6 real DOF (MCP joints only)
# Extracts: [thumbCMC, thumbMCP, indexMCP, middleMCP, ringMCP, littleMCP]
# = semantic order: [thumb_rot, thumb_bend, index, middle, ring, little]
HAND_SIM_TO_REAL_IDX = [0, 1, 2, 4, 6, 8]

# Sim hand joint URDF limits (from URDF, lower=0 for all joints)
# Order: [thumbCMC, thumbMCP, indexMCP, middleMCP, ringMCP, littleMCP]
HAND_SIM_LOWER = np.array([0.0,    0.0,    0.0,   0.0,   0.0,   0.0  ], dtype=np.float32)
HAND_SIM_UPPER = np.array([1.5708, 0.8727, 1.309, 1.309, 1.309, 1.309], dtype=np.float32)

# Real robot joint limits from _normalize_hand_joints()
# Order: [thumb1, thumb2, index, middle, ring, pinky] = same semantic order
# sim=0 (LOWER) -> open hand -> REAL_LOWER (negative)
# sim=upper     -> closed    -> REAL_UPPER (positive)
HAND_REAL_LOWER = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, -3.0], dtype=np.float32)
HAND_REAL_UPPER = np.array([ 1.5,  1.5,  1.5,  1.5,  3.0,  3.0], dtype=np.float32)


def sim_hand_to_real_radians(hand_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D sim hand (radians) to real robot radians via linear rescale.

    URDF limits: lower=0, upper varies per joint (all joints start at 0=open).
    Real robot:  LOWER=negative (open), UPPER=positive (closed/gripping).

    Linear rescale:
        t        = (sim_rad - SIM_LOWER) / (SIM_UPPER - SIM_LOWER)  in [0, 1]
        real_rad = REAL_LOWER + (REAL_UPPER - REAL_LOWER) * t

    Boundary behaviour:
        sim=0          -> t=0 -> REAL_LOWER (fully open)
        sim=SIM_UPPER  -> t=1 -> REAL_UPPER (fully closed)

    Args:
        hand_6d: (T, 6) sim radians, order:
                 [thumbCMC, thumbMCP, indexMCP, middleMCP, ringMCP, littleMCP]
    Returns:
        (T, 6) real robot radians within [REAL_LOWER, REAL_UPPER]
    """
    sim_range  = HAND_SIM_UPPER - HAND_SIM_LOWER   # [1.5708, 0.8727, 1.309, 1.309, 1.309, 1.309]
    real_range = HAND_REAL_UPPER - HAND_REAL_LOWER  # [3.0, 3.0, 3.0, 3.0, 6.0, 6.0]

    # Normalise to [0, 1], clip for safety against numerical overshoot
    t = np.clip((hand_6d - HAND_SIM_LOWER) / sim_range, 0.0, 1.0)

    # Rescale to real range
    real_rad = HAND_REAL_LOWER + real_range * t

    return real_rad.astype(np.float32)

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

    # Hands: 10D sim -> 6D (MCP extraction) -> percent inverse -> real radians
    left_hand_sim  = sim_actions[:, 14:24]   # (T, 10)
    right_hand_sim = sim_actions[:, 24:34]   # (T, 10)

    # Extract the 6 MCP joints (drops PIP duplicates)
    left_hand_6d  = left_hand_sim[:, HAND_SIM_TO_REAL_IDX]    # (T, 6) sim radians
    right_hand_6d = right_hand_sim[:, HAND_SIM_TO_REAL_IDX]   # (T, 6) sim radians

    # Convert sim radians -> real robot radians via 0-100 percent intermediate
    out[:, SEG_LEFT_HAND]  = sim_hand_to_real_radians(left_hand_6d)
    out[:, SEG_RIGHT_HAND] = sim_hand_to_real_radians(right_hand_6d)

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
# Trimming logic
# ─────────────────────────────────────────────

def compute_trim_steps(trim_seconds: float) -> int:
    """Convert trim duration in seconds to number of steps to remove.
    
    Each recorded step = STEP_DT seconds (default 0.05s).
    So 1.0 second = 20 steps, 2.0 seconds = 40 steps.
    """
    if trim_seconds <= 0:
        return 0
    steps = int(round(trim_seconds / STEP_DT))
    return steps


def compute_jerk_trim_steps(
    actions: np.ndarray,
    jerk_threshold: float = 0.5,
    window: int = 3,
    max_trim_steps: int = 60,
) -> int:
    """Auto-detect how many initial steps to trim based on joint velocity jerk.
    
    Computes frame-to-frame joint velocity magnitude. The "jerky snap to ready"
    shows up as large velocity spikes at the start. We find the last step within
    [0, max_trim_steps] where the max joint velocity exceeds jerk_threshold,
    then trim up to that point + a small buffer.
    
    Args:
        actions: (T, D) array of joint positions
        jerk_threshold: rad/step velocity above which we consider "jerky"
        window: extra steps to trim after the last detected jerk
        max_trim_steps: don't look beyond this many steps from the start
    
    Returns:
        Number of steps to trim from the start.
    """
    if actions.shape[0] < 3:
        return 0
    
    # Compute per-step velocity (first difference)
    vel = np.diff(actions, axis=0)  # (T-1, D)
    # Max absolute velocity across all joints per step
    max_vel = np.max(np.abs(vel), axis=1)  # (T-1,)
    
    # Only look at the first max_trim_steps
    search_range = min(max_trim_steps, len(max_vel))
    
    # Find the last step in the search range that exceeds the threshold
    last_jerk_idx = -1
    for i in range(search_range):
        if max_vel[i] > jerk_threshold:
            last_jerk_idx = i
    
    if last_jerk_idx < 0:
        return 0  # No jerk detected
    
    # Trim up to last_jerk_idx + window (the +1 accounts for diff offset)
    trim = min(last_jerk_idx + 1 + window, actions.shape[0] - 1)
    return trim


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
    trim_start_steps: int = 0,
    trim_end_steps: int = 0,
    auto_trim: bool = False,
    jerk_threshold: float = 0.5,
) -> dict:
    """
    Process a single demo from the HDF5 file.

    Returns episode metadata dict:
        {episode_index, length, tasks, trimmed_start, trimmed_end}
    """
    demo = hdf5_file[f"data/{demo_key}"]

    # ── Load raw arrays ──────────────────────────────────────────────
    actions_34d = np.array(demo["actions"], dtype=np.float32)             # (T, 34)
    obs_actions_34d = np.array(demo["obs"]["actions"], dtype=np.float32)  # (T, 34)

    T_original = actions_34d.shape[0]

    # ── Compute trim amounts ─────────────────────────────────────────
    actual_trim_start = trim_start_steps
    
    if auto_trim:
        # Use jerk detection to find optimal trim point
        auto_steps = compute_jerk_trim_steps(
            actions_34d, 
            jerk_threshold=jerk_threshold,
        )
        # Take the larger of manual and auto-detected trim
        actual_trim_start = max(trim_start_steps, auto_steps)
        if auto_steps > 0:
            print(f"    Auto-trim detected {auto_steps} jerky steps (threshold={jerk_threshold:.2f} rad/step)")
    
    actual_trim_end = trim_end_steps
    
    # Validate: ensure we don't trim more than the episode length (keep at least 2 steps)
    total_trim = actual_trim_start + actual_trim_end
    if total_trim >= T_original - 1:
        print(f"    [WARN] Trim ({actual_trim_start}+{actual_trim_end}={total_trim}) >= episode length ({T_original}). Reducing trim.")
        # Scale back proportionally, keep at least 2 steps
        max_trim = T_original - 2
        if actual_trim_start + actual_trim_end > max_trim:
            # Prioritize start trim (that's where the jerk is)
            actual_trim_start = min(actual_trim_start, max_trim)
            actual_trim_end = min(actual_trim_end, max_trim - actual_trim_start)
    
    # ── Apply trimming ───────────────────────────────────────────────
    end_idx = T_original - actual_trim_end if actual_trim_end > 0 else T_original
    
    actions_34d     = actions_34d[actual_trim_start:end_idx]
    obs_actions_34d = obs_actions_34d[actual_trim_start:end_idx]
    
    T = actions_34d.shape[0]
    
    if actual_trim_start > 0 or actual_trim_end > 0:
        trim_start_sec = actual_trim_start * STEP_DT
        trim_end_sec = actual_trim_end * STEP_DT
        print(f"    Trimmed: {T_original} → {T} steps "
              f"(removed {actual_trim_start} from start [{trim_start_sec:.2f}s], "
              f"{actual_trim_end} from end [{trim_end_sec:.2f}s])")

    # ── Convert to 44D ───────────────────────────────────────────────
    state_44d  = sim_34d_to_lerobot_44d(obs_actions_34d)   # (T, 44) - state at t
    action_44d = sim_34d_to_lerobot_44d(actions_34d)       # (T, 44) - command at t

    # ── Build timestamps (re-baselined to 0 after trim) ──────────────
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
            rgb_data = np.array(demo["obs"][hdf5_key], dtype=np.uint8)  # (T_orig, H, W, C)
            
            # Apply same trimming to video frames
            rgb_data = rgb_data[actual_trim_start:end_idx]
            
            video_path = output_root / "videos" / chunk_name / lerobot_key / f"episode_{episode_index:06d}.mp4"
            extract_frames_to_mp4(rgb_data, video_path, fps=fps)
            print(f"    Written: {video_path.relative_to(output_root)}")

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
            "task_index":   0,
            "annotation.human.validity": "valid",
            "next.reward":  float(is_last),
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
    print(f"    Written: {parquet_path.relative_to(output_root)}  ({T} rows)")

    return {
        "episode_index": episode_index,
        "tasks": [task_description, "valid"],
        "length": T,
        "original_length": T_original,
        "trimmed_start": actual_trim_start,
        "trimmed_end": actual_trim_end,
    }


# ─────────────────────────────────────────────
# Meta file writers
# ─────────────────────────────────────────────

def write_tasks_jsonl(output_root: Path, task_description: str) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": "valid"}) + "\n")


def write_episodes_jsonl(output_root: Path, episode_metas: list) -> None:
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_metas:
            # Write only the standard fields (don't leak internal trim metadata)
            out = {
                "episode_index": ep["episode_index"],
                "tasks": ep["tasks"],
                "length": ep["length"],
            }
            f.write(json.dumps(out) + "\n")


def write_info_json(output_root: Path, episode_metas: list, fps: float) -> None:
    total_rows = sum(ep["length"] for ep in episode_metas)
    total_episodes = len(episode_metas)
    num_video_keys = len(CAMERA_MAP)
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
    names = []
    names += [f"left_arm_joint_{i+1}" for i in range(7)]
    names += ["left_thumbCMC", "left_thumbMCP", "left_indexMCP", "left_middleMCP", "left_ringMCP", "left_littleMCP"]
    names += [f"left_leg_{i}" for i in range(6)]
    names += [f"neck_{i}" for i in range(3)]
    names += [f"right_arm_joint_{i+1}" for i in range(7)]
    names += ["right_thumbCMC", "right_thumbMCP", "right_indexMCP", "right_middleMCP", "right_rightMCP", "right_littleMCP"]
    names += [f"right_leg_{i}" for i in range(6)]
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

    # 3. Episode length consistency
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

    # 4. is_last_row check
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


def print_trim_summary(episode_metas: list) -> None:
    """Print a summary of trimming applied across all episodes."""
    trimmed_episodes = [ep for ep in episode_metas if ep["trimmed_start"] > 0 or ep["trimmed_end"] > 0]
    if not trimmed_episodes:
        return
    
    print(f"\n── Trimming Summary ──────────────────────────────────────")
    total_original = sum(ep["original_length"] for ep in episode_metas)
    total_after = sum(ep["length"] for ep in episode_metas)
    total_removed = total_original - total_after
    
    start_trims = [ep["trimmed_start"] for ep in episode_metas]
    end_trims = [ep["trimmed_end"] for ep in episode_metas]
    
    print(f"  Episodes trimmed: {len(trimmed_episodes)}/{len(episode_metas)}")
    print(f"  Total steps: {total_original} → {total_after} (removed {total_removed}, {100*total_removed/total_original:.1f}%)")
    print(f"  Start trim range: {min(start_trims)}–{max(start_trims)} steps ({min(start_trims)*STEP_DT:.2f}–{max(start_trims)*STEP_DT:.2f}s)")
    if any(t > 0 for t in end_trims):
        print(f"  End trim range: {min(end_trims)}–{max(end_trims)} steps")
    print(f"──────────────────────────────────────────────────────────\n")


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
    
    # ── Trimming arguments ───────────────────────────────────────────
    trim_group = parser.add_argument_group("Trimming", "Remove jerky start/end frames from episodes")
    trim_group.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Trim this many seconds from the START of each episode. "
             "Removes the jerky snap-to-ready frames. (default: 0, typical: 0.5–2.0)",
    )
    trim_group.add_argument(
        "--trim-end",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Trim this many seconds from the END of each episode. (default: 0)",
    )
    trim_group.add_argument(
        "--auto-trim",
        action="store_true",
        default=False,
        help="Automatically detect and trim jerky start frames using joint velocity analysis. "
             "Takes the max of --trim-start and auto-detected trim. "
             "Useful when jerk duration varies per episode.",
    )
    trim_group.add_argument(
        "--jerk-threshold",
        type=float,
        default=0.5,
        metavar="RAD_PER_STEP",
        help="Joint velocity threshold (rad/step) for auto-trim jerk detection. "
             "Lower = more aggressive trimming. (default: 0.5)",
    )
    
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_root = Path(args.output)
    export_video = not args.no_video
    
    # Convert trim seconds to steps
    trim_start_steps = compute_trim_steps(args.trim_start)
    trim_end_steps = compute_trim_steps(args.trim_end)

    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_path}")

    print(f"\n{'='*60}")
    print(f"  Isaac Lab HDF5 → LeRobot Converter")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_root}")
    print(f"  FPS:    {args.fps}")
    print(f"  Video:  {'yes' if export_video else 'no (--no-video)'}")
    if args.trim_start > 0 or args.trim_end > 0 or args.auto_trim:
        print(f"  Trim:   start={args.trim_start:.2f}s ({trim_start_steps} steps)"
              f", end={args.trim_end:.2f}s ({trim_end_steps} steps)"
              f"{', auto-trim ON' if args.auto_trim else ''}")
        if args.auto_trim:
            print(f"          jerk threshold={args.jerk_threshold:.2f} rad/step")
    print(f"{'='*60}\n")

    # Write static meta files first
    write_tasks_jsonl(output_root, args.task)
    write_modality_json(output_root)

    episode_metas = []
    global_index = 0

    with h5py.File(input_path, "r") as f:
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
                trim_start_steps=trim_start_steps,
                trim_end_steps=trim_end_steps,
                auto_trim=args.auto_trim,
                jerk_threshold=args.jerk_threshold,
            )
            episode_metas.append(meta)
            global_index += meta["length"]

    # Write remaining meta files
    write_episodes_jsonl(output_root, episode_metas)
    write_info_json(output_root, episode_metas, args.fps)

    print(f"\nConversion complete!")
    print(f"  Total episodes: {len(episode_metas)}")
    print(f"  Total frames:   {global_index}")

    # Print trim summary
    print_trim_summary(episode_metas)

    # Run integrity validation
    validate_dataset(output_root, episode_metas)

    print(f"Dataset written to: {output_root}")


if __name__ == "__main__":
    main()