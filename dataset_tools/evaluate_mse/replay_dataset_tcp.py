#!/usr/bin/env python3
"""
Replay LeRobot-style dataset episodes (parquet + mp4s) through a Kuavo TCP policy server,
print predicted actions, and compute MSE vs ground-truth actions.

Expected dataset layout (as you described):
  <root>/data/chunk-000000/episode_000123.parquet
  <root>/videos/chunk-000/observation.images.ego_view/episode_000123.mp4
  <root>/videos/chunk-000/observation.images.left_wrist_view/episode_000123.mp4
  <root>/videos/chunk-000/observation.images.right_wrist_view/episode_000123.mp4

Wire protocol: msgpack dict with 4-byte big-endian length prefix
  request: {"state": [...], "images": {cam_name: jpeg_bytes, ...}, "meta": {...}}
  response: {"action": [...], "mode": "delta"|"absolute"}


python replay_dataset_tcp.py \
  --root /path/to/lerobot_dataset \
  --data-chunk chunk-000000 \
  --videos-chunk chunk-000 \
  --host 127.0.0.1 --port 5555 \
  --episodes 1 \
  --print-every 20


  python replay_dataset_tcp.py \
  --root /home/sensethreat/lab_mimic/VLA_IL/capstone-vla/dataset_tools/pourLeftCereal/Lusmse/pourLeftCereal \
  --data-chunk chunk-000 \
  --videos-chunk chunk-000 \
  --host 127.0.0.1 --port 5555 \
  --episodes 10 \
  --print-every 20

"""

import argparse
import os
import re
import socket
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgpack
import numpy as np

# Video I/O
import cv2

# Parquet I/O (fast + reliable)
try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pyarrow. Install with:\n"
        "  pip install pyarrow\n"
    ) from e


EP_RE = re.compile(r"episode_(\d+)\.parquet$")


# -----------------------
# TCP helpers (matches tcp_policy_server.py)
# -----------------------
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server disconnected")
        buf += chunk
    return buf


def send_msg(conn: socket.socket, obj: Dict) -> None:
    payload = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(conn: socket.socket) -> Dict:
    header = recv_exact(conn, 4)
    (n,) = struct.unpack("!I", header)
    payload = recv_exact(conn, n)
    return msgpack.unpackb(payload, raw=False)


# -----------------------
# Dataset helpers
# -----------------------
def list_episode_parquets(data_dir: Path) -> List[Path]:
    eps = sorted([p for p in data_dir.glob("episode_*.parquet") if p.is_file()])
    return eps


def parse_episode_id(parquet_path: Path) -> int:
    m = EP_RE.search(parquet_path.name)
    if not m:
        raise ValueError(f"Not an episode parquet: {parquet_path}")
    return int(m.group(1))


def read_parquet_table(parquet_path: Path):
    # Keep as Arrow table then convert columns as needed
    return pq.read_table(parquet_path)


def get_column(table, name: str):
    if name not in table.column_names:
        raise KeyError(
            f"Column '{name}' not found in {table.schema}\n"
            f"Available columns: {table.column_names}"
        )
    return table[name]


def to_numpy_2d(col) -> np.ndarray:
    """
    Converts an Arrow column to numpy. Supports:
    - fixed-size list / list columns
    - nested arrays
    Output shape should be [T, D]
    """
    arr = col.to_numpy(zero_copy_only=False)
    # Common case: dtype=object, each row is list/np array
    if arr.dtype == object:
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)
    else:
        arr = np.asarray(arr, dtype=np.float32)
        # If scalar per row, make it (T,1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    return arr.astype(np.float32)


# -----------------------
# Video helpers
# -----------------------
def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise EOFError("Video ended (no frame)")
    return frame_bgr


def jpeg_bytes_from_bgr(frame_bgr: np.ndarray, jpeg_quality: int = 90) -> bytes:
    # Server decodes as JPEG and converts BGR->RGB internally
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError("Failed to JPEG-encode frame")
    return buf.tobytes()


# -----------------------
# Core replay + eval
# -----------------------
def resolve_video_paths(
    root: Path,
    videos_chunk: str,
    episode_id: int,
    ego_dir: str,
    left_dir: str,
    right_dir: str,
) -> Tuple[Path, Path, Path]:
    ep_name = f"episode_{episode_id:06d}.mp4"
    base = root / "videos" / videos_chunk
    ego = base / ego_dir / ep_name
    left = base / left_dir / ep_name
    right = base / right_dir / ep_name
    return ego, left, right


def connect_tcp(host: str, port: int, timeout_s: float = 10.0) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect((host, port))
    s.settimeout(None)  # switch to blocking
    return s


def replay_episode(
    conn: socket.socket,
    parquet_path: Path,
    root: Path,
    videos_chunk: str,
    state_col: str,
    action_col: str,
    ego_dir: str,
    left_dir: str,
    right_dir: str,
    cam_name_map: Dict[str, str],
    max_steps: Optional[int],
    print_every: int,
    jpeg_quality: int,
) -> Dict[str, float]:
    """
    Returns per-episode stats: mse, steps, etc.
    """
    episode_id = parse_episode_id(parquet_path)
    table = read_parquet_table(parquet_path)

    state = to_numpy_2d(get_column(table, state_col))    # [T, Ds]
    gt_action = to_numpy_2d(get_column(table, action_col))  # [T, Da]

    T = min(len(state), len(gt_action))
    if max_steps is not None:
        T = min(T, max_steps)

    ego_v, left_v, right_v = resolve_video_paths(
        root=root,
        videos_chunk=videos_chunk,
        episode_id=episode_id,
        ego_dir=ego_dir,
        left_dir=left_dir,
        right_dir=right_dir,
    )

    cap_ego = open_video(ego_v)
    cap_left = open_video(left_v)
    cap_right = open_video(right_v)

    sq_err_sum = 0.0
    n_el = 0

    print(f"\n=== Episode {episode_id:06d} ===")
    print(f"Parquet: {parquet_path}")
    print(f"Videos:  \n  {ego_v}\n  {left_v}\n  {right_v}")
    print(f"Steps: {T}  |  state_dim={state.shape[1]} action_dim={gt_action.shape[1]}")

    for t in range(T):
        try:
            f_ego = read_frame(cap_ego)
            f_left = read_frame(cap_left)
            f_right = read_frame(cap_right)
        except EOFError:
            print(f"[WARN] Video shorter than parquet at t={t}. Stopping episode early.")
            break

        images_payload = {
            cam_name_map["ego"]: jpeg_bytes_from_bgr(f_ego, jpeg_quality=jpeg_quality),
            cam_name_map["left"]: jpeg_bytes_from_bgr(f_left, jpeg_quality=jpeg_quality),
            cam_name_map["right"]: jpeg_bytes_from_bgr(f_right, jpeg_quality=jpeg_quality),
        }

        req = {
            "state": state[t].astype(np.float32).tolist(),
            "images": images_payload,
            "meta": {"episode": int(episode_id), "t": int(t)},
        }

        send_msg(conn, req)
        resp = recv_msg(conn)

        pred_action = np.asarray(resp["action"], dtype=np.float32)
        gt = gt_action[t].astype(np.float32)

        if pred_action.shape != gt.shape:
            raise ValueError(f"Shape mismatch at t={t}: pred={pred_action.shape} gt={gt.shape}")

        err = pred_action - gt
        sq_err_sum += float(np.sum(err * err))
        n_el += int(err.size)

        if (t % print_every) == 0:
            mse_t = float(np.mean(err * err))
            print(f"t={t:04d} mode={resp.get('mode','?')} mse_t={mse_t:.6f}")
            print(f"  pred[:8]={pred_action[:8].tolist()}")
            print(f"  gt  [:8]={gt[:8].tolist()}")

    cap_ego.release()
    cap_left.release()
    cap_right.release()

    mse = (sq_err_sum / max(n_el, 1)) if n_el > 0 else float("nan")
    steps_done = int(n_el / gt_action.shape[1]) if n_el > 0 else 0

    print(f"Episode {episode_id:06d} done: steps={steps_done} mse={mse:.8f}")
    return {"episode_id": float(episode_id), "steps": float(steps_done), "mse": float(mse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root folder (contains data/ and videos/)")
    ap.add_argument("--data-chunk", default="chunk-000000", help="data/<chunk-xxxxxx> folder name")
    ap.add_argument("--videos-chunk", default="chunk-000", help="videos/<chunk-xxx> folder name")

    # Parquet columns (you can override to match your schema)
    ap.add_argument("--state-col", default="observation.state", help="Parquet column for state (list/array per row)")
    ap.add_argument("--action-col", default="action", help="Parquet column for GT action (list/array per row)")

    # Video directories (relative to videos/<chunk>/)
    ap.add_argument("--ego-dir", default="observation.images.ego_view")
    ap.add_argument("--left-dir", default="observation.images.left_wrist_view")
    ap.add_argument("--right-dir", default="observation.images.right_wrist_view")

    # TCP server
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)

    # Episode selection
    ap.add_argument("--episodes", type=int, default=1, help="How many episodes to replay (from start of chunk)")
    ap.add_argument("--start-episode-idx", type=int, default=0, help="Index within sorted parquet list to start from")
    ap.add_argument("--max-steps", type=int, default=None, help="Max timesteps per episode (default: all)")
    ap.add_argument("--print-every", type=int, default=10, help="Print predicted/gt every N steps")
    ap.add_argument("--jpeg-quality", type=int, default=90)

    # Camera names that the tcp_policy_server expects as keys in request['images'].
    # By default, tcp_policy_server.py maps:
    #   head_cam_h -> observation.images.ego_view
    #   wrist_cam_l -> observation.images.left_wrist_view
    #   wrist_cam_r -> observation.images.right_wrist_view
    ap.add_argument("--cam-ego", default="head_cam_h")
    ap.add_argument("--cam-left", default="wrist_cam_l")
    ap.add_argument("--cam-right", default="wrist_cam_r")

    args = ap.parse_args()

    root = Path(args.root)
    data_dir = root / "data" / args.data_chunk
    if not data_dir.exists():
        raise SystemExit(f"Missing data dir: {data_dir}")

    parquets = list_episode_parquets(data_dir)
    if not parquets:
        raise SystemExit(f"No episode_*.parquet found in: {data_dir}")

    start = args.start_episode_idx
    end = min(start + args.episodes, len(parquets))
    selected = parquets[start:end]

    print(f"Dataset root: {root}")
    print(f"Selected {len(selected)} episode parquet(s) from {data_dir} (idx {start}..{end-1})")
    print(f"Connecting to TCP server {args.host}:{args.port} ...")

    conn = connect_tcp(args.host, args.port)
    print("Connected.\n")

    cam_name_map = {"ego": args.cam_ego, "left": args.cam_left, "right": args.cam_right}

    all_mse = []
    all_steps = 0

    try:
        for pq_path in selected:
            stats = replay_episode(
                conn=conn,
                parquet_path=pq_path,
                root=root,
                videos_chunk=args.videos_chunk,
                state_col=args.state_col,
                action_col=args.action_col,
                ego_dir=args.ego_dir,
                left_dir=args.left_dir,
                right_dir=args.right_dir,
                cam_name_map=cam_name_map,
                max_steps=args.max_steps,
                print_every=args.print_every,
                jpeg_quality=args.jpeg_quality,
            )
            if not np.isnan(stats["mse"]):
                all_mse.append(stats["mse"])
            all_steps += int(stats["steps"])

    finally:
        conn.close()

    if all_mse:
        avg_mse = float(np.mean(all_mse))
        print("\n====================")
        print("OVERALL SUMMARY")
        print("====================")
        print(f"Episodes: {len(all_mse)}")
        print(f"Total steps: {all_steps}")
        print(f"Avg episode MSE: {avg_mse:.8f}")
    else:
        print("\nNo valid MSE computed (check schema / videos / server output).")


if __name__ == "__main__":
    main()