#!/usr/bin/env python3
"""
lerobot_health.py

Health checker for LeRobot (v2.0-style) datasets.

Checks (configurable by mode):
- meta: meta files exist + tasks/episodes/info sanity
- light: meta + parquet integrity + video file existence + basic decode sanity
- heavy: light + per-video frame counts + dataset total_frames consistency

Exit codes:
- 0 PASS (no errors)
- 1 FAIL (errors present)
- 2 FAIL (warnings treated as errors via --strict)

Examples:
  python lerobot_health.py --root /path/to/dataset --mode light --report health_report.json
  python lerobot_health.py --root /path/to/dataset --mode heavy --strict --tolerance-frames 1

Notes:
- Requires: pandas, pyarrow (for parquet), numpy
- For fast/accurate video frame counting in heavy mode: ffprobe recommended (from FFmpeg).
  If ffprobe is not available, OpenCV will be used as a fallback (slower/less reliable).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def get_episode_last_timestamp_seconds(parquet_path: Path) -> Optional[float]:
    """Read only timestamp column and return last non-null value as float."""
    try:
        # columns-only read (fast)
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        if df.empty:
            return None
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            return None
        return float(ts.iloc[-1])
    except Exception:
        return None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path} line {i}: {e}") from e
    return rows

def safe_int(x: Any, field: str, ctx: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"{ctx}: field '{field}' is not int-like: {x!r}") from e

def fmt_chunk(n: int) -> str:
    return f"chunk-{n:03d}"

def fmt_episode(n: int) -> str:
    return f"episode_{n:06d}"

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def parse_vector_cell(v: Any) -> Optional[np.ndarray]:
    """
    LeRobot often stores vectors as:
      - numpy array / list
      - python list nested in object dtype
      - string representation like "[ 0.1 0.2 ... ]" or "[0.1,0.2,...]"
    This function tries to return np.ndarray or None if can't parse.
    """
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (list, tuple)):
        try:
            return np.array(v, dtype=np.float64)
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Strip quotes already handled by pandas; attempt to parse inside []
        # Accept comma-separated or whitespace-separated floats
        s = s.replace("\n", " ").replace("\t", " ")
        # Remove outer quotes and brackets
        s2 = s
        # keep the content inside brackets if possible
        m = re.search(r"\[(.*)\]", s2, flags=re.DOTALL)
        if m:
            s2 = m.group(1)
        s2 = s2.strip()
        if not s2:
            return np.array([], dtype=np.float64)
        # replace commas with spaces, collapse multiple spaces
        s2 = s2.replace(",", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        try:
            arr = np.fromstring(s2, sep=" ", dtype=np.float64)
            return arr
        except Exception:
            return None
    # unknown type
    return None

def ensure_dir_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing path: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory but got file: {path}")


# -----------------------------
# Report model
# -----------------------------

@dataclass
class Finding:
    code: str
    message: str
    context: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclass
class Report:
    dataset_root: str
    mode: str
    strict: bool
    stats: Dict[str, Any] = dataclasses.field(default_factory=dict)
    errors: List[Finding] = dataclasses.field(default_factory=list)
    warnings: List[Finding] = dataclasses.field(default_factory=list)
    passed: List[str] = dataclasses.field(default_factory=list)

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(Finding(code=code, message=message, context=context))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(Finding(code=code, message=message, context=context))

    def add_pass(self, code: str) -> None:
        self.passed.append(code)

    def status(self) -> str:
        if self.errors:
            return "fail"
        if self.strict and self.warnings:
            return "fail_warnings"
        return "pass"

    def exit_code(self) -> int:
        if self.errors:
            return 1
        if self.strict and self.warnings:
            return 2
        return 0

    def to_json(self) -> Dict[str, Any]:
        def ser_findings(xs: List[Finding]) -> List[Dict[str, Any]]:
            return [{"code": x.code, "message": x.message, "context": x.context} for x in xs]
        return {
            "dataset_root": self.dataset_root,
            "mode": self.mode,
            "strict": self.strict,
            "status": self.status(),
            "stats": self.stats,
            "errors": ser_findings(self.errors),
            "warnings": ser_findings(self.warnings),
            "passed": self.passed,
        }


# -----------------------------
# Video helpers
# -----------------------------

def ffprobe_video_duration_seconds(video_path: Path) -> Optional[float]:
    """Return container duration in seconds using ffprobe; otherwise None."""
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        fmt = data.get("format") or {}
        dur = fmt.get("duration")
        if dur is None:
            return None
        return float(dur)
    except Exception:
        return None


def ffprobe_last_frame_time_seconds(video_path: Path, tail_seconds: float = 1.0) -> Optional[float]:
    """
    Return the timestamp_time (seconds) of the last decodable frame.
    Uses -sseof to seek near the end and reads frames there, then takes max timestamp_time.
    """
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None

    # Seek from end by tail_seconds and list frames with timestamps
    # We keep it small so it doesn't scan whole video.
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-sseof", f"-{tail_seconds}",
        "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        frames = data.get("frames") or []
        if not frames:
            return None

        times: List[float] = []
        for fr in frames:
            # prefer best_effort_timestamp_time, fallback to pkt_pts_time
            t = fr.get("best_effort_timestamp_time")
            if t is None:
                t = fr.get("pkt_pts_time")
            if t is None:
                continue
            try:
                times.append(float(t))
            except Exception:
                continue
        if not times:
            return None
        return max(times)
    except Exception:
        return None

def ffprobe_frame_count(video_path: Path) -> Optional[int]:
    """
    Return frame count using ffprobe if available; otherwise None.
    Uses nb_read_frames if possible, else estimates.
    """
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None

    # Try: count frames accurately
    # nb_read_frames is available with -count_frames on some builds.
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames",
        "-of", "json",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        streams = data.get("streams") or []
        if not streams:
            return None
        st = streams[0]
        # Prefer nb_read_frames, then nb_frames
        for k in ("nb_read_frames", "nb_frames"):
            v = st.get(k)
            if v is not None and str(v).isdigit():
                return int(v)
        return None
    except Exception:
        return None

def basic_video_readable(video_path: Path) -> bool:
    """
    Light sanity: file exists + can read metadata.
    Prefer ffprobe; fallback to OpenCV.
    """
    ffprobe = which("ffprobe")
    if ffprobe:
        cmd = [ffprobe, "-v", "error", "-show_format", "-show_streams", str(video_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return proc.returncode == 0
        except Exception:
            pass

    # Fallback to OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        ok = cap.isOpened()
        if ok:
            ok = cap.read()[0]  # try to read one frame
        cap.release()
        return bool(ok)
    except Exception:
        return False

def opencv_frame_count(video_path: Path) -> Optional[int]:
    """
    Fallback frame count via OpenCV. May be slow and/or unreliable.
    """
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None
        # Try CAP_PROP_FRAME_COUNT first
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            cap.release()
            return n
        # Otherwise iterate (slow)
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        return count
    except Exception:
        return None


# -----------------------------
# Core checks
# -----------------------------

REQUIRED_META = ["info.json", "episodes.jsonl", "tasks.jsonl"]

def discover_video_keys_from_info(info: Dict[str, Any]) -> List[str]:
    feats = info.get("features") or {}
    vids = feats.get("observation") or {}
    imgs = vids.get("images") or {}
    # keys under observation.images.* should be video keys
    keys = list(imgs.keys())
    # stable order
    keys.sort()
    return keys

def modality_video_folder_name(video_key: str) -> str:
    # your structure: videos/chunk-000/observation.images.ego_view/episode_000000.mp4
    return f"observation.images.{video_key}"

def compute_episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        return 0
    return episode_index // chunks_size

def expected_parquet_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    ch = compute_episode_chunk(episode_index, chunks_size)
    return root / "data" / fmt_chunk(ch) / f"{fmt_episode(episode_index)}.parquet"

def expected_video_path(root: Path, episode_index: int, chunks_size: int, video_key: str) -> Path:
    ch = compute_episode_chunk(episode_index, chunks_size)
    return root / "videos" / fmt_chunk(ch) / modality_video_folder_name(video_key) / f"{fmt_episode(episode_index)}.mp4"

def required_parquet_columns() -> List[str]:
    return [
        "episode_index",
        "index",
        "timestamp",
        "observation.state",
        "action",
        "annotation.human.action.task_description",
        "task_index",
        "annotation.human.validity",
        "next.reward",
        "next.done",
    ]

def check_meta(report: Report, root: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    meta_dir = root / "meta"
    ensure_dir_exists(root)
    ensure_dir_exists(meta_dir)

    # required meta files
    missing = [name for name in REQUIRED_META if not (meta_dir / name).exists()]
    if missing:
        report.add_error("E_META_MISSING", f"Missing meta files: {missing}", meta_dir=str(meta_dir))
        return None, None, [], []
    report.add_pass("P_META_FILES_PRESENT")

    info = read_json(meta_dir / "info.json")
    modality = None
    mod_path = meta_dir / "modality.json"
    if mod_path.exists():
        try:
            modality = read_json(mod_path)
            report.add_pass("P_MODALITY_JSON_READ")
        except Exception as e:
            report.add_warning("W_MODALITY_JSON_READ", f"Failed to read modality.json: {e}", path=str(mod_path))

    episodes = read_jsonl(meta_dir / "episodes.jsonl")
    tasks = read_jsonl(meta_dir / "tasks.jsonl")

    # tasks sanity
    task_indices = []
    task_map: Dict[int, str] = {}
    dup = False
    for i, row in enumerate(tasks):
        if "task_index" not in row or "task" not in row:
            report.add_error("E_TASKS_FORMAT", "tasks.jsonl row missing task_index/task", row=i, value=row)
            continue
        ti = safe_int(row["task_index"], "task_index", f"tasks.jsonl row {i}")
        t = str(row["task"])
        if ti in task_map:
            dup = True
        task_map[ti] = t
        task_indices.append(ti)

    if dup:
        report.add_error("E_TASKS_DUP_INDEX", "Duplicate task_index values in tasks.jsonl")
    else:
        report.add_pass("P_TASKS_UNIQUE_INDICES")

    if 1 not in task_map or task_map.get(1) != "valid":
        report.add_warning("W_TASKS_VALID_RESERVED",
                           'Expected task_index 1 to be "valid" (per your convention).',
                           found=task_map.get(1))
    else:
        report.add_pass("P_TASKS_VALID_RESERVED")

    # episodes sanity
    ep_indices = []
    for i, row in enumerate(episodes):
        if "episode_index" not in row or "length" not in row or "tasks" not in row:
            report.add_error("E_EPISODES_FORMAT", "episodes.jsonl row missing episode_index/length/tasks", row=i, value=row)
            continue
        ei = safe_int(row["episode_index"], "episode_index", f"episodes.jsonl row {i}")
        ln = safe_int(row["length"], "length", f"episodes.jsonl row {i}")
        ep_indices.append(ei)
        if ln <= 0:
            report.add_error("E_EPISODE_LENGTH_NONPOSITIVE", f"Episode length must be > 0, got {ln}", episode_index=ei)
        tasks_list = row.get("tasks")
        if not isinstance(tasks_list, list) or not tasks_list:
            report.add_error("E_EPISODE_TASKS_EMPTY", "Episode tasks list missing/empty", episode_index=ei, tasks=tasks_list)
        else:
            # ensure contains "valid" string
            if "valid" not in [str(x) for x in tasks_list]:
                report.add_warning("W_EPISODE_TASKS_NO_VALID", 'Episode tasks does not include "valid"', episode_index=ei, tasks=tasks_list)

    # zero-index + consecutive (warn, not error, because some datasets may sample)
    if ep_indices:
        ep_sorted = sorted(ep_indices)
        if ep_sorted[0] != 0:
            report.add_warning("W_EPISODES_NOT_ZERO_INDEXED", "First episode_index is not 0", first=ep_sorted[0])
        # check gaps
        expected = list(range(ep_sorted[0], ep_sorted[-1] + 1))
        missing_eps = sorted(set(expected) - set(ep_sorted))
        if missing_eps:
            report.add_warning("W_EPISODES_GAPS", "episode_index has gaps (may be ok, but unusual)", missing_count=len(missing_eps), examples=missing_eps[:10])
        else:
            report.add_pass("P_EPISODES_CONSECUTIVE_OR_COMPLETE")

    # info sanity cross-checks
    total_episodes_info = info.get("total_episodes")
    if total_episodes_info is not None:
        try:
            te = int(total_episodes_info)
            if te != len(episodes):
                report.add_error("E_INFO_TOTAL_EPISODES_MISMATCH",
                                 "info.json total_episodes != number of episodes.jsonl lines",
                                 info_total_episodes=te, episodes_jsonl=len(episodes))
            else:
                report.add_pass("P_INFO_TOTAL_EPISODES_MATCH")
        except Exception:
            report.add_warning("W_INFO_TOTAL_EPISODES_BAD", "info.json total_episodes not int-like", value=total_episodes_info)

    chunks_size = int(info.get("chunks_size") or 0)
    if chunks_size <= 0:
        report.add_warning("W_INFO_CHUNKS_SIZE", "info.json chunks_size missing or <= 0; chunk checks may be unreliable", chunks_size=chunks_size)

    # total_chunks check (warn if mismatch)
    total_chunks_info = info.get("total_chunks")
    if total_chunks_info is not None and chunks_size > 0:
        try:
            tc = int(total_chunks_info)
            computed = int(math.ceil(len(episodes) / chunks_size)) if len(episodes) else 0
            if tc != computed:
                report.add_warning("W_INFO_TOTAL_CHUNKS_MISMATCH",
                                   "info.json total_chunks != computed ceil(total_episodes/chunks_size)",
                                   info_total_chunks=tc, computed_total_chunks=computed)
            else:
                report.add_pass("P_INFO_TOTAL_CHUNKS_MATCH")
        except Exception:
            report.add_warning("W_INFO_TOTAL_CHUNKS_BAD", "info.json total_chunks not int-like", value=total_chunks_info)

    return info, modality, episodes, tasks

def check_parquets_and_indices(
    report: Report,
    root: Path,
    info: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    *,
    max_episodes: Optional[int],
    allow_missing_annotations: bool,
    relax_done_check: bool,
    check_timestamps: bool,
) -> Tuple[int, Optional[int], Optional[int], Dict[int, int]]:
    """
    Returns:
      total_rows, global_min_index, global_max_index, per_episode_rows
    """
    task_map = {safe_int(r["task_index"], "task_index", "tasks.jsonl"): str(r["task"]) for r in tasks if "task_index" in r and "task" in r}

    chunks_size = int(info.get("chunks_size") or 0)
    if chunks_size <= 0:
        chunks_size = 10**9  # effectively one chunk

    required_cols = required_parquet_columns()
    if allow_missing_annotations:
        required_cols = [c for c in required_cols if not c.startswith("annotation.")]

    total_rows = 0
    global_indices: List[int] = []
    per_episode_rows: Dict[int, int] = {}

    # iterate episodes
    eps_iter = episodes[: max_episodes] if (max_episodes is not None and max_episodes > 0) else episodes

    for i, ep in enumerate(eps_iter):
        ei = safe_int(ep.get("episode_index"), "episode_index", f"episodes.jsonl row {i}")
        expected_len = safe_int(ep.get("length"), "length", f"episodes.jsonl row {i}")

        pq_path = expected_parquet_path(root, ei, chunks_size)
        if not pq_path.exists():
            report.add_error("E_PARQUET_MISSING", "Missing parquet file for episode", episode_index=ei, path=str(pq_path))
            continue

        try:
            df = pd.read_parquet(pq_path)
        except Exception as e:
            report.add_error("E_PARQUET_READ", f"Failed to read parquet: {e}", episode_index=ei, path=str(pq_path))
            continue

        n = len(df)
        per_episode_rows[ei] = n
        total_rows += n

        if n != expected_len:
            report.add_error("E_EPISODE_ROWCOUNT_MISMATCH",
                             "Parquet row count != episodes.jsonl length",
                             episode_index=ei, expected=expected_len, found=n, path=str(pq_path))

        # required columns
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report.add_error("E_PARQUET_MISSING_COLS",
                             "Parquet missing required columns",
                             episode_index=ei, missing_cols=missing_cols, path=str(pq_path))
            # still continue with what we can

        # episode consistency fields
        for col in ("episode_index", "episode_chunk", "episode_id"):
            if col in df.columns:
                uniq = df[col].dropna().unique()
                if len(uniq) > 1:
                    report.add_error("E_EPISODE_FIELD_NOT_CONSTANT",
                                     f"Column '{col}' varies within episode parquet",
                                     episode_index=ei, column=col, unique_values=[str(x) for x in uniq[:5]], path=str(pq_path))

        if "episode_index" in df.columns:
            # ensure correct
            bad = df["episode_index"].dropna().astype(int) != int(ei)
            if bad.any():
                report.add_error("E_EPISODE_INDEX_MISMATCH_IN_PARQUET",
                                 "episode_index column does not match episode file index",
                                 episode_index=ei, bad_rows=int(bad.sum()), path=str(pq_path))

        # shapes of vectors (sample a few rows for speed)
        for vec_col, expected_dim in (("observation.state", 44), ("action", 44)):
            if vec_col in df.columns and n > 0:
                sample_idx = [0, n // 2, n - 1] if n >= 3 else list(range(n))
                for si in sample_idx:
                    arr = parse_vector_cell(df.iloc[si][vec_col])
                    if arr is None:
                        report.add_error("E_VECTOR_PARSE",
                                         f"Failed to parse vector column '{vec_col}'",
                                         episode_index=ei, row=int(si), column=vec_col, value=str(df.iloc[si][vec_col])[:120])
                        break
                    if arr.size != expected_dim:
                        report.add_error("E_VECTOR_DIM",
                                         f"Vector dim mismatch for '{vec_col}' (expected {expected_dim})",
                                         episode_index=ei, row=int(si), column=vec_col, expected=expected_dim, found=int(arr.size))
                        break

        # task_index validity
        if "task_index" in df.columns:
            try:
                # allow nulls
                ti_vals = df["task_index"].dropna().astype(int).unique()
                unknown = [int(t) for t in ti_vals if int(t) not in task_map]
                if unknown:
                    report.add_error("E_TASK_INDEX_UNKNOWN",
                                     "Parquet contains task_index not present in tasks.jsonl",
                                     episode_index=ei, unknown_task_indices=unknown[:20])
            except Exception as e:
                report.add_warning("W_TASK_INDEX_PARSE",
                                   f"Could not validate task_index values: {e}",
                                   episode_index=ei)

        # timestamp monotonic
        if check_timestamps and "timestamp" in df.columns and n > 1:
            ts = df["timestamp"]
            # coerce numeric where possible
            try:
                ts_num = pd.to_numeric(ts, errors="coerce")
                if ts_num.isna().all():
                    report.add_warning("W_TIMESTAMP_NONNUMERIC", "timestamp could not be parsed as numeric", episode_index=ei)
                else:
                    diffs = ts_num.diff()
                    bad = diffs.dropna() < 0
                    if bad.any():
                        # first offending index
                        first_bad_pos = int(bad.idxmax())
                        report.add_warning("W_TS_NONMONOTONIC",
                                           "timestamp decreases within episode",
                                           episode_index=ei, first_bad_row=first_bad_pos,
                                           prev=float(ts_num.iloc[first_bad_pos - 1]),
                                           curr=float(ts_num.iloc[first_bad_pos]))
            except Exception as e:
                report.add_warning("W_TIMESTAMP_CHECK_FAIL", f"Failed timestamp monotonic check: {e}", episode_index=ei)

        # done check
        if not relax_done_check and "next.done" in df.columns and n > 0:
            try:
                done = df["next.done"]
                # accept booleans or strings
                done_bool = done.astype(str).str.lower().isin(["true", "1", "t", "yes", "y"]) if done.dtype == object else done.astype(bool)
                # last row should be True
                if not bool(done_bool.iloc[-1]):
                    report.add_warning("W_DONE_LAST_FALSE",
                                       "Last row next.done is not True (may be dataset-specific)",
                                       episode_index=ei)
                # earlier rows should be False
                if done_bool.iloc[:-1].any():
                    report.add_warning("W_DONE_EARLY_TRUE",
                                       "Some non-last rows have next.done True (may be dataset-specific)",
                                       episode_index=ei, count=int(done_bool.iloc[:-1].sum()))
            except Exception as e:
                report.add_warning("W_DONE_CHECK_FAIL", f"Failed done check: {e}", episode_index=ei)

        # collect global indices
        if "index" in df.columns:
            try:
                idx = df["index"].dropna().astype(int).tolist()
                global_indices.extend(idx)
            except Exception as e:
                report.add_error("E_INDEX_PARSE", f"Failed to parse global index column: {e}", episode_index=ei, path=str(pq_path))

    # global index checks
    global_min = None
    global_max = None
    if global_indices:
        global_min = int(min(global_indices))
        global_max = int(max(global_indices))

        if global_min != 0:
            report.add_error("E_IDX_NOT_ZERO_BASED", "Global index min is not 0", min_index=global_min)

        # uniqueness + consecutiveness
        idx_sorted = sorted(global_indices)
        # duplicates
        dup_count = len(idx_sorted) - len(set(idx_sorted))
        if dup_count > 0:
            report.add_error("E_IDX_DUPLICATE", "Duplicate global indices found", duplicate_count=dup_count)

        expected_n = len(idx_sorted)
        # expected values 0..N-1
        if expected_n > 0:
            # Fast gap check without building huge list if dataset is big:
            # Compare against arithmetic series properties and spot-check missing ranges.
            # But to be exact, we do set difference if N isn't too large; else use run-length scan.
            N = expected_n
            if N <= 5_000_000:
                missing = sorted(set(range(0, N)) - set(idx_sorted))
                if missing:
                    report.add_error("E_IDX_GAP",
                                     "Global index has gaps (missing values in 0..N-1)",
                                     missing_count=len(missing),
                                     examples=missing[:20])
                else:
                    report.add_pass("P_IDX_CONSECUTIVE")
            else:
                # streaming gap scan
                gaps = []
                prev = idx_sorted[0]
                for cur in idx_sorted[1:]:
                    if cur == prev:
                        continue
                    if cur != prev + 1:
                        gaps.append((prev, cur))
                        if len(gaps) >= 20:
                            break
                    prev = cur
                if gaps:
                    report.add_error("E_IDX_GAP",
                                     "Global index has gaps (sampled)",
                                     gap_examples=gaps[:20],
                                     note="Dataset too large for full set difference; showing first gaps.")
                else:
                    report.add_pass("P_IDX_CONSECUTIVE_SAMPLE")

        # max should be N-1 if consecutive and 0-based
        if global_max != len(global_indices) - 1:
            report.add_error("E_IDX_MAX_MISMATCH",
                             "Global index max != total_rows-1 (indicates gaps/duplicates/mismatch)",
                             max_index=global_max, total_rows=len(global_indices), expected_max=len(global_indices) - 1)

    else:
        report.add_error("E_IDX_MISSING", "No global indices collected (missing 'index' column or parse failures)")

    return total_rows, global_min, global_max, per_episode_rows


def check_videos(
    report: Report,
    root: Path,
    info: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    video_keys: List[str],
    *,
    max_episodes: Optional[int],
    mode: str,
    tolerance_frames: int,
    tolerance_seconds: float,
) -> Tuple[Optional[int], Dict[Tuple[int, str], Optional[int]]]:
    """
    Returns:
      total_frames_sum (if counted), per_video_frames map ((episode_index, key)->frames or None)
    """
    chunks_size = int(info.get("chunks_size") or 0)
    if chunks_size <= 0:
        chunks_size = 10**9

    eps_iter = episodes[: max_episodes] if (max_episodes is not None and max_episodes > 0) else episodes

    per_video_frames: Dict[Tuple[int, str], Optional[int]] = {}
    total_frames_sum = 0
    counted_any = False

    for i, ep in enumerate(eps_iter):
        ei = safe_int(ep.get("episode_index"), "episode_index", f"episodes.jsonl row {i}")
        expected_len = safe_int(ep.get("length"), "length", f"episodes.jsonl row {i}")
        # We’ll collect per-key timing info for this episode
        last_frame_sec_by_key: Dict[str, float] = {}
        duration_sec_by_key: Dict[str, float] = {}

        # parquet last timestamp (episode-level)
        pq_path = expected_parquet_path(root, ei, chunks_size)
        parquet_last_ts = get_episode_last_timestamp_seconds(pq_path)
        for k in video_keys:
            vp = expected_video_path(root, ei, chunks_size, k)
            if not vp.exists():
                report.add_error("E_VIDEO_MISSING", "Missing video file", episode_index=ei, video_key=k, path=str(vp))
                per_video_frames[(ei, k)] = None
                continue

            # light decode sanity
            if mode in ("light", "heavy"):
                if not basic_video_readable(vp):
                    report.add_error("E_VIDEO_UNREADABLE", "Video file not readable/decodable", episode_index=ei, video_key=k, path=str(vp))

            if mode == "heavy":
                # count frames
                frames = ffprobe_frame_count(vp)
                if frames is None:
                    frames = opencv_frame_count(vp)

                per_video_frames[(ei, k)] = frames
                if frames is not None:
                    counted_any = True
                    total_frames_sum += frames
                    # compare frames with episode length (tolerant)
                    if abs(frames - expected_len) > tolerance_frames:
                        report.add_warning("W_VIDEO_FRAMES_MISMATCH",
                                           "Video frame count differs from episode length",
                                           episode_index=ei, video_key=k, expected_len=expected_len,
                                           frames=frames, tolerance=tolerance_frames, path=str(vp))
                else:
                    report.add_warning("W_VIDEO_FRAMES_UNKNOWN",
                                       "Could not determine video frame count (no ffprobe and OpenCV failed)",
                                       episode_index=ei, video_key=k, path=str(vp))
            # Timing checks require ffprobe
            if which("ffprobe"):
                dur = ffprobe_video_duration_seconds(vp)
                last_t = ffprobe_last_frame_time_seconds(vp, tail_seconds=1.0)

                if dur is not None:
                    duration_sec_by_key[k] = dur
                else:
                    report.add_warning("W_VIDEO_DURATION_UNKNOWN",
                                       "Could not read video duration via ffprobe",
                                       episode_index=ei, video_key=k, path=str(vp))

                if last_t is not None:
                    last_frame_sec_by_key[k] = last_t
                else:
                    report.add_warning("W_VIDEO_LASTFRAME_TIME_UNKNOWN",
                                       "Could not read last-frame timestamp via ffprobe",
                                       episode_index=ei, video_key=k, path=str(vp))

                # parquet timestamp <= video duration
                if parquet_last_ts is not None and dur is not None:
                    if parquet_last_ts > (dur + tolerance_seconds):
                        report.add_warning(
                            "W_PARQUET_TS_EXCEEDS_VIDEO_DURATION",
                            "Last parquet timestamp is greater than video duration",
                            episode_index=ei, video_key=k,
                            parquet_last_ts=parquet_last_ts,
                            video_duration_sec=dur,
                            tolerance_seconds=tolerance_seconds,
                            path=str(vp),
                        )
            else:
                # If no ffprobe, we can’t reliably do timing checks
                # (avoid spamming: warn once per episode)
                pass
        # Cross-camera last-frame seconds should match within tolerance
        if mode in ("light", "heavy"):
            if which("ffprobe"):
                if len(last_frame_sec_by_key) >= 2:
                    vals = list(last_frame_sec_by_key.values())
                    spread = max(vals) - min(vals)
                    if spread > tolerance_seconds:
                        report.add_warning(
                            "W_LAST_FRAME_SECONDS_MISMATCH",
                            "Last-frame timestamps differ across video keys for the same episode",
                            episode_index=ei,
                            spread_seconds=spread,
                            tolerance_seconds=tolerance_seconds,
                            last_frame_sec_by_key=last_frame_sec_by_key,
                        )
            else:
                # warn once per run would be nicer, but keep it simple:
                report.add_warning(
                    "W_NO_FFPROBE_TIMING_SKIPPED",
                    "ffprobe not found; skipped last-frame seconds and duration/timestamp checks",
                )

    return (total_frames_sum if counted_any else None), per_video_frames


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot dataset health checker")
    ap.add_argument("--root", required=True, help="Dataset root directory")
    ap.add_argument("--tolerance-seconds", type=float, default=0.05,
                help="Allowed seconds difference for last-frame alignment and timestamp<=duration checks")
    ap.add_argument("--mode", choices=["meta", "light", "heavy"], default="light")
    ap.add_argument("--report", default=None, help="Write machine-readable JSON report to this path")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures (exit code 2)")
    ap.add_argument("--max-episodes", type=int, default=None, help="Only check first N episodes (fast sampling)")
    ap.add_argument("--allow-missing-annotations", action="store_true", help="Do not error if annotation.* columns are missing")
    ap.add_argument("--relax-done-check", action="store_true", help="Do not warn about next.done conventions")
    ap.add_argument("--no-timestamp-check", action="store_true", help="Skip timestamp monotonic checks")
    ap.add_argument("--tolerance-frames", type=int, default=0, help="Allowed abs diff between video frames and episode length (heavy mode)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    report = Report(dataset_root=str(root), mode=args.mode, strict=args.strict)

    try:
        info, modality, episodes, tasks = check_meta(report, root)
        if info is None:
            # fatal meta failure
            finalize(report, args.report)
            return report.exit_code()

        # stats from info / meta
        video_keys = discover_video_keys_from_info(info)
        report.stats["video_keys"] = video_keys
        report.stats["num_video_keys"] = len(video_keys)
        report.stats["chunks_size"] = int(info.get("chunks_size") or 0)
        report.stats["fps"] = info.get("fps")
        report.stats["info_total_frames"] = info.get("total_frames")
        report.stats["info_total_episodes"] = info.get("total_episodes")
        report.stats["episodes_jsonl_count"] = len(episodes)

        # meta-only mode ends here
        if args.mode == "meta":
            finalize(report, args.report)
            return report.exit_code()

        # parquet checks
        total_rows, global_min, global_max, per_episode_rows = check_parquets_and_indices(
            report,
            root,
            info,
            episodes,
            tasks,
            max_episodes=args.max_episodes,
            allow_missing_annotations=args.allow_missing_annotations,
            relax_done_check=args.relax_done_check,
            check_timestamps=not args.no_timestamp_check,
        )
        report.stats["checked_episodes"] = len(episodes[: args.max_episodes] if args.max_episodes else episodes)
        report.stats["total_rows_parquet_sum"] = total_rows
        report.stats["global_index_min"] = global_min
        report.stats["global_index_max"] = global_max

        # video checks
        total_frames_sum, _ = check_videos(
            report,
            root,
            info,
            episodes,
            video_keys,
            max_episodes=args.max_episodes,
            mode=args.mode,
            tolerance_frames=max(0, args.tolerance_frames),
            tolerance_seconds=float(args.tolerance_seconds),
        )

        if args.mode == "heavy":
            report.stats["counted_total_frames_sum"] = total_frames_sum

            # dataset-level total_frames check if available
            info_total_frames = info.get("total_frames")
            if total_frames_sum is not None and info_total_frames is not None:
                try:
                    itf = int(info_total_frames)
                    if itf != total_frames_sum:
                        report.add_warning(
                            "W_TOTAL_FRAMES_MISMATCH",
                            "info.json total_frames != sum of counted video frames",
                            info_total_frames=itf,
                            counted_total_frames_sum=total_frames_sum,
                        )
                    else:
                        report.add_pass("P_TOTAL_FRAMES_MATCH")
                except Exception:
                    report.add_warning("W_INFO_TOTAL_FRAMES_BAD", "info.json total_frames not int-like", value=info_total_frames)

            # rule-of-thumb: total_frames / num_video_keys == total_rows
            if info_total_frames is not None and len(video_keys) > 0 and total_rows > 0:
                try:
                    itf = int(info_total_frames)
                    per_stream = itf / float(len(video_keys))
                    # allow tiny float error; should be integer-ish
                    if abs(per_stream - round(per_stream)) < 1e-6:
                        per_stream = int(round(per_stream))
                    if int(per_stream) != int(total_rows):
                        report.add_warning(
                            "W_FRAMES_PER_STREAM_ROW_MISMATCH",
                            "info.json total_frames/num_video_keys != total_rows (global index count)",
                            total_frames=itf,
                            num_video_keys=len(video_keys),
                            frames_per_stream=per_stream,
                            total_rows=total_rows,
                        )
                    else:
                        report.add_pass("P_FRAMES_PER_STREAM_MATCH_ROWS")
                except Exception:
                    report.add_warning("W_FRAMES_RULE_CHECK_FAIL", "Failed to validate total_frames/num_video_keys rule")

        finalize(report, args.report)
        return report.exit_code()

    except Exception as e:
        report.add_error("E_FATAL", f"Unhandled exception: {e}", exception_type=type(e).__name__)
        finalize(report, args.report)
        return report.exit_code()


def finalize(report: Report, report_path: Optional[str]) -> None:
    # Print summary
    print("=" * 80)
    print(f"Dataset: {report.dataset_root}")
    print(f"Mode: {report.mode} | Strict: {report.strict}")
    if report.stats:
        # concise stats
        keys = ["episodes_jsonl_count", "checked_episodes", "total_rows_parquet_sum", "num_video_keys", "chunks_size", "fps"]
        parts = []
        for k in keys:
            if k in report.stats and report.stats[k] is not None:
                parts.append(f"{k}={report.stats[k]}")
        if parts:
            print("Stats:", ", ".join(parts))

    print(f"Passed: {len(report.passed)} | Warnings: {len(report.warnings)} | Errors: {len(report.errors)}")
    status = report.status()
    status_str = "✅ PASS" if status == "pass" else ("⚠️ FAIL (warnings treated as errors)" if status == "fail_warnings" else "❌ FAIL")
    print("Status:", status_str)
    print("-" * 80)

    if report.errors:
        print(f"ERRORS ({len(report.errors)}):")
        for e in report.errors[:50]:
            print(f"  - {e.code}: {e.message}")
            if e.context:
                print(f"    context: {json.dumps(e.context, ensure_ascii=False)}")
        if len(report.errors) > 50:
            print(f"  ... {len(report.errors) - 50} more errors omitted")
        print("-" * 80)

    if report.warnings:
        print(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings[:50]:
            print(f"  - {w.code}: {w.message}")
            if w.context:
                print(f"    context: {json.dumps(w.context, ensure_ascii=False)}")
        if len(report.warnings) > 50:
            print(f"  ... {len(report.warnings) - 50} more warnings omitted")
        print("-" * 80)

    # Write JSON report
    if report_path:
        rp = Path(report_path).expanduser().resolve()
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            with rp.open("w", encoding="utf-8") as f:
                json.dump(report.to_json(), f, indent=2, ensure_ascii=False)
            print(f"Wrote report: {rp}")
        except Exception as e:
            print(f"Failed to write report to {rp}: {e}", file=sys.stderr)

    print("=" * 80)


if __name__ == "__main__":
    raise SystemExit(main())