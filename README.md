# capstone-vla

**Scaling VLA Fine-Tuning with Synthetic Data Generation**

Investigates whether simulation-generated synthetic demonstrations can augment real-world data to improve fine-tuned Vision-Language-Action (VLA) policy robustness for humanoid robot manipulation. Evaluated on the **KuavoV4Pro 34-DOF humanoid** (LejuRobotics) performing a bimanual liquid-pouring task.

> **Research question:** Can synthetic data generated via simulation augment real demonstrations for a target task and embodiment to improve fine-tuned VLA policy robustness?

---

## Repository Structure

```
capstone-vla/
├── dataset_tools/                      ← Shared data utilities (conversion, mixing, validation)
│   ├── hdf5_to_lerobot.py             ← Isaac Lab HDF5 → LeRobot v2.1 converter
│   ├── data_mixer.py                  ← Episode-level dataset mixer (A + x% B → C)
│   ├── bake_task_prompts_into_parquet.py ← Bakes task string into Parquet for Pi0
│   ├── converter.py                   ← LeRobot v2.0 → v2.1 format converter
│   ├── lerobot_health.py              ← Dataset integrity validator (meta / light / heavy modes)
│   ├── download_dataset.py            ← Downloads datasets from HuggingFace
│   ├── upload_checkpoints.py          ← Uploads fine-tuned checkpoints to HuggingFace
│   ├── parquet_to_csv.py              ← Exports Parquet to CSV for inspection
│   ├── environment.yml                ← Conda env for dataset tools
│   └── evaluate_mse/
│       ├── replay_dataset_tcp.py      ← Open-loop MSE evaluation via TCP/policy server
│       └── environment.yml
│
├── lab/
│   └── IsaacLab/                      ← Submodule: Isaac Lab (sim + synthetic data gen)
│
└── third_party/
    ├── gr00t/                          ← Submodule: GR00T N1.6-3B (primary VLA)
    ├── diffusion_policy/               ← Submodule: Diffusion Policy (IL baseline)
    └── openpi/                         ← Submodule: OpenPI / π₀.₅ (VLA)
```

---

## Full Pipeline

```
1. Collect real demos       Meta Quest 3 → Drake IK → Isaac Lab teleoperation
                            Output: raw HDF5 (34D joint actions + 3× RGB cameras)

2. Generate synthetic data  MimicGen in Isaac Lab (annotate subtask boundaries → generate)
                            Output: synthetic HDF5 (same schema)

3. Convert to LeRobot       hdf5_to_lerobot.py
                            34D sim layout → 44D real layout
                            10D sim hand → 6D real hand (MCP joints only)
                            RGBA frames → MP4 video per camera
                            Output: LeRobot v2.1 dataset (Parquet + MP4)

4. Mix datasets             data_mixer.py (e.g. 90% real + 10% synthetic, seed=0)
                            Episode-level random sampling, provenance tracked in info.json
                            Output: mixed LeRobot dataset

5. Fine-tune models         GR00T N1.6-3B on RunPod cloud GPUs
                            (Diffusion Policy and π₀.₅ were scoped out due to time)
                            Output: model checkpoints → HuggingFace

6. Deploy & evaluate        Flask HTTP inference server → ROS1 Noetic controller → KuavoV4Pro
                            All models share a common /predict HTTP interface
```

---

## Dataset Tools

### `hdf5_to_lerobot.py`

Converts Isaac Lab HDF5 output to LeRobot v2.1 format for the KuavoV4Pro pouring task.

**Joint layout mapping (34D sim → 44D real):**

| Segment | Real indices | Source |
|---------|-------------|--------|
| Left arm (7 DOF) | [0:7] | `actions[0:7]` |
| Left hand (6 DOF) | [7:13] | `actions[14:24]`, extract MCP joints [0,1,2,4,6,8] |
| Left leg (6 DOF) | [13:19] | Zeros |
| Neck (3 DOF) | [19:22] | Zeros |
| Right arm (7 DOF) | [22:29] | `actions[7:14]` |
| Right hand (6 DOF) | [29:35] | `actions[24:34]`, extract MCP joints [0,1,2,4,6,8] |
| Right leg (6 DOF) | [35:41] | Zeros |
| Waist (3 DOF) | [41:44] | Zeros |

**Hand sim→real conversion:** 10D sim hand (thumbCMC, thumbMCP, indexMCP, indexPIP, middleMCP, middlePIP, ringMCP, ringPIP, littleMCP, littlePIP) is reduced to 6D by selecting MCP joints only, then linearly rescaled from sim URDF limits [0, upper] to real robot limits [REAL_LOWER, REAL_UPPER].

**Camera mapping:**

| HDF5 key | LeRobot folder |
|----------|---------------|
| `cam_egoview_rgb` | `observation.images.ego_view` |
| `cam_leftwrist_rgb` | `observation.images.left_wrist_view` |
| `cam_rightwrist_rgb` | `observation.images.right_wrist_view` |

**Trimming options:** `--trim-start <seconds>`, `--trim-end <seconds>`, `--auto-trim` (jerk detection via joint velocity). Useful for removing snap-to-ready frames at episode start.

**Usage:**
```bash
python hdf5_to_lerobot.py \
  --input /path/to/output_dataset.hdf5 \
  --output /path/to/lerobot_dataset \
  --task "Use both hands to pour contents from the cup into the bowl" \
  --fps 20.0 \
  --trim-start 0.5
```

---

### `data_mixer.py`

Mixes two LeRobot datasets at a configurable episode ratio (A + x% of B → C).

- Chunking is episode-based: `episode_chunk = floor(episode_index / chunks_size)`
- Global `index` column is rewritten to be contiguous 0..N-1 across the output dataset
- `task_index` values are remapped on merge (deduplication by task string)
- Provenance is stored in `info.json` under a `"mixing"` key for later validation
- Refuses to mix if `modality.json` differs between A and B
- Includes a `check` subcommand for standalone validation

**Usage:**
```bash
# Mix
python data_mixer.py mix \
  --dataset-a /path/to/real_dataset \
  --dataset-b /path/to/synthetic_dataset \
  --out /path/to/mixed_dataset \
  --percent-b 10 \
  --seed 0 \
  --force

# Validate
python data_mixer.py check --dataset /path/to/mixed_dataset
```

---

### `bake_task_prompts_into_parquet.py`

Writes canonical task description strings into Parquet files by mapping `task_index → task_text` from `tasks.jsonl`. Required for Pi0 / π₀.₅ which consumes task strings directly rather than integer indices.

---

### `converter.py`

Converts LeRobot datasets from codebase version `v2.0` to `v2.1` (generates per-episode stats, removes deprecated `stats.json`). Can operate on local datasets or push to HuggingFace.

```bash
python converter.py \
  --repo-id data/Hf_data/Lusmse/realWorldPouring \
  --root . \
  --push-to-hub false
```

---

### `lerobot_health.py`

Multi-mode dataset validator. Checks episode counts, Parquet schema, video file existence, frame count consistency, and global index integrity.

```bash
python lerobot_health.py --root /path/to/dataset --mode heavy --strict
```

Exit codes: `0` = pass, `1` = errors, `2` = warnings (with `--strict`).

---

### `evaluate_mse/replay_dataset_tcp.py`

Open-loop MSE evaluation: replays Parquet + MP4 episodes through a running policy server over TCP, prints predicted actions, and computes MSE vs ground-truth actions. Wire protocol: msgpack dict with 4-byte big-endian length prefix.

```bash
python replay_dataset_tcp.py \
  --root /path/to/lerobot_dataset \
  --host 127.0.0.1 --port 5555 \
  --episodes 10 \
  --print-every 20
```

---

## Submodules

| Submodule | Upstream | What was modified |
|-----------|----------|-------------------|
| `lab/IsaacLab` | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | KuavoV4Pro 34-DOF environment, Pink IK controller (NullSpacePostureTask for elbow flip fix), MimicGen pouring task, ZeroMQ teleoperation device, liquid particle terminations, domain randomisation, `InteractiveScene` state patch for `RigidObjectCollection`, cloud setup scripts |
| `third_party/gr00t` | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | KuavoV4Pro embodiment config, Flask HTTP inference server (`/predict` endpoint), policy fallback patch, 44D modality config |
| `third_party/diffusion_policy` | [LejuRobotics/kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) | TCP and HTTP deployment servers matching the GR00T `/predict` interface |
| `third_party/openpi` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | KuavoV4Pro training config (26D arm+hand slice), Flask/TCP servers, cloud automation scripts |

Each submodule has its own README.

---

## Setup

```bash
git clone --recurse-submodules https://github.com/Muslinmin/capstone-vla.git
cd capstone-vla

# Dataset tools conda environment
conda env create -f dataset_tools/environment.yml
conda activate capstone-vla

# For MSE evaluation tools
conda env create -f dataset_tools/evaluate_mse/environment.yml

# Individual model repos — see each submodule's README for setup
```

---

## Model Deployment

All three model servers expose the same HTTP `/predict` endpoint, enabling the ROS1 controller to switch between them by changing `model_url`.

| Model | Server | Default port |
|-------|--------|-------------|
| GR00T N1.6-3B | `third_party/gr00t/scripts/deployment/http_gr00t_server.py` | 5050 |
| Diffusion Policy | `third_party/diffusion_policy/kuavo_deploy/diffusion_http_server.py` | 8000 |
| π₀ / π₀.₅ | `third_party/openpi/src/openpi/serving/openpi_http_server.py` | 8001 |

The ROS1 controller runs inside a Docker container (ROS1 Noetic) and sends proprioception + camera frames to the inference server, which returns a 44D joint position command.

---

## Data & Checkpoints

| Resource | Link |
|----------|------|
| Real + synthetic datasets | [Lusmse/syn_realDataset](https://huggingface.co/datasets/Lusmse/syn_realDataset) |
| Fine-tuned checkpoints | [Lusmse/capstone-vla-checkpoints](https://huggingface.co/Lusmse/capstone-vla-checkpoints) |

Checkpoints can be uploaded/managed via `dataset_tools/upload_checkpoints.py`.

---

## Hardware & Software Stack

| Component | Specification |
|-----------|--------------|
| Robot | KuavoV4Pro 34-DOF humanoid (LejuRobotics) |
| Primary VLA | NVIDIA GR00T N1.6-3B |
| Simulation | Isaac Lab + Isaac Sim + PhysX 5 |
| Synthetic data | MimicGen (integrated into Isaac Lab) |
| Dataset format | LeRobot v2.1 (Parquet + MP4) |
| IK solver | Pink (Pinocchio-based, with NullSpacePostureTask) |
| Teleoperation | Meta Quest 3 → ZeroMQ bridge |
| Cameras | Intel RealSense D435 (ego), D405 (wrist) |
| Deployment | ROS1 Noetic, Flask HTTP inference server |
| Cloud compute | RunPod |

---

## Known Bugs Fixed

| Bug | Fix |
|-----|-----|
| IK elbow flips during teleoperation | Added `NullSpacePostureTask` in Pink IK config |
| `RigidObjectCollection` state not captured by `InteractiveScene.get_state()` | Monkey-patched `get_state()` / `reset_to()` |
| `task_index` stuck at 0 in MimicGen-generated parquet | Fixed prompt indexing in annotation loop |
| ROS control loop sleep stacking under load | Replaced with dedicated fixed-rate control thread |
| MimicGen EEF action field incorrect | Sourced from `obs/robot_joint_pos` instead of EEF |

---

## License

MIT. Individual submodules retain their upstream licenses: BSD-3 for IsaacLab, Apache 2.0 for GR00T and OpenPI, project-specific license for Diffusion Policy.
