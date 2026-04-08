# capstone-vla

**Scaling VLA Fine-Tuning with Synthetic Data Generation** — benchmarking Vision-Language-Action and Imitation Learning models on the KuavoV4Pro 34-DOF humanoid robot with real and MimicGen-generated synthetic demonstrations.

---

## Overview

This is the top-level monorepo that ties together the full capstone pipeline: synthetic data generation, data conversion, model fine-tuning, and real-robot deployment. It contains shared dataset utilities in `dataset_tools/` and references the four component repositories as Git submodules.

```
capstone-vla/
├── dataset_tools/          ← Shared data conversion, mixing, and evaluation utilities
├── lab/IsaacLab            ← Submodule: Isaac Lab (synthetic data generation)
└── third_party/
    ├── gr00t               ← Submodule: GR00T N1.6 (VLA fine-tuning & deployment)
    ├── diffusion_policy    ← Submodule: Diffusion Policy (IL baseline)
    └── openpi              ← Submodule: OpenPI / π₀.₅ (VLA fine-tuning & deployment)
```

---

## Pipeline

```
1. Collect demos          Meta Quest 3 → Drake IK → Isaac Lab teleoperation
                          Output: raw HDF5 (34D joint actions + 3× RGB)

2. Generate synthetic     MimicGen in Isaac Lab (annotate → generate)
   data                   Output: synthetic HDF5

3. Convert to LeRobot     hdf5_to_lerobot.py (34D sim → 44D real layout, 10D→6D hands)
                          Output: LeRobot v2.1 (Parquet + MP4)

4. Mix datasets           data_mixer.py (e.g. 90% real + 10% synthetic)
                          Output: mixed LeRobot dataset

5. Fine-tune models       GR00T / Diffusion Policy / π₀.₅
                          Output: checkpoints

6. Deploy & evaluate      Flask HTTP server → ROS controller → KuavoV4Pro
                          All three models share the same /predict interface
```

---

## Dataset Tools

| Script | Purpose |
|--------|---------|
| `hdf5_to_lerobot.py` | Converts Isaac Lab HDF5 to LeRobot v2.1 format — handles joint layout mapping (34D sim → 44D real), hand joint reduction (10D → 6D MCP-only), and auto-detects teleop vs MimicGen source |
| `data_mixer.py` | Mixes two LeRobot datasets at a configurable ratio (e.g. 90:10 real-to-synthetic) with episode-based chunking and provenance tracking |
| `bake_task_prompts_into_parquet.py` | Writes canonical task prompts into Parquet files for consistent `task_index` alignment across real and synthetic data |
| `converter.py` | General-purpose data format conversion utilities |
| `lerobot_health.py` | Validates LeRobot dataset integrity (episode counts, video keys, Parquet schema) |
| `download_dataset.py` | Downloads datasets from HuggingFace |
| `upload_checkpoints.py` | Uploads fine-tuned checkpoints to HuggingFace |
| `parquet_to_csv.py` | Exports Parquet data to CSV for inspection |
| `evaluate_mse/replay_dataset_tcp.py` | Open-loop MSE evaluation via TCP replay against a running policy server |

---

## Submodules

| Submodule | Upstream | What was modified |
|-----------|----------|-------------------|
| `lab/IsaacLab` | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | KuavoV4Pro environment, Pink IK controller, MimicGen pouring task, teleoperation device, liquid particle terminations, domain randomisation, cloud setup scripts |
| `third_party/gr00t` | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | KuavoV4Pro embodiment config, Flask HTTP inference server, policy fallback patch |
| `third_party/diffusion_policy` | [LejuRobotics/kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) | TCP and HTTP deployment servers matching the GR00T `/predict` interface |
| `third_party/openpi` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | KuavoV4Pro training config (26D arm+hand slice), Flask/TCP servers, cloud automation scripts |

Each submodule has its own detailed README.

---

## Setup

```bash
git clone --recurse-submodules https://github.com/Muslinmin/capstone-vla.git
cd capstone-vla

# Dataset tools
conda env create -f dataset_tools/environment.yml
conda activate capstone-vla

# Individual model repos — see each submodule's README for setup
```

---

## Model Deployment (Interchangeable)

All three model servers expose the same HTTP `/predict` endpoint. The ROS controller switches between them by changing `model_url`:

| Model | Server location | Default port |
|-------|----------------|-------------|
| GR00T N1.6 | `third_party/gr00t/scripts/deployment/http_gr00t_server.py` | 5050 |
| Diffusion Policy | `third_party/diffusion_policy/kuavo_deploy/diffusion_http_server.py` | 8000 |
| π₀ / π₀.₅ | `third_party/openpi/src/openpi/serving/openpi_http_server.py` | 8001 |

---

## Data & Checkpoints (HuggingFace)

| Resource | Link |
|----------|------|
| Real + synthetic datasets | [Lusmse/syn_realDataset](https://huggingface.co/datasets/Lusmse/syn_realDataset) |
| Fine-tuned checkpoints | [Lusmse/capstone-vla-checkpoints](https://huggingface.co/Lusmse/capstone-vla-checkpoints) |

---

## License

MIT. Individual submodules retain their upstream licenses (BSD-3 for IsaacLab, Apache 2.0 for GR00T/OpenPI/Diffusion Policy).
