#!/usr/bin/env python3
"""
upload_checkpoints.py v2
========================
Uses upload_folder() instead of upload_file() to batch all files into
a single commit per run — avoids HuggingFace's 128 commits/hour limit.

Usage
-----
  python upload_checkpoints_v2.py \\
      --repo  Lusmse/capstone-vla-checkpoints \\
      --base  /workspace/capstone-vla

Optional flags:
  --model   gr00t|openpi|dp   upload only one model (default: all)
  --dry-run                   print what would be uploaded, don't upload
  --reauth                    force re-entry of token even if cached
"""

import argparse
import getpass
import os
import sys
import tempfile
import shutil
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Run:  pip install huggingface_hub")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "gr00t": {
        "third_party_dir": "gr00t",
        "chkpt_subdir":    "chkpt_rlwd",
        "hf_prefix":       "gr00t",
    },
    "openpi": {
        "third_party_dir": "openpi",
        "chkpt_subdir":    "checkpoints",
        "hf_prefix":       "openpi",
    },
    "dp": {
        "third_party_dir": "diffusion_policy",
        "chkpt_subdir":    os.path.join("outputs", "train"),
        "hf_prefix":       "dp",
    },
}

TOKEN_FILENAME = ".hf_token"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def resolve_token(base: Path, reauth: bool) -> str:
    token_file = base / TOKEN_FILENAME

    if not reauth:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            print("[AUTH] Using token from HF_TOKEN environment variable.")
            return env_token

    if not reauth and token_file.exists():
        cached = token_file.read_text().strip()
        if cached:
            print(f"[AUTH] Using cached token from {token_file}")
            return cached

    print("\n" + "=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print("  Make sure it has WRITE permission.")
    print()
    token = getpass.getpass("  Paste your HF token (input hidden): ").strip()

    if not token:
        print("[ERROR] No token entered. Exiting.")
        sys.exit(1)

    save = input("  Save token to disk for future runs? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        token_file.write_text(token)
        token_file.chmod(0o600)
        print(f"  Token saved to {token_file}")

    return token


def resolve_base(base_arg: str) -> Path:
    candidates = [
        Path(base_arg),
        Path("/workspace/capstone-vla"),
        Path("/home/sensethreat/lab_mimic/VLA_IL/capstone-vla"),
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    print("[ERROR] Could not find capstone-vla directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core upload logic  — ONE commit per run via upload_folder()
# ---------------------------------------------------------------------------

def upload_model(api: HfApi, repo_id: str, chkpt_root: Path,
                 hf_prefix: str, dry_run: bool) -> list:
    if not chkpt_root.exists():
        print(f"  [WARN] Checkpoint directory does not exist: {chkpt_root}")
        return []

    run_dirs = sorted([d for d in chkpt_root.iterdir() if d.is_dir()])
    if not run_dirs:
        print(f"  No run directories found under {chkpt_root}")
        return []

    uploaded_runs = []

    for run_dir in run_dirs:
        file_count = sum(1 for _ in run_dir.rglob("*") if _.is_file())
        if file_count == 0:
            print(f"  [SKIP] Empty run dir: {run_dir.name}")
            continue

        path_in_repo = f"{hf_prefix}/{run_dir.name}"
        total_mb = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / 1024**2

        print(f"\n  Run: {run_dir.name}  ({file_count} files, {total_mb:.1f} MB total)")
        print(f"  -> uploading as single commit to: {path_in_repo}/")

        if dry_run:
            print(f"  [DRY-RUN] Would upload {run_dir} -> {path_in_repo}")
            uploaded_runs.append(run_dir.name)
            continue

        try:
            api.upload_folder(
                folder_path=str(run_dir),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add {hf_prefix}/{run_dir.name} ({file_count} files)",
                # upload_folder handles large files automatically via LFS
            )
            print(f"  [OK] Uploaded {run_dir.name}")
            uploaded_runs.append(run_dir.name)
        except Exception as e:
            print(f"  [ERROR] Failed to upload {run_dir.name}: {e}")

    return uploaded_runs


def make_readme(models_uploaded: dict) -> str:
    lines = [
        "---", "license: mit", "tags:", "  - robotics", "  - vla",
        "  - manipulation", "  - sim-to-real", "---", "",
        "# Capstone VLA Checkpoints", "",
        "Training checkpoints for VLA models trained on a pouring task",
        "using a KuavoV4Pro humanoid robot.", "", "## Models", "",
    ]
    for key, runs in models_uploaded.items():
        lines.append(f"### `{key}/`")
        for r in runs:
            lines.append(f"- `{r}`")
        if not runs:
            lines.append("- *(no runs uploaded yet)*")
        lines.append("")
    lines += [
        "## Pipeline", "",
        "1. Teleoperation recording — Isaac Lab + ROS ZeroMQ bridge (34 DOF)",
        "2. Synthetic data generation — Isaac Lab Mimic",
        "3. Photorealistic augmentation — NVIDIA Cosmos Transfer 2.5",
        "4. VLA training — gr00t n1.6 · pi0.5 · Diffusion Policy", "",
        "---", "_Auto-generated by `upload_checkpoints_v2.py`_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload VLA checkpoints to HuggingFace.")
    parser.add_argument("--repo",    type=str, required=True)
    parser.add_argument("--base",    type=str, default="auto")
    parser.add_argument("--model",   type=str, default=None, choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reauth",  action="store_true")
    args = parser.parse_args()

    base = resolve_base(args.base if args.base != "auto" else "")
    third_party = base / "third_party"

    print(f"[INFO] capstone-vla root : {base}")
    print(f"[INFO] Target HF repo    : {args.repo}")
    print(f"[INFO] Dry run           : {args.dry_run}")

    token = resolve_token(base, args.reauth)
    api = HfApi(token=token)

    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        sys.exit(1)

    if not args.dry_run:
        create_repo(repo_id=args.repo, repo_type="model",
                    private=False, exist_ok=True, token=token)
        print(f"[INFO] Repo ready: https://huggingface.co/{args.repo}")

    models_to_upload = (
        {args.model: MODEL_CONFIG[args.model]}
        if args.model else MODEL_CONFIG
    )

    all_uploaded = {}
    for model_key, cfg in models_to_upload.items():
        chkpt_root = third_party / cfg["third_party_dir"] / cfg["chkpt_subdir"]
        print(f"\n{'='*60}")
        print(f"Model  : {model_key}")
        print(f"Source : {chkpt_root}")
        print(f"{'='*60}")

        runs = upload_model(
            api=api,
            repo_id=args.repo,
            chkpt_root=chkpt_root,
            hf_prefix=cfg["hf_prefix"],
            dry_run=args.dry_run,
        )
        all_uploaded[model_key] = runs

    # Update README
    readme_content = make_readme(all_uploaded)
    if not args.dry_run:
        try:
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="model",
                commit_message="Update README",
            )
            print("\n[INFO] README.md updated.")
        except Exception as e:
            print(f"\n[WARN] Could not update README: {e}")

    # Summary
    print("\n" + "="*60)
    print("UPLOAD SUMMARY")
    print("="*60)
    for model_key, runs in all_uploaded.items():
        status = f"{len(runs)} run(s)" if runs else "nothing uploaded"
        print(f"  {model_key:20s} -> {status}")
        for r in runs:
            print(f"    - {r}")

    if not args.dry_run:
        print(f"\nView your repo: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()