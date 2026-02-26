#!/usr/bin/env python3
"""
upload_checkpoints.py
=====================
Authenticate to HuggingFace and upload / update VLA model checkpoints.

Supported models (third_party subdirs):
  - gr00t             -> chkpts/
  - openpi            -> checkpoints/
  - diffusion_policy  -> outputs/train/

HF repo structure produced:
  <HF_USERNAME>/<REPO_NAME>/
  ├── README.md
  ├── gr00t/
  │   └── <run_folder>/   (full nested checkpoint structure preserved)
  ├── openpi/
  │   └── <run_folder>/
  └── dp/
      └── <run_folder>/

Usage
-----
  python upload_checkpoints.py \\
      --repo  your-username/capstone-vla-checkpoints \\
      --base  /workspace/capstone-vla

  On first run you will be prompted for your HF token.
  It is saved to <base>/.hf_token on your network volume so you only
  need to paste it once — subsequent runs load it automatically.

Optional flags:
  --model   gr00t|openpi|dp   upload only one model (default: all)
  --dry-run                   print what would be uploaded, don't upload
  --resume                    skip files already present in the repo
  --reauth                    force re-entry of token even if cached
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import EntryNotFoundError
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
        "chkpt_subdir":    "chkpts",
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
    """
    Token resolution priority:
      1. HF_TOKEN environment variable (if already exported in this shell)
      2. <base>/.hf_token file on disk  (persists on network volume across restarts)
      3. Interactive prompt  ->  optionally saved to disk for next time
    """
    token_file = base / TOKEN_FILENAME

    # 1. Env var
    if not reauth:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            print("[AUTH] Using token from HF_TOKEN environment variable.")
            return env_token

    # 2. Cached file on network volume
    if not reauth and token_file.exists():
        cached = token_file.read_text().strip()
        if cached:
            print(f"[AUTH] Using cached token from {token_file}")
            return cached

    # 3. Interactive prompt
    print()
    print("=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print("  Make sure it has  WRITE  permission.")
    print()
    token = getpass.getpass("  Paste your HF token (input hidden): ").strip()

    if not token:
        print("[ERROR] No token entered. Exiting.")
        sys.exit(1)

    # Offer to save to network volume
    save = input("  Save token to disk for future runs? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        token_file.write_text(token)
        token_file.chmod(0o600)  # owner read/write only
        print(f"  Token saved to {token_file}")
        print(f"  (Delete that file or run --reauth to change it)")
    else:
        print("  Token not saved — you will be prompted again next run.")

    print()
    return token


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_base(base_arg: str) -> Path:
    """Return the absolute path to the capstone-vla root."""
    candidates = [
        Path(base_arg),
        Path("/workspace/capstone-vla"),
        Path("/home/sensethreat/lab_mimic/VLA_IL/capstone-vla"),
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    print("[ERROR] Could not find capstone-vla directory.")
    print(f"        Tried: {[str(c) for c in candidates]}")
    print("        Pass the correct path with --base /path/to/capstone-vla")
    sys.exit(1)


def get_run_dirs(chkpt_root: Path) -> list:
    if not chkpt_root.exists():
        print(f"  [WARN] Checkpoint directory does not exist: {chkpt_root}")
        return []
    return sorted([d for d in chkpt_root.iterdir() if d.is_dir()])


def collect_files(run_dir: Path) -> list:
    return sorted([f for f in run_dir.rglob("*") if f.is_file()])


def remote_file_exists(api: HfApi, repo_id: str, path_in_repo: str) -> bool:
    try:
        api.get_paths_info(repo_id=repo_id, paths=[path_in_repo], repo_type="model")
        return True
    except EntryNotFoundError:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

def make_readme(models_uploaded: dict) -> str:
    lines = [
        "---",
        "license: mit",
        "tags:",
        "  - robotics",
        "  - vla",
        "  - manipulation",
        "  - sim-to-real",
        "---",
        "",
        "# Capstone VLA Checkpoints",
        "",
        "Training checkpoints for Vision-Language-Action (VLA) models trained on a",
        "pouring task using a KuavoV4Pro humanoid robot.",
        "",
        "## Models",
        "",
    ]
    for key, runs in models_uploaded.items():
        lines.append(f"### `{key}/`")
        if runs:
            for r in runs:
                lines.append(f"- `{r}`")
        else:
            lines.append("- *(no runs uploaded yet)*")
        lines.append("")

    lines += [
        "## Pipeline",
        "",
        "1. **Teleoperation recording** — Isaac Lab + ROS ZeroMQ bridge (34 DOF)",
        "2. **Synthetic data generation** — Isaac Lab Mimic",
        "3. **Photorealistic augmentation** — NVIDIA Cosmos Transfer 2.5",
        "4. **VLA training** — gr00t n1.6 · pi0.5 · Diffusion Policy",
        "",
        "---",
        "_Auto-generated by `upload_checkpoints.py`_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core upload logic
# ---------------------------------------------------------------------------

def upload_model(api, repo_id, chkpt_root, hf_prefix, dry_run, resume) -> list:
    run_dirs = get_run_dirs(chkpt_root)
    if not run_dirs:
        print(f"  No run directories found under {chkpt_root}")
        return []

    uploaded_runs = []

    for run_dir in run_dirs:
        files = collect_files(run_dir)
        if not files:
            print(f"  [SKIP] Empty run dir: {run_dir.name}")
            continue

        print(f"\n  Run: {run_dir.name}  ({len(files)} files)")
        run_uploaded = False

        for local_file in files:
            relative_to_chkpt = local_file.relative_to(chkpt_root)
            path_in_repo = f"{hf_prefix}/{relative_to_chkpt}".replace("\\", "/")

            if resume and remote_file_exists(api, repo_id, path_in_repo):
                print(f"    [SKIP - exists] {path_in_repo}")
                continue

            file_size_mb = local_file.stat().st_size / (1024 ** 2)
            tag = "[DRY-RUN] " if dry_run else ""
            print(f"    {tag}Uploading  {path_in_repo}  ({file_size_mb:.1f} MB)")

            if not dry_run:
                try:
                    api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"Add {hf_prefix}/{run_dir.name}",
                    )
                    run_uploaded = True
                except Exception as e:
                    print(f"    [ERROR] Failed: {e}")

        if run_uploaded or dry_run:
            uploaded_runs.append(run_dir.name)

    return uploaded_runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload VLA checkpoints to HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo",     type=str, required=True,
                        help="HF repo id, e.g. your-username/capstone-vla-checkpoints")
    parser.add_argument("--base",     type=str, default="auto",
                        help="Path to capstone-vla root (auto-detected if omitted)")
    parser.add_argument("--model",    type=str, default=None,
                        choices=list(MODEL_CONFIG.keys()),
                        help="Upload only this model (default: all)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be uploaded without actually uploading")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip files already present in the remote repo")
    parser.add_argument("--reauth",   action="store_true",
                        help="Force re-entry of token even if one is cached")
    args = parser.parse_args()

    # --- Base path ---
    base = resolve_base(args.base if args.base != "auto" else "")
    third_party = base / "third_party"

    print(f"[INFO] capstone-vla root : {base}")
    print(f"[INFO] third_party dir   : {third_party}")
    print(f"[INFO] Target HF repo    : {args.repo}")
    print(f"[INFO] Dry run           : {args.dry_run}")
    print(f"[INFO] Resume mode       : {args.resume}")

    # --- Token (interactive prompt if needed) ---
    token = resolve_token(base, args.reauth)

    # --- API client ---
    api = HfApi(token=token)

    # --- Verify token before doing anything ---
    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("        Run with --reauth to enter a new token.")
        sys.exit(1)

    # --- Create repo if it doesn't exist ---
    if not args.dry_run:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=token,
        )
        print(f"[INFO] Repo ready: https://huggingface.co/{args.repo}")
    else:
        print(f"[DRY-RUN] Would create/verify repo: {args.repo}")

    # --- Select models ---
    models_to_upload = (
        {args.model: MODEL_CONFIG[args.model]}
        if args.model
        else MODEL_CONFIG
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
            resume=args.resume,
        )
        all_uploaded[model_key] = runs

    # --- Upload README ---
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
    else:
        print("\n[DRY-RUN] Would upload README.md")

    # --- Summary ---
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