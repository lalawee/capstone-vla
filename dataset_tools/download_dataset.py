#!/usr/bin/env python3
"""
download_datasets.py
====================
Downloads one or more HuggingFace datasets to a local directory,
reusing the same token-cache logic as upload_checkpoints.py.

Usage
-----
  python download_datasets.py \\
      --repos  Lusmse/pourLeftCereal Lusmse/pourRightCereal \\
      --base   /workspace/capstone-vla \\
      --outdir datasets

Optional flags:
  --outdir   subdirectory under --base to store downloads (default: datasets)
  --dry-run  print what would be downloaded, don't download
  --reauth   force re-entry of token even if cached
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed.")
    print("        Run:  pip install huggingface_hub")
    sys.exit(1)

TOKEN_FILENAME = ".hf_token"


# ---------------------------------------------------------------------------
# Auth helpers  (identical to upload_checkpoints.py)
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
    print("  Make sure it has READ permission.")
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
# Core download logic
# ---------------------------------------------------------------------------

def download_dataset(repo_id: str, local_dir: Path, token: str, dry_run: bool) -> bool:
    repo_name = repo_id.split("/")[-1]
    dest = local_dir / repo_name

    print(f"\n  Repo  : {repo_id}")
    print(f"  Dest  : {dest}")

    if dry_run:
        print(f"  [DRY-RUN] Would download {repo_id} -> {dest}")
        return True

    dest.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  [OK] Downloaded {repo_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {repo_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets.")
    parser.add_argument(
        "--repos", type=str, nargs="+", required=True,
        help="One or more HuggingFace dataset repo IDs, e.g. Lusmse/pourLeftCereal",
    )
    parser.add_argument("--base",    type=str, default="auto",
                        help="Root of capstone-vla project (auto-detected if omitted)")
    parser.add_argument("--outdir",  type=str, default="datasets",
                        help="Subdirectory under --base for downloads (default: datasets)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reauth",  action="store_true")
    args = parser.parse_args()

    base = resolve_base(args.base if args.base != "auto" else "")
    local_dir = base / args.outdir

    print(f"[INFO] capstone-vla root : {base}")
    print(f"[INFO] Download target   : {local_dir}")
    print(f"[INFO] Repos             : {args.repos}")
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
        local_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for repo_id in args.repos:
        print(f"\n{'='*60}")
        print(f"Dataset: {repo_id}")
        print(f"{'='*60}")
        ok = download_dataset(repo_id, local_dir, token, args.dry_run)
        results[repo_id] = ok

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for repo_id, ok in results.items():
        status = "OK" if ok else "FAILED"
        dest = local_dir / repo_id.split("/")[-1]
        print(f"  [{status}] {repo_id}")
        if ok and not args.dry_run:
            print(f"         -> {dest}")

    if not args.dry_run:
        print(f"\nDatasets saved under: {local_dir}")


if __name__ == "__main__":
    main()