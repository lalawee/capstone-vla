#!/usr/bin/env python3
"""
download_dataset_folder.py
==========================
Download a specific subfolder from a HuggingFace dataset repo.

Usage
-----
  # Download a specific folder (direct)
  python download_dataset_folder.py --dataset Lusmse/sd --folder both_400

  # Download a specific folder to a custom output dir
  python download_dataset_folder.py --dataset Lusmse/sd --folder both_400 --output /workspace/data

  # Interactive — list folders in a repo and pick one
  python download_dataset_folder.py --dataset Lusmse/sd

  # Multiple folders at once
  python download_dataset_folder.py --dataset Lusmse/sd --folder both_400,left_only

Optional flags:
  --output  /custom/output/dir   Override default output directory
  --reauth                       Force re-entry of HF token
  --flat                         Download folder contents directly into output dir
                                 (no subfolder created)
"""

import argparse
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
# Token — same pattern as download_dataset.py / upload_checkpoints.py
# ---------------------------------------------------------------------------

def resolve_token(script_dir: Path, reauth: bool) -> str:
    import getpass
    token_file = script_dir / TOKEN_FILENAME

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

    print()
    print("=" * 60)
    print("  HuggingFace Authentication")
    print("=" * 60)
    print("  Get your token at: https://huggingface.co/settings/tokens")
    print("  Read permission is sufficient for downloading.")
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
        print(f"  (Run with --reauth to change it)")
    else:
        print("  Token not saved — you will be prompted again next run.")

    print()
    return token


# ---------------------------------------------------------------------------
# List top-level folders in a dataset repo
# ---------------------------------------------------------------------------

def list_repo_folders(api: HfApi, repo_id: str, token: str) -> list[str]:
    """Return unique top-level directory names in the dataset repo."""
    try:
        files = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        folders = set()
        for f in files:
            # f.path is e.g. "both_400/chunk-000/episode_000000.parquet"
            parts = Path(f.path).parts
            if len(parts) > 1:
                folders.add(parts[0])
        return sorted(folders)
    except Exception as e:
        print(f"[ERROR] Could not list repo contents: {e}")
        sys.exit(1)


def interactive_folder_picker(api: HfApi, repo_id: str, token: str) -> list[str]:
    """Show top-level folders and let user pick one or more."""
    print(f"\n[INFO] Fetching folder listing for {repo_id} ...")
    folders = list_repo_folders(api, repo_id, token)

    if not folders:
        print("[WARN] No subfolders found — repo may be flat.")
        print("       Use the parent download_dataset.py to grab everything.")
        sys.exit(0)

    print()
    print("=" * 60)
    print(f"  Folders in  {repo_id}")
    print("=" * 60)
    for i, name in enumerate(folders, 1):
        print(f"  [{i:>2}] {name}")
    print()
    print("  Enter numbers to download (e.g.  1  or  1,3  or  all)")
    raw = input("  Selection: ").strip().lower()

    if raw == "all":
        return folders

    selected = []
    for part in raw.replace(" ", "").split(","):
        try:
            idx = int(part) - 1
            if 0 <= idx < len(folders):
                selected.append(folders[idx])
            else:
                print(f"  [WARN] Index {part} out of range, skipping.")
        except ValueError:
            print(f"  [WARN] Could not parse '{part}', skipping.")

    if not selected:
        print("[ERROR] No valid folders selected.")
        sys.exit(1)

    return selected


# ---------------------------------------------------------------------------
# Download a single subfolder
# ---------------------------------------------------------------------------

def download_folder(
    repo_id: str,
    folder: str,
    output_root: Path,
    token: str,
    flat: bool = False,
):
    """
    Download all files under `folder/` in the dataset repo.

    By default, files land at:
        output_root/<dataset_name>/<folder>/...

    With --flat:
        output_root/<folder>/...
    """
    dataset_name = repo_id.split("/")[-1]

    if flat:
        local_dir = output_root / folder
    else:
        local_dir = output_root / dataset_name / folder

    local_dir.mkdir(parents=True, exist_ok=True)

    # allow_patterns filters to only files inside the requested folder
    pattern = f"{folder}/**"

    print(f"\n  Repo   : {repo_id}")
    print(f"  Folder : {folder}/")
    print(f"  Saving : {local_dir}")
    print(f"  Filter : {pattern}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
            allow_patterns=[pattern],
            ignore_patterns=["*.gitattributes", ".gitattributes"],
        )

        # snapshot_download preserves the subfolder structure inside local_dir,
        # so files land at local_dir/folder/... — unwrap one level if needed.
        nested = local_dir / folder
        if nested.is_dir():
            # Move contents up one level so local_dir == the folder root
            import shutil
            for item in nested.iterdir():
                dest = local_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
                shutil.move(str(item), str(local_dir))
            nested.rmdir()

        print(f"  [OK] Downloaded to {local_dir}")
        return local_dir

    except Exception as e:
        print(f"  [ERROR] Failed to download {repo_id}/{folder}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download a specific subfolder from a HuggingFace dataset repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="HF dataset repo id, e.g.  Lusmse/sd",
    )
    parser.add_argument(
        "--folder", type=str, default=None,
        help="Folder(s) to download, e.g.  both_400  or  both_400,left_only  "
             "(comma-separated). If omitted, shows an interactive picker.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output root directory (default: ./data/)",
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Place folder contents directly in output_root/<folder>/ "
             "instead of output_root/<dataset>/<folder>/",
    )
    parser.add_argument(
        "--reauth", action="store_true",
        help="Force re-entry of HF token even if one is cached",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    token = resolve_token(script_dir, args.reauth)

    output_root = Path(args.output) if args.output else script_dir / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output root: {output_root}")

    api = HfApi(token=token)
    try:
        user = api.whoami()
        print(f"[AUTH] Logged in as: {user['name']}")
    except Exception as e:
        print(f"[ERROR] Token validation failed: {e}")
        print("        Run with --reauth to enter a new token.")
        sys.exit(1)

    # Resolve target folders
    if args.folder:
        folders = [f.strip() for f in args.folder.split(",") if f.strip()]
    else:
        folders = interactive_folder_picker(api, args.dataset, token)

    # Download
    print(f"\n{'='*60}")
    print(f"Downloading {len(folders)} folder(s) from {args.dataset}")
    print(f"{'='*60}")

    results = {}
    for folder in folders:
        local_dir = download_folder(
            repo_id=args.dataset,
            folder=folder,
            output_root=output_root,
            token=token,
            flat=args.flat,
        )
        results[folder] = local_dir

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for folder, local_dir in results.items():
        if local_dir and local_dir.exists():
            files = [f for f in local_dir.rglob("*") if f.is_file()]
            total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
            print(f"  {args.dataset}/{folder}")
            print(f"    -> {local_dir}  ({len(files)} files, {total_mb:.1f} MB)")
        else:
            print(f"  {args.dataset}/{folder}  -> [FAILED]")


if __name__ == "__main__":
    main()
