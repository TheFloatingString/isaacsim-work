"""
Helper script to download training results from Modal volume.
Usage:
  python download_results.py --list                    # List all runs
  python download_results.py --download RUN_NAME       # Download a specific run
"""

import modal
import argparse
from pathlib import Path

# Connect to the same app and volume
app = modal.App("isaaclab-quickstart")
volume = modal.Volume.from_name("isaaclab-training-data", create_if_missing=True)


@app.function(volumes={"/outputs": volume})
def list_runs():
    """List all available training runs."""
    import os

    print("\n=== Available Training Runs ===\n")
    if os.path.exists("/outputs"):
        runs = [
            d
            for d in os.listdir("/outputs")
            if os.path.isdir(os.path.join("/outputs", d))
        ]
        if runs:
            for run in sorted(runs):
                run_path = os.path.join("/outputs", run)
                print(f"\n📁 {run}")

                # Count files
                total_size = 0
                file_count = 0
                videos = []
                checkpoints = []

                for root, dirs, files in os.walk(run_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                        file_count += 1

                        if file.endswith((".mp4", ".avi")):
                            videos.append(file)
                        elif "checkpoint" in file.lower() or file.endswith(".pt"):
                            checkpoints.append(file)

                print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
                print(f"  Files: {file_count}")
                if checkpoints:
                    print(f"  Checkpoints: {len(checkpoints)}")
                if videos:
                    print(f"  Videos: {len(videos)}")
        else:
            print("No training runs found.")
    else:
        print("No outputs directory found.")


@app.function(volumes={"/outputs": volume})
def download_run_data(run_name: str):
    """Download a specific training run as a tarball."""
    import os
    import tarfile

    run_path = f"/outputs/{run_name}"
    if not os.path.exists(run_path):
        available = [
            d
            for d in os.listdir("/outputs")
            if os.path.isdir(os.path.join("/outputs", d))
        ]
        print(f"❌ Run '{run_name}' not found!")
        print(f"Available runs: {', '.join(available)}")
        return None

    # Create a tarball
    tar_path = f"/tmp/{run_name}.tar.gz"
    print(f"📦 Creating archive of {run_name}...")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(run_path, arcname=run_name)

    # Read the tarball
    with open(tar_path, "rb") as f:
        data = f.read()

    print(f"✓ Archive created: {len(data) / 1024 / 1024:.2f} MB")
    return data


@app.local_entrypoint()
def main(
    list_flag: bool = False, download: str = None, output_dir: str = "./downloads"
):
    """Main entry point for downloading results."""
    if list_flag:
        list_runs.remote()
    elif download:
        print(f"\nDownloading run: {download}")
        data = download_run_data.remote(download)

        if data:
            # Save to local file
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            file_path = output_path / f"{download}.tar.gz"

            with open(file_path, "wb") as f:
                f.write(data)

            print(f"\n✓ Downloaded to: {file_path}")
            print(f"\nTo extract:")
            print(f"  tar -xzf {file_path}")
    else:
        print("Usage:")
        print("  modal run download_results.py --list")
        print("  modal run download_results.py --download RUN_NAME")
