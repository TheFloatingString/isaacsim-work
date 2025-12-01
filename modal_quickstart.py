"""
Modal script to run IsaacSim/IsaacLab setup and training on GPU.
Matches CUDA 12.8 from quickstart.txt requirements.
"""

import modal

# Create Modal app
app = modal.App("isaaclab-quickstart")

# Create a volume to persist training outputs, models, and videos
volume = modal.Volume.from_name("isaaclab-training-data", create_if_missing=True)

# Build image with all dependencies
# Using CUDA 12.8 to match torch cu128 requirement
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "wget",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libglu1-mesa",
        "libgl1-mesa-glx",
        "libxt6",
        "libegl1-mesa",
        "libvulkan1",
    )
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "isaacsim[all,extscache]==5.1.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    .run_commands(
        "git clone https://github.com/isaac-sim/IsaacLab.git /root/IsaacLab",
        "cd /root/IsaacLab && yes | ./isaaclab.sh --install",
    )
)


@app.function(
    image=image,
    gpu="A10G",  # Can change to "A100", "T4", etc. based on needs
    timeout=3600,  # 1 hour timeout
    volumes={"/outputs": volume},  # Mount volume to persist outputs
)
def train_ant():
    """Run Isaac-Ant-v0 training in headless mode with video recording."""
    import subprocess
    import os
    import shutil
    import threading
    import time

    # Verify GPU is accessible
    print("=" * 80)
    print("GPU DIAGNOSTICS:")
    print("=" * 80)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("WARNING: nvidia-smi failed!")
        print(result.stderr)
    print("=" * 80)

    # Background sync function to continuously copy logs to volume
    def sync_logs_to_volume():
        """Periodically sync training outputs to volume."""
        logs_dir = "/root/logs/skrl/ant"
        sync_interval = 30  # Sync every 30 seconds

        while not stop_syncing.is_set():
            try:
                if os.path.exists(logs_dir):
                    run_dirs = [
                        d
                        for d in os.listdir(logs_dir)
                        if os.path.isdir(os.path.join(logs_dir, d))
                    ]
                    if run_dirs:
                        latest_run = sorted(run_dirs)[-1]
                        src_path = os.path.join(logs_dir, latest_run)
                        dst_path = f"/outputs/{latest_run}"

                        # Copy incrementally
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                        volume.commit()
                        print(f"[SYNC] Saved outputs to volume: {dst_path}", flush=True)
            except Exception as e:
                print(f"[SYNC] Error during sync: {e}", flush=True)

            time.sleep(sync_interval)

    # Start background sync thread
    stop_syncing = threading.Event()
    sync_thread = threading.Thread(target=sync_logs_to_volume, daemon=True)
    sync_thread.start()

    # Set environment variables for headless rendering with EGL (allows video export)
    env = os.environ.copy()
    env["OMNI_KIT_ALLOW_ROOT"] = "1"  # Allow running as root
    env["ACCEPT_EULA"] = "Y"  # Accept EULA automatically

    # GPU/CUDA environment variables
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

    # Force EGL for headless rendering
    env["OMNI_KIT_RENDERER_MODE"] = "rtx"
    env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    print("Environment variables set:")
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "OMNI_KIT_ALLOW_ROOT",
        "OMNI_KIT_RENDERER_MODE",
    ]:
        print(f"  {key} = {env.get(key)}")
    print()

    # Stream output in real-time
    print("=" * 80)
    print("TRAINING OUTPUT (streaming):")
    print("=" * 80)

    process = subprocess.Popen(
        [
            "python",
            "-u",  # Unbuffered output
            "/root/IsaacLab/scripts/reinforcement_learning/skrl/train.py",
            "--task=Isaac-Ant-v0",
            "--headless",
            "--video",  # Enable video recording
            "--video_length=200",  # Record 200 steps per video
            "--video_interval=2000",  # Record video every 2000 steps
            "--max_iterations=200",  # Total number of training iterations
            "--num_envs=1",  # Use only 1 environment for testing
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line-buffered
        env=env,
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end="")  # Print without extra newline

    process.wait()

    # Stop background syncing thread
    print("\n" + "=" * 80)
    print("Stopping background sync and performing final save...")
    print("=" * 80)
    stop_syncing.set()
    sync_thread.join(timeout=10)  # Wait up to 10 seconds for sync to finish

    # Store return code
    return_code = process.returncode
    training_status = (
        "succeeded" if return_code == 0 else f"failed (exit code {return_code})"
    )

    # FINAL SYNC: Copy all training outputs to persistent volume one last time
    logs_dir = "/root/logs/skrl/ant"
    if os.path.exists(logs_dir):
        print(f"\nPerforming final sync of training outputs to volume...")
        # Find the latest training run directory
        run_dirs = [
            d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))
        ]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            src_path = os.path.join(logs_dir, latest_run)
            dst_path = f"/outputs/{latest_run}"
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"Saved training outputs to: {dst_path}")

            # List what was saved
            print(f"\nSaved files:")
            for root, dirs, files in os.walk(dst_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), dst_path)
                    print(f"  - {rel_path}")

        volume.commit()  # Persist changes to volume
        print(f"\nVolume committed successfully! Training {training_status}")
    else:
        print(f"\nNo logs directory found at {logs_dir}")

    # Raise exception AFTER saving outputs if training failed
    if return_code != 0:
        raise Exception(f"Training {training_status} (outputs saved to volume)")

    return "Training completed successfully"


@app.function(volumes={"/outputs": volume})
def list_outputs():
    """List all saved training runs and their contents."""
    import os

    print("\n=== Saved Training Runs ===\n")
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

                # List checkpoints
                checkpoints_dir = os.path.join(run_path, "checkpoints")
                if os.path.exists(checkpoints_dir):
                    checkpoints = os.listdir(checkpoints_dir)
                    if checkpoints:
                        print(f"  Checkpoints ({len(checkpoints)}):")
                        for ckpt in sorted(checkpoints)[:5]:  # Show first 5
                            print(f"    - {ckpt}")
                        if len(checkpoints) > 5:
                            print(f"    ... and {len(checkpoints) - 5} more")

                # List videos
                for root, dirs, files in os.walk(run_path):
                    videos = [f for f in files if f.endswith((".mp4", ".avi"))]
                    if videos:
                        print(f"  Videos ({len(videos)}):")
                        for vid in sorted(videos):
                            rel_path = os.path.relpath(
                                os.path.join(root, vid), run_path
                            )
                            print(f"    - {rel_path}")
        else:
            print("No training runs found yet.")
    else:
        print("No outputs directory found yet.")


@app.function(volumes={"/outputs": volume})
def download_run(run_name: str, local_path: str = "./downloads"):
    """Download a specific training run to local machine."""
    import os
    import tarfile
    from pathlib import Path

    run_path = f"/outputs/{run_name}"
    if not os.path.exists(run_path):
        print(f"Run '{run_name}' not found!")
        return None

    # Create a tarball of the run
    tar_path = f"/tmp/{run_name}.tar.gz"
    print(f"Creating archive of {run_name}...")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(run_path, arcname=run_name)

    # Read the tarball
    with open(tar_path, "rb") as f:
        data = f.read()

    print(f"Archive created: {len(data) / 1024 / 1024:.2f} MB")
    return data


@app.local_entrypoint()
def main():
    """Main entry point - runs the training."""
    print("Starting IsaacLab training on Modal GPU...")
    output = train_ant.remote()
    print("\nTraining completed successfully!")
    print(output)

    # List what was saved
    print("\n" + "=" * 50)
    list_outputs.remote()
