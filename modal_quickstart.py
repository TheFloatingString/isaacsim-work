"""
Modal script to run IsaacSim/IsaacLab setup and training on GPU.
Matches CUDA 12.8 from quickstart.txt requirements.
"""

import modal

# Create Modal app
app = modal.App("isaaclab-quickstart")

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
)
def train_ant():
    """Run Isaac-Ant-v0 training in headless mode."""
    import subprocess

    result = subprocess.run(
        [
            "python",
            "/root/IsaacLab/scripts/reinforcement_learning/skrl/train.py",
            "--task=Isaac-Ant-v0",
            "--headless",
        ],
        capture_output=True,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        raise Exception(f"Training failed with return code {result.returncode}")

    return result.stdout


@app.local_entrypoint()
def main():
    """Main entry point - runs the training."""
    print("Starting IsaacLab training on Modal GPU...")
    output = train_ant.remote()
    print("\nTraining completed successfully!")
    print(output)
