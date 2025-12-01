# Isaac Sim on Modal

Setup for running NVIDIA Isaac Sim on Modal with GPU support.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (for local dependency management)
- [Modal](https://modal.com) account

## Setup

1. Install dependencies locally with uv:
```bash
uv pip install -e .
```

2. Set up Modal authentication:
```bash
modal token new
```

3. Run Isaac Sim on Modal:
```bash
modal run isaacsim_modal.py
```

Or using the custom command from your config:
```bash
export PYTHONIOENCODING="utf-8"; modal run isaacsim_modal.py
```

## Requirements

Isaac Sim requires:
- NVIDIA GPU (A10G, A100, or T4 recommended)
- 16GB+ memory
- Vulkan or RTX graphics support

## Notes

The current setup is a template. Full Isaac Sim installation requires:
1. Download from NVIDIA Omniverse
2. NVIDIA NGC authentication
3. License agreement acceptance
