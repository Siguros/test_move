#!/usr/bin/env python3

import requests
import os
import json

# GitHub configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "nmdlkg"
REPO_NAME = "ml"
WHEEL_PATH = "./dist/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl"

# Release configuration
RELEASE_TAG = "v0.9.0-cuda-a100"
RELEASE_TITLE = "AIHWKit v0.9.0 - CUDA A100 Optimized Build"

RELEASE_BODY = """# AIHWKit v0.9.0 - CUDA A100 Optimized Build

This release contains a custom-built AIHWKit wheel optimized specifically for NVIDIA A100 GPUs (compute capability 8.0).

## 🚀 Features

- **CUDA-enabled** AIHWKit with full GPU acceleration
- **A100-optimized** compilation (compute capability 8.0 only)
- **LR-TT (Low-Rank Tensor Transfer)** functionality with latest modifications
- **PyTorch 2.3** compatibility
- **Python 3.10** support

## 📋 Requirements

- NVIDIA A100 GPU (compute capability 8.0)
- Python 3.10
- PyTorch 2.3.x with CUDA 11.8
- CUDA Toolkit 11.8 or later

## 💾 Installation

### Option 1: Direct Installation from Release
```bash
# Download and install the wheel directly
pip install https://github.com/nmdlkg/ml/releases/download/v0.9.0-cuda-a100/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl
```

### Option 2: Download and Install Locally
```bash
# Download the wheel file
wget https://github.com/nmdlkg/ml/releases/download/v0.9.0-cuda-a100/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl

# Install the wheel
pip install aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl
```

### Option 3: Using conda environment (recommended)
```bash
# Create or activate conda environment with Python 3.10
conda create -n ml python=3.10
conda activate ml

# Install PyTorch 2.3 with CUDA 11.8
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install AIHWKit from this release
pip install https://github.com/nmdlkg/ml/releases/download/v0.9.0-cuda-a100/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl
```

## ✅ Verification

After installation, verify CUDA support:

```python
import aihwkit
from aihwkit.simulator import rpu_base

print(f"AIHWKit version: {aihwkit.__version__}")
print(f"CUDA compiled: {rpu_base.cuda.is_compiled()}")
```

## 🧪 LR-TT Example

```python
import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, TransferCompound
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.algorithms.lr_tt import LRTransferHook

# Create transfer device configuration
rpu_config = SingleRPUConfig(
    device=TransferCompound(
        unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()]
    )
)

# Create analog layer
layer = AnalogLinear(in_features=128, out_features=64, rpu_config=rpu_config)
if torch.cuda.is_available():
    layer = layer.cuda()

# Set up LR-TT transfer hook
hook = LRTransferHook(
    modules_or_tiles=[layer],
    rank=16,
    transfer_every=100
)

# Your training loop here...
```

## 📊 Build Information

- **Built on**: August 25, 2025
- **Compiler**: NVCC with GCC
- **CUDA Architecture**: compute_80 (A100 only)
- **PyTorch**: 2.3.1+cu118
- **Python**: 3.10.18
- **Wheel size**: ~173MB

## ⚠️ Important Notes

1. This wheel is **A100-specific** and will not work on other GPU architectures
2. Requires **exact Python 3.10** - other Python versions are not supported
3. Built with **PyTorch 2.3** - ensure compatibility with your environment
4. Contains **latest LR-TT CUDA modifications** as of build date

## 🛠️ Troubleshooting

If you encounter issues:

1. Verify you have an A100 GPU: `nvidia-smi`
2. Check Python version: `python --version` (should be 3.10.x)
3. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check CUDA compute capability: `python -c "import torch; print(torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)"` (should be 8.0)

---

**Maintainer**: Siguors (duddns157@snu.ac.kr)  
**Repository**: https://github.com/nmdlkg/ml
"""

def create_github_release():
    """Create a GitHub release using the GitHub API."""
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    # Create the release
    release_data = {
        "tag_name": RELEASE_TAG,
        "target_commitish": "main",  # or "master" depending on your default branch
        "name": RELEASE_TITLE,
        "body": RELEASE_BODY,
        "draft": True,  # Create as draft first
        "prerelease": False
    }
    
    print(f"Creating release {RELEASE_TAG}...")
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases"
    
    response = requests.post(url, headers=headers, data=json.dumps(release_data))
    
    if response.status_code == 201:
        release = response.json()
        print(f"✓ Release created successfully!")
        print(f"  Release ID: {release['id']}")
        print(f"  URL: {release['html_url']}")
        return release
    else:
        print(f"✗ Failed to create release: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def upload_asset(release, asset_path):
    """Upload an asset to the GitHub release."""
    
    if not os.path.exists(asset_path):
        print(f"✗ Asset file not found: {asset_path}")
        return False
    
    asset_name = os.path.basename(asset_path)
    upload_url = release['upload_url'].replace('{?name,label}', f'?name={asset_name}')
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    
    print(f"Uploading {asset_name}...")
    file_size = os.path.getsize(asset_path)
    print(f"  File size: {file_size / (1024*1024):.1f} MB")
    
    with open(asset_path, 'rb') as f:
        response = requests.post(upload_url, headers=headers, data=f)
    
    if response.status_code == 201:
        asset = response.json()
        print(f"✓ Asset uploaded successfully!")
        print(f"  Download URL: {asset['browser_download_url']}")
        return True
    else:
        print(f"✗ Failed to upload asset: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def main():
    print("GitHub Release Creator for AIHWKit CUDA A100 Build")
    print("=" * 60)
    
    # Create the release
    release = create_github_release()
    if not release:
        return False
    
    # Upload the wheel file
    if not upload_asset(release, WHEEL_PATH):
        return False
    
    print("=" * 60)
    print("🎉 Release created successfully!")
    print(f"📦 Release URL: {release['html_url']}")
    print(f"💾 Wheel download: {release['html_url'].replace('/tag/', '/download/')}/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl")
    print()
    print("📝 Next steps:")
    print("1. Review the draft release on GitHub")
    print("2. Publish the release when ready")
    print("3. Test installation from the release URL")
    
    return True

if __name__ == "__main__":
    main()