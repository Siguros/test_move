#!/bin/bash
set -ex

# Clean everything
rm -rf build dist _skbuild *.egg-info

# Force full CUDA build with all features
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release -DRPU_USE_CUDA=ON"
export FORCE_CUDA=1

conda activate ml

echo "Building with full CUDA support..."
python -m pip wheel . -w dist --no-deps --no-build-isolation -v 2>&1 | grep -E "Building|Linking|Created wheel|Failed"
