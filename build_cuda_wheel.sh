#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ml

# Set CUDA environment variables
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 compute capability
export CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80"

echo "=== Build Environment ==="
echo "USE_CUDA=$USE_CUDA"
echo "CUDA_HOME=$CUDA_HOME"
echo "CMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "========================="

# Build wheel without build isolation
python -m pip wheel --no-build-isolation --wheel-dir dist .
