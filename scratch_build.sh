#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate ml

export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.0"
export CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release -DRPU_USE_CUDA=ON"
export FORCE_CUDA=1

echo "=== Build Environment ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "USE_CUDA=$USE_CUDA"
echo "CMAKE_ARGS=$CMAKE_ARGS"
nvcc --version | tail -1
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "========================"
echo ""
echo "=== Starting scratch build ==="
python -m pip wheel . -w dist --no-deps --no-build-isolation -v
