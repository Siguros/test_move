# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 1 with LRTT: simple network with one layer using LR-TT transfer learning.

Simple network that consist of one analog layer with Low-Rank Tiki-Taka (LR-TT) 
transfer learning. The network aims to learn to sum all the elements from one array
using low-rank adaptation.
"""
# pylint: disable=invalid-name

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss
import torch

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network using LR-TT transfer learning
# CRITICAL FIXES:
# 1. Increase dw_min from 0.0000001 to 0.01
# 2. Use rank=1 instead of rank=2
# 3. Reduce transfer_every from 100 to 10
device = ConstantStepDevice(
    dw_min=0.01,  # Increased significantly
    dw_min_dtod=0.0, 
    up_down_dtod=0.0
)

lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],  # fastA, fastB, visible
    rank=1,  # Reduced from 2 to 1
    transfer_every=10,  # Reduced from 100
    transfer_lr=1.0,  # Keep at 1.0
    forward_inject=True,  # Use W_eff = W_visible + α*(A@B) in forward pass
    lora_alpha=1.0,  # LoRA scaling factor
    units_in_mbatch=False,
)
rpu_config = UnitCellRPUConfig(device=lrtt_config)
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# CRITICAL: Initialize visible weights
init_weights = torch.randn(2, 4) * 0.1
if cuda.is_compiled():
    init_weights = init_weights.cuda()
model.set_weights(init_weights)

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.5)  # Increased from 0.1 to 0.5
opt.regroup_param_groups(model)

print("Training with LR-TT (Low-Rank Tiki-Taka) transfer learning - FIXED VERSION")
print(f"Rank: {lrtt_config.rank}, Transfer every: {lrtt_config.transfer_every} updates")
print(f"dw_min: {device.dw_min}, Learning rate: 0.5")
print("-" * 50)

losses = []
for epoch in range(1000):
    # Delete old gradient
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    losses.append(loss.item())
    # Run training (backward propagation).
    loss.backward()
    
    opt.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} - Loss: {loss:.8f}")
    
    # Transfer happens automatically every transfer_every updates
    # A@B is accumulated into visible weights, then A and B are reinitialized

print("-" * 50)
print(f"Final loss: {loss:.8f}")

# Analyze training
initial_loss = losses[0]
final_loss = losses[-1]
min_loss = min(losses)
print(f"\nTraining Analysis:")
print(f"  Initial loss: {initial_loss:.8f}")
print(f"  Final loss: {final_loss:.8f}")
print(f"  Min loss: {min_loss:.8f}")
print(f"  Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")

if final_loss < initial_loss * 0.5:
    print("  ✓ Training successful!")
else:
    print("  ⚠️  Training did not converge well")

# Test the trained model
with torch.no_grad():
    test_pred = model(x)
    test_loss = mse_loss(test_pred, y)
    print(f"\nTest loss: {test_loss:.8f}")
    print(f"Predictions:\n{test_pred}")
    print(f"Targets:\n{y}")