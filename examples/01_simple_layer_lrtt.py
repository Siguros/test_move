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
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a single-layer network using LR-TT transfer learning
# LR-TT uses three devices: fastA, fastB for low-rank updates, and visible for accumulated weights
device = ConstantStepDevice(dw_min=0.001)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],  # fastA, fastB, visible
    rank=2,  # Low-rank dimension
    transfer_every=9,  # Transfer A@B to visible every 10 updates
    transfer_lr=0.5,  # Learning rate for transfer
    forward_inject=True,  # Use W_eff = W_visible + α*(A@B) in forward pass
    lora_alpha=1.0,  # LoRA scaling factor
)

rpu_config = SingleRPUConfig(device=lrtt_config)
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

print("Training with LR-TT (Low-Rank Tiki-Taka) transfer learning")
print(f"Rank: 2, Transfer every: 10 updates")
print("-" * 50)


for epoch in range(100):
    # Delete old gradient
    opt.zero_grad()
    # Add the training Tensor to the model (input).
    pred = model(x)
    # Add the expected output Tensor.
    loss = mse_loss(pred, y)
    # Run training (backward propagation).
    loss.backward()
    
    opt.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} - Loss: {loss:.8f}")
    
    # Transfer happens automatically every transfer_every updates
    # A@B is accumulated into visible weights, then A and B are reinitialized

print("-" * 50)
print(f"Final loss: {loss:.8f}")

# Test the trained model
with torch.no_grad():
    test_pred = model(x)
    test_loss = mse_loss(test_pred, y)
    print(f"Test loss: {test_loss:.8f}")
    print(f"Predictions:\n{test_pred}")
    print(f"Targets:\n{y}")