#!/usr/bin/env python
"""Very simple LRTT test focusing on the core issue."""

import os
os.environ["AIHWKIT_DEBUG_LRTT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, '/workspace/site-packages')

import torch
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

print("Simple LRTT Test")
print("=" * 50)

# Create minimal config
rpu_config = SingleRPUConfig()
rpu_config.device = LRTTTransferCompound(
    unit_cell_devices=[
        ConstantStepDevice(),
        ConstantStepDevice(),
        ConstantStepDevice(),
    ],
    transfer_lr=0.1,
    transfer_every=1,  # Transfer every update
    rank=2,  # Very small rank
    forward_inject=False,  # Start without injection
)

print(f"Config: rank={rpu_config.device.rank}, transfer_every={rpu_config.device.transfer_every}")

# Create small layer
layer = AnalogLinear(4, 2, rpu_config=rpu_config, bias=False)
layer = layer.cuda()

# Initialize weights
with torch.no_grad():
    layer.set_weights(torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                    [5.0, 6.0, 7.0, 8.0]]))

# Create optimizer
opt = AnalogSGD(layer.parameters(), lr=0.01)

print("\nInitial weights set")

# Test forward pass
x = torch.ones(1, 4).cuda()
y = layer(x)
print(f"Forward pass output: {y}")

# Test backward and update
target = torch.tensor([[1.0, 1.0]]).cuda()
loss = torch.nn.functional.mse_loss(y, target)

opt.zero_grad()
loss.backward()
opt.step()

print(f"After update - loss: {loss.item():.4f}")

# Another forward pass
y2 = layer(x)
print(f"Second forward pass: {y2}")

print("\nTest completed successfully!")