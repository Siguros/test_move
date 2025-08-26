#!/usr/bin/env python
"""Test weight binding for LRTT device."""

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

print("=" * 80)
print("LRTT Weight Binding Test")
print("=" * 80)

# First test without forward_inject to confirm basic functionality
print("\n1. Testing WITHOUT forward_inject:")
print("-" * 40)

rpu_config = SingleRPUConfig()
rpu_config.mapping.max_input_size = 64
rpu_config.mapping.max_output_size = 64

rpu_config.device = LRTTTransferCompound(
    unit_cell_devices=[
        ConstantStepDevice(),  # fastA
        ConstantStepDevice(),  # fastB  
        ConstantStepDevice(),  # visible
    ],
    transfer_lr=0.1,
    transfer_every=2,
    rank=4,
    forward_inject=False,  # Disabled
)

layer = AnalogLinear(8, 4, rpu_config=rpu_config, bias=False)
with torch.no_grad():
    layer.set_weights(torch.ones(4, 8) * 0.5)
layer = layer.cuda()

x = torch.ones(2, 8).cuda()
y = layer(x)
print(f"Output without injection: {y}")
print(f"Output sum: {y.sum().item():.4f}")

# Now test with forward_inject
print("\n2. Testing WITH forward_inject:")
print("-" * 40)

rpu_config2 = SingleRPUConfig()
rpu_config2.mapping.max_input_size = 64
rpu_config2.mapping.max_output_size = 64

rpu_config2.device = LRTTTransferCompound(
    unit_cell_devices=[
        ConstantStepDevice(),  # fastA
        ConstantStepDevice(),  # fastB  
        ConstantStepDevice(),  # visible
    ],
    transfer_lr=0.1,
    transfer_every=2,
    rank=4,
    forward_inject=True,  # Enabled
    lora_alpha=1.0,
)

layer2 = AnalogLinear(8, 4, rpu_config=rpu_config2, bias=False)
with torch.no_grad():
    layer2.set_weights(torch.ones(4, 8) * 0.5)
layer2 = layer2.cuda()

# Get the tile to check internal state
tile = layer2.analog_tile
print(f"Tile type: {type(tile)}")
print(f"Tile has get_weights: {hasattr(tile, 'get_weights')}")

# Get weights from the tile
weights = layer2.get_weights()[0]
print(f"Weights from layer: sum = {weights.sum().item():.4f}")

x2 = torch.ones(2, 8).cuda()
y2 = layer2(x2)
print(f"Output with injection: {y2}")
print(f"Output sum: {y2.sum().item():.4f}")

# Try a backward pass to trigger weight updates
print("\n3. Testing backward pass:")
print("-" * 40)
loss = y2.sum()
loss.backward()
print("Backward pass completed")

print("\n" + "=" * 80)