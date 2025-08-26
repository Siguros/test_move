#!/usr/bin/env python
"""Test weight initialization for LRTT device."""

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
print("LRTT Weight Initialization Test")
print("=" * 80)

# Create configuration
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
    forward_inject=True,
    lora_alpha=1.0,
)

print("\n[Creating AnalogLinear layer...]")
layer = AnalogLinear(8, 4, rpu_config=rpu_config, bias=False)

# Set weights manually before moving to CUDA
print("[Setting initial weights...]")
with torch.no_grad():
    # Set the analog weights to non-zero values
    layer.set_weights(torch.ones(4, 8) * 0.5)
    
print("[Moving to CUDA...]")
layer = layer.cuda()

print("[Getting weights after CUDA...]")
weights = layer.get_weights()[0]
print(f"Weight shape: {weights.shape}")
print(f"Weight values (first row): {weights[0]}")
print(f"Weight sum: {weights.sum().item():.4f}")

# Test forward pass
print("\n[Testing forward pass...]")
x = torch.ones(2, 8).cuda()
y = layer(x)

print(f"Output shape: {y.shape}")
print(f"Output values: {y}")
print(f"Output sum: {y.sum().item():.4f}")

# Expected output should be approximately 4.0 per element (8 * 0.5)
expected = x.mm(weights.t())
print(f"\nExpected output (digital): {expected}")
print(f"Difference from expected: {(y - expected).abs().mean().item():.6f}")

print("\n" + "=" * 80)