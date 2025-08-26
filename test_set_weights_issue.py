#!/usr/bin/env python
"""Test to isolate the set_weights CUBLAS issue."""

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

print("Testing set_weights CUBLAS issue")
print("=" * 50)

# Test 1: Without LRTT (baseline)
print("\n1. Testing regular ConstantStepDevice (baseline):")
try:
    rpu_config = SingleRPUConfig()
    rpu_config.device = ConstantStepDevice()
    
    layer = AnalogLinear(4, 2, rpu_config=rpu_config, bias=False)
    layer = layer.cuda()
    
    # Try to set weights
    weights = torch.ones(2, 4) * 0.5
    layer.set_weights(weights)
    print("  ✓ set_weights successful with ConstantStepDevice")
    
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: With LRTT device
print("\n2. Testing LRTT device:")
try:
    rpu_config = SingleRPUConfig()
    rpu_config.device = LRTTTransferCompound(
        unit_cell_devices=[
            ConstantStepDevice(),
            ConstantStepDevice(),
            ConstantStepDevice(),
        ],
        transfer_lr=0.1,
        transfer_every=2,
        rank=2,
        forward_inject=False,
    )
    
    layer = AnalogLinear(4, 2, rpu_config=rpu_config, bias=False)
    print("  - Layer created")
    
    layer = layer.cuda()
    print("  - Moved to CUDA")
    
    # Try to set weights
    weights = torch.ones(2, 4) * 0.5
    print(f"  - Setting weights of shape {weights.shape}")
    layer.set_weights(weights)
    print("  ✓ set_weights successful with LRTT device")
    
except RuntimeError as e:
    print(f"  ✗ RuntimeError: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try get_weights first
print("\n3. Testing get_weights before set_weights:")
try:
    rpu_config = SingleRPUConfig()
    rpu_config.device = LRTTTransferCompound(
        unit_cell_devices=[
            ConstantStepDevice(),
            ConstantStepDevice(),
            ConstantStepDevice(),
        ],
        transfer_lr=0.1,
        transfer_every=2,
        rank=2,
        forward_inject=False,
    )
    
    layer = AnalogLinear(4, 2, rpu_config=rpu_config, bias=False).cuda()
    
    # Get weights first
    current_weights = layer.get_weights()[0]
    print(f"  - Current weights shape: {current_weights.shape}")
    print(f"  - Current weights sum: {current_weights.sum().item():.4f}")
    
    # Now try to set
    new_weights = torch.ones(2, 4) * 0.5
    layer.set_weights(new_weights)
    print("  ✓ set_weights successful after get_weights")
    
except RuntimeError as e:
    print(f"  ✗ RuntimeError: {e}")

print("\n" + "=" * 50)