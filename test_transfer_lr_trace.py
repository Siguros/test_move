#!/usr/bin/env python
"""Trace transfer_lr parameter from Python to C++."""

import os
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice, SingleRPUConfig
from aihwkit.simulator.rpu_base import cuda

DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")

# Test with different transfer_lr values
test_values = [0.001, 0.1, 0.5, 1.0]

for test_lr in test_values:
    print(f"\n{'='*60}")
    print(f"Testing transfer_lr = {test_lr}")
    print(f"{'='*60}")
    
    # Create config
    device = ConstantStepDevice(dw_min=0.0001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=2,
        transfer_every=2,  # Transfer frequently to see the value
        transfer_lr=test_lr,
        forward_inject=True,
        lora_alpha=1.0,
        transfer_use_bl_management=False,
        transfer_use_update_management=False,
    )
    
    print(f"Python config: transfer_lr = {lrtt_config.transfer_lr}")
    
    # Create the binding and check
    rpu_config = SingleRPUConfig(device=lrtt_config)
    
    # Create a layer
    layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
    layer = layer.to(DEVICE)
    
    # Create optimizer
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Simple data
    x = torch.randn(8, 4).to(DEVICE)
    target = torch.randn(8, 3).to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    
    print("\nRunning 2 training steps to trigger transfer...")
    for step in range(2):
        optimizer.zero_grad()
        output = layer(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # The debug output should show the transfer_lr value
        # Look for: "lr_scale=X.XXXe+00, lr_eff=X.XXXe+00"
    
    print(f"\nCheck debug output above for lr_scale and lr_eff values")
    print(f"Expected: lr_scale should be {test_lr}")
    print(f"Actual: Look for 'lr_scale=' in the output above")