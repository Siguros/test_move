#!/usr/bin/env python
"""Debug transfer_lr issue - check if dw_min is limiting the transfer."""

import torch
import torch.nn as nn
import os

# Imports from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Enable debug output
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")

def test_with_dw_min(dw_min_val, transfer_lr_val):
    """Test transfer with specific dw_min and transfer_lr."""
    print(f"\n{'='*60}")
    print(f"Testing dw_min={dw_min_val}, transfer_lr={transfer_lr_val}")
    print(f"{'='*60}")
    
    # Create device with specific dw_min
    device = ConstantStepDevice(dw_min=dw_min_val)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=2,
        transfer_every=3,
        transfer_lr=transfer_lr_val,
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
    layer = layer.to(DEVICE)
    
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Simple data
    x = torch.randn(8, 4).to(DEVICE)
    target = torch.randn(8, 3).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    # Get tile
    tiles = list(layer.analog_tiles())
    cpp_tile = tiles[0].tile
    
    # Train for 6 steps (transfer should happen at step 3 and 6)
    for step in range(6):
        # Get weights before
        vis_before = cpp_tile.lrtt_get_visible_weights().clone()
        A_before = cpp_tile.lrtt_get_A_lr().clone()
        B_before = cpp_tile.lrtt_get_B_lr().clone()
        
        # Training step
        optimizer.zero_grad()
        output = layer(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # Get weights after
        vis_after = cpp_tile.lrtt_get_visible_weights()
        A_after = cpp_tile.lrtt_get_A_lr()
        B_after = cpp_tile.lrtt_get_B_lr()
        
        # Check for transfer
        vis_delta = torch.norm(vis_after - vis_before).item()
        A_norm_before = torch.norm(A_before).item()
        A_norm_after = torch.norm(A_after).item()
        
        if A_norm_before > 0.01 and A_norm_after < 0.01:
            # Transfer detected
            AB_before = torch.matmul(A_before, B_before)
            expected_delta = transfer_lr_val * torch.norm(AB_before).item()
            
            print(f"\n🔄 TRANSFER at step {step + 1}:")
            print(f"  Visible delta: {vis_delta:.6f}")
            print(f"  Expected (transfer_lr * ||A@B||): {expected_delta:.6f}")
            print(f"  Ratio: {vis_delta/expected_delta if expected_delta > 0 else 0:.3f}")
            
            # Check if dw_min is limiting
            max_possible_delta = dw_min_val * 3 * 4  # dw_min * d_size * x_size
            if vis_delta < expected_delta * 0.5:
                print(f"  ⚠️  WARNING: Transfer appears limited!")
                print(f"  Max possible delta (dw_min * size): {max_possible_delta:.6f}")

# Test different combinations
print("TESTING DW_MIN AND TRANSFER_LR INTERACTION")
print("="*60)

# Test 1: Large dw_min with small transfer_lr
test_with_dw_min(0.01, 0.001)

# Test 2: Small dw_min with small transfer_lr  
test_with_dw_min(0.0001, 0.001)

# Test 3: Small dw_min with large transfer_lr
test_with_dw_min(0.0001, 1.0)

# Test 4: Very small dw_min with very small transfer_lr
test_with_dw_min(0.00001, 0.0001)