#!/usr/bin/env python
"""Test to verify transfer_lr is being applied correctly."""

import torch
import torch.nn as nn

# Imports from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")

def create_layer_with_transfer_lr(transfer_lr):
    """Create LRTT layer with specific transfer_lr."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=4,
        transfer_every=5,
        transfer_lr=transfer_lr,  # Test different values
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(8, 4, bias=False, rpu_config=rpu_config)
    return layer.to(DEVICE)

def get_weights(layer):
    """Get LRTT weights."""
    tiles = list(layer.analog_tiles())
    cpp_tile = tiles[0].tile
    return {
        'A': cpp_tile.lrtt_get_A_lr().clone(),
        'B': cpp_tile.lrtt_get_B_lr().clone(),
        'visible': cpp_tile.lrtt_get_visible_weights().clone(),
    }

def test_transfer_lr(transfer_lr_value):
    """Test transfer with specific transfer_lr."""
    print(f"\n{'='*60}")
    print(f"Testing transfer_lr = {transfer_lr_value}")
    print(f"{'='*60}")
    
    layer = create_layer_with_transfer_lr(transfer_lr_value)
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Simple data
    x = torch.randn(16, 8).to(DEVICE)
    target = torch.randn(16, 4).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    # Train for 10 steps (transfer should happen at step 5 and 10)
    transfer_deltas = []
    
    for step in range(10):
        # Get weights before
        w_before = get_weights(layer)
        
        # Training step
        optimizer.zero_grad()
        output = layer(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # Get weights after
        w_after = get_weights(layer)
        
        # Check for transfer
        vis_delta = torch.norm(w_after['visible'] - w_before['visible']).item()
        A_norm_before = torch.norm(w_before['A']).item()
        A_norm_after = torch.norm(w_after['A']).item()
        
        if A_norm_before > 0.01 and A_norm_after < 0.01 and vis_delta > 0.001:
            # Transfer detected
            transfer_deltas.append(vis_delta)
            
            # Calculate expected transfer
            AB_before = torch.matmul(w_before['A'], w_before['B'])
            expected_delta = transfer_lr_value * torch.norm(AB_before).item()
            
            print(f"\nTransfer at step {step + 1}:")
            print(f"  A norm: {A_norm_before:.6f} → {A_norm_after:.6f}")
            print(f"  Visible delta: {vis_delta:.6f}")
            print(f"  Expected delta (transfer_lr * ||A@B||): {expected_delta:.6f}")
            print(f"  Ratio (actual/expected): {vis_delta/expected_delta if expected_delta > 0 else 0:.3f}")
            
            # Show actual visible weight change
            delta_C = w_after['visible'] - w_before['visible']
            print(f"  First 2x2 of ΔC:")
            delta_show = delta_C[:2, :2].cpu().numpy()
            for row in delta_show:
                print(f"    [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    return transfer_deltas

# Test different transfer_lr values
print("TESTING TRANSFER_LR EFFECT ON VISIBLE WEIGHT CHANGES")
print("="*60)

results = {}
for transfer_lr in [1.0, 0.5, 0.1, 0.01, 0.001]:
    deltas = test_transfer_lr(transfer_lr)
    results[transfer_lr] = deltas

# Summary
print("\n" + "="*60)
print("SUMMARY: Transfer deltas for different transfer_lr values")
print("="*60)
for lr, deltas in results.items():
    if deltas:
        print(f"transfer_lr = {lr:6.3f}: deltas = {[f'{d:.6f}' for d in deltas]}")
    else:
        print(f"transfer_lr = {lr:6.3f}: No transfers detected")

# Check if deltas scale with transfer_lr
if len(results[1.0]) > 0 and len(results[0.001]) > 0:
    ratio = results[1.0][0] / results[0.001][0] if results[0.001][0] > 0 else 0
    print(f"\nRatio of deltas (1.0 / 0.001): {ratio:.2f}")
    print(f"Expected ratio: 1000.0")
    if abs(ratio - 1000) > 100:
        print("❌ WARNING: Transfer deltas DO NOT scale correctly with transfer_lr!")
    else:
        print("✓ Transfer deltas scale correctly with transfer_lr")