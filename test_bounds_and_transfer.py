#!/usr/bin/env python3
"""Test B weight bounds and transfer rate effect."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized
from aihwkit.optim import AnalogSGD

# Set debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("=" * 80)
print("Testing B Weight Bounds and Transfer Rate Effect")
print("=" * 80)

def test_with_transfer_rate(transfer_lr, label):
    """Test with specific transfer learning rate."""
    print(f"\n{'='*60}")
    print(f"Test: {label} (transfer_lr={transfer_lr})")
    print(f"{'='*60}")
    
    # Configuration
    d_size = 8
    x_size = 4
    rank = 2
    batch_size = 4
    
    # Create LRTT config
    config = lrtt_idealized(rank=rank)
    device = config.device
    device.transfer_every = 1
    device.transfer_lr = transfer_lr
    device.lora_alpha = 4.0
    device.forward_inject = True
    
    # Create model
    model = nn.Sequential(
        AnalogLinear(x_size, d_size, bias=False, rpu_config=config)
    ).cuda()
    
    # Create optimizer  
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    
    # Get tile
    layer = model[0]
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    
    # Get initial state
    params_init = tile.get_hidden_parameters()
    A_init = params_init['hidden_weights_0'].cpu().numpy()
    B_init = params_init['hidden_weights_1'].cpu().numpy()
    C_init = params_init['hidden_weights_2'].cpu().numpy()
    visible_init = tile.get_weights()[0].cpu().numpy()
    
    print(f"Initial state:")
    print(f"  A norm: {np.linalg.norm(A_init):.6f}")
    print(f"  B norm: {np.linalg.norm(B_init):.6f}")
    print(f"  C norm: {np.linalg.norm(C_init):.6f}")
    print(f"  Visible norm: {np.linalg.norm(visible_init):.6f}")
    
    # Perform 5 updates
    for i in range(5):
        x = torch.randn(batch_size, x_size, device='cuda') * 0.1
        target = torch.randn(batch_size, d_size, device='cuda') * 0.1
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
    
    # Get final state
    params_final = tile.get_hidden_parameters()
    A_final = params_final['hidden_weights_0'].cpu().numpy()
    B_final = params_final['hidden_weights_1'].cpu().numpy()
    C_final = params_final['hidden_weights_2'].cpu().numpy()
    visible_final = tile.get_weights()[0].cpu().numpy()
    
    # Check B bounds
    B_lr = B_final[:rank, :]  # First rank rows
    B_min = B_lr.min()
    B_max = B_lr.max()
    
    print(f"\nAfter 5 updates:")
    print(f"  A norm change: {np.linalg.norm(A_final - A_init):.6f}")
    print(f"  B norm change: {np.linalg.norm(B_final - B_init):.6f}")
    print(f"  C norm change: {np.linalg.norm(C_final - C_init):.6f}")
    print(f"  Visible norm change: {np.linalg.norm(visible_final - visible_init):.6f}")
    
    print(f"\nB weight bounds (first {rank} rows):")
    print(f"  B min: {B_min:.6f}")
    print(f"  B max: {B_max:.6f}")
    if B_min < -1.0 or B_max > 1.0:
        print(f"  ⚠️ WARNING: B exceeds [-1, 1] bounds!")
    else:
        print(f"  ✓ B within [-1, 1] bounds")
    
    # Check C bounds too
    C_min = C_final.min()
    C_max = C_final.max()
    print(f"\nC weight bounds:")
    print(f"  C min: {C_min:.6f}")
    print(f"  C max: {C_max:.6f}")
    
    # Compute expected transfer effect
    A_lr = A_final[:, :rank]
    B_lr = B_final[:rank, :]
    AB_product = A_lr @ B_lr
    expected_transfer_per_update = transfer_lr * np.linalg.norm(AB_product)
    
    print(f"\nTransfer analysis:")
    print(f"  A*B product norm: {np.linalg.norm(AB_product):.6f}")
    print(f"  Expected transfer per update: {expected_transfer_per_update:.8f}")
    print(f"  Expected total transfer (5 updates): {5 * expected_transfer_per_update:.8f}")
    print(f"  Actual C change: {np.linalg.norm(C_final - C_init):.8f}")
    
    return {
        'transfer_lr': transfer_lr,
        'C_change': np.linalg.norm(C_final - C_init),
        'visible_change': np.linalg.norm(visible_final - visible_init),
        'B_min': B_min,
        'B_max': B_max,
        'AB_norm': np.linalg.norm(AB_product)
    }

# Test with different transfer rates
transfer_rates = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
results = []

for tr in transfer_rates:
    result = test_with_transfer_rate(tr, f"Transfer LR = {tr}")
    results.append(result)

# Summary
print("\n" + "=" * 80)
print("SUMMARY OF ALL TESTS")
print("=" * 80)

print("\nTransfer Rate vs C Change:")
print(f"{'Transfer LR':<15} {'C Change':<15} {'Visible Change':<15} {'B min':<10} {'B max':<10}")
print("-" * 75)
for r in results:
    print(f"{r['transfer_lr']:<15.6f} {r['C_change']:<15.8f} {r['visible_change']:<15.8f} {r['B_min']:<10.6f} {r['B_max']:<10.6f}")

# Check if transfer rate has effect
c_changes = [r['C_change'] for r in results]
if max(c_changes) - min(c_changes) < 0.01:
    print("\n⚠️ WARNING: Transfer rate has NO effect on C weight changes!")
    print(f"   All C changes are around {np.mean(c_changes):.8f}")
else:
    print(f"\n✓ Transfer rate affects C changes (range: {min(c_changes):.8f} to {max(c_changes):.8f})")

# Check bounds violations
bounds_violations = [(r['transfer_lr'], r['B_min'], r['B_max']) 
                     for r in results if r['B_min'] < -1.0 or r['B_max'] > 1.0]
if bounds_violations:
    print(f"\n⚠️ BOUNDS VIOLATIONS DETECTED in {len(bounds_violations)} tests:")
    for tr, bmin, bmax in bounds_violations:
        print(f"   Transfer LR {tr}: B range [{bmin:.6f}, {bmax:.6f}]")

print("\n" + "=" * 80)
print("Test Complete")