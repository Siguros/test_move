#!/usr/bin/env python3
"""Test transfer mechanism: A*B -> C (visible weights)."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized
from aihwkit.optim import AnalogSGD

# Set debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

print("=" * 80)
print("Testing Transfer Mechanism: A*B -> C")
print("=" * 80)

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 4

# Create LRTT config with transfer_every=5
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = 5  # Transfer every 5 updates
device.transfer_lr = 1.0  # Full transfer rate

print(f"\nConfiguration:")
print(f"  Dimensions: {d_size}x{x_size}")
print(f"  Rank: {rank}")
print(f"  Transfer every: {device.transfer_every}")
print(f"  Transfer LR: {device.transfer_lr}")

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

def check_all_weights(label):
    """Check A, B, C weights and compute A*B."""
    params = tile.get_hidden_parameters()
    weight_A = params['hidden_weights_0'].cpu().numpy()
    weight_B = params['hidden_weights_1'].cpu().numpy()
    weight_C = params['hidden_weights_2'].cpu().numpy()
    
    # Get LoRA components
    A_lr = weight_A[:, :rank]  # First rank columns
    B_lr = weight_B[:rank, :]  # First rank rows
    AB_product = A_lr @ B_lr
    
    # Get visible weights
    visible = tile.get_weights()[0].cpu().numpy()
    
    print(f"\n{label}:")
    print(f"  A_lr norm: {np.linalg.norm(A_lr):.8f}")
    print(f"  B_lr norm: {np.linalg.norm(B_lr):.8f}")
    print(f"  C norm: {np.linalg.norm(weight_C):.8f}")
    print(f"  A*B norm: {np.linalg.norm(AB_product):.8f}")
    print(f"  Visible norm: {np.linalg.norm(visible):.8f}")
    
    # Show first few values
    print(f"  A_lr[0,0]: {A_lr[0,0]:.8f}")
    print(f"  B_lr[0,0]: {B_lr[0,0]:.8f}")
    print(f"  C[0,0]: {weight_C[0,0]:.8f}")
    print(f"  Visible[0,0]: {visible[0,0]:.8f}")
    
    return weight_A, weight_B, weight_C, A_lr, B_lr, AB_product, visible

# Initial state
print("\n" + "=" * 60)
print("Initial State:")
print("=" * 60)
A_init, B_init, C_init, A_lr_init, B_lr_init, AB_init, vis_init = check_all_weights("Initial")

# Do 4 updates (no transfer yet)
print("\n" + "=" * 60)
print("After 4 Updates (before transfer):")
print("=" * 60)

for i in range(4):
    x = torch.randn(batch_size, x_size, device='cuda') * 0.1
    target = torch.randn(batch_size, d_size, device='cuda') * 0.1
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    print(f"Update {i+1} done")

A_4, B_4, C_4, A_lr_4, B_lr_4, AB_4, vis_4 = check_all_weights("After 4 updates")

# Do 5th update (should trigger transfer)
print("\n" + "=" * 60)
print("5th Update (should trigger transfer):")
print("=" * 60)

x = torch.randn(batch_size, x_size, device='cuda') * 0.1
target = torch.randn(batch_size, d_size, device='cuda') * 0.1
optimizer.zero_grad()
output = model(x)
loss = nn.MSELoss()(output, target)
loss.backward()
optimizer.step()
print("Update 5 done - TRANSFER SHOULD OCCUR")

A_5, B_5, C_5, A_lr_5, B_lr_5, AB_5, vis_5 = check_all_weights("After 5 updates (after transfer)")

# Analysis
print("\n" + "=" * 60)
print("Transfer Analysis:")
print("=" * 60)

print("\nChanges from update 4->5 (transfer point):")
print(f"  A_lr change: {np.linalg.norm(A_lr_5 - A_lr_4):.8f}")
print(f"  B_lr change: {np.linalg.norm(B_lr_5 - B_lr_4):.8f}")
print(f"  C change: {np.linalg.norm(C_5 - C_4):.8f}")
print(f"  Visible change: {np.linalg.norm(vis_5 - vis_4):.8f}")

# Expected transfer
print("\nExpected vs Actual Transfer:")
print(f"  A*B before transfer (update 4): {np.linalg.norm(AB_4):.8f}")
print(f"  Expected C change (transfer_lr * A*B): {1.0 * np.linalg.norm(AB_4):.8f}")
print(f"  Actual C change: {np.linalg.norm(C_5 - C_4):.8f}")

# Check if A was reset
if np.linalg.norm(A_lr_5) < 1e-8:
    print("\n⚠️  A_lr was reset to 0 after transfer (expected)")
else:
    print(f"\n⚠️  A_lr is non-zero after transfer: {np.linalg.norm(A_lr_5):.8f}")

# Check if B was reinitialized
if np.linalg.norm(B_lr_5 - B_lr_4) > 0.1:
    print("⚠️  B_lr was reinitialized after transfer")
else:
    print("✓ B_lr unchanged after transfer")

# Check transfer correctness
if np.linalg.norm(C_5 - C_4) < 1e-8:
    print("\n❌ CRITICAL: No transfer occurred (C didn't change)!")
else:
    print(f"\n✓ Transfer occurred: C changed by {np.linalg.norm(C_5 - C_4):.8f}")

print("\n" + "=" * 80)
print("Test Complete")