#!/usr/bin/env python3
"""Detailed test to check if B weights change during updates 1-100."""

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
print("Detailed B Weight Update Test")
print("=" * 80)

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 4
transfer_every = 100

# Create LRTT config
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = transfer_every
device.transfer_lr = 0.01

print(f"\nConfiguration:")
print(f"  Dimensions: {d_size}x{x_size}")
print(f"  Rank: {rank}")
print(f"  Transfer every: {transfer_every}")

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
A_init = params_init['hidden_weights_0'].cpu().numpy().copy()
B_init = params_init['hidden_weights_1'].cpu().numpy().copy()
C_init = params_init['hidden_weights_2'].cpu().numpy().copy()

print(f"\nInitial state:")
print(f"  A shape: {A_init.shape}, norm: {np.linalg.norm(A_init):.8f}")
print(f"  B shape: {B_init.shape}, norm: {np.linalg.norm(B_init):.8f}")
print(f"  C shape: {C_init.shape}, norm: {np.linalg.norm(C_init):.8f}")

# Track B changes update by update
B_norms = []
B_changes = []
A_norms = []
A_changes = []

print("\n" + "=" * 60)
print("Tracking A and B changes for first 20 updates:")
print("=" * 60)

B_prev = B_init.copy()
A_prev = A_init.copy()

for update in range(20):
    # Generate data
    x = torch.randn(batch_size, x_size, device='cuda') * 0.1
    target = torch.randn(batch_size, d_size, device='cuda') * 0.1
    
    # Forward pass
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Get current weights
    params = tile.get_hidden_parameters()
    A_curr = params['hidden_weights_0'].cpu().numpy()
    B_curr = params['hidden_weights_1'].cpu().numpy()
    
    # Calculate changes
    A_change = np.linalg.norm(A_curr - A_prev)
    B_change = np.linalg.norm(B_curr - B_prev)
    
    A_norms.append(np.linalg.norm(A_curr))
    B_norms.append(np.linalg.norm(B_curr))
    A_changes.append(A_change)
    B_changes.append(B_change)
    
    print(f"Update {update+1:2d}: "
          f"A norm={A_norms[-1]:.8f} (Δ={A_change:.8f}), "
          f"B norm={B_norms[-1]:.8f} (Δ={B_change:.8f})")
    
    # Update previous
    A_prev = A_curr.copy()
    B_prev = B_curr.copy()

# Continue to update 100 to see the transfer
print("\n" + "=" * 60)
print("Continuing to update 100 (transfer point):")
print("=" * 60)

# Skip to update 95
for update in range(20, 95):
    x = torch.randn(batch_size, x_size, device='cuda') * 0.1
    target = torch.randn(batch_size, d_size, device='cuda') * 0.1
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()

# Track closely around transfer point
for update in range(95, 105):
    # Generate data
    x = torch.randn(batch_size, x_size, device='cuda') * 0.1
    target = torch.randn(batch_size, d_size, device='cuda') * 0.1
    
    # Forward pass
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Get current weights
    params = tile.get_hidden_parameters()
    A_curr = params['hidden_weights_0'].cpu().numpy()
    B_curr = params['hidden_weights_1'].cpu().numpy()
    C_curr = params['hidden_weights_2'].cpu().numpy()
    
    # Get low-rank parts
    A_lr = A_curr[:, :rank]
    B_lr = B_curr[:rank, :]
    
    print(f"Update {update+1:3d}: "
          f"A_lr norm={np.linalg.norm(A_lr):.6f}, "
          f"B_lr norm={np.linalg.norm(B_lr):.6f}, "
          f"C norm={np.linalg.norm(C_curr):.6f}")
    
    if update == 99:
        B_before_transfer = B_curr.copy()
        A_before_transfer = A_curr.copy()
        print("  >>> Before transfer")
    elif update == 100:
        B_after_transfer = B_curr.copy()
        A_after_transfer = A_curr.copy()
        print("  >>> After transfer")
        print(f"  >>> B change at transfer: {np.linalg.norm(B_after_transfer - B_before_transfer):.6f}")
        print(f"  >>> A change at transfer: {np.linalg.norm(A_after_transfer - A_before_transfer):.6f}")

print("\n" + "=" * 60)
print("Analysis:")
print("=" * 60)

# Check if B changed during first 20 updates
total_B_change = sum(B_changes)
total_A_change = sum(A_changes)

print(f"\nTotal changes in first 20 updates:")
print(f"  A total change: {total_A_change:.8f}")
print(f"  B total change: {total_B_change:.8f}")

if total_B_change < 1e-8:
    print("  ⚠️ B does NOT change during normal updates (frozen)")
else:
    print(f"  ✓ B DOES change during normal updates")

if total_A_change < 1e-8:
    print("  ⚠️ A does NOT change during normal updates")
else:
    print(f"  ✓ A DOES change during normal updates")

# Check what happens at initialization
print(f"\nAt initialization (update 0->1):")
if B_changes[0] > 1e-8:
    print(f"  B changed by {B_changes[0]:.8f} (likely Kaiming init)")
else:
    print(f"  B did not change")

if A_changes[0] > 1e-8:
    print(f"  A changed by {A_changes[0]:.8f}")
else:
    print(f"  A did not change")

print("\n" + "=" * 80)
print("Test Complete")