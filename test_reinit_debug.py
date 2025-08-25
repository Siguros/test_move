#!/usr/bin/env python3
"""Debug reinit operation in LRTT."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized
from aihwkit.optim import AnalogSGD

# Set debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'  # Enable LRTT debug output

print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 80)

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 4

# Create LRTT config
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = 1  # Transfer every update
device.transfer_lr = 0.01  # Larger transfer rate to see effect
device.lora_alpha = 4.0
device.forward_inject = True

print(f"Configuration:")
print(f"  Rank: {rank}")
print(f"  Transfer every: {device.transfer_every}")
print(f"  Transfer LR: {device.transfer_lr}")
print(f"  Reinit gain: {device.reinit_gain if hasattr(device, 'reinit_gain') else 'N/A'}")

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

def check_AB_weights(tile, label):
    """Check A and B weights specifically."""
    params = tile.get_hidden_parameters()
    weight_A = params['hidden_weights_0'].cpu().numpy()
    weight_B = params['hidden_weights_1'].cpu().numpy()
    weight_C = params['hidden_weights_2'].cpu().numpy()
    
    # Check first rank columns of A and first rank rows of B
    A_lr = weight_A[:, :rank]
    B_lr = weight_B[:rank, :]
    
    print(f"\n{label}:")
    print(f"  A full shape: {weight_A.shape}")
    print(f"  A_lr (first {rank} cols) norm: {np.linalg.norm(A_lr):.8f}")
    print(f"  A_lr max: {np.abs(A_lr).max():.8f}")
    print(f"  B full shape: {weight_B.shape}")
    print(f"  B_lr (first {rank} rows) norm: {np.linalg.norm(B_lr):.8f}")
    print(f"  B_lr max: {np.abs(B_lr).max():.8f}")
    print(f"  C norm: {np.linalg.norm(weight_C):.8f}")
    
    # Check if A is zeros and B is non-zero (expected after reinit)
    if np.linalg.norm(A_lr) < 1e-8 and np.linalg.norm(B_lr) > 1e-8:
        print("  ✓ A_lr is zero, B_lr is initialized (correct reinit)")
    elif np.linalg.norm(A_lr) < 1e-8 and np.linalg.norm(B_lr) < 1e-8:
        print("  ⚠ Both A_lr and B_lr are zero (reinit may not have run)")
    else:
        print(f"  ℹ A_lr and B_lr both have values")
    
    return weight_A, weight_B, weight_C

# Initial state
print("\n" + "=" * 40)
print("Initial State (before any updates):")
print("=" * 40)
A_init, B_init, C_init = check_AB_weights(tile, "Initial")

# Do one update to trigger transfer and reinit
print("\n" + "=" * 40)
print("First Update (should trigger transfer & reinit):")
print("=" * 40)

x = torch.randn(batch_size, x_size, device='cuda') * 0.1
target = torch.randn(batch_size, d_size, device='cuda') * 0.1

optimizer.zero_grad()
output = model(x)
loss = nn.MSELoss()(output, target)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.6f}")

# Check after first update
A_after1, B_after1, C_after1 = check_AB_weights(tile, "After 1st update")

# Do another update
print("\n" + "=" * 40)
print("Second Update:")
print("=" * 40)

x = torch.randn(batch_size, x_size, device='cuda') * 0.1
target = torch.randn(batch_size, d_size, device='cuda') * 0.1

optimizer.zero_grad()
output = model(x)
loss = nn.MSELoss()(output, target)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.6f}")

# Check after second update
A_after2, B_after2, C_after2 = check_AB_weights(tile, "After 2nd update")

# Analysis
print("\n" + "=" * 40)
print("Analysis:")
print("=" * 40)

print("\nA changes:")
print(f"  Initial->1st: {np.linalg.norm(A_after1[:,:rank] - A_init[:,:rank]):.8f}")
print(f"  1st->2nd: {np.linalg.norm(A_after2[:,:rank] - A_after1[:,:rank]):.8f}")

print("\nB changes:")
print(f"  Initial->1st: {np.linalg.norm(B_after1[:rank,:] - B_init[:rank,:]):.8f}")
print(f"  1st->2nd: {np.linalg.norm(B_after2[:rank,:] - B_after1[:rank,:]):.8f}")

print("\nC changes:")
print(f"  Initial->1st: {np.linalg.norm(C_after1 - C_init):.8f}")
print(f"  1st->2nd: {np.linalg.norm(C_after2 - C_after1):.8f}")

# Check if reinit pattern is happening
if np.linalg.norm(A_after1[:,:rank]) < 1e-8:
    print("\n✓ A_lr stays at zero after updates (correct)")
else:
    print(f"\n⚠ A_lr is non-zero after updates: {np.linalg.norm(A_after1[:,:rank]):.8f}")
    
if np.linalg.norm(B_after1[:rank,:]) > 1e-8:
    print("✓ B_lr gets initialized to non-zero (correct)")
else:
    print("⚠ B_lr remains zero (reinit might not be working)")

print("\n" + "=" * 80)
print("Test Complete")