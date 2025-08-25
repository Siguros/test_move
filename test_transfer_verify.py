#!/usr/bin/env python3
"""Verify LRTT transfer with small learning rate."""

import torch
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized

# Set CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 80)

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 2
transfer_lr = 0.0000001  # Very small transfer rate as requested

# Create LRTT config
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = 1
device.transfer_lr = transfer_lr
device.lora_alpha = 4.0
device.forward_inject = True
device.transfer_use_bl_management = False
device.ab_use_bl_management = False

print(f"Configuration:")
print(f"  Rank: {rank}")
print(f"  Transfer every: {device.transfer_every}")
print(f"  Transfer LR: {device.transfer_lr:.10f}")
print(f"  LoRA alpha: {device.lora_alpha}")
print(f"  Forward inject: {device.forward_inject}")
print(f"  Transfer BL management: {device.transfer_use_bl_management}")

# Create layer
layer = AnalogLinear(x_size, d_size, bias=False, rpu_config=config).cuda()

# Get tile
tiles = list(layer.analog_tiles())
tile = tiles[0]

# Get initial weights
print("\n" + "=" * 40)
print("Initial State:")
print("=" * 40)

params = tile.get_hidden_parameters()
weight_A = params['hidden_weights_0'].cpu().numpy()
weight_B = params['hidden_weights_1'].cpu().numpy()
weight_C = params['hidden_weights_2'].cpu().numpy()

print(f"A shape: {weight_A.shape}, norm: {np.linalg.norm(weight_A):.6f}")
print(f"B shape: {weight_B.shape}, norm: {np.linalg.norm(weight_B):.6f}")
print(f"C shape: {weight_C.shape}, norm: {np.linalg.norm(weight_C):.6f}")

# Compute initial A*B
A_subset = weight_A[:, :rank]  # First rank columns of A
B_subset = weight_B[:rank, :]  # First rank rows of B
AB_initial = A_subset @ B_subset
print(f"\nInitial A*B product:")
print(f"  Shape: {AB_initial.shape}")
print(f"  Norm: {np.linalg.norm(AB_initial):.6f}")
print(f"  Range: [{AB_initial.min():.6f}, {AB_initial.max():.6f}]")

# Get visible weights
visible_initial = tile.get_weights()[0].cpu().numpy()
print(f"\nInitial visible weights:")
print(f"  Shape: {visible_initial.shape}")
print(f"  Norm: {np.linalg.norm(visible_initial):.6f}")

# Perform one update
print("\n" + "=" * 40)
print("Performing Single Update:")
print("=" * 40)

x = torch.randn(batch_size, x_size, device='cuda')
y = layer(x)
loss = y.sum()
loss.backward()

print(f"Loss: {loss.item():.6f}")

# Get weights after update
print("\n" + "=" * 40)
print("After Update (should trigger transfer):")
print("=" * 40)

params_after = tile.get_hidden_parameters()
weight_A_after = params_after['hidden_weights_0'].cpu().numpy()
weight_B_after = params_after['hidden_weights_1'].cpu().numpy()
weight_C_after = params_after['hidden_weights_2'].cpu().numpy()

# Check changes
A_change = np.linalg.norm(weight_A_after - weight_A)
B_change = np.linalg.norm(weight_B_after - weight_B)
C_change = np.linalg.norm(weight_C_after - weight_C)

print(f"A change: {A_change:.10f}")
print(f"B change: {B_change:.10f}")
print(f"C change: {C_change:.10f}")

# Compute new A*B
A_subset_after = weight_A_after[:, :rank]
B_subset_after = weight_B_after[:rank, :]
AB_after = A_subset_after @ B_subset_after

print(f"\nA*B after update:")
print(f"  Norm: {np.linalg.norm(AB_after):.6f}")
print(f"  Change from initial: {np.linalg.norm(AB_after - AB_initial):.6f}")

# Expected transfer
expected_transfer = transfer_lr * AB_after
expected_C = weight_C + expected_transfer

print(f"\nTransfer Analysis:")
print(f"  Transfer LR: {transfer_lr:.10f}")
print(f"  A*B norm: {np.linalg.norm(AB_after):.6f}")
print(f"  Expected transfer amount norm: {np.linalg.norm(expected_transfer):.10f}")
print(f"  Expected C after transfer norm: {np.linalg.norm(expected_C):.6f}")
print(f"  Actual C after transfer norm: {np.linalg.norm(weight_C_after):.6f}")
print(f"  Difference from expected: {np.linalg.norm(weight_C_after - expected_C):.10f}")

# Check if transfer happened at all
if C_change < 1e-10:
    print("\n⚠️  WARNING: C did not change! Transfer may not be working!")
else:
    print(f"\n✓ C changed by {C_change:.10f}")

# Check visible weights
visible_after = tile.get_weights()[0].cpu().numpy()
visible_change = np.linalg.norm(visible_after - visible_initial)

print(f"\nVisible weights after update:")
print(f"  Norm: {np.linalg.norm(visible_after):.6f}")
print(f"  Change: {visible_change:.10f}")

if device.forward_inject:
    expected_visible = weight_C_after + (device.lora_alpha / rank) * AB_after
    print(f"\nWith forward_inject, expected visible = C + (alpha/rank)*AB:")
    print(f"  alpha/rank = {device.lora_alpha/rank:.2f}")
    print(f"  Expected visible norm: {np.linalg.norm(expected_visible):.6f}")
    print(f"  Actual visible norm: {np.linalg.norm(visible_after):.6f}")
    print(f"  Difference: {np.linalg.norm(visible_after - expected_visible):.10f}")

# Do multiple updates to see pattern
print("\n" + "=" * 40)
print("Multiple Updates (5 more):")
print("=" * 40)

for i in range(5):
    x = torch.randn(batch_size, x_size, device='cuda')
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    params_i = tile.get_hidden_parameters()
    weight_C_i = params_i['hidden_weights_2'].cpu().numpy()
    C_norm = np.linalg.norm(weight_C_i)
    C_change_i = np.linalg.norm(weight_C_i - weight_C)
    
    print(f"Update {i+2}: C norm = {C_norm:.6f}, total C change = {C_change_i:.10f}")

print("\n" + "=" * 80)
print("Test Complete")