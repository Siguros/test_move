#!/usr/bin/env python3
"""Full LRTT test with proper training loop."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized
from aihwkit.optim import AnalogSGD

# Set CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"CUDA available: {torch.cuda.is_available()}")
print("=" * 80)

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 4
transfer_lr = 0.0000001  # Very small transfer rate
learning_rate = 0.01

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
print(f"  Learning rate: {learning_rate}")

# Create model
model = nn.Sequential(
    AnalogLinear(x_size, d_size, bias=False, rpu_config=config)
).cuda()

# Create optimizer
optimizer = AnalogSGD(model.parameters(), lr=learning_rate)

# Get tile
layer = model[0]
tiles = list(layer.analog_tiles())
tile = tiles[0]

def check_weights(tile, step_name):
    """Check weight states."""
    params = tile.get_hidden_parameters()
    weight_A = params['hidden_weights_0'].cpu().numpy()
    weight_B = params['hidden_weights_1'].cpu().numpy()
    weight_C = params['hidden_weights_2'].cpu().numpy()
    
    A_subset = weight_A[:, :rank]
    B_subset = weight_B[:rank, :]
    AB = A_subset @ B_subset
    
    visible = tile.get_weights()[0].cpu().numpy()
    
    print(f"\n{step_name}:")
    print(f"  A norm: {np.linalg.norm(weight_A):.6f}")
    print(f"  B norm: {np.linalg.norm(weight_B):.6f}")
    print(f"  C norm: {np.linalg.norm(weight_C):.6f}")
    print(f"  A*B norm: {np.linalg.norm(AB):.6f}")
    print(f"  Visible norm: {np.linalg.norm(visible):.6f}")
    
    return weight_A, weight_B, weight_C, AB, visible

# Initial check
print("\n" + "=" * 40)
print("Initial State:")
print("=" * 40)
A_init, B_init, C_init, AB_init, vis_init = check_weights(tile, "Initial")

# Training loop
print("\n" + "=" * 40)
print("Training Loop:")
print("=" * 40)

for epoch in range(10):
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
    
    if epoch % 2 == 0:
        params = tile.get_hidden_parameters()
        weight_A = params['hidden_weights_0'].cpu().numpy()
        weight_B = params['hidden_weights_1'].cpu().numpy()
        weight_C = params['hidden_weights_2'].cpu().numpy()
        
        A_change = np.linalg.norm(weight_A - A_init)
        B_change = np.linalg.norm(weight_B - B_init)
        C_change = np.linalg.norm(weight_C - C_init)
        
        print(f"Epoch {epoch}: Loss={loss.item():.6f}, A_change={A_change:.8f}, B_change={B_change:.8f}, C_change={C_change:.8f}")

# Final check
print("\n" + "=" * 40)
print("Final State:")
print("=" * 40)
A_final, B_final, C_final, AB_final, vis_final = check_weights(tile, "Final")

# Analysis
print("\n" + "=" * 40)
print("Change Analysis:")
print("=" * 40)
print(f"A total change: {np.linalg.norm(A_final - A_init):.8f}")
print(f"B total change: {np.linalg.norm(B_final - B_init):.8f}")
print(f"C total change: {np.linalg.norm(C_final - C_init):.8f}")
print(f"A*B change: {np.linalg.norm(AB_final - AB_init):.8f}")
print(f"Visible change: {np.linalg.norm(vis_final - vis_init):.8f}")

# Check transfer calculation
if np.linalg.norm(A_final) > 1e-6 and np.linalg.norm(B_final) > 1e-6:
    expected_transfer_per_step = transfer_lr * np.linalg.norm(AB_final)
    print(f"\nExpected transfer per step: {expected_transfer_per_step:.10f}")
    print(f"After 10 updates, expected total transfer: {10 * expected_transfer_per_step:.10f}")
    print(f"Actual C change: {np.linalg.norm(C_final - C_init):.10f}")
    
    if np.linalg.norm(C_final - C_init) < 1e-8:
        print("\n⚠️  WARNING: C barely changed despite A*B being non-zero!")
        print("    Transfer may not be working correctly.")
else:
    print("\n⚠️  WARNING: A and B are still near zero after training!")

print("\n" + "=" * 80)
print("Test Complete")