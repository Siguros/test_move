#!/usr/bin/env python3
"""Test LRTT with transfer_every=100 to check A weight changes."""

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
print("Testing LRTT with transfer_every=100")
print("=" * 80)

# Configuration
d_size = 32
x_size = 16
rank = 4
batch_size = 8
transfer_every = 100  # Transfer every 100 updates
transfer_lr = 0.01

# Create LRTT config
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = transfer_every
device.transfer_lr = transfer_lr
device.lora_alpha = 4.0
device.forward_inject = True

print(f"\nConfiguration:")
print(f"  Dimensions: {d_size}x{x_size}")
print(f"  Rank: {rank}")
print(f"  Transfer every: {transfer_every}")
print(f"  Transfer LR: {transfer_lr}")
print(f"  LoRA alpha: {device.lora_alpha}")

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

def check_weights(tile, label):
    """Check A, B, C weights."""
    params = tile.get_hidden_parameters()
    weight_A = params['hidden_weights_0'].cpu().numpy()
    weight_B = params['hidden_weights_1'].cpu().numpy()
    weight_C = params['hidden_weights_2'].cpu().numpy()
    
    # Get LoRA components
    A_lr = weight_A[:, :rank]  # First rank columns
    B_lr = weight_B[:rank, :]  # First rank rows
    AB_product = A_lr @ B_lr
    
    visible = tile.get_weights()[0].cpu().numpy()
    
    print(f"\n{label}:")
    print(f"  A_lr norm: {np.linalg.norm(A_lr):.8f}, max: {np.abs(A_lr).max():.8f}")
    print(f"  B_lr norm: {np.linalg.norm(B_lr):.8f}, max: {np.abs(B_lr).max():.8f}")
    print(f"  C norm: {np.linalg.norm(weight_C):.8f}")
    print(f"  A*B norm: {np.linalg.norm(AB_product):.8f}")
    print(f"  Visible norm: {np.linalg.norm(visible):.8f}")
    
    return weight_A, weight_B, weight_C, A_lr, B_lr, AB_product

# Initial state
print("\n" + "=" * 60)
print("Initial State:")
print("=" * 60)
A_init, B_init, C_init, A_lr_init, B_lr_init, AB_init = check_weights(tile, "Initial")

# Store A changes over time
A_norms = []
B_norms = []
C_norms = []
AB_norms = []

# Training loop
print("\n" + "=" * 60)
print("Training Loop (monitoring every 10 updates):")
print("=" * 60)

for update in range(200):
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
    
    # Monitor every 10 updates
    if (update + 1) % 10 == 0:
        params = tile.get_hidden_parameters()
        weight_A = params['hidden_weights_0'].cpu().numpy()
        weight_B = params['hidden_weights_1'].cpu().numpy()
        weight_C = params['hidden_weights_2'].cpu().numpy()
        
        A_lr = weight_A[:, :rank]
        B_lr = weight_B[:rank, :]
        AB = A_lr @ B_lr
        
        A_norms.append(np.linalg.norm(A_lr))
        B_norms.append(np.linalg.norm(B_lr))
        C_norms.append(np.linalg.norm(weight_C))
        AB_norms.append(np.linalg.norm(AB))
        
        print(f"Update {update+1:3d}: A_lr={A_norms[-1]:.6f}, B_lr={B_norms[-1]:.6f}, "
              f"C={C_norms[-1]:.6f}, A*B={AB_norms[-1]:.6f}, Loss={loss.item():.6f}")
        
        # Special check at transfer points
        if (update + 1) == transfer_every:
            print("\n" + "-" * 40)
            print(f"AT TRANSFER POINT (update {update+1}):")
            print("-" * 40)
            _, _, _, A_lr_before, B_lr_before, AB_before = check_weights(tile, f"Before transfer {update+1}")
            
        elif (update + 1) == transfer_every + 1:
            print("\n" + "-" * 40)
            print(f"AFTER TRANSFER (update {update+1}):")
            print("-" * 40)
            _, _, _, A_lr_after, B_lr_after, AB_after = check_weights(tile, f"After transfer {update+1}")
            
            # Check what changed
            print("\nChanges due to transfer:")
            print(f"  A_lr change: {np.linalg.norm(A_lr_after - A_lr_before):.8f}")
            print(f"  B_lr change: {np.linalg.norm(B_lr_after - B_lr_before):.8f}")
            print(f"  A*B change: {np.linalg.norm(AB_after - AB_before):.8f}")
            
            if np.linalg.norm(A_lr_after) < 1e-8:
                print("  ⚠️ A_lr was reset to zero!")
            if np.linalg.norm(B_lr_after - B_lr_before) > 0.1:
                print("  ⚠️ B_lr was reinitialized!")

# Final state
print("\n" + "=" * 60)
print("Final State (after 200 updates):")
print("=" * 60)
A_final, B_final, C_final, A_lr_final, B_lr_final, AB_final = check_weights(tile, "Final")

# Analysis
print("\n" + "=" * 60)
print("Analysis:")
print("=" * 60)

print("\nTotal changes from initial:")
print(f"  A_lr: {np.linalg.norm(A_lr_final - A_lr_init):.8f}")
print(f"  B_lr: {np.linalg.norm(B_lr_final - B_lr_init):.8f}")
print(f"  C: {np.linalg.norm(C_final - C_init):.8f}")
print(f"  A*B: {np.linalg.norm(AB_final - AB_init):.8f}")

# Check A accumulation before transfer
print("\nA_lr norm progression (every 10 updates):")
for i, a_norm in enumerate(A_norms[:10]):
    print(f"  Update {(i+1)*10:3d}: {a_norm:.8f}")
    
print("\nKey observations:")
if max(A_norms[:10]) > 1e-6:
    print("✓ A accumulates values before first transfer")
else:
    print("⚠️ A remains near zero even before transfer")

# Check transfer effect
if len(A_norms) > 10:
    before_transfer = A_norms[9]  # Update 100
    after_transfer = A_norms[10]  # Update 110
    print(f"\nA_lr norm at update 100 (before transfer): {before_transfer:.8f}")
    print(f"A_lr norm at update 110 (after transfer): {after_transfer:.8f}")
    
    if after_transfer < before_transfer * 0.1:
        print("⚠️ A_lr appears to be reset after transfer!")

# Plot if matplotlib available
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    updates = [(i+1)*10 for i in range(len(A_norms))]
    
    axes[0,0].plot(updates, A_norms, 'b-')
    axes[0,0].axvline(x=transfer_every, color='r', linestyle='--', label=f'Transfer at {transfer_every}')
    axes[0,0].axvline(x=transfer_every*2, color='r', linestyle='--')
    axes[0,0].set_title('A_lr norm over updates')
    axes[0,0].set_xlabel('Update')
    axes[0,0].set_ylabel('Norm')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].plot(updates, B_norms, 'g-')
    axes[0,1].axvline(x=transfer_every, color='r', linestyle='--', label=f'Transfer at {transfer_every}')
    axes[0,1].axvline(x=transfer_every*2, color='r', linestyle='--')
    axes[0,1].set_title('B_lr norm over updates')
    axes[0,1].set_xlabel('Update')
    axes[0,1].set_ylabel('Norm')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    axes[1,0].plot(updates, C_norms, 'm-')
    axes[1,0].axvline(x=transfer_every, color='r', linestyle='--', label=f'Transfer at {transfer_every}')
    axes[1,0].axvline(x=transfer_every*2, color='r', linestyle='--')
    axes[1,0].set_title('C norm over updates')
    axes[1,0].set_xlabel('Update')
    axes[1,0].set_ylabel('Norm')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    axes[1,1].plot(updates, AB_norms, 'r-')
    axes[1,1].axvline(x=transfer_every, color='r', linestyle='--', label=f'Transfer at {transfer_every}')
    axes[1,1].axvline(x=transfer_every*2, color='r', linestyle='--')
    axes[1,1].set_title('A*B product norm over updates')
    axes[1,1].set_xlabel('Update')
    axes[1,1].set_ylabel('Norm')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('lrtt_transfer_every_100.png')
    print("\nPlot saved as lrtt_transfer_every_100.png")
except ImportError:
    print("\nMatplotlib not available, skipping plot")

print("\n" + "=" * 80)
print("Test Complete")