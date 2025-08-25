#!/usr/bin/env python
"""Comprehensive test of LRTT operations for MNIST example."""

import torch
import torch.nn as nn
import numpy as np

# Imports from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Check device
DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")
print("=" * 70)

# Create a simple LRTT layer for testing
def create_lrtt_layer(in_features=4, out_features=2, rank=2, transfer_every=3):
    """Create a single LRTT layer for testing."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],  # fastA, fastB, visible
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=0.5,
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(in_features, out_features, bias=False, rpu_config=rpu_config)
    return layer.to(DEVICE)

# Test 1: Check A and B receive gradients and update
print("TEST 1: Checking if A and B receive gradients and update correctly")
print("-" * 70)

layer = create_lrtt_layer()
optimizer = AnalogSGD(layer.parameters(), lr=0.1)

# Access the C++ tile directly
tiles = list(layer.analog_tiles())
cpp_tile = tiles[0].tile

# Check if LRTT methods exist
if not hasattr(cpp_tile, 'lrtt_get_visible_weights'):
    print("ERROR: LRTT methods not found in tile!")
    exit(1)

# Get initial A, B, and visible weights using LRTT-specific methods
initial_visible = cpp_tile.lrtt_get_visible_weights().clone()
initial_A = cpp_tile.lrtt_get_A_lr().clone()
initial_B = cpp_tile.lrtt_get_B_lr().clone()

print(f"Initial A shape: {initial_A.shape}, norm: {torch.norm(initial_A):.6f}")
print(f"Initial B shape: {initial_B.shape}, norm: {torch.norm(initial_B):.6f}")
print(f"Initial visible shape: {initial_visible.shape}, norm: {torch.norm(initial_visible):.6f}")

# Do forward-backward passes
x = torch.randn(8, 4).to(DEVICE)
target = torch.randn(8, 2).to(DEVICE)
loss_fn = nn.MSELoss()

print("\nTraining for 5 steps to observe weight changes:")
for step in range(5):
    optimizer.zero_grad()
    output = layer(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    
    # Get current weights
    current_visible = cpp_tile.lrtt_get_visible_weights()
    current_A = cpp_tile.lrtt_get_A_lr()
    current_B = cpp_tile.lrtt_get_B_lr()
    
    # Check changes
    A_change = torch.norm(current_A - initial_A).item()
    B_change = torch.norm(current_B - initial_B).item()
    visible_change = torch.norm(current_visible - initial_visible).item()
    
    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"ΔA={A_change:.6f}, ΔB={B_change:.6f}, Δvisible={visible_change:.6f}")

print("\n✓ A and B are receiving gradients and updating!" if A_change > 0 and B_change > 0 else "✗ Problem with A/B updates")

# Test 2: Check transfer to visible weights
print("\n" + "=" * 70)
print("TEST 2: Checking transfer to visible weights (transfer_every=3)")
print("-" * 70)

layer2 = create_lrtt_layer(transfer_every=3)
optimizer2 = AnalogSGD(layer2.parameters(), lr=0.1)
tiles2 = list(layer2.analog_tiles())
cpp_tile2 = tiles2[0].tile

print("Monitoring weights at each step (transfer expected at step 3):")
for step in range(6):
    # Get weights before update
    visible_before = cpp_tile2.lrtt_get_visible_weights().clone()
    A_before = cpp_tile2.lrtt_get_A_lr().clone()
    B_before = cpp_tile2.lrtt_get_B_lr().clone()
    
    # Compute expected transfer (A @ B)
    AB_product = torch.matmul(A_before, B_before)  # [d_size, x_size]
    
    optimizer2.zero_grad()
    output = layer2(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer2.step()
    
    # Get weights after update
    visible_after = cpp_tile2.lrtt_get_visible_weights()
    A_after = cpp_tile2.lrtt_get_A_lr()
    B_after = cpp_tile2.lrtt_get_B_lr()
    
    # Check changes
    visible_change = torch.norm(visible_after - visible_before).item()
    A_norm_before = torch.norm(A_before).item()
    A_norm_after = torch.norm(A_after).item()
    B_norm_before = torch.norm(B_before).item()
    B_norm_after = torch.norm(B_after).item()
    
    if step == 2:  # Transfer should happen at step 3 (0-indexed)
        print(f"\nStep {step+1} (TRANSFER EXPECTED):")
        print(f"  Visible change: {visible_change:.6f}")
        print(f"  A norm: {A_norm_before:.6f} → {A_norm_after:.6f} (should → 0)")
        print(f"  B norm: {B_norm_before:.6f} → {B_norm_after:.6f} (should reinit)")
        
        # Check if A reset to 0 and B reinitialized
        A_reset = A_norm_after < 0.01
        B_reinit = B_norm_after > 0.1
        transfer_detected = visible_change > 0.01 and A_reset
        
        print(f"  Transfer detected: {transfer_detected}")
        print(f"  A reset to 0: {A_reset}")
        print(f"  B reinitialized: {B_reinit}")
    else:
        print(f"Step {step+1}: vis_change={visible_change:.6f}, "
              f"A_norm={A_norm_after:.6f}, B_norm={B_norm_after:.6f}")

# Test 3: Check forward operation (C + AB)
print("\n" + "=" * 70)
print("TEST 3: Checking forward operation computes C + α*(A@B) correctly")
print("-" * 70)

layer3 = create_lrtt_layer(in_features=6, out_features=4, rank=3)
tiles3 = list(layer3.analog_tiles())
cpp_tile3 = tiles3[0].tile

# Set known values for A, B, and visible weights
# A shape should be [out_features, rank] = [4, 3]
# B shape should be [rank, in_features] = [3, 6]
A_test = torch.randn(4, 3).to(DEVICE) * 0.1  # [d_size, rank]
B_test = torch.randn(3, 6).to(DEVICE) * 0.1  # [rank, x_size]
C_test = torch.randn(4, 6).to(DEVICE) * 0.2  # [d_size, x_size]

cpp_tile3.lrtt_set_A_lr(A_test)
cpp_tile3.lrtt_set_B_lr(B_test)
cpp_tile3.set_weights(C_test)  # This should set visible weights

# Get the actual weights stored
A_stored = cpp_tile3.lrtt_get_A_lr()
B_stored = cpp_tile3.lrtt_get_B_lr()
C_stored = cpp_tile3.lrtt_get_visible_weights()

print(f"A shape: {A_stored.shape}, norm: {torch.norm(A_stored):.6f}")
print(f"B shape: {B_stored.shape}, norm: {torch.norm(B_stored):.6f}")
print(f"C (visible) shape: {C_stored.shape}, norm: {torch.norm(C_stored):.6f}")

# Check if compose_w_eff exists
if hasattr(cpp_tile3, 'lrtt_compose_w_eff'):
    W_eff_composed = cpp_tile3.lrtt_compose_w_eff(1.0)  # Pass alpha=1.0
    print(f"W_eff (composed) norm: {torch.norm(W_eff_composed):.6f}")
    
    # Manual computation
    # A is [d_size, rank], B is [rank, x_size]
    # A @ B = [d_size, x_size] = [out_features, in_features]
    AB = torch.matmul(A_stored, B_stored)  # [d_size, x_size]
    W_eff_manual = C_stored + 1.0 * AB  # lora_alpha = 1.0
    
    diff = torch.norm(W_eff_composed - W_eff_manual).item()
    print(f"Difference between composed and manual: {diff:.6e}")

# Test forward pass
x_test = torch.randn(10, 6).to(DEVICE)

# Forward with injection ON (default)
# We need to access the device's forward_inject parameter differently
layer3.eval()

# Get the RPU config to modify forward_inject
rpu_config = layer3.analog_module.rpu_config
original_inject = rpu_config.device.forward_inject

# Forward with injection ON (should be default)
rpu_config.device.forward_inject = True
# Re-initialize the tile with updated config
layer3.analog_module.tile = layer3.analog_module._create_simulator_tile(
    layer3.analog_module.tile.get_x_size(),
    layer3.analog_module.tile.get_d_size(),
    rpu_config
)
# Restore weights
layer3.analog_module.tile.set_weights(C_stored)
cpp_tile3_new = layer3.analog_module.tile
cpp_tile3_new.lrtt_set_A_lr(A_stored)
cpp_tile3_new.lrtt_set_B_lr(B_stored)

with torch.no_grad():
    output_with_inject = layer3(x_test)

# Forward with injection OFF (visible only)
rpu_config.device.forward_inject = False
layer3.analog_module.tile = layer3.analog_module._create_simulator_tile(
    layer3.analog_module.tile.get_x_size(),
    layer3.analog_module.tile.get_d_size(),
    rpu_config
)
layer3.analog_module.tile.set_weights(C_stored)
cpp_tile3_new2 = layer3.analog_module.tile
cpp_tile3_new2.lrtt_set_A_lr(A_stored)
cpp_tile3_new2.lrtt_set_B_lr(B_stored)

with torch.no_grad():
    output_visible_only = layer3(x_test)

# Restore
rpu_config.device.forward_inject = original_inject

# Manual computation
AB = torch.matmul(A_stored, B_stored)  # [d_size, x_size]
W_with_inject = C_stored + 1.0 * AB
expected_with_inject = torch.matmul(x_test, W_with_inject.T)
expected_visible_only = torch.matmul(x_test, C_stored.T)

print(f"\nForward pass comparison:")
print(f"  Output with injection norm: {torch.norm(output_with_inject):.6f}")
print(f"  Output visible-only norm: {torch.norm(output_visible_only):.6f}")
print(f"  Expected with inject norm: {torch.norm(expected_with_inject):.6f}")
print(f"  Expected visible-only norm: {torch.norm(expected_visible_only):.6f}")

inject_error = torch.norm(output_with_inject - expected_with_inject).item() / (torch.norm(expected_with_inject).item() + 1e-8)
visible_error = torch.norm(output_visible_only - expected_visible_only).item() / (torch.norm(expected_visible_only).item() + 1e-8)

print(f"  Relative error (with inject): {inject_error:.6f}")
print(f"  Relative error (visible only): {visible_error:.6f}")

# Test 4: Verify injection difference
print("\n" + "=" * 70)
print("TEST 4: Verifying forward_inject makes a difference")
print("-" * 70)

inject_diff = torch.norm(output_with_inject - output_visible_only).item()
AB_contribution = torch.norm(torch.matmul(x_test, AB.T)).item()

print(f"Difference between inject ON vs OFF: {inject_diff:.6f}")
print(f"Expected A@B contribution: {AB_contribution:.6f}")
print(f"Ratio: {inject_diff / (AB_contribution + 1e-8):.6f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"1. A/B gradient updates: {'✓' if A_change > 0 and B_change > 0 else '✗'}")
print(f"2. Transfer to visible: {'✓' if visible_change > 0.01 else '✗'}")
print(f"3. Forward (C+AB): {'✓' if inject_error < 0.1 else '✗'}")
print(f"4. Injection control: {'✓' if inject_diff > 0.01 else '✗'}")
print("=" * 70)