#!/usr/bin/env python
"""Comprehensive test of LRTT operations: gradient updates, transfer, and forward computation."""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

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
print("TEST 1: Checking if A and B receive gradients and update")
print("-" * 70)

layer = create_lrtt_layer()
optimizer = AnalogSGD(layer.parameters(), lr=0.1)

# Get initial A, B, and visible weights
tile = layer.analog_module.tile
# Get all weights - returns (A, B, visible)
weights = tile.get_weights()
if isinstance(weights, tuple) and len(weights) == 3:
    initial_A, initial_B, initial_visible = weights[0].clone(), weights[1].clone(), weights[2].clone()
else:
    # Fallback to regular weights
    initial_visible = tile.get_weights() if not isinstance(tile.get_weights(), tuple) else tile.get_weights()[0]
    initial_A = torch.zeros_like(initial_visible[:2, :])  # Mock for rank=2
    initial_B = torch.zeros_like(initial_visible[:, :2])

print(f"Initial A shape: {initial_A.shape}, norm: {torch.norm(initial_A):.6f}")
print(f"Initial B shape: {initial_B.shape}, norm: {torch.norm(initial_B):.6f}")
print(f"Initial visible shape: {initial_visible.shape}, norm: {torch.norm(initial_visible):.6f}")

# Do a forward-backward pass
x = torch.randn(8, 4).to(DEVICE)
target = torch.randn(8, 2).to(DEVICE)

loss_fn = nn.MSELoss()
losses = []

print("\nTraining for 5 steps to observe weight changes:")
for step in range(5):
    optimizer.zero_grad()
    output = layer(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    # Get current weights
    weights = tile.get_weights()
    if isinstance(weights, tuple) and len(weights) == 3:
        current_A, current_B, current_visible = weights[0], weights[1], weights[2]
    else:
        current_visible = weights if not isinstance(weights, tuple) else weights[0]
        current_A = initial_A
        current_B = initial_B
    
    # Check changes
    A_change = torch.norm(current_A - initial_A).item()
    B_change = torch.norm(current_B - initial_B).item()
    visible_change = torch.norm(current_visible - initial_visible).item()
    
    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"ΔA={A_change:.6f}, ΔB={B_change:.6f}, Δvisible={visible_change:.6f}")
    
    # Update references for next comparison
    if step == 2:  # Transfer should happen at step 3 (transfer_every=3)
        A_before_transfer = current_A.clone()
        B_before_transfer = current_B.clone()
        visible_before_transfer = current_visible.clone()

print("\n✓ A and B are receiving gradients and updating!" if A_change > 0 and B_change > 0 else "✗ Problem with A/B updates")

# Test 2: Check transfer to visible weights
print("\n" + "=" * 70)
print("TEST 2: Checking transfer to visible weights (transfer_every=3)")
print("-" * 70)

# Continue training to trigger transfer
layer2 = create_lrtt_layer(transfer_every=3)
optimizer2 = AnalogSGD(layer2.parameters(), lr=0.1)
tile2 = layer2.analog_module.tile

print("Monitoring transfer at step 3:")
for step in range(6):
    # Get weights before update
    weights_before = tile2.get_weights()
    if isinstance(weights_before, tuple) and len(weights_before) == 3:
        A_before = weights_before[0].clone()
        B_before = weights_before[1].clone()
        visible_before = weights_before[2].clone()
    else:
        visible_before = weights_before if not isinstance(weights_before, tuple) else weights_before[0]
        A_before = torch.zeros(2, 2).to(DEVICE)
        B_before = torch.zeros(2, 4).to(DEVICE)
    
    # Compute expected transfer (A @ B)
    AB_product = torch.matmul(B_before.T, A_before.T).T  # Adjust for dimension ordering
    
    optimizer2.zero_grad()
    output = layer2(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer2.step()
    
    # Get weights after update
    weights_after = tile2.get_weights()
    if isinstance(weights_after, tuple) and len(weights_after) == 3:
        A_after = weights_after[0]
        B_after = weights_after[1]
        visible_after = weights_after[2]
    else:
        visible_after = weights_after if not isinstance(weights_after, tuple) else weights_after[0]
        A_after = A_before
        B_after = B_before
    
    # Check if transfer happened
    visible_change = torch.norm(visible_after - visible_before).item()
    A_reset = torch.norm(A_after).item() < torch.norm(A_before).item() * 0.5  # A should reset
    B_reset = torch.norm(B_after).item() > torch.norm(B_before).item() * 0.5  # B reinits to non-zero
    
    if step == 2:  # Transfer should happen at step 3 (0-indexed)
        print(f"\nStep {step+1} (TRANSFER EXPECTED):")
        print(f"  Visible change: {visible_change:.6f}")
        print(f"  A norm: {torch.norm(A_before):.6f} → {torch.norm(A_after):.6f} (should → 0)")
        print(f"  B norm: {torch.norm(B_before):.6f} → {torch.norm(B_after):.6f} (should reinit)")
        print(f"  Transfer detected: {visible_change > 0.01 and A_reset}")
    else:
        print(f"Step {step+1}: visible_change={visible_change:.6f}")

# Test 3: Check forward operation (C + AB)
print("\n" + "=" * 70)
print("TEST 3: Checking forward operation computes C + α*(A@B) correctly")
print("-" * 70)

# Create a fresh layer
layer3 = create_lrtt_layer(in_features=6, out_features=4, rank=3)
tile3 = layer3.analog_module.tile

# Get the weights
weights3 = tile3.get_weights()
if isinstance(weights3, tuple) and len(weights3) == 3:
    A = weights3[0]  # shape: [rank, out_features]
    B = weights3[1]  # shape: [rank, in_features]
    C_visible = weights3[2]  # shape: [out_features, in_features]
else:
    C_visible = weights3 if not isinstance(weights3, tuple) else weights3[0]
    A = torch.randn(3, 4).to(DEVICE) * 0.1
    B = torch.randn(3, 6).to(DEVICE) * 0.1

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C (visible) shape: {C_visible.shape}")

# Manual computation of expected output
x_test = torch.randn(10, 6).to(DEVICE)

# Expected: W_eff = C + α*(A.T @ B)
# Note: Need to handle the dimension ordering correctly
AB = torch.matmul(A.T, B)  # [out_features, in_features]
W_effective = C_visible + 1.0 * AB  # lora_alpha = 1.0

expected_output = torch.matmul(x_test, W_effective.T)

# Actual forward pass
layer3.eval()
with torch.no_grad():
    actual_output = layer3(x_test)

# Compare
difference = torch.norm(actual_output - expected_output).item()
relative_error = difference / (torch.norm(expected_output).item() + 1e-8)

print(f"\nForward pass comparison:")
print(f"  Expected output norm: {torch.norm(expected_output):.6f}")
print(f"  Actual output norm: {torch.norm(actual_output):.6f}")
print(f"  Absolute difference: {difference:.6f}")
print(f"  Relative error: {relative_error:.6f}")

if relative_error < 0.1:
    print("✓ Forward operation (C + A@B) is working correctly!")
else:
    print("✗ Forward operation has issues")

# Test 4: Check forward with and without injection
print("\n" + "=" * 70)
print("TEST 4: Comparing forward with injection ON vs OFF")
print("-" * 70)

# With injection ON (default)
meta_params = tile3.get_meta_parameters()
meta_params.forward_inject = True
with torch.no_grad():
    output_with_inject = layer3(x_test)

# With injection OFF (visible only)
meta_params.forward_inject = False
with torch.no_grad():
    output_visible_only = layer3(x_test)

# Restore
meta_params.forward_inject = True

# Expected visible-only output
expected_visible_only = torch.matmul(x_test, C_visible.T)

print(f"Output with injection norm: {torch.norm(output_with_inject):.6f}")
print(f"Output visible-only norm: {torch.norm(output_visible_only):.6f}")
print(f"Difference (inject vs visible): {torch.norm(output_with_inject - output_visible_only):.6f}")
print(f"Visible-only accuracy: {torch.norm(output_visible_only - expected_visible_only):.6f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("1. A/B gradient updates: ✓" if A_change > 0 and B_change > 0 else "1. A/B gradient updates: ✗")
print("2. Transfer to visible: ✓" if visible_change > 0.01 else "2. Transfer to visible: ✗")
print("3. Forward (C+AB): ✓" if relative_error < 0.1 else "3. Forward (C+AB): ✗")
print("4. Injection control: ✓" if torch.norm(output_with_inject - output_visible_only) > 0.01 else "4. Injection control: ✗")