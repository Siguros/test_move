#!/usr/bin/env python
"""Comprehensive LR-TT Debug Test following the debug prompt exactly."""

import torch
import torch.nn as nn
import os
import sys

# Enable LRTT debug before importing aihwkit
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")
print("="*80)
print("LR-TT COMPREHENSIVE DEBUG TEST")
print("="*80)

def create_lrtt_layer(transfer_every=2, transfer_lr=0.5, dw_min=0.0001):
    """Create an LR-TT layer with specified parameters."""
    print(f"\nCreating LR-TT layer with:")
    print(f"  transfer_every = {transfer_every}")
    print(f"  transfer_lr = {transfer_lr}")
    print(f"  dw_min = {dw_min}")
    print(f"  transfer_use_bl_management = False")
    print(f"  transfer_use_update_management = False")
    
    device = ConstantStepDevice(dw_min=dw_min)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=2,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,  # Correct key name
        forward_inject=True,
        lora_alpha=1.0,
        # Disable management for clearer testing
        transfer_use_bl_management=False,
        transfer_use_update_management=False,
        ab_use_bl_management=False,
        ab_use_update_management=False,
    )
    
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
    layer = layer.to(DEVICE)
    
    return layer

def run_training_steps(layer, n_steps=6):
    """Run training steps and monitor debug output."""
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Simple data
    x = torch.randn(8, 4).to(DEVICE)
    target = torch.randn(8, 3).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    # Get tile
    tiles = list(layer.analog_tiles())
    cpp_tile = tiles[0].tile
    
    print(f"\n{'='*80}")
    print("TRAINING STEPS")
    print(f"{'='*80}")
    
    for step in range(n_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Get weights before
        try:
            vis_before = cpp_tile.lrtt_get_visible_weights().clone()
            A_before = cpp_tile.lrtt_get_A_lr().clone()
            B_before = cpp_tile.lrtt_get_B_lr().clone()
            A_norm_before = torch.norm(A_before).item()
            B_norm_before = torch.norm(B_before).item()
            vis_norm_before = torch.norm(vis_before).item()
        except:
            print("  [Error getting weights before]")
            A_norm_before = 0
            B_norm_before = 0
            vis_norm_before = 0
        
        # Training step
        optimizer.zero_grad()
        output = layer(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        # Get weights after
        try:
            vis_after = cpp_tile.lrtt_get_visible_weights()
            A_after = cpp_tile.lrtt_get_A_lr()
            B_after = cpp_tile.lrtt_get_B_lr()
            A_norm_after = torch.norm(A_after).item()
            B_norm_after = torch.norm(B_after).item()
            vis_norm_after = torch.norm(vis_after).item()
        except:
            print("  [Error getting weights after]")
            A_norm_after = 0
            B_norm_after = 0
            vis_norm_after = 0
        
        # Report norms
        print(f"  A norm: {A_norm_before:.6f} -> {A_norm_after:.6f} (Δ={A_norm_after - A_norm_before:.6f})")
        print(f"  B norm: {B_norm_before:.6f} -> {B_norm_after:.6f} (Δ={B_norm_after - B_norm_before:.6f})")
        print(f"  Visible norm: {vis_norm_before:.6f} -> {vis_norm_after:.6f} (Δ={vis_norm_after - vis_norm_before:.6f})")
        
        # Check for transfer (A reset to 0)
        if A_norm_before > 0.01 and A_norm_after < 0.01:
            print(f"  🔄 TRANSFER DETECTED (A reset to ~0)")
            vis_delta = torch.norm(vis_after - vis_before).item()
            print(f"     Visible weight change: {vis_delta:.6f}")

print("\n" + "="*80)
print("TEST 1: BASIC OPERATION CHECK")
print("="*80)
print("Expected: ")
print("  - Initial reinit messages")
print("  - LoRA updates on each step")
print("  - Transfer every 2 steps")
print("  - Reinit after each transfer")

layer1 = create_lrtt_layer(transfer_every=2, transfer_lr=0.5, dw_min=0.0001)
run_training_steps(layer1, n_steps=6)

print("\n" + "="*80)
print("TEST 2: PARAMETER SENSITIVITY - TRANSFER_EVERY")
print("="*80)

print("\nTest 2a: transfer_every=2 (frequent transfers)")
layer2a = create_lrtt_layer(transfer_every=2, transfer_lr=0.5, dw_min=0.0001)
run_training_steps(layer2a, n_steps=4)

print("\nTest 2b: transfer_every=100000 (no transfers expected)")
layer2b = create_lrtt_layer(transfer_every=100000, transfer_lr=0.5, dw_min=0.0001)
run_training_steps(layer2b, n_steps=4)

print("\n" + "="*80)
print("TEST 3: PARAMETER SENSITIVITY - TRANSFER_LR")
print("="*80)

print("\nTest 3a: transfer_lr=0.5")
layer3a = create_lrtt_layer(transfer_every=3, transfer_lr=0.5, dw_min=0.0001)
run_training_steps(layer3a, n_steps=3)

print("\nTest 3b: transfer_lr=0.05")
layer3b = create_lrtt_layer(transfer_every=3, transfer_lr=0.05, dw_min=0.0001)
run_training_steps(layer3b, n_steps=3)

print("\nTest 3c: transfer_lr=0.005")
layer3c = create_lrtt_layer(transfer_every=3, transfer_lr=0.005, dw_min=0.0001)
run_training_steps(layer3c, n_steps=3)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("Check the debug output above for:")
print("1. REINIT: 'reinitFastTiles' messages after transfers")
print("2. UPDATES: 'Updating A device' and 'Updating B device' messages")
print("3. TRANSFERS: 'Transfer triggered' and 'applyABOuterAsPulsedUpdate' messages")
print("4. PARAMETER EFFECT: Different behavior with different transfer_every/transfer_lr values")