#!/usr/bin/env python
"""Detailed monitoring of LRTT weights A, B, C and their deltas during training."""

import torch
import torch.nn as nn
from torch import Tensor

# Imports from aihwkit
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Check device
DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")
print("=" * 80)

def create_lrtt_layer():
    """Create a simple LRTT layer for detailed monitoring."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=2,  # Small rank for easier visualization
        transfer_every=5,  # Transfer every 5 steps
        transfer_lr=0.5,
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
    return layer.to(DEVICE)

def print_weight_details(weights, label=""):
    """Print detailed weight information."""
    A, B, C = weights['A'], weights['B'], weights['visible']
    
    print(f"\n{label}")
    print("-" * 60)
    
    # Print shapes and norms
    print(f"Shapes: A={list(A.shape)}, B={list(B.shape)}, C={list(C.shape)}")
    print(f"Norms:  A={torch.norm(A).item():.6f}, B={torch.norm(B).item():.6f}, C={torch.norm(C).item():.6f}")
    
    # Print actual values (first few elements)
    print(f"\nA matrix (first 2x2):")
    A_show = A[:2, :2].cpu().numpy() if A.shape[0] >= 2 and A.shape[1] >= 2 else A.cpu().numpy()
    for row in A_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    print(f"\nB matrix (first 2x2):")
    B_show = B[:2, :2].cpu().numpy() if B.shape[0] >= 2 and B.shape[1] >= 2 else B.cpu().numpy()
    for row in B_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    print(f"\nC (visible) matrix (first 2x2):")
    C_show = C[:2, :2].cpu().numpy() if C.shape[0] >= 2 and C.shape[1] >= 2 else C.cpu().numpy()
    for row in C_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    # Compute and show A @ B
    AB = torch.matmul(A, B)
    print(f"\nA @ B product (first 2x2):")
    AB_show = AB[:2, :2].cpu().numpy() if AB.shape[0] >= 2 and AB.shape[1] >= 2 else AB.cpu().numpy()
    for row in AB_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    print(f"\nA @ B norm: {torch.norm(AB).item():.6f}")

def print_weight_deltas(w_before, w_after, label=""):
    """Print weight changes."""
    print(f"\n{label}")
    print("-" * 60)
    
    delta_A = w_after['A'] - w_before['A']
    delta_B = w_after['B'] - w_before['B']
    delta_C = w_after['visible'] - w_before['visible']
    
    print(f"Delta norms: ΔA={torch.norm(delta_A).item():.6f}, "
          f"ΔB={torch.norm(delta_B).item():.6f}, "
          f"ΔC={torch.norm(delta_C).item():.6f}")
    
    # Check for transfer indicators
    A_norm_before = torch.norm(w_before['A']).item()
    A_norm_after = torch.norm(w_after['A']).item()
    B_norm_before = torch.norm(w_before['B']).item()
    B_norm_after = torch.norm(w_after['B']).item()
    
    if A_norm_before > 0.01 and A_norm_after < 0.01:
        print(f"🔄 TRANSFER DETECTED: A reset from {A_norm_before:.6f} to {A_norm_after:.6f}")
        print(f"                     B changed from {B_norm_before:.6f} to {B_norm_after:.6f}")
    
    # Print delta matrices
    print(f"\nΔA (first 2x2):")
    dA_show = delta_A[:2, :2].cpu().numpy() if delta_A.shape[0] >= 2 and delta_A.shape[1] >= 2 else delta_A.cpu().numpy()
    for row in dA_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    print(f"\nΔB (first 2x2):")
    dB_show = delta_B[:2, :2].cpu().numpy() if delta_B.shape[0] >= 2 and delta_B.shape[1] >= 2 else delta_B.cpu().numpy()
    for row in dB_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    print(f"\nΔC (visible) (first 2x2):")
    dC_show = delta_C[:2, :2].cpu().numpy() if delta_C.shape[0] >= 2 and delta_C.shape[1] >= 2 else delta_C.cpu().numpy()
    for row in dC_show:
        print(f"  [{' '.join(f'{x:7.4f}' for x in row)}]")
    
    # Check if ΔC ≈ transfer_lr * (A_before @ B_before)
    if torch.norm(delta_C).item() > 0.01:
        AB_before = torch.matmul(w_before['A'], w_before['B'])
        expected_transfer = 0.5 * AB_before  # transfer_lr = 0.5
        transfer_match = torch.norm(delta_C - expected_transfer).item()
        print(f"\nTransfer check: ΔC vs 0.5*(A@B)_before")
        print(f"  Difference norm: {transfer_match:.6f}")
        if transfer_match < 0.1:
            print(f"  ✓ Transfer matches expected value!")

def get_weights(layer):
    """Get LRTT weights from layer."""
    tiles = list(layer.analog_tiles())
    cpp_tile = tiles[0].tile
    
    return {
        'A': cpp_tile.lrtt_get_A_lr().clone(),
        'B': cpp_tile.lrtt_get_B_lr().clone(),
        'visible': cpp_tile.lrtt_get_visible_weights().clone(),
    }

def main():
    """Main monitoring function."""
    print("DETAILED LRTT WEIGHT MONITORING")
    print("=" * 80)
    print("Configuration: rank=2, transfer_every=5, transfer_lr=0.5")
    print("=" * 80)
    
    # Create layer and optimizer
    layer = create_lrtt_layer()
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Create simple data
    x = torch.randn(8, 4).to(DEVICE)
    target = torch.randn(8, 3).to(DEVICE)
    loss_fn = nn.MSELoss()
    
    # Set initial weights to non-zero for visibility
    tiles = list(layer.analog_tiles())
    cpp_tile = tiles[0].tile
    
    # Set initial C to small random values
    initial_C = torch.randn(3, 4).to(DEVICE) * 0.1
    cpp_tile.set_weights(initial_C)
    
    # Get initial state
    weights_initial = get_weights(layer)
    print_weight_details(weights_initial, "INITIAL WEIGHTS")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING (10 steps)")
    print("=" * 80)
    
    prev_weights = weights_initial
    
    for step in range(10):
        print(f"\n{'='*80}")
        print(f"STEP {step + 1}")
        print(f"{'='*80}")
        
        # Forward and backward pass
        optimizer.zero_grad()
        output = layer(x)
        loss = loss_fn(output, target)
        loss.backward()
        
        print(f"Loss: {loss.item():.6f}")
        
        # Get weights before optimizer step
        weights_before_opt = get_weights(layer)
        
        # Optimizer step
        optimizer.step()
        
        # Get weights after optimizer step
        weights_after_opt = get_weights(layer)
        
        # Print current weights
        print_weight_details(weights_after_opt, f"AFTER STEP {step + 1}")
        
        # Print deltas
        print_weight_deltas(weights_before_opt, weights_after_opt, 
                           f"CHANGES DURING STEP {step + 1}")
        
        # Check for significant changes from previous step
        if step > 0:
            cumulative_delta_C = torch.norm(weights_after_opt['visible'] - prev_weights['visible']).item()
            if cumulative_delta_C > 0.01:
                print(f"\n📊 Cumulative ΔC from step {step}: {cumulative_delta_C:.6f}")
        
        prev_weights = weights_after_opt
        
        # Extra details at transfer steps
        if (step + 1) % 5 == 0:
            print(f"\n{'*'*80}")
            print(f"EXPECTED TRANSFER POINT (step {step + 1})")
            print(f"{'*'*80}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    final_weights = get_weights(layer)
    
    print(f"\nTotal changes from initial:")
    total_delta_A = torch.norm(final_weights['A'] - weights_initial['A']).item()
    total_delta_B = torch.norm(final_weights['B'] - weights_initial['B']).item()
    total_delta_C = torch.norm(final_weights['visible'] - weights_initial['visible']).item()
    
    print(f"  Total ΔA: {total_delta_A:.6f}")
    print(f"  Total ΔB: {total_delta_B:.6f}")
    print(f"  Total ΔC: {total_delta_C:.6f}")
    
    # Show final effective weights
    print(f"\nFinal effective weights W_eff = C + A@B:")
    AB_final = torch.matmul(final_weights['A'], final_weights['B'])
    W_eff = final_weights['visible'] + AB_final
    print(f"  W_eff norm: {torch.norm(W_eff).item():.6f}")
    print(f"  C norm: {torch.norm(final_weights['visible']).item():.6f}")
    print(f"  A@B norm: {torch.norm(AB_final).item():.6f}")
    
    print("\n" + "=" * 80)
    print("MONITORING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()