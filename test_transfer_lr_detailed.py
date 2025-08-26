#!/usr/bin/env python3
"""Detailed test of LRTT transfer with different transfer_lr values showing the dw_min effect."""

import torch
import math
from torch import nn
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    LRTTTransferCompound,
    ConstantStepDevice
)

def create_lrtt_layer(transfer_lr, dw_min=0.001, rank=4, transfer_every=2):
    """Create an LRTT layer with specified parameters."""
    device = ConstantStepDevice(dw_min=dw_min)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],  # A, B, visible devices
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        forward_inject=True,
        lora_alpha=1.0,
        transfer_use_bl_management=True,  # This enables UBLM mode
        transfer_use_update_management=False
    )
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(10, 10, bias=False, rpu_config=rpu_config)
    
    if torch.cuda.is_available():
        layer = layer.cuda()
    
    return layer

def get_lrtt_weights(layer):
    """Extract LRTT weights from a layer."""
    tiles = list(layer.analog_tiles())
    if not tiles:
        return None
    
    cpp_tile = tiles[0].tile
    if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
        return {
            'visible': cpp_tile.lrtt_get_visible_weights().clone(),
            'A': cpp_tile.lrtt_get_A_lr().clone(),
            'B': cpp_tile.lrtt_get_B_lr().clone(),
        }
    return None

def analyze_transfer_probability(transfer_lr, dw_min, rank):
    """Calculate the theoretical transfer probability in UBLM mode."""
    # In StochasticCompressed with UBLM, the probability scaling is sqrt(lr/dw_min/K)
    # where K is the rank for the transfer operation
    lr_div_dwmin = abs(transfer_lr) / dw_min
    prob_scale = math.sqrt(lr_div_dwmin / rank)
    
    # The actual probability of a weight update happening
    # In stochastic mode, this is the probability that determines if update occurs
    return prob_scale

def test_transfer_effect(transfer_lr, dw_min=0.001, num_steps=4):
    """Test LRTT transfer with specified parameters and analyze the effect."""
    rank = 4
    print(f"\n{'='*70}")
    print(f"Testing: transfer_lr = {transfer_lr:.2e}, dw_min = {dw_min:.4f}")
    print(f"{'='*70}")
    
    # Calculate theoretical transfer probability
    prob_scale = analyze_transfer_probability(transfer_lr, dw_min, rank)
    print(f"\nTheoretical Analysis (StochasticCompressed + UBLM):")
    print(f"  lr/dw_min = {abs(transfer_lr)/dw_min:.6f}")
    print(f"  Transfer probability scale = sqrt(lr/dw_min/K) = sqrt({abs(transfer_lr)/dw_min:.6f}/{rank})")
    print(f"  Transfer probability scale = {prob_scale:.6f}")
    
    if prob_scale < 0.01:
        print(f"  ⚠️  WARNING: Probability < 1%, transfer updates will be VERY rare!")
    elif prob_scale < 0.1:
        print(f"  ⚠️  WARNING: Probability < 10%, transfer updates will be infrequent")
    elif prob_scale > 1.0:
        print(f"  ✓  Probability > 100%, transfer updates will be clipped but effective")
    else:
        print(f"  ✓  Probability = {prob_scale*100:.1f}%, reasonable transfer rate")
    
    # Create and test the layer
    layer = create_lrtt_layer(transfer_lr, dw_min=dw_min, rank=rank, transfer_every=2)
    optimizer = AnalogSGD([{'params': layer.parameters()}], lr=0.1)
    criterion = nn.MSELoss()
    
    # Track weight changes
    visible_changes = []
    transfer_steps = []
    
    print(f"\nRunning {num_steps} training steps (transfer_every=2):")
    print("-" * 50)
    
    for step in range(num_steps):
        weights_before = get_lrtt_weights(layer)
        
        # Training step
        x = torch.randn(5, 10)
        target = torch.randn(5, 10)
        if next(layer.parameters()).is_cuda:
            x = x.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = layer(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        weights_after = get_lrtt_weights(layer)
        
        if weights_before and weights_after:
            vis_change = torch.norm(weights_after['visible'] - weights_before['visible']).item()
            visible_changes.append(vis_change)
            
            # Check if this is a transfer step
            if step > 0 and (step + 1) % 2 == 0:
                a_norm_before = torch.norm(weights_before['A']).item()
                a_norm_after = torch.norm(weights_after['A']).item()
                
                print(f"Step {step + 1} (TRANSFER):")
                print(f"  A norm: {a_norm_before:.6f} → {a_norm_after:.6f}")
                print(f"  Visible weight change: {vis_change:.6f}")
                
                if vis_change < 0.001:
                    print(f"  ❌ No visible weight change detected (< 0.001)")
                elif vis_change < 0.01:
                    print(f"  ⚠️  Very small visible weight change")
                else:
                    print(f"  ✓  Visible weights updated")
                
                transfer_steps.append(vis_change)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    if transfer_steps:
        avg_transfer_change = sum(transfer_steps) / len(transfer_steps)
        print(f"  Average visible change during transfer: {avg_transfer_change:.6f}")
        
        if avg_transfer_change < 0.001:
            print(f"  ❌ Transfer is NOT working (changes < 0.001)")
            print(f"     Reason: transfer_lr too small for dw_min constraint")
        elif avg_transfer_change < 0.01:
            print(f"  ⚠️  Transfer is barely working (very small changes)")
        else:
            print(f"  ✓  Transfer is working properly")
    
    return visible_changes

# Main test
if __name__ == "__main__":
    print("="*70)
    print("LRTT Transfer Mechanism Analysis with dw_min Constraint")
    print("="*70)
    
    dw_min = 0.001  # Device constraint
    
    # Test 1: Very low transfer_lr (should show minimal/no transfer)
    test_transfer_effect(transfer_lr=0.00001, dw_min=dw_min)
    
    # Test 2: Medium transfer_lr (should show some transfer)
    test_transfer_effect(transfer_lr=0.01, dw_min=dw_min)
    
    # Test 3: High transfer_lr (should show clear transfer)
    test_transfer_effect(transfer_lr=1.0, dw_min=dw_min)
    
    # Test 4: Very high transfer_lr (should show strong transfer, possibly clipped)
    test_transfer_effect(transfer_lr=10000, dw_min=dw_min)
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("In StochasticCompressed mode with UBLM (update_bl_management=True):")
    print("- Transfer probability = sqrt(transfer_lr / dw_min / rank)")
    print("- When transfer_lr << dw_min, the probability becomes very small")
    print("- This means few or no weight updates occur during transfer")
    print("- For effective transfer, transfer_lr should be >= dw_min * rank")
    print(f"- Recommended minimum transfer_lr for dw_min={dw_min}, rank=4: {dw_min * 4:.4f}")