#!/usr/bin/env python3
"""Test LRTT transfer with different transfer_lr values to verify transfer mechanism."""

import torch
from torch import nn
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    LRTTTransferCompound,
    ConstantStepDevice
)

def create_lrtt_layer(transfer_lr, rank=4, transfer_every=2):
    """Create an LRTT layer with specified transfer_lr."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        forward_inject=True,
        lora_alpha=1.0,
        transfer_use_bl_management=False,
        transfer_use_update_management=False
    )
    rpu_config = SingleRPUConfig(device=lrtt_config)
    layer = AnalogLinear(10, 10, bias=False, rpu_config=rpu_config)
    
    # Move to CUDA if available for LRTT weight extraction
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

def test_transfer(transfer_lr, num_steps=5):
    """Test LRTT transfer with specified learning rate."""
    print(f"\n{'='*60}")
    print(f"Testing with transfer_lr = {transfer_lr}")
    print(f"{'='*60}")
    
    # Create model with LRTT layer
    layer = create_lrtt_layer(transfer_lr, rank=4, transfer_every=2)
    
    # Setup optimizer
    optimizer = AnalogSGD([{'params': layer.parameters()}], lr=0.1)
    criterion = nn.MSELoss()
    
    # Track weight changes
    visible_changes = []
    a_norms = []
    b_norms = []
    
    # Initial weights
    initial_weights = get_lrtt_weights(layer)
    if initial_weights:
        print(f"\nInitial state:")
        print(f"  A norm: {torch.norm(initial_weights['A']).item():.6f}")
        print(f"  B norm: {torch.norm(initial_weights['B']).item():.6f}")
        print(f"  Visible norm: {torch.norm(initial_weights['visible']).item():.6f}")
        print(f"  Visible[0,0]: {initial_weights['visible'][0,0].item():.6f}")
    
    print(f"\nTraining for {num_steps} steps (transfer_every=2):")
    
    for step in range(num_steps):
        # Get weights before update
        weights_before = get_lrtt_weights(layer)
        
        # Forward pass  
        x = torch.randn(5, 10)
        target = torch.randn(5, 10)
        
        # Move to CUDA if layer is on CUDA
        if next(layer.parameters()).is_cuda:
            x = x.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = layer(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Get weights after update
        weights_after = get_lrtt_weights(layer)
        
        if weights_before and weights_after:
            # Calculate changes
            vis_change = torch.norm(weights_after['visible'] - weights_before['visible']).item()
            a_norm = torch.norm(weights_after['A']).item()
            b_norm = torch.norm(weights_after['B']).item()
            
            visible_changes.append(vis_change)
            a_norms.append(a_norm)
            b_norms.append(b_norm)
            
            # Check if transfer happened (A should reset to ~0)
            if step > 0 and (step + 1) % 2 == 0:  # Transfer should happen every 2 steps
                a_norm_before = torch.norm(weights_before['A']).item()
                
                print(f"\nStep {step + 1} (TRANSFER EXPECTED):")
                print(f"  A norm: {a_norm_before:.6f} → {a_norm:.6f}")
                print(f"  B norm: {b_norm:.6f}")
                print(f"  Visible change: {vis_change:.6f}")
                print(f"  Visible[0,0]: {weights_before['visible'][0,0].item():.6f} → {weights_after['visible'][0,0].item():.6f}")
                
                # Calculate effective transfer amount (A@B product norm)
                if a_norm_before > 0.001:  # If A had accumulated gradients
                    ab_product = torch.matmul(weights_before['A'], weights_before['B'])
                    ab_norm = torch.norm(ab_product).item()
                    print(f"  A@B product norm (before transfer): {ab_norm:.6f}")
                    
                    # Expected visible change should be proportional to transfer_lr * A@B
                    expected_change = transfer_lr * ab_norm
                    print(f"  Expected visible change scale: transfer_lr * ||A@B|| = {transfer_lr:.2e} * {ab_norm:.6f} = {expected_change:.6f}")
                
                if a_norm < 0.001:
                    print(f"  ✓ A was reset (norm < 0.001)")
                else:
                    print(f"  ⚠ A was not fully reset")
                    
            else:
                print(f"\nStep {step + 1} (gradient accumulation):")
                print(f"  A norm: {a_norm:.6f}")
                print(f"  Visible change: {vis_change:.6f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for transfer_lr = {transfer_lr}:")
    print(f"  Average visible weight change: {sum(visible_changes)/len(visible_changes):.6f}")
    print(f"  Max visible weight change: {max(visible_changes):.6f}")
    print(f"  Final A norm: {a_norms[-1]:.6f}")
    print(f"  Final B norm: {b_norms[-1]:.6f}")
    
    final_weights = get_lrtt_weights(layer)
    if initial_weights and final_weights:
        total_change = torch.norm(final_weights['visible'] - initial_weights['visible']).item()
        print(f"  Total visible weight change from start: {total_change:.6f}")
    
    return visible_changes

# Test with different transfer_lr values
if __name__ == "__main__":
    print("Testing LRTT Transfer Mechanism with Different Learning Rates")
    print("="*60)
    
    # Test with high transfer_lr (10000)
    changes_high = test_transfer(transfer_lr=10000, num_steps=6)
    
    # Test with low transfer_lr (0.00001) 
    changes_low = test_transfer(transfer_lr=0.00001, num_steps=6)
    
    # Test with medium transfer_lr (1.0)
    changes_medium = test_transfer(transfer_lr=1.0, num_steps=6)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON OF TRANSFER EFFECTS:")
    print(f"{'='*60}")
    print(f"Average visible weight changes per step:")
    print(f"  transfer_lr = 10000:   {sum(changes_high)/len(changes_high):.6f}")
    print(f"  transfer_lr = 1.0:     {sum(changes_medium)/len(changes_medium):.6f}")
    print(f"  transfer_lr = 0.00001: {sum(changes_low)/len(changes_low):.6f}")
    print(f"\nMax visible weight changes:")
    print(f"  transfer_lr = 10000:   {max(changes_high):.6f}")
    print(f"  transfer_lr = 1.0:     {max(changes_medium):.6f}")
    print(f"  transfer_lr = 0.00001: {max(changes_low):.6f}")
    print(f"\nNote: Actual transfer is limited by device dw_min parameter (0.001 in our case)")