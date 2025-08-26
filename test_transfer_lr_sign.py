#!/usr/bin/env python3
"""Test that transfer_lr sign handling works correctly for both positive and negative values."""

import torch
from aihwkit.simulator.configs import lrtt_config
from aihwkit.nn import AnalogLinear

def test_transfer_lr_sign(transfer_lr_val):
    """Test transfer with specific transfer_lr value."""
    print(f"\n{'='*50}")
    print(f"Testing transfer_lr = {transfer_lr_val:.3f}")
    print('='*50)
    
    # Create config with specific transfer_lr
    config = lrtt_config(
        rank=4,
        transfer_every=2,
        transfer_lr=transfer_lr_val,
        dw_min=0.001,
        enforce_consistency=False,
        correct_gradient_magnitudes=False
    )
    
    # Create layer
    layer = AnalogLinear(10, 8, rpu_config=config, bias=False)
    layer.to('cuda')
    
    # Initialize weights  
    with torch.no_grad():
        initial = torch.randn(8, 10).cuda() * 0.1
        layer.set_weights(initial)
        
    # Store initial weights
    W_initial = layer.get_weights()[0].clone()
    print(f"Initial weights norm: {W_initial.norm().item():.6f}")
    
    # Run multiple updates to trigger transfers
    for i in range(5):
        x = torch.randn(32, 10).cuda()
        d = torch.randn(32, 8).cuda() * 0.01
        
        # Update
        layer.analog_tile.update(x.T, d.T)
        
        if i % 2 == 1:  # Transfer should happen after every 2 updates
            W_after = layer.get_weights()[0]
            delta = W_after - W_initial
            print(f"  After update {i+1}: Weight change norm = {delta.norm().item():.6f}")
            W_initial = W_after.clone()
            
    # Get final weights
    W_final = layer.get_weights()[0]
    total_change = W_final - initial
    print(f"\nFinal weight change statistics:")
    print(f"  Total change norm: {total_change.norm().item():.6f}")
    print(f"  Mean change: {total_change.mean().item():.6f}")
    print(f"  Std change: {total_change.std().item():.6f}")
    
    # Check sign behavior
    if transfer_lr_val > 0:
        print(f"  Expected: Weights should increase (positive transfer)")
    else:
        print(f"  Expected: Weights should decrease (negative transfer)")
    
    return W_final

if __name__ == "__main__":
    print("Testing LRTT transfer_lr sign handling")
    print("This verifies the D vector negation fix")
    
    # Test positive transfer_lr
    W_pos = test_transfer_lr_sign(0.1)
    
    # Test negative transfer_lr  
    W_neg = test_transfer_lr_sign(-0.1)
    
    # Test zero transfer_lr (no transfer)
    W_zero = test_transfer_lr_sign(0.0)
    
    print("\n" + "="*50)
    print("Sign handling test complete!")
    print("Check that positive and negative transfer_lr produce")
    print("opposite effects on the weight changes.")