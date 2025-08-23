#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Direct test of LRTT weight updates without using AnalogSGD.
This helps isolate whether the issue is in the optimizer or the tile itself.
"""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.presets.devices import ConstantStepDevice
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import PulseType


def test_direct_tile_update():
    """Test updating the tile directly without optimizer."""
    print("=== Testing Direct Tile Update ===\n")
    
    torch.manual_seed(42)
    
    # Create simple LRTT config with explicit positive dw_min
    fastA = ConstantStepDevice(dw_min=0.01, dw_min_dtod=0.0, dw_min_std=0.0)
    fastB = ConstantStepDevice(dw_min=0.01, dw_min_dtod=0.0, dw_min_std=0.0)
    visible = ConstantStepDevice(dw_min=0.01, dw_min_dtod=0.0, dw_min_std=0.0)
    
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=2,
        transfer_every=10,
        transfer_lr=0.1,
        forward_inject=True
    )
    
    # Minimal update parameters
    update_params = UpdateParameters(
        pulse_type=PulseType.STOCHASTIC_COMPRESSED,
        update_management=True,
        update_bl_management=True,
        desired_bl=31,
        fixed_bl=True
    )
    
    config = UnitCellRPUConfig(
        device=lrtt,
        update=update_params,
        forward=IOParameters(is_perfect=False)
    )
    
    # Create small layer for testing
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = layer.to(device)
    print(f"Device: {device}")
    
    # Get the analog tile
    tiles = layer.analog_tiles()
    tile = tiles[0]
    print(f"Tile type: {type(tile).__name__}")
    
    # Check if tile has learning rate methods
    if hasattr(tile, 'get_learning_rate'):
        print(f"Tile learning rate: {tile.get_learning_rate()}")
    
    if hasattr(tile, 'set_learning_rate'):
        print("Setting tile learning rate to 0.01...")
        tile.set_learning_rate(0.01)
        if hasattr(tile, 'get_learning_rate'):
            print(f"After setting: {tile.get_learning_rate()}")
    
    # Create input and gradient data
    batch_size = 2
    x = torch.randn(batch_size, 4, device=device)
    
    # Compute a forward pass to get output shape
    with torch.no_grad():
        y = layer(x)
    print(f"\nForward pass output shape: {y.shape}")
    
    # Create gradient for the output (as if from backprop)
    grad_output = torch.randn_like(y) * 0.1
    
    # Get initial weights
    initial_weights = layer.get_weights()[0].clone()
    print(f"Initial weight norm: {initial_weights.norm().item():.6f}")
    
    # Try different update approaches
    print("\n--- Approach 1: Direct tile.update() ---")
    try:
        # The tile.update() expects (x_input, d_input)
        # where d_input is the gradient w.r.t. output
        # Note: we need to transpose for proper dimensions
        if tile.in_trans:
            x_update = x.T
        else:
            x_update = x
            
        if tile.out_trans:
            d_update = grad_output.T
        else:
            d_update = grad_output
        
        # Manually scale the gradient (simulating learning rate)
        # Use POSITIVE gradient for proper SGD (the tile should negate internally)
        scaled_grad = d_update * 0.01  # Small positive "learning rate"
        
        tile.update(x_update, scaled_grad)
        print("✓ tile.update() succeeded!")
        
        # Check if weights changed
        updated_weights = layer.get_weights()[0]
        weight_change = (updated_weights - initial_weights).norm().item()
        print(f"Weight change norm: {weight_change:.6f}")
        
    except Exception as e:
        print(f"✗ tile.update() failed: {e}")
    
    print("\n--- Approach 2: Using backward() method ---")
    try:
        # Reset to initial weights
        layer.set_weights(initial_weights.to(device))
        
        # Perform a full forward-backward pass
        layer.zero_grad()
        output = layer(x)
        
        # Create a simple loss
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass (this should set up gradients)
        loss.backward()
        
        # Now manually call tile update with the computed gradients
        if layer.weight.grad is not None:
            print(f"Gradient norm: {layer.weight.grad.norm().item():.6f}")
            
            # Get the analog context if it exists
            if hasattr(layer.weight, 'analog_ctx'):
                ctx = layer.weight.analog_ctx
                if hasattr(ctx, 'analog_input') and hasattr(ctx, 'analog_grad_output'):
                    print("Found analog context with stored inputs/gradients")
                    # Try updating with stored context
                    try:
                        # The tile should use the stored input and gradient
                        tile.update(ctx.analog_input, ctx.analog_grad_output)
                        print("✓ Context-based update succeeded!")
                    except Exception as e:
                        print(f"✗ Context-based update failed: {e}")
            
        updated_weights = layer.get_weights()[0]
        weight_change = (updated_weights - initial_weights).norm().item()
        print(f"Weight change after backward: {weight_change:.6f}")
        
    except Exception as e:
        print(f"✗ Backward approach failed: {e}")
    
    print("\n--- Approach 3: Checking update parameter signs ---")
    # Let's check what the update parameters look like
    update_params = config.update
    print(f"Update management: {update_params.update_management}")
    print(f"BL management: {update_params.update_bl_management}")
    print(f"Desired BL: {update_params.desired_bl}")
    print(f"Fixed BL: {update_params.fixed_bl}")
    print(f"Pulse type: {update_params.pulse_type}")
    
    # Check device parameters
    print("\nDevice parameters:")
    for i, dev in enumerate([fastA, fastB, visible]):
        print(f"  Device {i}: dw_min={dev.dw_min}")


if __name__ == "__main__":
    test_direct_tile_update()