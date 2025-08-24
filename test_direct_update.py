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


def test_lrtt_with_transfer_every(transfer_every_val):
    """Test LRTT with specific transfer_every value."""
    print(f"\n=== Testing LRTT with transfer_every={transfer_every_val} ===")
    
    torch.manual_seed(42)  # Same seed for consistent comparison
    
    # Create simple LRTT config with very small dw_min to avoid thresholding issues
    fastA = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    fastB = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    visible = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=2,
        transfer_every=transfer_every_val,
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
    
    # Important: Set initial weights to trigger proper LRTT initialization
    print("Setting initial weights to trigger LRTT initialization...")
    initial_layer_weights = torch.randn(3, 4) * 0.1
    layer.set_weights(initial_layer_weights)
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer = layer.to(device)
    print(f"Device: {device}")
    
    # Get the analog tile
    tiles = list(layer.analog_tiles())
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
    
    # Check what get_weights actually returns for LRTT
    print(f"get_weights shape: {initial_weights.shape}")
    print(f"Expected layer shape: {layer.out_features} x {layer.in_features}")
    
    # Check if we're using forward injection
    tile_obj = tiles[0]  # Use the tile we already extracted
    if hasattr(tile_obj, 'tile'):
        cpp_tile = tile_obj.tile
        print(f"C++ tile type: {type(cpp_tile).__name__}")
        
        # Try to get individual device weights if available
        if hasattr(cpp_tile, 'get_device_weights'):
            try:
                device_weights = cpp_tile.get_device_weights()
                print(f"Individual device weights: {len(device_weights)} devices")
                for i, w in enumerate(device_weights):
                    print(f"  Device {i}: shape {w.shape}, norm {w.norm().item():.6f}")
            except Exception as e:
                print(f"Could not get device weights: {e}")
    
    # Check LRTT config
    print(f"LRTT rank: {lrtt.rank}")
    print(f"LRTT forward_inject: {lrtt.forward_inject}")
    print(f"LRTT lora_alpha: {lrtt.lora_alpha}")
    print(f"LRTT transfer_every: {lrtt.transfer_every}")
    
    # Check initial LRTT matrix state after set_weights
    print("\nLRTT matrices after set_weights (should show proper initialization):")
    try:
        if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
            visible_init = cpp_tile.lrtt_get_visible_weights()
            print(f"  Visible weights norm: {visible_init.norm().item():.6f}")
        if hasattr(cpp_tile, 'lrtt_get_A_lr'):
            A_init = cpp_tile.lrtt_get_A_lr()
            print(f"  A_lr weights norm: {A_init.norm().item():.6f}")
            print(f"  A_lr sample values: {A_init.flatten()[:5]}")
        if hasattr(cpp_tile, 'lrtt_get_B_lr'):
            B_init = cpp_tile.lrtt_get_B_lr()
            print(f"  B_lr weights norm: {B_init.norm().item():.6f}")
            print(f"  B_lr sample values: {B_init.flatten()[:5]}")
    except Exception as e:
        print(f"Could not get initial matrices: {e}")
    
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
        # Use MUCH LARGER gradient to see if anything happens
        scaled_grad = d_update * 1.0  # Large positive "learning rate"
        print(f"Input grad magnitude: {scaled_grad.norm().item():.6f}")
        print(f"Input X magnitude: {x_update.norm().item():.6f}")
        
        # Manually set both A and B matrices to non-zero values to test transfer
        print("Manually setting both A and B matrices to non-zero values...")
        try:
            if hasattr(cpp_tile, 'lrtt_set_A_lr') and hasattr(cpp_tile, 'lrtt_set_B_lr'):
                # Create non-zero A matrix [d_size, rank] = [3, 2] 
                A_test = torch.randn(3, 2, device=device) * 0.1
                print(f"Setting A_lr to tensor with norm: {A_test.norm().item():.6f}")
                cpp_tile.lrtt_set_A_lr(A_test)
                
                # Create non-zero B matrix [rank, x_size] = [2, 4]
                B_test = torch.randn(2, 4, device=device) * 0.1
                print(f"Setting B_lr to tensor with norm: {B_test.norm().item():.6f}")
                cpp_tile.lrtt_set_B_lr(B_test)
                
                # Verify they were set
                A_check = cpp_tile.lrtt_get_A_lr()
                B_check = cpp_tile.lrtt_get_B_lr()
                print(f"A_lr after setting norm: {A_check.norm().item():.6f}")
                print(f"B_lr after setting norm: {B_check.norm().item():.6f}")
                
                # Check current visible weights (should be zero before any transfer)
                visible_before_transfer = cpp_tile.lrtt_get_visible_weights()
                print(f"Visible weights before transfer: {visible_before_transfer.norm().item():.6f}")
                
                # Check if forward injection shows A@B contribution
                composed_after_AB = layer.get_weights()[0]
                print(f"get_weights() with A≠0, B≠0: {composed_after_AB.norm().item():.6f}")
                
                # Calculate expected A@B manually for verification
                AB_manual = torch.mm(A_check, B_check)
                print(f"Manual A@B norm: {AB_manual.norm().item():.6f}")
                
            else:
                print("lrtt_set_A_lr or lrtt_set_B_lr method not available")
        except Exception as e:
            print(f"Could not set A/B matrices: {e}")
        
        # Capture BEFORE state
        print("\nLRTT matrices BEFORE updates:")
        try:
            if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
                visible_before = cpp_tile.lrtt_get_visible_weights()
                print(f"  Visible weights norm: {visible_before.norm().item():.6f}")
            if hasattr(cpp_tile, 'lrtt_get_A_lr'):
                A_before = cpp_tile.lrtt_get_A_lr()
                print(f"  A_lr weights norm: {A_before.norm().item():.6f}")
            if hasattr(cpp_tile, 'lrtt_get_B_lr'):
                B_before = cpp_tile.lrtt_get_B_lr()
                print(f"  B_lr weights norm: {B_before.norm().item():.6f}")
        except Exception as e:
            print(f"Could not get BEFORE matrices: {e}")
        
        # Perform multiple updates to potentially trigger transfer
        # Focused test: just track B matrix updates across multiple steps  
        max_steps = min(15, transfer_every_val + 5)  # Test enough steps to see at least one transfer
        print(f"Tracking B matrix updates for {max_steps} steps...")
        
        b_updates_count = 0
        transfers_count = 0
        
        for step in range(max_steps):
            # Check B matrix BEFORE each update
            try:
                B_before_step = cpp_tile.lrtt_get_B_lr()
                B_norm_before = B_before_step.norm().item()
            except:
                B_norm_before = -1
            
            # Perform the update
            tile.update(x_update, scaled_grad)
            
            # Check B matrix AFTER each update
            try:
                A_after_step = cpp_tile.lrtt_get_A_lr()
                B_after_step = cpp_tile.lrtt_get_B_lr()
                A_norm_after = A_after_step.norm().item()
                B_norm_after = B_after_step.norm().item()
                
                # Track B matrix updates
                if abs(B_norm_after - B_norm_before) > 1e-6:
                    b_updates_count += 1
                
                # Check for transfer (A reinitialization)
                if A_norm_after > 1.0 and step > 0:
                    transfers_count += 1
                    print(f"  Step {step+1}: Transfer occurred - A={A_norm_after:.6f}, B zeroed to {B_norm_after:.6f}")
                elif step < 5 or step % 3 == 0:  # Show some details
                    print(f"  Step {step+1}: B={B_norm_before:.6f}->{B_norm_after:.6f}, A={A_norm_after:.6f}")
                        
            except Exception as e:
                print(f"  Step {step+1}: Could not check matrices: {e}")
                
            # Also check every 5 steps for summary
            if step % 5 == 4:  
                current_weights = layer.get_weights()[0]
                change_norm = (current_weights - initial_weights).norm().item()
                print(f"  === Step {step+1} Summary ===")
                print(f"    Weight change norm: {change_norm:.6f}")
                print(f"    A_lr norm: {A_norm_after:.6f}, B_lr norm: {B_norm_after:.6f}")
                
                composed_check = layer.get_weights()[0]
                print(f"    get_weights() norm: {composed_check.norm().item():.6f}")
                print()
        
        print("✓ Multiple tile.update() calls succeeded!")
        
        # Try manual transfer if available
        print("Attempting manual transfer to see Kaiming/zero initialization:")
        try:
            if hasattr(cpp_tile, 'doTransfer'):
                print("Calling doTransfer()...")
                cpp_tile.doTransfer()
                # Check A and B after manual transfer
                A_after_transfer = cpp_tile.lrtt_get_A_lr()
                B_after_transfer = cpp_tile.lrtt_get_B_lr()
                print(f"After manual transfer - A_lr norm: {A_after_transfer.norm().item():.6f}")
                print(f"After manual transfer - B_lr norm: {B_after_transfer.norm().item():.6f}")
                print(f"A sample values: {A_after_transfer.flatten()[:5]}")
                print(f"B sample values: {B_after_transfer.flatten()[:5]}")
            else:
                print("doTransfer() method not available")
        except Exception as e:
            print(f"Manual transfer failed: {e}")
        
        # Check if weights changed after all updates
        updated_weights = layer.get_weights()[0]
        weight_change = (updated_weights - initial_weights).norm().item()
        print(f"Final weight change norm: {weight_change:.6f}")
        
        # Check individual device weights after update
        if hasattr(cpp_tile, 'get_device_weights'):
            try:
                device_weights_after = cpp_tile.get_device_weights()
                print("Individual device weights after update:")
                for i, w in enumerate(device_weights_after):
                    print(f"  Device {i}: norm {w.norm().item():.6f}")
            except Exception as e:
                print(f"Could not get device weights after update: {e}")
        
        # Check individual LRTT matrices  
        print("Checking individual LRTT matrices after updates:")
        try:
            if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
                visible_weights = cpp_tile.lrtt_get_visible_weights()
                print(f"Visible weights norm: {visible_weights.norm().item():.6f}")
            if hasattr(cpp_tile, 'lrtt_get_A_lr'):
                A_lr = cpp_tile.lrtt_get_A_lr()
                print(f"A_lr weights norm: {A_lr.norm().item():.6f}")
            if hasattr(cpp_tile, 'lrtt_get_B_lr'):
                B_lr = cpp_tile.lrtt_get_B_lr()
                print(f"B_lr weights norm: {B_lr.norm().item():.6f}")
        except Exception as e:
            print(f"Could not get individual matrices: {e}")
            
        # Try to manually trigger forward injection to see composed weights
        if hasattr(cpp_tile, 'compose_effective_weights') or hasattr(cpp_tile, 'lrtt_compose_w_eff'):
            try:
                if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
                    print("Trying lrtt_compose_w_eff...")
                    composed_weights = cpp_tile.lrtt_compose_w_eff(lrtt.lora_alpha)
                elif hasattr(cpp_tile, 'compose_effective_weights'):
                    print("Trying compose_effective_weights...")
                    composed_weights = cpp_tile.compose_effective_weights()
                print(f"Composed weights norm: {composed_weights.norm().item():.6f}")
            except Exception as e:
                print(f"Could not compose weights: {e}")
        
        # Try to access LRTT-specific debug info
        try:
            # Look for debug methods or counters in the C++ tile
            if hasattr(cpp_tile, 'getHostCopyCount'):
                host_copies = cpp_tile.getHostCopyCount()
                print(f"Host copy count: {host_copies}")
            if hasattr(cpp_tile, 'printDebugStats'):
                print("Debug stats:")
                cpp_tile.printDebugStats()
            if hasattr(cpp_tile, 'getRank'):
                rank = cpp_tile.getRank()
                print(f"CUDA tile rank: {rank}")
        except Exception as e:
            print(f"Could not get debug info: {e}")
        
        # List all LRTT-specific methods on the cpp_tile
        print("Searching for LRTT-specific methods on CUDA tile:")
        methods = [m for m in dir(cpp_tile) if not m.startswith('_')]
        lrtt_methods = [m for m in methods if 'lrtt' in m.lower() or 'rank' in m.lower() or 'transfer' in m.lower()]
        if lrtt_methods:
            print("Found LRTT methods:")
            for method in lrtt_methods:
                print(f"  {method}")
        else:
            print("No obvious LRTT-specific methods found")
            
        # Show first 15 methods to see what's available
        print("First 15 methods:")
        for method in methods[:15]:
            print(f"  {method}")
        if len(methods) > 15:
            print(f"  ... and {len(methods)-15} more")
        
    except Exception as e:
        print(f"✗ tile.update() failed: {e}")
    
    print("\n--- Approach 4: Full layer training path ---")
    try:
        # Reset weights
        layer.set_weights(initial_weights.to(device))
        
        # Use AnalogSGD to trigger proper update path
        from aihwkit.optim import AnalogSGD
        optimizer = AnalogSGD(layer.parameters(), lr=0.1)
        
        initial_weights_4 = layer.get_weights()[0].clone()
        print(f"Initial weight norm (approach 4): {initial_weights_4.norm().item():.6f}")
        
        # Perform training steps
        for step in range(12):
            optimizer.zero_grad()
            
            # Forward pass  
            output = layer(x)
            
            # Create loss
            target = torch.randn_like(output)
            loss = nn.functional.mse_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights using AnalogSGD
            optimizer.step()
            
            if step % 5 == 4:
                current_weights = layer.get_weights()[0]
                change_norm = (current_weights - initial_weights_4).norm().item()
                print(f"  Step {step+1}: weight change norm: {change_norm:.6f}, loss: {loss.item():.6f}")
        
        # Final check
        final_weights_4 = layer.get_weights()[0]
        weight_change_4 = (final_weights_4 - initial_weights_4).norm().item()
        print(f"Final weight change norm (approach 4): {weight_change_4:.6f}")
        print("✓ Full layer training succeeded!")
        
    except Exception as e:
        print(f"✗ Full layer training failed: {e}")
    
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