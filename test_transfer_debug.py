#!/usr/bin/env python3
"""Debug script to test transfer AB to C with BL management disabled."""

import torch
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized

def test_transfer_ab_to_c():
    """Test transfer from A*B to C with debug output."""
    print("=" * 80)
    print("Testing LRTT Transfer AB->C with BL Management Disabled")
    print("=" * 80)
    
    # Configuration
    d_size = 32
    x_size = 16
    rank = 4
    batch_size = 8
    
    # Create LRTT config with BL management disabled
    config = lrtt_idealized(rank=rank)
    
    # Modify device settings
    device = config.device
    device.transfer_every = 1
    device.transfer_lr = 0.1
    device.lora_alpha = 4.0
    device.forward_inject = True
    # Explicitly disable BL management for transfer
    device.transfer_use_bl_management = False
    device.ab_use_bl_management = False
    device.transfer_desired_bl = -1.0
    device.ab_desired_bl = -1.0
    
    # Create analog layer
    layer = AnalogLinear(x_size, d_size, bias=False, rpu_config=config).cuda()
    
    print(f"\nConfiguration:")
    print(f"  Input size: {x_size}")
    print(f"  Output size: {d_size}")
    print(f"  Rank: {rank}")
    print(f"  Transfer every: {device.transfer_every}")
    print(f"  Transfer LR: {device.transfer_lr}")
    print(f"  LoRA alpha: {device.lora_alpha}")
    print(f"  Forward inject: {device.forward_inject}")
    print(f"  Transfer BL management: {device.transfer_use_bl_management}")
    print(f"  AB BL management: {device.ab_use_bl_management}")
    
    # Get the tile and access individual devices
    tiles = list(layer.analog_tiles())
    if tiles:
        tile = tiles[0]
    else:
        print("ERROR: No analog tiles found!")
        return
    
    # Get initial weights
    print("\n" + "=" * 40)
    print("Initial State:")
    print("=" * 40)
    
    # Get weights from each device
    weights_dict = tile.get_hidden_parameters()
    
    # Extract A, B, C weights
    if 'weight_0' in weights_dict:
        weight_A = weights_dict['weight_0'].cpu().numpy()
        print(f"A shape: {weight_A.shape}")
        print(f"A range: [{weight_A.min():.6f}, {weight_A.max():.6f}]")
        print(f"A norm: {np.linalg.norm(weight_A):.6f}")
    
    if 'weight_1' in weights_dict:
        weight_B = weights_dict['weight_1'].cpu().numpy()
        print(f"B shape: {weight_B.shape}")
        print(f"B range: [{weight_B.min():.6f}, {weight_B.max():.6f}]")
        print(f"B norm: {np.linalg.norm(weight_B):.6f}")
    
    if 'weight_2' in weights_dict:
        weight_C = weights_dict['weight_2'].cpu().numpy()
        print(f"C shape: {weight_C.shape}")
        print(f"C range: [{weight_C.min():.6f}, {weight_C.max():.6f}]")
        print(f"C norm: {np.linalg.norm(weight_C):.6f}")
    
    # Get initial visible weights
    if hasattr(layer, 'weight') and layer.weight is not None:
        visible_weights_before = layer.weight.data.cpu().numpy()
        print(f"\nInitial visible weights:")
        print(f"  Shape: {visible_weights_before.shape}")
        print(f"  Norm: {np.linalg.norm(visible_weights_before):.6f}")
    else:
        # Get visible weights directly from tile
        visible_weights_before = tile.get_weights()[0].cpu().numpy()
        print(f"\nInitial visible weights (from tile):")
        print(f"  Shape: {visible_weights_before.shape}")
        print(f"  Norm: {np.linalg.norm(visible_weights_before):.6f}")
    
    # Perform forward and backward pass to trigger update
    print("\n" + "=" * 40)
    print("Performing Update:")
    print("=" * 40)
    
    x = torch.randn(batch_size, x_size, device='cuda')
    target = torch.randn(batch_size, d_size, device='cuda')
    
    # Forward pass
    y = layer(x)
    print(f"Forward output shape: {y.shape}")
    
    # Compute loss and backward
    loss = ((y - target) ** 2).mean()
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    
    # Get weights after update
    print("\n" + "=" * 40)
    print("After Update (should trigger transfer):")
    print("=" * 40)
    
    weights_dict_after = tile.get_hidden_parameters()
    
    # Check A weights (should change)
    if 'weight_0' in weights_dict_after:
        weight_A_after = weights_dict_after['weight_0'].cpu().numpy()
        A_change = np.linalg.norm(weight_A_after - weight_A)
        print(f"A change norm: {A_change:.6f}")
        print(f"A range after: [{weight_A_after.min():.6f}, {weight_A_after.max():.6f}]")
    
    # Check B weights (should change)
    if 'weight_1' in weights_dict_after:
        weight_B_after = weights_dict_after['weight_1'].cpu().numpy()
        B_change = np.linalg.norm(weight_B_after - weight_B)
        print(f"B change norm: {B_change:.6f}")
        print(f"B range after: [{weight_B_after.min():.6f}, {weight_B_after.max():.6f}]")
    
    # Check C weights (should change due to transfer)
    if 'weight_2' in weights_dict_after:
        weight_C_after = weights_dict_after['weight_2'].cpu().numpy()
        C_change = np.linalg.norm(weight_C_after - weight_C)
        print(f"C change norm: {C_change:.6f}")
        print(f"C range after: [{weight_C_after.min():.6f}, {weight_C_after.max():.6f}]")
        
        # Compute A*B product
        A_subset = weight_A_after[:, :rank]  # First rank columns of A
        B_subset = weight_B_after[:rank, :]  # First rank rows of B
        AB_product = A_subset @ B_subset
        
        print(f"\n=== Transfer Analysis ===")
        print(f"A*B product:")
        print(f"  Shape: {AB_product.shape}")
        print(f"  Norm: {np.linalg.norm(AB_product):.6f}")
        print(f"  Range: [{AB_product.min():.6f}, {AB_product.max():.6f}]")
        
        # Expected transfer to C
        expected_transfer = device.transfer_lr * AB_product
        expected_C = weight_C + expected_transfer
        
        print(f"\nExpected C after transfer (C + transfer_lr * AB):")
        print(f"  Transfer amount norm: {np.linalg.norm(expected_transfer):.6f}")
        print(f"  Expected C norm: {np.linalg.norm(expected_C):.6f}")
        print(f"  Actual C norm: {np.linalg.norm(weight_C_after):.6f}")
        print(f"  Difference: {np.linalg.norm(weight_C_after - expected_C):.6f}")
        
        # Check visible weights
        if hasattr(layer, 'weight') and layer.weight is not None:
            visible_weights_after = layer.weight.data.cpu().numpy()
        else:
            visible_weights_after = tile.get_weights()[0].cpu().numpy()
        
        print(f"\n=== Visible Weight Verification ===")
        print(f"Visible weights after update:")
        print(f"  Shape: {visible_weights_after.shape}")
        print(f"  Norm: {np.linalg.norm(visible_weights_after):.6f}")
        print(f"  Change from initial: {np.linalg.norm(visible_weights_after - visible_weights_before):.6f}")
        
        # With forward_inject, visible = C + (lora_alpha/rank) * A*B
        if device.forward_inject:
            expected_visible = weight_C_after + (device.lora_alpha / rank) * AB_product
            print(f"\nExpected visible (C + alpha/rank * AB):")
            print(f"  alpha/rank = {device.lora_alpha/rank:.2f}")
            print(f"  Expected norm: {np.linalg.norm(expected_visible):.6f}")
            print(f"  Actual norm: {np.linalg.norm(visible_weights_after):.6f}")
            print(f"  Difference: {np.linalg.norm(visible_weights_after - expected_visible):.6f}")
            
            # Check if AB contribution is correct
            AB_contribution = (device.lora_alpha / rank) * AB_product
            print(f"\nAB contribution to visible:")
            print(f"  Norm: {np.linalg.norm(AB_contribution):.6f}")
            print(f"  Max value: {np.abs(AB_contribution).max():.6f}")
        else:
            print(f"\nWithout forward_inject, visible should equal C:")
            print(f"  C norm: {np.linalg.norm(weight_C_after):.6f}")
            print(f"  Visible norm: {np.linalg.norm(visible_weights_after):.6f}")
            print(f"  Difference: {np.linalg.norm(visible_weights_after - weight_C_after):.6f}")
    
    # Multiple updates to see transfer pattern
    print("\n" + "=" * 40)
    print("Multiple Updates Test:")
    print("=" * 40)
    
    for i in range(5):
        x = torch.randn(batch_size, x_size, device='cuda')
        target = torch.randn(batch_size, d_size, device='cuda')
        
        y = layer(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        
        weights_dict_i = tile.get_hidden_parameters()
        weight_C_i = weights_dict_i['weight_2'].cpu().numpy()
        C_norm = np.linalg.norm(weight_C_i)
        
        if hasattr(layer, 'weight') and layer.weight is not None:
            visible_i = layer.weight.data.cpu().numpy()
        else:
            visible_i = tile.get_weights()[0].cpu().numpy()
        visible_norm = np.linalg.norm(visible_i)
        
        print(f"Update {i+1}: C norm = {C_norm:.6f}, Visible norm = {visible_norm:.6f}, Loss = {loss.item():.6f}")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    # Set CUDA debugging flags
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run test
    test_transfer_ab_to_c()