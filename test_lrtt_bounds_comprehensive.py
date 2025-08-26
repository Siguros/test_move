#!/usr/bin/env python3
"""Comprehensive test for LRTT weight bounds and update mechanisms."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.optim import AnalogSGD
import os

# Enable debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

def create_lrtt_config(rank=2, transfer_lr=1.0, w_min=-1.0, w_max=1.0):
    """Create LRTT configuration with specified bounds."""
    
    # Create devices with specific bounds
    fastA = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
    fastB = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
    visible = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.0001)
    
    # Create LRTT compound
    device = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=rank,
        transfer_lr=transfer_lr,
        transfer_every=1,
        forward_inject=True,
        lora_alpha=1.0,
        reinit_gain=1.0
    )
    
    return UnitCellRPUConfig(device=device)

def test_weight_bounds_and_updates():
    """Test weight bounds enforcement and update mechanisms."""
    print("=" * 80)
    print("COMPREHENSIVE LRTT BOUNDS AND UPDATE TEST")
    print("=" * 80)
    
    # Test parameters
    d_size = 8
    x_size = 4
    rank = 2
    batch_size = 2
    
    # Test configurations with different bounds
    test_cases = [
        {"w_min": -1.0, "w_max": 1.0, "transfer_lr": 1.0, "label": "Standard bounds [-1,1]"},
        {"w_min": -0.5, "w_max": 0.5, "transfer_lr": 1.0, "label": "Tight bounds [-0.5,0.5]"},
        {"w_min": -2.0, "w_max": 2.0, "transfer_lr": 1.0, "label": "Wide bounds [-2,2]"},
        {"w_min": -1.0, "w_max": 1.0, "transfer_lr": 0.1, "label": "Low transfer LR"},
        {"w_min": -1.0, "w_max": 1.0, "transfer_lr": 10.0, "label": "High transfer LR"},
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test_case['label']}")
        print(f"  w_min={test_case['w_min']}, w_max={test_case['w_max']}, transfer_lr={test_case['transfer_lr']}")
        print(f"{'='*60}")
        
        # Create config and model
        config = create_lrtt_config(
            rank=rank, 
            transfer_lr=test_case['transfer_lr'],
            w_min=test_case['w_min'],
            w_max=test_case['w_max']
        )
        
        model = nn.Sequential(
            AnalogLinear(x_size, d_size, bias=False, rpu_config=config)
        ).cuda()
        
        optimizer = AnalogSGD(model.parameters(), lr=0.01)
        
        # Get tile and initial state
        layer = model[0]
        tiles = list(layer.analog_tiles())
        tile = tiles[0]
        
        # Force weight initialization
        tile.set_weights(torch.randn(d_size, x_size).cuda() * 0.1)
        
        # Get initial state
        params_init = tile.get_hidden_parameters()
        A_init = params_init['hidden_weights_0'].cpu().numpy()
        B_init = params_init['hidden_weights_1'].cpu().numpy()
        C_init = params_init['hidden_weights_2'].cpu().numpy()
        
        print(f"\nInitial state:")
        print(f"  A shape: {A_init.shape}, norm: {np.linalg.norm(A_init):.6f}")
        print(f"  B shape: {B_init.shape}, norm: {np.linalg.norm(B_init):.6f}")
        print(f"  C shape: {C_init.shape}, norm: {np.linalg.norm(C_init):.6f}")
        
        # Check initial A initialization
        A_lr = A_init[:, :rank]
        print(f"  A_lr (first {rank} cols) stats:")
        print(f"    min: {A_lr.min():.6f}, max: {A_lr.max():.6f}, mean: {A_lr.mean():.6f}")
        
        # Perform updates
        for i in range(3):
            x = torch.randn(batch_size, x_size, device='cuda') * 0.5
            target = torch.randn(batch_size, d_size, device='cuda') * 0.5
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
            
            # Check state after each update
            params = tile.get_hidden_parameters()
            A = params['hidden_weights_0'].cpu().numpy()
            B = params['hidden_weights_1'].cpu().numpy()
            C = params['hidden_weights_2'].cpu().numpy()
            
            print(f"\nAfter update {i+1}:")
            print(f"  A change: {np.linalg.norm(A - A_init):.6f}")
            print(f"  B change: {np.linalg.norm(B - B_init):.6f}")
            print(f"  C change: {np.linalg.norm(C - C_init):.6f}")
            
            # Check bounds for each device
            A_lr = A[:, :rank]
            B_lr = B[:rank, :]
            
            # Check A bounds
            A_violations = (A_lr < test_case['w_min']) | (A_lr > test_case['w_max'])
            if A_violations.any():
                print(f"  ⚠️ A bounds violation: {A_violations.sum()} elements out of bounds")
                print(f"     A range: [{A_lr.min():.6f}, {A_lr.max():.6f}]")
            else:
                print(f"  ✓ A within bounds [{A_lr.min():.6f}, {A_lr.max():.6f}]")
            
            # Check B bounds
            B_violations = (B_lr < test_case['w_min']) | (B_lr > test_case['w_max'])
            if B_violations.any():
                print(f"  ⚠️ B bounds violation: {B_violations.sum()} elements out of bounds")
                print(f"     B range: [{B_lr.min():.6f}, {B_lr.max():.6f}]")
            else:
                print(f"  ✓ B within bounds [{B_lr.min():.6f}, {B_lr.max():.6f}]")
            
            # Check C bounds
            C_violations = (C < test_case['w_min']) | (C > test_case['w_max'])
            if C_violations.any():
                print(f"  ⚠️ C bounds violation: {C_violations.sum()} elements out of bounds")
                print(f"     C range: [{C.min():.6f}, {C.max():.6f}]")
            else:
                print(f"  ✓ C within bounds [{C.min():.6f}, {C.max():.6f}]")
            
            # Check A*B product
            AB = A_lr @ B_lr
            print(f"  A*B product norm: {np.linalg.norm(AB):.6f}")

def test_initialization_and_reinit():
    """Test weight initialization and reinit mechanisms."""
    print("\n" + "=" * 80)
    print("TESTING INITIALIZATION AND REINIT")
    print("=" * 80)
    
    d_size = 8
    x_size = 4
    rank = 2
    
    config = create_lrtt_config(rank=rank)
    model = nn.Sequential(
        AnalogLinear(x_size, d_size, bias=False, rpu_config=config)
    ).cuda()
    
    layer = model[0]
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    
    # Check initial state
    params = tile.get_hidden_parameters()
    A = params['hidden_weights_0'].cpu().numpy()
    B = params['hidden_weights_1'].cpu().numpy()
    C = params['hidden_weights_2'].cpu().numpy()
    
    print("\nInitial weights after creation:")
    print(f"  A[:, :rank] norm: {np.linalg.norm(A[:, :rank]):.6f}")
    print(f"  B[:rank, :] norm: {np.linalg.norm(B[:rank, :]):.6f}")
    print(f"  C norm: {np.linalg.norm(C):.6f}")
    
    # Try manual initialization
    print("\nManually setting weights...")
    new_weights = torch.randn(d_size, x_size).cuda() * 0.5
    tile.set_weights(new_weights)
    
    params = tile.get_hidden_parameters()
    A = params['hidden_weights_0'].cpu().numpy()
    B = params['hidden_weights_1'].cpu().numpy()
    C = params['hidden_weights_2'].cpu().numpy()
    
    print("After manual weight setting:")
    print(f"  A[:, :rank] norm: {np.linalg.norm(A[:, :rank]):.6f}")
    print(f"  B[:rank, :] norm: {np.linalg.norm(B[:rank, :]):.6f}")
    print(f"  C norm: {np.linalg.norm(C):.6f}")
    
    # Check if A gets initialized on first update
    print("\nPerforming first update...")
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    x = torch.randn(2, x_size, device='cuda')
    target = torch.randn(2, d_size, device='cuda')
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    
    params = tile.get_hidden_parameters()
    A = params['hidden_weights_0'].cpu().numpy()
    B = params['hidden_weights_1'].cpu().numpy()
    C = params['hidden_weights_2'].cpu().numpy()
    
    print("After first update:")
    print(f"  A[:, :rank] norm: {np.linalg.norm(A[:, :rank]):.6f}")
    print(f"  B[:rank, :] norm: {np.linalg.norm(B[:rank, :]):.6f}")
    print(f"  C norm: {np.linalg.norm(C):.6f}")
    
    if np.linalg.norm(A[:, :rank]) < 1e-6:
        print("  ⚠️ WARNING: A weights still zero after update!")
    else:
        print("  ✓ A weights initialized")

if __name__ == "__main__":
    test_weight_bounds_and_updates()
    test_initialization_and_reinit()
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 80)