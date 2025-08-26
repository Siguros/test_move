#!/usr/bin/env python3
"""Test LRTT set_weights and get_weights behavior with over-boundary values."""

import torch
import torch.nn as nn
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs import UnitCellRPUConfig
import os

# Enable debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

def test_lrtt_bounds_overflow():
    """Test LRTT behavior when setting weights beyond device bounds."""
    print("=" * 80)
    print("TESTING LRTT SET_WEIGHTS WITH OVER-BOUNDARY VALUES")
    print("=" * 80)
    
    # Configuration
    d_size = 4
    x_size = 4
    rank = 2
    
    # Create devices with specific bounds [-1, 1]
    w_min, w_max = -1.0, 1.0
    
    fastA = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
    fastB = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
    visible = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.0001)
    
    device = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=rank,
        transfer_lr=1.0,
        transfer_every=1,
        forward_inject=True,
        lora_alpha=1.0
    )
    
    config = UnitCellRPUConfig(device=device)
    
    # Create model
    model = nn.Sequential(
        AnalogLinear(x_size, d_size, bias=False, rpu_config=config)
    ).cuda()
    
    layer = model[0]
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    
    print(f"\nDevice bounds configuration:")
    print(f"  w_min: {w_min}")
    print(f"  w_max: {w_max}")
    
    # Test 1: Set weights with values exceeding bounds
    print("\n" + "="*60)
    print("TEST 1: Setting weights with over-boundary values")
    print("="*60)
    
    # Create weight matrix with values outside [-1, 1]
    test_weights = torch.tensor([
        [3.0, -3.0, 2.5, -2.5],   # Row with extreme values
        [1.5, -1.5, 0.5, -0.5],    # Row with moderate overflow
        [0.8, -0.8, 0.3, -0.3],    # Row within bounds
        [2.0, -2.0, 1.2, -1.2]     # Row with mixed overflow
    ], device='cuda', dtype=torch.float32)
    
    print(f"\nInput weights to set_weights:")
    print(f"  Shape: {test_weights.shape}")
    print(f"  Min value: {test_weights.min().item():.3f}")
    print(f"  Max value: {test_weights.max().item():.3f}")
    print(f"  Values exceeding bounds: {((test_weights < w_min) | (test_weights > w_max)).sum().item()} / {test_weights.numel()}")
    print("\nWeight matrix:")
    print(test_weights.cpu().numpy())
    
    # Set the weights
    tile.set_weights(test_weights)
    
    # Get the weights back
    retrieved_weights = tile.get_weights()[0]
    
    print(f"\nRetrieved weights after set_weights:")
    print(f"  Shape: {retrieved_weights.shape}")
    print(f"  Min value: {retrieved_weights.min().item():.3f}")
    print(f"  Max value: {retrieved_weights.max().item():.3f}")
    print("\nWeight matrix:")
    print(retrieved_weights.cpu().numpy())
    
    # Check if weights were clipped
    diff = (retrieved_weights.cpu() - test_weights.cpu()).abs()
    clipped_elements = (diff > 1e-6).sum().item()
    
    print(f"\nBounds enforcement analysis:")
    print(f"  Elements that were clipped: {clipped_elements} / {test_weights.numel()}")
    
    # Check each element
    for i in range(d_size):
        for j in range(x_size):
            original = test_weights[i, j].item()
            retrieved = retrieved_weights[i, j].item()
            expected = np.clip(original, w_min, w_max)
            
            if abs(retrieved - expected) > 1e-6:
                print(f"  ⚠️ [{i},{j}]: Original={original:.3f}, Expected={expected:.3f}, Got={retrieved:.3f}")
            elif abs(original - retrieved) > 1e-6:
                print(f"  ✓ [{i},{j}]: Clipped {original:.3f} -> {retrieved:.3f}")
    
    # Test 2: Check hidden weights (A, B, C)
    print("\n" + "="*60)
    print("TEST 2: Checking hidden device weights (A, B, C)")
    print("="*60)
    
    params = tile.get_hidden_parameters()
    A = params['hidden_weights_0'].cpu().numpy()
    B = params['hidden_weights_1'].cpu().numpy()
    C = params['hidden_weights_2'].cpu().numpy()
    
    print(f"\nDevice A (fastA) - shape {A.shape}:")
    print(f"  Min: {A.min():.3f}, Max: {A.max():.3f}")
    A_lr = A[:, :rank]
    print(f"  Low-rank part [:, :rank] - Min: {A_lr.min():.3f}, Max: {A_lr.max():.3f}")
    if (A_lr < w_min).any() or (A_lr > w_max).any():
        print(f"  ⚠️ A has values outside [{w_min}, {w_max}]")
    else:
        print(f"  ✓ A within bounds [{w_min}, {w_max}]")
    
    print(f"\nDevice B (fastB) - shape {B.shape}:")
    print(f"  Min: {B.min():.3f}, Max: {B.max():.3f}")
    B_lr = B[:rank, :]
    print(f"  Low-rank part [:rank, :] - Min: {B_lr.min():.3f}, Max: {B_lr.max():.3f}")
    if (B_lr < w_min).any() or (B_lr > w_max).any():
        print(f"  ⚠️ B has values outside [{w_min}, {w_max}]")
    else:
        print(f"  ✓ B within bounds [{w_min}, {w_max}]")
    
    print(f"\nDevice C (visible) - shape {C.shape}:")
    print(f"  Min: {C.min():.3f}, Max: {C.max():.3f}")
    if (C < w_min).any() or (C > w_max).any():
        print(f"  ⚠️ C has values outside [{w_min}, {w_max}]")
    else:
        print(f"  ✓ C within bounds [{w_min}, {w_max}]")
    
    # Test 3: Set extreme values and check bounds after update
    print("\n" + "="*60)
    print("TEST 3: Extreme values followed by update")
    print("="*60)
    
    # Set extreme weights
    extreme_weights = torch.ones(d_size, x_size, device='cuda') * 5.0
    extreme_weights[::2, ::2] = -5.0  # Checkerboard pattern
    
    print(f"\nSetting extreme weights (±5.0)...")
    tile.set_weights(extreme_weights)
    
    # Perform an update
    from aihwkit.optim import AnalogSGD
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    
    x = torch.randn(2, x_size, device='cuda') * 0.1
    target = torch.randn(2, d_size, device='cuda') * 0.1
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    
    # Check weights after update
    final_weights = tile.get_weights()[0]
    params_final = tile.get_hidden_parameters()
    A_final = params_final['hidden_weights_0'].cpu().numpy()
    B_final = params_final['hidden_weights_1'].cpu().numpy()
    C_final = params_final['hidden_weights_2'].cpu().numpy()
    
    print(f"\nAfter update:")
    print(f"  Visible weights - Min: {final_weights.min().item():.3f}, Max: {final_weights.max().item():.3f}")
    print(f"  A weights - Min: {A_final.min():.3f}, Max: {A_final.max():.3f}")
    print(f"  B weights - Min: {B_final.min():.3f}, Max: {B_final.max():.3f}")
    print(f"  C weights - Min: {C_final.min():.3f}, Max: {C_final.max():.3f}")
    
    # Final check
    all_within_bounds = True
    if final_weights.min().item() < w_min or final_weights.max().item() > w_max:
        print(f"  ⚠️ Visible weights exceed bounds!")
        all_within_bounds = False
    if A_final[:, :rank].min() < w_min or A_final[:, :rank].max() > w_max:
        print(f"  ⚠️ A low-rank weights exceed bounds!")
        all_within_bounds = False
    if B_final[:rank, :].min() < w_min or B_final[:rank, :].max() > w_max:
        print(f"  ⚠️ B low-rank weights exceed bounds!")
        all_within_bounds = False
    if C_final.min() < w_min or C_final.max() > w_max:
        print(f"  ⚠️ C weights exceed bounds!")
        all_within_bounds = False
    
    if all_within_bounds:
        print(f"  ✓ All weights within bounds [{w_min}, {w_max}]")
    
    return all_within_bounds

def test_different_bound_configurations():
    """Test with different bound configurations."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT BOUND CONFIGURATIONS")
    print("=" * 80)
    
    test_configs = [
        {"w_min": -1.0, "w_max": 1.0, "label": "Standard [-1, 1]"},
        {"w_min": -0.5, "w_max": 0.5, "label": "Tight [-0.5, 0.5]"},
        {"w_min": -2.0, "w_max": 2.0, "label": "Wide [-2, 2]"},
        {"w_min": 0.0, "w_max": 1.0, "label": "Positive only [0, 1]"},
        {"w_min": -1.0, "w_max": 0.0, "label": "Negative only [-1, 0]"},
    ]
    
    for cfg in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {cfg['label']}")
        print(f"{'='*50}")
        
        w_min, w_max = cfg['w_min'], cfg['w_max']
        
        # Create simple LRTT config
        fastA = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
        fastB = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.001)
        visible = ConstantStepDevice(w_min=w_min, w_max=w_max, dw_min=0.0001)
        
        device = LRTTTransferCompound(
            unit_cell_devices=[fastA, fastB, visible],
            rank=2
        )
        
        config = UnitCellRPUConfig(device=device)
        model = AnalogLinear(4, 4, bias=False, rpu_config=config).cuda()
        
        tiles = list(model.analog_tiles())
        tile = tiles[0]
        
        # Test with values outside the bounds
        test_values = torch.tensor([
            [3.0, -3.0, 0.0, cfg['w_max'] * 2],
            [cfg['w_min'] * 2, 0.5, -0.5, 1.5],
            [0.0, 0.0, cfg['w_min'] / 2, cfg['w_max'] / 2],
            [-2.5, 2.5, cfg['w_min'], cfg['w_max']]
        ], device='cuda', dtype=torch.float32)
        
        # Set and retrieve
        tile.set_weights(test_values)
        retrieved = tile.get_weights()[0]
        
        # Check bounds
        violations = (retrieved < w_min) | (retrieved > w_max)
        
        print(f"  Input range: [{test_values.min().item():.2f}, {test_values.max().item():.2f}]")
        print(f"  Output range: [{retrieved.min().item():.2f}, {retrieved.max().item():.2f}]")
        print(f"  Bounds: [{w_min}, {w_max}]")
        
        if violations.any():
            print(f"  ⚠️ {violations.sum().item()} values exceed bounds!")
        else:
            print(f"  ✓ All values within bounds")

if __name__ == "__main__":
    # Run main test
    success = test_lrtt_bounds_overflow()
    
    # Run configuration tests
    test_different_bound_configurations()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED: Bounds are properly enforced")
    else:
        print("⚠️ TEST FAILED: Bounds violations detected")
    print("=" * 80)