#!/usr/bin/env python
"""
Test to verify that lrtt_get_visible_weights() returns the correct visible device weights.
Compare different ways of accessing the visible weights to see if there's a discrepancy.
"""

import torch
import torch.nn as nn
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.presets.devices import ConstantStepDevice
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import PulseType


def test_visible_weights_access():
    """Test different methods of accessing visible weights."""
    print("=== Testing Visible Weights Access Methods ===\n")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create LRTT config
    fastA = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    fastB = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    visible = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=2,
        transfer_every=5,
        transfer_lr=1.0,
        forward_inject=True
    )
    
    config = UnitCellRPUConfig(
        device=lrtt,
        update=UpdateParameters(pulse_type=PulseType.STOCHASTIC_COMPRESSED),
        forward=IOParameters(is_perfect=False)
    )
    
    # Create layer
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    
    # Set specific weights to track
    test_weights = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0], 
        [9.0, 10.0, 11.0, 12.0]
    ], dtype=torch.float32)
    
    layer.set_weights(test_weights)
    layer = layer.to(device)
    
    # Get CUDA tile
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    cpp_tile = tile.tile
    
    print("=== Initial Weight Comparison ===")
    
    # Method 1: layer.get_weights() (composed/effective weights)
    effective_weights = layer.get_weights()[0]
    print(f"1. layer.get_weights() shape: {effective_weights.shape}")
    print(f"1. layer.get_weights() norm: {effective_weights.norm().item():.6f}")
    print(f"1. layer.get_weights() sample:\n{effective_weights[:2, :2]}")
    
    # Method 2: cpp_tile.lrtt_get_visible_weights() (should be visible device only)
    try:
        visible_weights = cpp_tile.lrtt_get_visible_weights()
        print(f"\n2. lrtt_get_visible_weights() shape: {visible_weights.shape}")
        print(f"2. lrtt_get_visible_weights() norm: {visible_weights.norm().item():.6f}")
        print(f"2. lrtt_get_visible_weights() sample:\n{visible_weights[:2, :2]}")
    except Exception as e:
        print(f"\n2. lrtt_get_visible_weights() failed: {e}")
        return
    
    # Method 3: Check A and B matrices
    try:
        A_lr = cpp_tile.lrtt_get_A_lr()
        B_lr = cpp_tile.lrtt_get_B_lr()
        print(f"\n3. A_lr norm: {A_lr.norm().item():.6f}")
        print(f"3. B_lr norm: {B_lr.norm().item():.6f}")
        
        # Manual composition: should match layer.get_weights()
        if A_lr.norm().item() > 1e-6 and B_lr.norm().item() > 1e-6:
            manual_composition = visible_weights + lrtt.lora_alpha * torch.mm(A_lr, B_lr)
            print(f"3. Manual composition norm: {manual_composition.norm().item():.6f}")
            print(f"3. Manual composition sample:\n{manual_composition[:2, :2]}")
        else:
            manual_composition = visible_weights
            print("3. Manual composition = visible_weights (A or B is zero)")
    except Exception as e:
        print(f"\n3. A/B matrices failed: {e}")
        return
    
    # Method 4: Try to access individual device weights directly (if possible)
    try:
        # Check if we can access device weights directly
        if hasattr(cpp_tile, 'get_device_weights'):
            device_weights = cpp_tile.get_device_weights()
            print(f"\n4. get_device_weights() returned {len(device_weights)} devices")
            for i, dw in enumerate(device_weights):
                print(f"4. Device {i} norm: {dw.norm().item():.6f}")
                if i == lrtt.idx_visible:  # Should be index 2
                    print(f"4. Device {i} (visible) sample:\n{dw[:2, :2]}")
        else:
            print("\n4. get_device_weights() not available")
    except Exception as e:
        print(f"\n4. get_device_weights() failed: {e}")
    
    # Compare the methods (ensure same device)
    print(f"\n=== Comparison ===")
    manual_composition_cpu = manual_composition.cpu()
    visible_weights_cpu = visible_weights.cpu()
    effective_weights_cpu = effective_weights.cpu()
    
    effective_vs_manual = (effective_weights_cpu - manual_composition_cpu).norm().item()
    effective_vs_visible = (effective_weights_cpu - visible_weights_cpu).norm().item()
    
    print(f"Difference (effective vs manual composition): {effective_vs_manual:.6f}")
    print(f"Difference (effective vs visible only): {effective_vs_visible:.6f}")
    
    if effective_vs_manual < 1e-6:
        print("✓ Manual composition matches layer.get_weights() - forward injection working")
    else:
        print("✗ Manual composition doesn't match layer.get_weights() - forward injection broken")
    
    if effective_vs_visible < 1e-6:
        print("✓ Effective weights == visible weights (A@B contribution is zero)")
    else:
        print("✗ Effective weights != visible weights (A@B contribution exists)")
    
    print(f"\n=== Testing Weight Modification ===")
    
    # Now modify the visible weights directly and see if it reflects
    try:
        modified_visible = visible_weights + 0.5
        cpp_tile.lrtt_set_visible_weights(modified_visible)
        print("Set visible weights to original + 0.5")
        
        # Check all methods again
        new_effective = layer.get_weights()[0]
        new_visible = cpp_tile.lrtt_get_visible_weights()
        
        print(f"After modification:")
        print(f"  layer.get_weights() norm: {new_effective.norm().item():.6f}")
        print(f"  lrtt_get_visible_weights() norm: {new_visible.norm().item():.6f}")
        
        modification_reflected = (new_effective - new_visible).norm().item() < 1e-6
        if modification_reflected:
            print("✓ Visible weight modifications reflect in layer.get_weights()")
        else:
            print("✗ Visible weight modifications don't reflect in layer.get_weights()")
            print(f"  Difference: {(new_effective - new_visible).norm().item():.6f}")
        
        # Check if the modification actually happened
        expected_modified = test_weights + 0.5
        actual_change = (new_visible - expected_modified.to(device)).norm().item()
        print(f"  Expected vs actual visible change: {actual_change:.6f}")
        
    except Exception as e:
        print(f"Weight modification test failed: {e}")


if __name__ == "__main__":
    test_visible_weights_access()