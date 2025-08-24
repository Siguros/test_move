#!/usr/bin/env python
"""
Test AnalogLinear compatibility with LRTT configuration.
Check if AnalogLinear properly initializes and manages LRTT device weights.
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
from aihwkit.simulator.tiles import AnalogTile


def test_analog_linear_lrtt():
    """Test AnalogLinear with LRTT configuration."""
    print("=== Testing AnalogLinear with LRTT Configuration ===\n")
    
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
    
    print("=== 1. Create AnalogLinear Layer ===")
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    print(f"Layer created: in_features={layer.in_features}, out_features={layer.out_features}")
    
    # Check the tile type
    tiles = list(layer.analog_tiles())
    if tiles:
        tile = tiles[0]
        print(f"Tile type: {type(tile).__name__}")
        if hasattr(tile, 'tile'):
            cpp_tile = tile.tile
            print(f"C++ tile type: {type(cpp_tile).__name__}")
            # Check if it's actually an LRTT tile
            if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
                print("✓ LRTT-specific methods available")
            else:
                print("✗ LRTT-specific methods NOT available - wrong tile type!")
    
    print("\n=== 2. Test Weight Setting Methods ===")
    
    # Test weights
    test_weights = torch.randn(3, 4) * 0.1
    print(f"Test weights norm: {test_weights.norm().item():.6f}")
    
    # Method 1: Standard set_weights
    print("\nMethod 1: layer.set_weights()")
    layer.set_weights(test_weights)
    layer = layer.to(device)
    
    # Check what was actually set
    retrieved_weights = layer.get_weights()[0]
    print(f"  Retrieved weights norm: {retrieved_weights.norm().item():.6f}")
    
    # Check LRTT internal weights
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    cpp_tile = tile.tile
    
    visible_weights = cpp_tile.lrtt_get_visible_weights()
    A_weights = cpp_tile.lrtt_get_A_lr()
    B_weights = cpp_tile.lrtt_get_B_lr()
    
    print(f"  Visible weights norm: {visible_weights.norm().item():.6f}")
    print(f"  A weights norm: {A_weights.norm().item():.6f}")
    print(f"  B weights norm: {B_weights.norm().item():.6f}")
    
    if visible_weights.norm().item() < 1e-6:
        print("  ✗ Visible weights not set by layer.set_weights()")
    else:
        print("  ✓ Visible weights were set")
    
    print("\n=== 3. Try Direct Tile Weight Setting ===")
    
    # Try to set weights directly on the tile
    print("Attempting direct tile weight setting...")
    try:
        # Method 2: Direct tile set_weights
        tile.set_weights(test_weights)
        
        # Check results
        visible_after = cpp_tile.lrtt_get_visible_weights()
        print(f"  Visible weights after tile.set_weights(): {visible_after.norm().item():.6f}")
        
        if visible_after.norm().item() > 1e-6:
            print("  ✓ Direct tile.set_weights() worked")
        else:
            print("  ✗ Direct tile.set_weights() didn't set visible weights")
            
    except Exception as e:
        print(f"  ✗ Direct tile.set_weights() failed: {e}")
    
    print("\n=== 4. Try LRTT-specific Weight Setting ===")
    
    # Try LRTT-specific method
    print("Attempting LRTT-specific weight setting...")
    try:
        cpp_tile.lrtt_set_visible_weights(test_weights.to(device))
        
        # Check results
        visible_after_lrtt = cpp_tile.lrtt_get_visible_weights()
        retrieved_after_lrtt = layer.get_weights()[0]
        
        print(f"  Visible weights after lrtt_set_visible_weights(): {visible_after_lrtt.norm().item():.6f}")
        print(f"  layer.get_weights() after LRTT set: {retrieved_after_lrtt.norm().item():.6f}")
        
        if visible_after_lrtt.norm().item() > 1e-6:
            print("  ✓ lrtt_set_visible_weights() worked")
        else:
            print("  ✗ lrtt_set_visible_weights() failed")
            
    except Exception as e:
        print(f"  ✗ LRTT-specific weight setting failed: {e}")
    
    print("\n=== 5. Check Weight Synchronization ===")
    
    # Now check if layer.get_weights() properly combines visible + A@B
    visible_final = cpp_tile.lrtt_get_visible_weights()
    A_final = cpp_tile.lrtt_get_A_lr()
    B_final = cpp_tile.lrtt_get_B_lr()
    
    # Manual composition
    if A_final.norm().item() > 1e-6 and B_final.norm().item() > 1e-6:
        manual_weights = visible_final + lrtt.lora_alpha * torch.mm(A_final, B_final)
    else:
        manual_weights = visible_final
    
    layer_weights = layer.get_weights()[0]
    
    print(f"Manual composition norm: {manual_weights.cpu().norm().item():.6f}")
    print(f"layer.get_weights() norm: {layer_weights.norm().item():.6f}")
    
    diff = (manual_weights.cpu() - layer_weights).norm().item()
    print(f"Difference: {diff:.6f}")
    
    if diff < 1e-6:
        print("✓ layer.get_weights() correctly composes visible + A@B")
    else:
        print("✗ layer.get_weights() does NOT match manual composition")
        print("  This suggests AnalogLinear is not using LRTT forward injection properly")
    
    print("\n=== 6. Test Alternative: Direct AnalogTile Creation ===")
    
    # Try creating tile directly without AnalogLinear
    print("Creating AnalogTile directly...")
    try:
        direct_tile = AnalogTile(3, 4, config, bias=False)
        direct_tile.set_weights(test_weights)
        direct_tile = direct_tile.to(device)
        
        # Check LRTT weights
        if hasattr(direct_tile.tile, 'lrtt_get_visible_weights'):
            direct_visible = direct_tile.tile.lrtt_get_visible_weights()
            print(f"  Direct tile visible weights norm: {direct_visible.norm().item():.6f}")
            
            if direct_visible.norm().item() > 1e-6:
                print("  ✓ Direct AnalogTile creation sets visible weights correctly")
            else:
                print("  ✗ Even direct AnalogTile doesn't set visible weights")
        else:
            print("  ✗ Direct tile doesn't have LRTT methods")
            
    except Exception as e:
        print(f"  ✗ Direct AnalogTile creation failed: {e}")
    
    print("\n=== SUMMARY ===")
    print("The issue appears to be that AnalogLinear's set_weights() method")
    print("doesn't properly route weights to the LRTT visible device.")
    print("This breaks the entire LRTT mechanism since visible weights stay at zero.")


if __name__ == "__main__":
    test_analog_linear_lrtt()