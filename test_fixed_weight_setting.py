#!/usr/bin/env python
"""
Test the correct order of operations for LRTT weight setting.
"""

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.presets.devices import ConstantStepDevice
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import PulseType


def test_correct_weight_setting():
    """Test correct weight setting procedure for LRTT."""
    print("=== Testing Correct Weight Setting for LRTT ===\n")
    
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
    
    test_weights = torch.randn(3, 4) * 0.1
    
    print("=== Method 1: Set weights BEFORE moving to CUDA (WRONG) ===")
    layer1 = AnalogLinear(4, 3, rpu_config=config, bias=False)
    layer1.set_weights(test_weights)  # Set before CUDA
    layer1 = layer1.to(device)  # Then move to CUDA
    
    tile1 = list(layer1.analog_tiles())[0]
    cpp_tile1 = tile1.tile
    visible1 = cpp_tile1.lrtt_get_visible_weights()
    print(f"Visible weights norm: {visible1.norm().item():.6f}")
    print(f"Result: {'✗ FAILED' if visible1.norm().item() < 1e-6 else '✓ SUCCESS'}")
    
    print("\n=== Method 2: Move to CUDA first, then set weights (CORRECT) ===")
    layer2 = AnalogLinear(4, 3, rpu_config=config, bias=False)
    layer2 = layer2.to(device)  # Move to CUDA first
    layer2.set_weights(test_weights)  # Then set weights
    
    tile2 = list(layer2.analog_tiles())[0]
    cpp_tile2 = tile2.tile
    visible2 = cpp_tile2.lrtt_get_visible_weights()
    print(f"Visible weights norm: {visible2.norm().item():.6f}")
    print(f"Result: {'✓ SUCCESS' if visible2.norm().item() > 1e-6 else '✗ FAILED'}")
    
    print("\n=== Method 3: Use tile.set_weights() after CUDA (CORRECT) ===")
    layer3 = AnalogLinear(4, 3, rpu_config=config, bias=False)
    layer3 = layer3.to(device)  # Move to CUDA first
    tile3 = list(layer3.analog_tiles())[0]
    tile3.set_weights(test_weights)  # Set directly on tile
    
    cpp_tile3 = tile3.tile
    visible3 = cpp_tile3.lrtt_get_visible_weights()
    print(f"Visible weights norm: {visible3.norm().item():.6f}")
    print(f"Result: {'✓ SUCCESS' if visible3.norm().item() > 1e-6 else '✗ FAILED'}")
    
    print("\n=== Test Transfer with Properly Set Weights ===")
    
    # Use method 3 which works
    layer = layer3
    tile = tile3
    cpp_tile = cpp_tile3
    
    # Create inputs
    x = torch.randn(2, 4, device=device)
    grad = torch.randn(2, 3, device=device) * 1.0
    
    # Run updates to accumulate A@B
    print("Running 6 updates (should trigger transfer at step 5)...")
    for step in range(6):
        tile.update(x, grad)
        
        if step == 4 or step == 5:  # Around transfer
            A = cpp_tile.lrtt_get_A_lr()
            B = cpp_tile.lrtt_get_B_lr()
            V = cpp_tile.lrtt_get_visible_weights()
            print(f"  Step {step+1}: A={A.norm().item():.6f}, B={B.norm().item():.6f}, V={V.norm().item():.6f}")
    
    final_visible = cpp_tile.lrtt_get_visible_weights()
    print(f"\nFinal visible weights norm: {final_visible.norm().item():.6f}")
    
    if abs(final_visible.norm().item() - visible3.norm().item()) > 0.001:
        print("✓ Visible weights changed after transfer (transfer working!)")
    else:
        print("✗ Visible weights unchanged (transfer still broken)")


if __name__ == "__main__":
    test_correct_weight_setting()