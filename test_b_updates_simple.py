#!/usr/bin/env python
"""
Simple test to verify B matrix updates work with different transfer_every values.
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


def test_b_updates_simple(transfer_every_val, num_steps=12):
    """Simple test focusing only on B matrix updates."""
    print(f"\n=== Testing B Updates with transfer_every={transfer_every_val} ===")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create LRTT config with small dw_min
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
    
    config = UnitCellRPUConfig(
        device=lrtt,
        update=UpdateParameters(pulse_type=PulseType.STOCHASTIC_COMPRESSED),
        forward=IOParameters(is_perfect=False)
    )
    
    # Create layer
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    layer.set_weights(torch.randn(3, 4) * 0.1)
    layer = layer.to(device)
    
    # Get CUDA tile
    tiles = list(layer.analog_tiles())
    tile = tiles[0]
    cpp_tile = tile.tile
    
    # Set initial B matrix to non-zero
    B_initial = torch.randn(2, 4, device=device) * 0.1
    cpp_tile.lrtt_set_B_lr(B_initial)
    print(f"Set B_lr initial norm: {B_initial.norm().item():.6f}")
    
    # Create inputs
    x = torch.randn(2, 4, device=device)
    grad = torch.randn(2, 3, device=device) * 0.5
    
    # Track updates
    b_updates = 0
    transfers = 0
    b_norms = []
    
    for step in range(num_steps):
        # Get B before update
        B_before = cpp_tile.lrtt_get_B_lr()
        b_norm_before = B_before.norm().item()
        
        # Perform update
        tile.update(x, grad)
        
        # Get B after update  
        B_after = cpp_tile.lrtt_get_B_lr()
        b_norm_after = B_after.norm().item()
        A_after = cpp_tile.lrtt_get_A_lr()
        a_norm_after = A_after.norm().item()
        
        # Track metrics
        b_norms.append(b_norm_after)
        if abs(b_norm_after - b_norm_before) > 1e-6:
            b_updates += 1
        if a_norm_after > 1.0 and step > 0:  # Transfer detected
            transfers += 1
            print(f"  Step {step+1}: TRANSFER - A={a_norm_after:.6f}, B={b_norm_after:.6f}")
        elif step < 3 or step % 4 == 0:
            print(f"  Step {step+1}: B={b_norm_before:.6f}→{b_norm_after:.6f}")
    
    print(f"Results: {b_updates}/{num_steps} B updates, {transfers} transfers")
    return b_updates, transfers, num_steps


def main():
    """Test B updates across different transfer_every values."""
    print("=== Testing B Matrix Updates Across Transfer Frequencies ===")
    
    results = {}
    for transfer_every in [2, 10, 100]:
        b_updates, transfers, total_steps = test_b_updates_simple(transfer_every, 12)
        results[transfer_every] = {
            'b_updates': b_updates,
            'transfers': transfers, 
            'total_steps': total_steps,
            'update_rate': b_updates / total_steps if total_steps > 0 else 0
        }
    
    print(f"\n{'='*60}")
    print("SUMMARY: B Matrix Update Performance")  
    print(f"{'='*60}")
    print(f"{'transfer_every':<12} | {'B updates':<9} | {'Transfers':<9} | {'Update rate':<11}")
    print("-" * 60)
    
    for te, data in results.items():
        rate = f"{data['update_rate']:.1%}"
        print(f"{te:<12} | {data['b_updates']:<9} | {data['transfers']:<9} | {rate:<11}")


if __name__ == "__main__":
    main()