#!/usr/bin/env python
"""
Test different transfer_lr values to check if A@B → visible transfer works.
Starting with B=0 (natural initialization) and letting A and B accumulate.
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


def test_transfer_lr(transfer_lr_val, transfer_every_val=5):
    """Test transfer with specific transfer_lr value."""
    print(f"\n=== Testing transfer_lr={transfer_lr_val} ===")
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create LRTT config with small dw_min and specified transfer_lr
    fastA = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    fastB = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    visible = ConstantStepDevice(dw_min=0.00001, dw_min_dtod=0.0, dw_min_std=0.0)
    
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=2,
        transfer_every=transfer_every_val,
        transfer_lr=transfer_lr_val,  # This is what we're testing
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
    
    # Check natural initialization (should be zeros for A and B)
    A_init = cpp_tile.lrtt_get_A_lr()
    B_init = cpp_tile.lrtt_get_B_lr()
    visible_init = cpp_tile.lrtt_get_visible_weights()
    
    print(f"Initial state: A={A_init.norm().item():.6f}, B={B_init.norm().item():.6f}, Visible={visible_init.norm().item():.6f}")
    
    # Create inputs - larger gradients to accumulate meaningful A@B  
    x = torch.randn(2, 4, device=device)
    grad = torch.randn(2, 3, device=device) * 1.0  # Larger gradient
    
    # Track transfer effects
    transfers = 0
    visible_changes = []
    ab_products = []
    
    # Run enough steps to get at least one transfer
    num_steps = min(transfer_every_val + 5, 110)  # Cap at 110 steps for performance
    
    for step in range(num_steps):
        # Get state before update
        A_before = cpp_tile.lrtt_get_A_lr()
        B_before = cpp_tile.lrtt_get_B_lr()
        visible_before = cpp_tile.lrtt_get_visible_weights()
        
        a_norm_before = A_before.norm().item()
        b_norm_before = B_before.norm().item()
        visible_norm_before = visible_before.norm().item()
        
        # Calculate A@B before update if both are non-zero
        if a_norm_before > 1e-6 and b_norm_before > 1e-6:
            ab_before = torch.mm(A_before, B_before)
            ab_norm_before = ab_before.norm().item()
        else:
            ab_norm_before = 0.0
        
        # Perform update
        tile.update(x, grad)
        
        # Get state after update
        A_after = cpp_tile.lrtt_get_A_lr()
        B_after = cpp_tile.lrtt_get_B_lr()
        visible_after = cpp_tile.lrtt_get_visible_weights()
        
        a_norm_after = A_after.norm().item()
        b_norm_after = B_after.norm().item()
        visible_norm_after = visible_after.norm().item()
        
        # Check for transfer (A reinitialized)
        is_transfer = (a_norm_after > 1.0 and step > 0)
        if is_transfer:
            transfers += 1
            visible_change = visible_norm_after - visible_norm_before
            ab_products.append(ab_norm_before)
            visible_changes.append(visible_change)
            
            print(f"  Step {step+1}: TRANSFER!")
            print(f"    A@B before transfer: {ab_norm_before:.6f}")
            print(f"    Visible before: {visible_norm_before:.6f} → after: {visible_norm_after:.6f}")
            print(f"    Expected change: {transfer_lr_val * ab_norm_before:.6f}, Actual change: {visible_change:+.6f}")
            print(f"    A: {a_norm_before:.6f} → {a_norm_after:.6f} (reinitialized)")
            print(f"    B: {b_norm_before:.6f} → {b_norm_after:.6f} (zeroed)")
        elif step < 3 or step % 20 == 0 or step == num_steps - 1:
            print(f"  Step {step+1}: A={a_norm_after:.6f}, B={b_norm_after:.6f}, V={visible_norm_after:.6f}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Transfers: {transfers}")
    print(f"  Visible changes during transfers: {visible_changes}")
    print(f"  A@B norms before transfers: {ab_products}")
    
    if transfers > 0 and len(visible_changes) > 0:
        total_expected = sum(transfer_lr_val * ab for ab in ab_products)
        total_actual = sum(visible_changes) 
        print(f"  Total expected visible change: {total_expected:.6f}")
        print(f"  Total actual visible change: {total_actual:+.6f}")
        success = abs(total_actual) > 0.001
        print(f"  Transfer working: {'✓' if success else '✗'}")
    else:
        print(f"  No transfers occurred (need {transfer_every_val} steps)")
        success = False
    
    return success, transfers, visible_changes


def main():
    """Test different transfer_lr values with long accumulation period."""
    print("=== Testing A@B → Visible Transfer with Long Accumulation (transfer_every=100) ===")
    
    transfer_lrs = [1.0, 10.0, 100.0]  # Skip 0.1, focus on bigger values
    results = {}
    
    for lr in transfer_lrs:
        success, transfers, changes = test_transfer_lr(lr, 100)  # Much longer accumulation
        results[lr] = {
            'success': success,
            'transfers': transfers, 
            'visible_changes': changes,
            'max_change': max([abs(c) for c in changes]) if changes else 0.0
        }
    
    print(f"\n{'='*70}")
    print("TRANSFER LEARNING RATE ANALYSIS")  
    print(f"{'='*70}")
    print(f"{'transfer_lr':<12} | {'Working':<8} | {'Transfers':<9} | {'Max Change':<12}")
    print("-" * 70)
    
    for lr, data in results.items():
        status = "✓" if data['success'] else "✗"
        max_change = f"{data['max_change']:.6f}"
        print(f"{lr:<12} | {status:<8} | {data['transfers']:<9} | {max_change:<12}")
    
    # Final verdict
    working_lrs = [lr for lr, data in results.items() if data['success']]
    if working_lrs:
        print(f"\n✓ Transfer mechanism works with lr: {working_lrs}")
    else:
        print(f"\n✗ Transfer mechanism appears broken - no visible weight changes detected")


if __name__ == "__main__":
    main()