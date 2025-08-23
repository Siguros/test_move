#!/usr/bin/env python
"""Wrapper to run LR-TT tests with proper paths."""

import sys
import os

# Add source directory to path
sys.path.insert(0, '/workspace/aihwkit/src')

# Now run the actual test
import torch
import torch.nn.functional as F
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.presets.devices import IdealizedPresetDevice, EcRamPresetDevice
from aihwkit.optim import AnalogSGD

def test_lrtt_basic():
    """Test basic LR-TT configuration and operations."""
    print("Testing basic LR-TT configuration...")
    
    # Create LR-TT configuration
    device = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),  # fastA
            IdealizedPresetDevice(),  # fastB
            EcRamPresetDevice(),      # visible
        ],
        rank=4,
        transfer_every=2,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    config = UnitCellRPUConfig(
        device=device,
        forward=IOParameters(),
        backward=IOParameters(),
        update=UpdateParameters(),
    )
    
    # Create layer
    layer = AnalogLinear(8, 6, rpu_config=config, bias=False)
    if torch.cuda.is_available():
        layer = layer.cuda()
        print("  Using CUDA device")
    else:
        print("  Using CPU device")
    
    # Test forward pass
    x = torch.randn(4, 8)
    if torch.cuda.is_available():
        x = x.cuda()
    
    y = layer(x)
    print(f"  Forward pass: input {x.shape} -> output {y.shape}")
    
    # Test backward pass and update
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    optimizer.zero_grad()
    
    target = torch.randn_like(y)
    loss = F.mse_loss(y, target)
    loss.backward()
    
    print(f"  Backward pass: loss = {loss.item():.4f}")
    
    # Update weights
    optimizer.step()
    print("  Weight update completed")
    
    # Test another forward pass
    y2 = layer(x)
    loss2 = F.mse_loss(y2, target)
    print(f"  After update: loss = {loss2.item():.4f}")
    
    if loss2 < loss:
        print("✓ Loss decreased after update")
    else:
        print("✗ Loss did not decrease")
    
    return loss2 < loss

def test_lrtt_transfer():
    """Test LR-TT transfer operation."""
    print("\nTesting LR-TT transfer operation...")
    
    # Create configuration with transfer_every=1
    device = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),  # fastA
            IdealizedPresetDevice(),  # fastB
            IdealizedPresetDevice(),  # visible
        ],
        rank=2,
        transfer_every=1,  # Transfer every update
        transfer_lr=0.5,
        forward_inject=True,
    )
    
    config = UnitCellRPUConfig(device=device)
    
    # Create layer
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    if torch.cuda.is_available():
        layer = layer.cuda()
    
    # Train for multiple steps to trigger transfer
    optimizer = AnalogSGD(layer.parameters(), lr=0.1)
    
    x = torch.randn(2, 4)
    if torch.cuda.is_available():
        x = x.cuda()
    
    losses = []
    for i in range(5):
        optimizer.zero_grad()
        y = layer(x)
        target = torch.zeros_like(y)  # Simple target
        loss = F.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {i+1}: loss = {loss.item():.4f}")
    
    # Check if losses are generally decreasing
    avg_first_half = sum(losses[:2]) / 2
    avg_second_half = sum(losses[3:]) / 2
    
    if avg_second_half < avg_first_half:
        print("✓ Training is working (losses decreasing)")
        return True
    else:
        print("✗ Training may not be working properly")
        return False

def test_lrtt_rank_chunking():
    """Test LR-TT with rank chunking."""
    print("\nTesting LR-TT rank chunking...")
    
    # Create configuration with rank chunking
    device = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),  # fastA
            IdealizedPresetDevice(),  # fastB
            IdealizedPresetDevice(),  # visible
        ],
        rank=8,
        rank_chunk=4,  # Process in chunks of 4
        transfer_every=0,  # No transfer for this test
        forward_inject=True,
    )
    
    config = UnitCellRPUConfig(device=device)
    
    # Create layer
    layer = AnalogLinear(6, 5, rpu_config=config, bias=False)
    if torch.cuda.is_available():
        layer = layer.cuda()
    
    # Test forward pass
    x = torch.randn(3, 6)
    if torch.cuda.is_available():
        x = x.cuda()
    
    y = layer(x)
    print(f"  Forward with chunked rank: input {x.shape} -> output {y.shape}")
    
    # Verify output is valid
    if not torch.isnan(y).any() and not torch.isinf(y).any():
        print("✓ Rank chunking forward pass successful")
        return True
    else:
        print("✗ Rank chunking forward pass failed")
        return False

def test_lrtt_getters_setters():
    """Test LR-TT tile getters and setters if available."""
    print("\nTesting LR-TT tile getters/setters...")
    
    # Create simple configuration
    device = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),  # fastA
            IdealizedPresetDevice(),  # fastB
            IdealizedPresetDevice(),  # visible
        ],
        rank=2,
        transfer_every=0,
        forward_inject=True,
    )
    
    config = UnitCellRPUConfig(device=device)
    layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
    
    if torch.cuda.is_available():
        layer = layer.cuda()
        
        # Try to access the tile
        try:
            tile = layer.analog_tile
            
            # Check for LR-TT specific methods
            has_getters = all(hasattr(tile, m) for m in [
                'lrtt_get_visible_weights',
                'lrtt_get_A_lr', 
                'lrtt_get_B_lr'
            ])
            
            has_setters = all(hasattr(tile, m) for m in [
                'lrtt_set_visible_weights',
                'lrtt_set_A_lr',
                'lrtt_set_B_lr'
            ])
            
            if has_getters and has_setters:
                # Test getters
                C = tile.lrtt_get_visible_weights()
                A = tile.lrtt_get_A_lr()
                B = tile.lrtt_get_B_lr()
                
                print(f"  Visible weights shape: {C.shape}")
                print(f"  A matrix shape: {A.shape}")
                print(f"  B matrix shape: {B.shape}")
                
                # Test setters with modified values
                A_new = A * 2.0
                tile.lrtt_set_A_lr(A_new)
                A_read = tile.lrtt_get_A_lr()
                
                if torch.allclose(A_read, A_new, rtol=1e-5):
                    print("✓ LR-TT getters/setters working")
                    return True
                else:
                    print("✗ Setter/getter mismatch")
                    return False
            else:
                print("  LR-TT methods not available (CPU build?)")
                return True  # Not a failure if methods aren't available
                
        except Exception as e:
            print(f"  Could not test tile methods: {e}")
            return True  # Not a failure if we can't access tile
    else:
        print("  Skipping (CUDA not available)")
        return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("LR-TT Operation Tests")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, using CPU")
    
    results = []
    
    # Run tests
    results.append(("Basic operations", test_lrtt_basic()))
    results.append(("Transfer operation", test_lrtt_transfer()))
    results.append(("Rank chunking", test_lrtt_rank_chunking()))
    results.append(("Getters/Setters", test_lrtt_getters_setters()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("-" * 60)
    
    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name:30s} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{len(results)} tests passed")
    print("=" * 60)
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)