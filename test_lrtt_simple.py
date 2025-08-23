#!/usr/bin/env python
"""Simple test for LR-TT operations without complex imports."""

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

def main():
    """Run all tests."""
    print("=" * 60)
    print("LR-TT Simple Operation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Basic operations", test_lrtt_basic()))
    results.append(("Transfer operation", test_lrtt_transfer()))
    results.append(("Rank chunking", test_lrtt_rank_chunking()))
    
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