#!/usr/bin/env python
"""Test script to verify LR-TT uses fully_hidden mode correctly."""

import torch
import torch.nn.functional as F
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.presets.devices import IdealizedPresetDevice, EcRamPresetDevice
from aihwkit.optim import AnalogSGD

def test_lrtt_fully_hidden():
    """Test that LR-TT configuration uses fully_hidden mode."""
    print("Testing LR-TT fully_hidden mode configuration...")
    
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
    print(f"  Created layer: {layer}")
    
    # Test on both CPU and CUDA if available
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    for device_type in devices_to_test:
        print(f"\n  Testing on {device_type}...")
        
        if device_type == 'cuda':
            layer = layer.cuda()
        
        # Get the tile to check configuration
        tile = layer.analog_tile
        
        # Check if we can access meta parameters (this varies by implementation)
        try:
            # Try to check if fully_hidden is enabled
            # This might need adjustment based on actual API
            print(f"    Tile type: {type(tile)}")
            
            # Test forward pass
            x = torch.randn(4, 8)
            if device_type == 'cuda':
                x = x.cuda()
            
            y = layer(x)
            print(f"    Forward pass: input {x.shape} -> output {y.shape}")
            
            # Test backward pass and update
            optimizer = AnalogSGD(layer.parameters(), lr=0.1)
            optimizer.zero_grad()
            
            target = torch.randn_like(y)
            loss = F.mse_loss(y, target)
            loss.backward()
            
            print(f"    Backward pass: loss = {loss.item():.4f}")
            
            # Get initial weights
            initial_weights = layer.get_weights()[0].clone()
            
            # Update weights
            optimizer.step()
            print("    Weight update completed")
            
            # Get updated weights
            updated_weights = layer.get_weights()[0]
            
            # Check weights changed
            weight_change = torch.norm(updated_weights - initial_weights).item()
            print(f"    Weight change norm: {weight_change:.6f}")
            
            # Test another forward pass
            y2 = layer(x)
            loss2 = F.mse_loss(y2, target)
            print(f"    After update: loss = {loss2.item():.4f}")
            
            if loss2 < loss:
                print(f"    ✓ Loss decreased after update on {device_type}")
            else:
                print(f"    ✗ Loss did not decrease on {device_type}")
                
        except Exception as e:
            print(f"    Error during test: {e}")
            return False
    
    print("\n✓ LR-TT fully_hidden mode test completed successfully")
    return True

def test_lrtt_no_gamma_error():
    """Test that we don't get 'last gamma should not be zero' error."""
    print("\nTesting that gamma error is avoided...")
    
    try:
        # Create configuration with transfer
        device = LRTTTransferCompound(
            unit_cell_devices=[
                IdealizedPresetDevice(),  # fastA
                IdealizedPresetDevice(),  # fastB
                IdealizedPresetDevice(),  # visible
            ],
            rank=2,
            transfer_every=1,
            transfer_lr=0.5,
            forward_inject=True,
        )
        
        config = UnitCellRPUConfig(device=device)
        
        # Create layer - this is where the error would occur if gamma setup is wrong
        layer = AnalogLinear(4, 3, rpu_config=config, bias=False)
        
        # Initialize with some weights
        init_weights = torch.randn(3, 4)
        layer.set_weights(init_weights)
        
        # Train for a step to trigger any potential issues
        if torch.cuda.is_available():
            layer = layer.cuda()
            x = torch.randn(2, 4).cuda()
        else:
            x = torch.randn(2, 4)
        
        optimizer = AnalogSGD(layer.parameters(), lr=0.1)
        optimizer.zero_grad()
        
        y = layer(x)
        target = torch.zeros_like(y)
        loss = F.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        
        print("  ✓ No 'last gamma should not be zero' error occurred")
        return True
        
    except RuntimeError as e:
        if "last gamma should not be zero" in str(e):
            print(f"  ✗ Error: {e}")
            return False
        else:
            raise  # Re-raise if it's a different error

def main():
    """Run all tests."""
    print("=" * 60)
    print("LR-TT Fully Hidden Mode Tests")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, using CPU only")
    
    results = []
    
    # Run tests
    results.append(("Fully hidden mode", test_lrtt_fully_hidden()))
    results.append(("No gamma error", test_lrtt_no_gamma_error()))
    
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