#!/usr/bin/env python
"""Comprehensive test for LRTT parameter passing and functionality."""

import os
os.environ["AIHWKIT_DEBUG_LRTT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, '/workspace/site-packages')

import torch
import numpy as np
from aihwkit.simulator.configs import InferenceRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def test_lrtt_basic_functionality():
    """Test basic LRTT functionality with different configurations."""
    
    print_section("LRTT Basic Functionality Test")
    
    # Create a simple configuration
    rpu_config = SingleRPUConfig()
    rpu_config.mapping.max_input_size = 256
    rpu_config.mapping.max_output_size = 256
    
    # Configure LRTT device with visible parameters
    rpu_config.device = LRTTTransferCompound(
        unit_cell_devices=[
            ConstantStepDevice(),  # fastA
            ConstantStepDevice(),  # fastB  
            ConstantStepDevice(),  # visible
        ],
        transfer_lr=0.1,
        transfer_every=2,
        rank=8,
        ab_use_bl_management=True,
        ab_desired_bl=2.0,
        transfer_use_bl_management=False,
        transfer_desired_bl=-1.0,
        correct_gradient_magnitudes=True,
        swap_xd=False,
        forward_inject=True,
        lora_alpha=1.0,
        reinit_gain=1.0,
    )
    
    print("\nConfiguration:")
    print(f"  transfer_lr: {rpu_config.device.transfer_lr}")
    print(f"  transfer_every: {rpu_config.device.transfer_every}")
    print(f"  rank: {rpu_config.device.rank}")
    print(f"  forward_inject: {rpu_config.device.forward_inject}")
    
    # Create analog layer
    print("\n[Creating AnalogLinear layer with LRTT device...]")
    layer = AnalogLinear(64, 32, rpu_config=rpu_config, bias=False)
    layer = layer.cuda()
    
    # Create optimizer
    opt = AnalogSGD(layer.parameters(), lr=0.01)
    
    # Test forward pass
    print("\n[Testing forward pass...]")
    x = torch.randn(16, 64).cuda()
    y = layer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
    # Test backward pass and updates
    print("\n[Testing backward pass and weight updates...]")
    for i in range(5):
        print(f"\n  Update {i+1}:")
        x = torch.randn(16, 64).cuda()
        target = torch.randn(16, 32).cuda()
        
        # Forward
        output = layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        print(f"    Loss: {loss.item():.6f}")
        
        # Backward
        opt.zero_grad()
        loss.backward()
        
        # Update
        opt.step()
        
        # Check if transfer should happen
        if (i + 1) % rpu_config.device.transfer_every == 0:
            print(f"    -> Transfer should occur at update {i+1}")
    
    print("\n[Basic functionality test completed]")
    return True

def test_lrtt_parameter_variations():
    """Test different LRTT parameter combinations."""
    
    print_section("LRTT Parameter Variations Test")
    
    test_configs = [
        {
            "name": "High transfer rate, frequent updates",
            "transfer_lr": 0.5,
            "transfer_every": 1,
            "rank": 4,
        },
        {
            "name": "Low transfer rate, sparse updates",
            "transfer_lr": 0.01,
            "transfer_every": 10,
            "rank": 16,
        },
        {
            "name": "With boundary management",
            "transfer_lr": 0.1,
            "transfer_every": 3,
            "rank": 8,
            "transfer_use_bl_management": True,
            "transfer_desired_bl": 3.0,
        },
        {
            "name": "With swap_xd enabled",
            "transfer_lr": 0.2,
            "transfer_every": 2,
            "rank": 8,
            "swap_xd": True,
        },
    ]
    
    for config in test_configs:
        print(f"\n[Testing: {config['name']}]")
        
        # Create configuration
        rpu_config = SingleRPUConfig()
        rpu_config.mapping.max_input_size = 128
        rpu_config.mapping.max_output_size = 128
        
        # Set device parameters
        device_params = {
            "unit_cell_devices": [
                ConstantStepDevice(),
                ConstantStepDevice(),
                ConstantStepDevice(),
            ],
            "transfer_lr": config.get("transfer_lr", 0.1),
            "transfer_every": config.get("transfer_every", 5),
            "rank": config.get("rank", 8),
            "ab_use_bl_management": config.get("ab_use_bl_management", False),
            "ab_desired_bl": config.get("ab_desired_bl", -1.0),
            "transfer_use_bl_management": config.get("transfer_use_bl_management", False),
            "transfer_desired_bl": config.get("transfer_desired_bl", -1.0),
            "swap_xd": config.get("swap_xd", False),
            "forward_inject": True,
            "lora_alpha": 1.0,
        }
        
        rpu_config.device = LRTTTransferCompound(**device_params)
        
        # Create and test layer
        layer = AnalogLinear(32, 16, rpu_config=rpu_config, bias=False).cuda()
        
        # Quick functionality test
        x = torch.randn(8, 32).cuda()
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        print(f"  ✓ Configuration tested successfully")
    
    print("\n[Parameter variations test completed]")
    return True

def test_lrtt_training_convergence():
    """Test if LRTT device can learn a simple pattern."""
    
    print_section("LRTT Training Convergence Test")
    
    # Create a simple dataset
    torch.manual_seed(42)
    X = torch.randn(100, 20).cuda()
    W_true = torch.randn(10, 20).cuda() * 0.1
    y_true = torch.mm(X, W_true.t())
    
    print(f"\nDataset:")
    print(f"  X shape: {X.shape}")
    print(f"  True W shape: {W_true.shape}")
    print(f"  y shape: {y_true.shape}")
    
    # Create LRTT model
    rpu_config = SingleRPUConfig()
    rpu_config.mapping.max_input_size = 256
    rpu_config.mapping.max_output_size = 256
    
    rpu_config.device = LRTTTransferCompound(
        unit_cell_devices=[
            ConstantStepDevice(),
            ConstantStepDevice(),
            ConstantStepDevice(),
        ],
        transfer_lr=0.05,
        transfer_every=5,
        rank=4,
        forward_inject=True,
        lora_alpha=1.0,
    )
    
    layer = AnalogLinear(20, 10, rpu_config=rpu_config, bias=False).cuda()
    opt = AnalogSGD(layer.parameters(), lr=0.1)
    
    # Training loop
    print("\nTraining:")
    losses = []
    for epoch in range(20):
        # Forward
        y_pred = layer(X)
        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        losses.append(loss.item())
        
        # Backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss.item():.6f}")
    
    # Check convergence
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")
    print(f"  Improvement:  {improvement:.1f}%")
    
    if final_loss < initial_loss * 0.5:
        print("  ✓ Model converged successfully!")
    else:
        print("  ⚠ Model did not converge well")
    
    print("\n[Training convergence test completed]")
    return final_loss < initial_loss

def main():
    """Run all LRTT tests."""
    print("=" * 80)
    print(" LRTT Comprehensive Test Suite")
    print("=" * 80)
    print("\nThis test will verify:")
    print("  1. Basic LRTT functionality")
    print("  2. Parameter variations")
    print("  3. Training convergence")
    
    # Run tests
    results = []
    
    try:
        results.append(("Basic Functionality", test_lrtt_basic_functionality()))
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        results.append(("Basic Functionality", False))
    
    try:
        results.append(("Parameter Variations", test_lrtt_parameter_variations()))
    except Exception as e:
        print(f"\n❌ Parameter variations test failed: {e}")
        results.append(("Parameter Variations", False))
    
    try:
        results.append(("Training Convergence", test_lrtt_training_convergence()))
    except Exception as e:
        print(f"\n❌ Training convergence test failed: {e}")
        results.append(("Training Convergence", False))
    
    # Summary
    print_section("Test Summary")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:25s}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)