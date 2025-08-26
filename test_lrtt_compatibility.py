#!/usr/bin/env python
"""Test LRTTTransferCompound compatibility with all analog layer types."""

import os
os.environ["AIHWKIT_DEBUG_LRTT"] = "0"  # Disable debug for cleaner output

import sys
sys.path.insert(0, '/workspace/aihwkit/lib/python3.10/site-packages')

import torch
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import (
    AnalogLinear, 
    AnalogConv1d, 
    AnalogConv2d, 
    AnalogConv3d,
    AnalogLinearMapped,
    AnalogConv1dMapped,
    AnalogConv2dMapped,
    AnalogConv3dMapped
)
from aihwkit.optim import AnalogSGD

print("LRTTTransferCompound Compatibility Test")
print("=" * 50)

# Create LRTT config
def create_lrtt_config(config_class=SingleRPUConfig, rank=2):
    """Create an LRTT configuration."""
    rpu_config = config_class()
    
    # For SingleRPUConfig
    if hasattr(rpu_config, 'device'):
        rpu_config.device = LRTTTransferCompound(
            unit_cell_devices=[
                ConstantStepDevice(),
                ConstantStepDevice(),
                ConstantStepDevice(),
            ],
            transfer_lr=0.1,
            transfer_every=1,
            rank=rank,
            forward_inject=False,
        )
    
    return rpu_config

# Test function
def test_layer(layer_class, *args, config_class=SingleRPUConfig, **kwargs):
    """Test a single layer type with LRTT."""
    layer_name = layer_class.__name__
    print(f"\nTesting {layer_name}...")
    
    try:
        # Create config
        config = create_lrtt_config(config_class)
        
        # Create layer
        layer = layer_class(*args, rpu_config=config, **kwargs)
        layer = layer.cuda()
        
        # Get appropriate input shape
        if "Conv" in layer_name:
            if "1d" in layer_name:
                x = torch.randn(1, args[0], 10).cuda()  # batch, channels, length
            elif "2d" in layer_name:
                x = torch.randn(1, args[0], 8, 8).cuda()  # batch, channels, height, width
            elif "3d" in layer_name:
                x = torch.randn(1, args[0], 4, 4, 4).cuda()  # batch, channels, depth, height, width
        else:  # Linear layers
            x = torch.randn(1, args[0]).cuda()  # batch, features
        
        # Forward pass
        y = layer(x)
        
        # Backward pass
        loss = y.sum()
        opt = AnalogSGD(layer.parameters(), lr=0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print(f"  ✓ {layer_name} works with LRTT")
        print(f"    - Input shape: {list(x.shape)}")
        print(f"    - Output shape: {list(y.shape)}")
        return True
        
    except Exception as e:
        print(f"  ✗ {layer_name} failed: {e}")
        return False

# Run tests
print("\n" + "="*50)
print("Testing Standard Layers with SingleRPUConfig:")
print("="*50)

results = {}

# Test AnalogLinear
results['AnalogLinear'] = test_layer(AnalogLinear, 10, 5, bias=False)

# Test AnalogConv1d
results['AnalogConv1d'] = test_layer(AnalogConv1d, 3, 16, 3, bias=False)

# Test AnalogConv2d
results['AnalogConv2d'] = test_layer(AnalogConv2d, 3, 16, (3, 3), bias=False)

# Test AnalogConv3d
results['AnalogConv3d'] = test_layer(AnalogConv3d, 3, 8, (3, 3, 3), bias=False)

print("\n" + "="*50)
print("Testing Mapped Layers with SingleRPUConfig:")
print("="*50)

# For mapped layers, we need to set mapping parameters
def create_mapped_lrtt_config(rank=2):
    """Create an LRTT configuration for mapped layers."""
    from aihwkit.simulator.configs.utils import MappingParameter
    
    rpu_config = SingleRPUConfig()
    rpu_config.mapping = MappingParameter(max_input_size=512, max_output_size=512)
    rpu_config.device = LRTTTransferCompound(
        unit_cell_devices=[
            ConstantStepDevice(),
            ConstantStepDevice(), 
            ConstantStepDevice(),
        ],
        transfer_lr=0.1,
        transfer_every=1,
        rank=rank,
        forward_inject=False,
    )
    return rpu_config

# Test mapped layers
config = create_mapped_lrtt_config()

results['AnalogLinearMapped'] = test_layer(
    AnalogLinearMapped, 10, 5, 
    config_class=None, 
    bias=False
)

results['AnalogConv1dMapped'] = test_layer(
    AnalogConv1dMapped, 3, 16, 3,
    config_class=None,
    bias=False
)

results['AnalogConv2dMapped'] = test_layer(
    AnalogConv2dMapped, 3, 16, (3, 3),
    config_class=None,
    bias=False  
)

results['AnalogConv3dMapped'] = test_layer(
    AnalogConv3dMapped, 3, 8, (3, 3, 3),
    config_class=None,
    bias=False
)

# Summary
print("\n" + "="*50)
print("SUMMARY:")
print("="*50)

passed = sum(1 for v in results.values() if v)
total = len(results)

for layer, success in results.items():
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {layer:25s}: {status}")

print(f"\nTotal: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 All layers are compatible with LRTTTransferCompound!")
else:
    print(f"\n⚠️  {total - passed} layer(s) failed compatibility test")