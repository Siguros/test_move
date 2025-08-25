#!/usr/bin/env python
"""Direct test of C++ binding for transfer_lr parameter."""

import os
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice, SingleRPUConfig
from aihwkit.simulator.rpu_base import cuda

DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")

# Create config with specific transfer_lr
device = ConstantStepDevice(dw_min=0.0001)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=3,
    transfer_lr=0.456,  # Specific value that should appear in debug output
    forward_inject=True,
    lora_alpha=1.0,
)

print(f"Python config: transfer_lr = {lrtt_config.transfer_lr}")

# Create a layer and get the tile
rpu_config = SingleRPUConfig(device=lrtt_config)
layer = AnalogLinear(4, 3, bias=False, rpu_config=rpu_config)
layer = layer.to(DEVICE)

# Get the actual tile
tiles = list(layer.analog_tiles())
cpp_tile = tiles[0].tile

print(f"\nTile type: {type(cpp_tile)}")
print(f"Tile attributes: {[a for a in dir(cpp_tile) if 'transfer' in a.lower() or 'param' in a.lower()]}")

# Try to get the parameters from the tile
if hasattr(cpp_tile, 'get_meta_parameters'):
    try:
        meta_params = cpp_tile.get_meta_parameters()
        print(f"\nMeta parameters type: {type(meta_params)}")
        if hasattr(meta_params, 'transfer_lr'):
            print(f"Meta parameters transfer_lr: {meta_params.transfer_lr}")
    except Exception as e:
        print(f"Error getting meta parameters: {e}")

# Try to get device parameters
if hasattr(cpp_tile, 'get_device_parameters'):
    try:
        device_params = cpp_tile.get_device_parameters()
        print(f"\nDevice parameters type: {type(device_params)}")
        if hasattr(device_params, 'transfer_lr'):
            print(f"Device parameters transfer_lr: {device_params.transfer_lr}")
    except Exception as e:
        print(f"Error getting device parameters: {e}")

# Check the actual binding structure
binding = rpu_config.as_bindings()
print(f"\nBinding type: {type(binding)}")
print(f"Binding attributes: {[a for a in dir(binding) if not a.startswith('_')]}")

if hasattr(binding, 'device_par'):
    dev_par = binding.device_par
    print(f"\ndevice_par type: {type(dev_par)}")
    print(f"device_par transfer_lr: {dev_par.transfer_lr if hasattr(dev_par, 'transfer_lr') else 'NOT FOUND'}")
    
    # Check all numeric attributes
    print("\nAll numeric attributes in device_par:")
    for attr in dir(dev_par):
        if not attr.startswith('_'):
            try:
                val = getattr(dev_par, attr)
                if isinstance(val, (int, float)):
                    print(f"  {attr} = {val}")
            except:
                pass