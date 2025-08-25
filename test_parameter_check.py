#!/usr/bin/env python
"""Check if transfer_lr parameter is properly set in LRTTTransferCompound."""

import os
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig

# Create config with specific transfer_lr
device = ConstantStepDevice(dw_min=0.0001)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=3,
    transfer_lr=0.123,  # Specific value
    forward_inject=True,
    lora_alpha=1.0,
)

print("Python-side LRTTTransferCompound config:")
print(f"  transfer_lr = {lrtt_config.transfer_lr}")
print(f"  transfer_every = {lrtt_config.transfer_every}")

# Create the C++ parameter object
rpu_config = SingleRPUConfig(device=lrtt_config)

# Get the actual C++ parameter object
cpp_params = rpu_config.as_bindings()

print("\nC++ bindings object:")
print(f"  Type: {type(cpp_params)}")

# Get device params from the tile parameters
if hasattr(cpp_params, 'device_par'):
    device_params = cpp_params.device_par
    print(f"  device_par type: {type(device_params)}")
    print(f"  Has transfer_lr: {hasattr(device_params, 'transfer_lr')}")
    if hasattr(device_params, 'transfer_lr'):
        print(f"  transfer_lr value: {device_params.transfer_lr}")
        
    print("\nTrying to access transfer_lr directly...")
    try:
        val = device_params.transfer_lr
        print(f"  SUCCESS: transfer_lr = {val}")
    except AttributeError as e:
        print(f"  FAILED: {e}")
    
    print("\nChecking device hierarchy...")
    print(f"  device_params type: {type(device_params).__name__}")
    print(f"  MRO: {[c.__name__ for c in type(device_params).__mro__]}")

    # Check parent class attributes
    print("\nAll attributes of device_params:")
    attrs = [a for a in dir(device_params) if not a.startswith('_')]
    for attr in sorted(attrs):
        try:
            val = getattr(device_params, attr)
            if 'transfer' in attr.lower() or 'lr' in attr.lower():
                print(f"  {attr} = {val}")
        except:
            pass