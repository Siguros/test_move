#!/usr/bin/env python
"""Debug test for LRTT - test forward injection."""

import os
os.environ["AIHWKIT_DEBUG_LRTT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.insert(0, '/workspace/site-packages')

import torch
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import AnalogLinear

print("=" * 80)
print("LRTT Forward Injection Test")
print("=" * 80)

# Test with forward_inject enabled
rpu_config = SingleRPUConfig()
rpu_config.mapping.max_input_size = 64
rpu_config.mapping.max_output_size = 64

rpu_config.device = LRTTTransferCompound(
    unit_cell_devices=[
        ConstantStepDevice(),  # fastA
        ConstantStepDevice(),  # fastB  
        ConstantStepDevice(),  # visible
    ],
    transfer_lr=0.1,
    transfer_every=2,
    rank=4,
    forward_inject=True,  # Enable injection
    lora_alpha=1.0,
)

print("\nConfiguration:")
print(f"  transfer_lr: {rpu_config.device.transfer_lr}")
print(f"  transfer_every: {rpu_config.device.transfer_every}")
print(f"  rank: {rpu_config.device.rank}")
print(f"  forward_inject: {rpu_config.device.forward_inject}")
print(f"  lora_alpha: {rpu_config.device.lora_alpha}")

try:
    print("\n[Creating AnalogLinear layer...]")
    layer = AnalogLinear(8, 4, rpu_config=rpu_config, bias=False)
    
    print("[Moving to CUDA...]")
    layer = layer.cuda()
    
    print("[Creating input tensor...]")
    x = torch.ones(2, 8).cuda()
    
    print("[Running forward pass with injection...]")
    y = layer(x)
    
    print(f"\nForward pass successful with injection!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output values: {y}")
    
    # Now test backward pass
    print("\n[Testing backward pass...]")
    loss = y.sum()
    loss.backward()
    print("Backward pass successful!")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)