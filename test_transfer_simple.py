#!/usr/bin/env python3
"""Simple test to check LRTT transfer functionality."""

import torch
import numpy as np
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized

# Set CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print(f"CUDA available: {torch.cuda.is_available()}")

# Configuration
d_size = 8
x_size = 4
rank = 2
batch_size = 2

# Create LRTT config
config = lrtt_idealized(rank=rank)
device = config.device
device.transfer_every = 1
device.transfer_lr = 0.1
device.lora_alpha = 4.0
device.forward_inject = True

print(f"\nConfiguration:")
print(f"  Rank: {rank}")
print(f"  Transfer every: {device.transfer_every}")
print(f"  Transfer LR: {device.transfer_lr}")
print(f"  Forward inject: {device.forward_inject}")

# Create layer
layer = AnalogLinear(x_size, d_size, bias=False, rpu_config=config).cuda()

# Get tile
tiles = list(layer.analog_tiles())
tile = tiles[0]

print("\nTile info:")
print(f"  Tile type: {type(tile)}")

# Check what's available
print("\nChecking available methods:")
if hasattr(tile, 'get_hidden_parameters'):
    print("  has get_hidden_parameters")
    params = tile.get_hidden_parameters()
    print(f"  Keys: {params.keys()}")
    for key, val in params.items():
        print(f"    {key}: shape={val.shape}")
        
if hasattr(tile, 'get_weights'):
    print("  has get_weights")
    weights = tile.get_weights()
    print(f"  Weights: {type(weights)}, len={len(weights)}")
    for i, w in enumerate(weights):
        print(f"    Weight {i}: shape={w.shape}")

print("\nPerforming forward/backward:")
x = torch.randn(batch_size, x_size, device='cuda')
y = layer(x)
loss = y.sum()
loss.backward()

print("\nAfter update:")
if hasattr(tile, 'get_hidden_parameters'):
    params_after = tile.get_hidden_parameters()
    print(f"  Keys after: {params_after.keys()}")

print("\nTest complete")