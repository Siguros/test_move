#!/usr/bin/env python
"""Quick test of MNIST LRTT validation functionality."""

import os
import torch
from torch import nn
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    SingleRPUConfig, 
    FloatingPointRPUConfig,
    LRTTTransferCompound, 
    ConstantStepDevice, 
    FloatingPointDevice
)
from aihwkit.simulator.rpu_base import cuda

# Check device
DEVICE = torch.device("cuda" if cuda.is_compiled() else "cpu")
print(f"Using device: {DEVICE}")

# Create a small model for testing
def create_test_model():
    """Create a small test model with LRTT."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = SingleRPUConfig(
        device=LRTTTransferCompound(
            unit_cell_devices=[device, device, device],
            rank=4,
            transfer_every=5,
            transfer_lr=0.5,
            forward_inject=True,
            lora_alpha=1.0,
        )
    )
    
    model = AnalogSequential(
        AnalogLinear(784, 32, True, rpu_config=lrtt_config),
        nn.ReLU(),
        AnalogLinear(32, 10, True, rpu_config=FloatingPointRPUConfig(device=FloatingPointDevice())),
    )
    return model.to(DEVICE)

# Load a small batch of MNIST data
transform = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST("data/DATASET", download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# Create model
model = create_test_model()
print(f"Model created: {model}")

# Test validation with and without forward_inject
def test_with_inject_control(model, data_loader, disable_inject=False):
    """Test model with or without forward_inject."""
    model.eval()
    
    # Store and modify forward_inject states
    original_states = []
    if disable_inject:
        for layer in model.modules():
            if isinstance(layer, AnalogLinear):
                try:
                    if hasattr(layer.analog_module, 'tile'):
                        tile = layer.analog_module.tile
                        if hasattr(tile, 'get_meta_parameters'):
                            meta_params = tile.get_meta_parameters()
                            if hasattr(meta_params, 'forward_inject'):
                                print(f"  Found LRTT layer, forward_inject was: {meta_params.forward_inject}")
                                original_states.append((meta_params, meta_params.forward_inject))
                                meta_params.forward_inject = False
                except:
                    pass
    
    # Get one batch and test
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE).view(data.size(0), -1)
            output = model(data)
            pred = output.argmax(dim=1)
            accuracy = (pred == target.to(DEVICE)).float().mean().item() * 100
            break
    
    # Restore states
    for meta_params, original_state in original_states:
        meta_params.forward_inject = original_state
        print(f"  Restored forward_inject to: {original_state}")
    
    return accuracy

# Train for a few steps
print("\nTraining for 10 steps...")
optimizer = AnalogSGD(model.parameters(), lr=0.01)
optimizer.regroup_param_groups(model)
criterion = nn.CrossEntropyLoss()

model.train()
train_set = datasets.MNIST("data/DATASET", download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

step = 0
for data, target in train_loader:
    if step >= 10:
        break
    data = data.to(DEVICE).view(data.size(0), -1)
    target = target.to(DEVICE)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"  Step {step}: Loss = {loss.item():.4f}")
    step += 1

# Test validation
print("\nTesting validation modes:")
print("With forward_inject enabled (A@B included):")
acc_with_inject = test_with_inject_control(model, test_loader, disable_inject=False)
print(f"  Accuracy: {acc_with_inject:.2f}%")

print("\nWith forward_inject disabled (visible weights only):")
acc_without_inject = test_with_inject_control(model, test_loader, disable_inject=True)
print(f"  Accuracy: {acc_without_inject:.2f}%")

print(f"\nDifference (inject - visible): {acc_with_inject - acc_without_inject:.2f}%")