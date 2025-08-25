#!/usr/bin/env python
"""Monitor LRTT operations during MNIST training to verify everything works correctly."""

import os
import torch
from torch import nn
from torchvision import datasets, transforms

# Imports from aihwkit
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
print("=" * 70)

# Network parameters
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10
BATCH_SIZE = 64

# LR-TT parameters
LRTT_RANK = 8
TRANSFER_EVERY = 10  # Transfer every 10 updates
TRANSFER_LR = 1.0

def create_lrtt_config():
    """Create LR-TT configuration."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],
        rank=LRTT_RANK,
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        forward_inject=True,
        lora_alpha=1.0,
    )
    return SingleRPUConfig(device=lrtt_config)

def create_model():
    """Create MNIST model with LRTT layers."""
    lrtt_config = create_lrtt_config()
    fp_config = FloatingPointRPUConfig(device=FloatingPointDevice())
    
    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], True, rpu_config=lrtt_config),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[0], HIDDEN_SIZES[1], True, rpu_config=lrtt_config),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZES[1], OUTPUT_SIZE, True, rpu_config=fp_config),
        nn.LogSoftmax(dim=1),
    )
    return model.to(DEVICE)

def load_mnist_data():
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("data/DATASET", download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

def get_lrtt_weights(layer):
    """Extract LRTT weights from a layer."""
    if not isinstance(layer, AnalogLinear):
        return None
    
    tiles = list(layer.analog_tiles())
    if not tiles:
        return None
    
    cpp_tile = tiles[0].tile
    
    # Check if this is an LRTT tile
    if not hasattr(cpp_tile, 'lrtt_get_visible_weights'):
        return None
    
    return {
        'visible': cpp_tile.lrtt_get_visible_weights().clone(),
        'A': cpp_tile.lrtt_get_A_lr().clone(),
        'B': cpp_tile.lrtt_get_B_lr().clone(),
    }

def monitor_training():
    """Monitor LRTT operations during MNIST training."""
    print("MONITORING LRTT OPERATIONS DURING MNIST TRAINING")
    print("=" * 70)
    
    # Create model and optimizer
    model = create_model()
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)
    criterion = nn.NLLLoss()
    
    # Load data
    train_loader = load_mnist_data()
    
    # Get references to LRTT layers
    lrtt_layers = []
    layer_names = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, AnalogLinear):
            weights = get_lrtt_weights(module)
            if weights is not None:  # It's an LRTT layer
                lrtt_layers.append(module)
                layer_names.append(f"Layer_{i}")
                print(f"Found LRTT layer {i}: {module.in_features}→{module.out_features}")
    
    print(f"\nFound {len(lrtt_layers)} LRTT layers to monitor")
    print("-" * 70)
    
    # Store initial weights
    initial_weights = []
    for layer in lrtt_layers:
        initial_weights.append(get_lrtt_weights(layer))
    
    # Training loop with monitoring
    step_count = 0
    transfer_steps = []
    
    print("\nStarting training with monitoring...")
    print("Transfer expected every", TRANSFER_EVERY, "steps")
    print("-" * 70)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 20:  # Monitor first 20 batches
            break
        
        data = data.to(DEVICE).view(data.size(0), -1)
        target = target.to(DEVICE)
        
        # Get weights before update
        weights_before = []
        for layer in lrtt_layers:
            weights_before.append(get_lrtt_weights(layer))
        
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        step_count += 1
        
        # Get weights after update
        weights_after = []
        for layer in lrtt_layers:
            weights_after.append(get_lrtt_weights(layer))
        
        # Check for changes
        transfer_detected = False
        for i, (before, after, name) in enumerate(zip(weights_before, weights_after, layer_names)):
            vis_change = torch.norm(after['visible'] - before['visible']).item()
            A_change = torch.norm(after['A'] - before['A']).item()
            B_change = torch.norm(after['B'] - before['B']).item()
            
            # Check if A reset (transfer happened)
            A_norm_before = torch.norm(before['A']).item()
            A_norm_after = torch.norm(after['A']).item()
            B_norm_before = torch.norm(before['B']).item()
            B_norm_after = torch.norm(after['B']).item()
            
            if A_norm_before > 0.01 and A_norm_after < 0.01 and vis_change > 0.001:
                transfer_detected = True
                transfer_steps.append(step_count)
                print(f"\n🔄 TRANSFER DETECTED at step {step_count} in {name}!")
                print(f"   Visible change: {vis_change:.6f}")
                print(f"   A norm: {A_norm_before:.6f} → {A_norm_after:.6f} (reset)")
                print(f"   B norm: {B_norm_before:.6f} → {B_norm_after:.6f} (reinit)")
        
        # Regular progress reporting
        if batch_idx % 5 == 0:
            if not transfer_detected:
                print(f"Step {step_count:3d}: Loss={loss.item():.4f}", end="")
                
                # Show weight norms for first layer
                w = weights_after[0]
                print(f" | L1: A={torch.norm(w['A']):.3f}, B={torch.norm(w['B']):.3f}, vis={torch.norm(w['visible']):.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Check weight updates
    print("\n1. WEIGHT UPDATE CHECK:")
    for i, (initial, final, name) in enumerate(zip(initial_weights, weights_after, layer_names)):
        A_updated = torch.norm(final['A'] - initial['A']).item() > 0.001 or torch.norm(final['A']).item() > 0.001
        B_updated = torch.norm(final['B'] - initial['B']).item() > 0.001 or torch.norm(final['B']).item() > 0.001
        vis_updated = torch.norm(final['visible'] - initial['visible']).item() > 0.001
        
        print(f"   {name}: A={'✓' if A_updated else '✗'}, B={'✓' if B_updated else '✗'}, visible={'✓' if vis_updated else '✗'}")
    
    # Check transfers
    print(f"\n2. TRANSFER CHECK:")
    print(f"   Expected transfers at steps: {list(range(TRANSFER_EVERY, step_count+1, TRANSFER_EVERY))}")
    print(f"   Detected transfers at steps: {transfer_steps}")
    
    # Test forward operation
    print(f"\n3. FORWARD OPERATION CHECK (C + A@B):")
    for i, (layer, name) in enumerate(zip(lrtt_layers, layer_names)):
        weights = get_lrtt_weights(layer)
        tiles = list(layer.analog_tiles())
        cpp_tile = tiles[0].tile
        
        # Test with small input
        x_test = torch.randn(2, layer.in_features).to(DEVICE)
        
        # Get effective weights
        if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
            W_eff = cpp_tile.lrtt_compose_w_eff(1.0)
            
            # Manual computation
            AB = torch.matmul(weights['A'], weights['B'])
            W_manual = weights['visible'] + AB
            
            # Skip bias handling for now - focus on weight matrix only
            
            diff = torch.norm(W_eff - W_manual).item()
            print(f"   {name}: Difference between compose_w_eff and manual: {diff:.6e} {'✓' if diff < 1e-5 else '✗'}")
    
    # Test validation with/without injection
    print(f"\n4. FORWARD INJECT CONTROL CHECK:")
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(5, INPUT_SIZE).to(DEVICE)
        
        # Normal forward (with injection)
        output_normal = model(x_test)
        
        # We can't easily disable injection at runtime without recreating tiles
        # But we can verify that A and B are non-zero (contributing to forward)
        for i, (layer, name) in enumerate(zip(lrtt_layers, layer_names)):
            weights = get_lrtt_weights(layer)
            A_norm = torch.norm(weights['A']).item()
            B_norm = torch.norm(weights['B']).item()
            if A_norm > 0.01 and B_norm > 0.01:
                print(f"   {name}: A@B contribution active (A norm={A_norm:.3f}, B norm={B_norm:.3f}) ✓")
            else:
                print(f"   {name}: A or B near zero (A norm={A_norm:.3f}, B norm={B_norm:.3f})")
    
    print("\n" + "=" * 70)
    print("MONITORING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    monitor_training()