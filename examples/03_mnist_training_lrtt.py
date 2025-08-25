# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 3 with LRTT: MNIST training with Low-Rank Tiki-Taka transfer learning.

MNIST training example using LR-TT (Low-Rank Tiki-Taka) for efficient transfer learning
on analog hardware. Based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name, redefined-outer-name

import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
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
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 3  # Reduced for testing with monitoring
BATCH_SIZE = 64
MONITOR_LRTT = True  # Enable LRTT operation monitoring

# LR-TT parameters
LRTT_RANK = 8  # Low-rank dimension
TRANSFER_EVERY = 100  # Transfer every N updates
TRANSFER_LR = 0.00000001  # Transfer learning rate (Note: actual transfer is limited by device dw_min)


def get_lrtt_weights(layer):
    """Extract LRTT weights from a layer for monitoring."""
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


def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data


def create_lrtt_config():
    """Create LR-TT configuration for analog layers."""
    device = ConstantStepDevice(dw_min=0.001)
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],  # fastA, fastB, visible
        rank=LRTT_RANK,
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        forward_inject=True,  # Use effective weights in forward pass
        lora_alpha=8.0,
    )
    return SingleRPUConfig(device=lrtt_config)


def create_analog_network_lrtt(input_size, hidden_sizes, output_size):
    """Create the neural network using analog layers with LR-TT.
    
    The first two layers use LR-TT for efficient transfer learning,
    while the last layer uses FloatingPointDevice for precise classification.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model with LR-TT for hidden layers
    """
    lrtt_config = create_lrtt_config()
    fp_config = FloatingPointRPUConfig(device=FloatingPointDevice())
    
    model = AnalogSequential(
        AnalogLinear(
            input_size,
            hidden_sizes[0],
            True,
            rpu_config=lrtt_config,
        ),
        nn.Sigmoid(),
        AnalogLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            True,
            rpu_config=lrtt_config,
        ),
        nn.Sigmoid(),
        AnalogLinear(
            hidden_sizes[1],
            output_size,
            True,
            rpu_config=fp_config,  # Use FloatingPointDevice for last layer
        ),
        nn.LogSoftmax(dim=1),
    )

    return model


def train(model, train_data, val_data=None):
    """Train the network with periodic validation.

    Args:
        model (nn.Module): network model.
        train_data (DataLoader): dataset with training data.
        val_data (DataLoader): optional validation dataset for periodic evaluation.

    Returns:
        nn.Module, float: model and loss for the epoch.
    """
    model.train()

    # Define the loss function and optimizer.
    optimizer = AnalogSGD(model.parameters(), lr=0.1)
    optimizer.regroup_param_groups(model)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    total_loss = 0
    
    # Get LRTT layers for monitoring
    lrtt_layers = []
    if MONITOR_LRTT:
        for i, module in enumerate(model.modules()):
            if isinstance(module, AnalogLinear):
                weights = get_lrtt_weights(module)
                if weights is not None:
                    lrtt_layers.append((module, f"L{i}"))
        if lrtt_layers:
            print(f"Monitoring {len(lrtt_layers)} LRTT layers")
    
    global_step = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_start = time()
        
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.view(data.size(0), -1)
            
            # Monitor before update if enabled
            weights_before = []
            if MONITOR_LRTT and global_step % TRANSFER_EVERY == (TRANSFER_EVERY - 1):
                for layer, _ in lrtt_layers:
                    weights_before.append(get_lrtt_weights(layer))
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            
            # Check for transfer if monitoring
            if MONITOR_LRTT and weights_before:
                for i, ((layer, name), w_before) in enumerate(zip(lrtt_layers, weights_before)):
                    w_after = get_lrtt_weights(layer)
                    vis_change = torch.norm(w_after['visible'] - w_before['visible']).item()
                    A_norm_before = torch.norm(w_before['A']).item()
                    A_norm_after = torch.norm(w_after['A']).item()
                    B_norm_before = torch.norm(w_before['B']).item()
                    B_norm_after = torch.norm(w_after['B']).item()
                    
                    if vis_change > 0.01 and A_norm_after < 0.01:
                        print(f'\n  🔄 TRANSFER at step {global_step} in {name}:')
                        print(f'     A: {A_norm_before:.4f} → {A_norm_after:.4f} (reset)')
                        print(f'     B: {B_norm_before:.4f} → {B_norm_after:.4f} (reinit)')
                        print(f'     C: Δ={vis_change:.4f} (transfer of A@B)')
                        
                        # Show actual weight values (first 2x2)
                        if epoch == 0 and i == 0:  # Show details for first layer in first epoch
                            print(f'\n     Weight details for {name}:')
                            A_show = w_after['A'][:2, :2].cpu().numpy()
                            B_show = w_after['B'][:2, :2].cpu().numpy()
                            C_show = w_after['visible'][:2, :2].cpu().numpy()
                            print(f'     A[0:2,0:2]: [{A_show[0,0]:.4f}, {A_show[0,1]:.4f}]')
                            print(f'     B[0:2,0:2]: [{B_show[0,0]:.4f}, {B_show[0,1]:.4f}]')
                            print(f'     C[0:2,0:2]: [{C_show[0,0]:.4f}, {C_show[0,1]:.4f}]')
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f'  Batch {batch_idx:3d}/{len(train_data)} - Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_data)
        total_loss += avg_loss
        
        print(f'Epoch {epoch + 1:2d}/{EPOCHS} - '
              f'Avg Loss: {avg_loss:.4f} - '
              f'LR: {optimizer.param_groups[0]["lr"]:.5f} - '
              f'Time: {time() - epoch_start:.1f}s')
        
        # Validate every 5 epochs
        if val_data and (epoch + 1) % 5 == 0:
            print("\nValidation check:")
            # Test with visible weights only
            val_acc_visible = test_evaluation(model, val_data, use_visible_only=True)
            # Test with A@B injection
            val_acc_inject = test_evaluation(model, val_data, use_visible_only=False)
            print(f"Accuracy difference (inject - visible): {val_acc_inject - val_acc_visible:.2f}%")
            print("-" * 70)

    return model, total_loss / EPOCHS


def test_evaluation(model, test_data, use_visible_only=True):
    """Test the trained network.
    
    For LRTT layers, we can evaluate using only visible weights (without A@B contribution)
    by temporarily disabling forward_inject during evaluation.

    Args:
        model (nn.Module): trained network.
        test_data (DataLoader): dataset with testing data.
        use_visible_only (bool): If True, use only visible weights for LRTT layers.

    Returns:
        float: test accuracy.
    """
    model.eval()
    
    # Store original forward_inject states and disable for validation
    original_states = []
    if use_visible_only:
        for layer in model.modules():
            if isinstance(layer, AnalogLinear):
                try:
                    # Check if this is an LRTT layer by accessing the tile
                    if hasattr(layer.analog_module, 'tile'):
                        tile = layer.analog_module.tile
                        if hasattr(tile, 'get_meta_parameters'):
                            meta_params = tile.get_meta_parameters()
                            if hasattr(meta_params, 'forward_inject'):
                                original_states.append((meta_params, meta_params.forward_inject))
                                meta_params.forward_inject = False
                except:
                    pass

    correct = 0
    criterion = nn.NLLLoss()
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(DEVICE), target.to(DEVICE)
            data = data.view(data.size(0), -1)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Restore original forward_inject states
    for meta_params, original_state in original_states:
        meta_params.forward_inject = original_state

    test_loss /= len(test_data)
    accuracy = 100.0 * correct / len(test_data.dataset)
    
    if use_visible_only:
        print(f'\nTest set (visible weights only): Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_data.dataset)} ({accuracy:.2f}%)\n')
    else:
        print(f'\nTest set (with A@B injection): Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_data.dataset)} ({accuracy:.2f}%)\n')

    return accuracy


def main():
    """Main function to run MNIST training with LR-TT."""
    print("=" * 70)
    print("MNIST Training with LR-TT (Low-Rank Tiki-Taka)")
    print(f"Rank: {LRTT_RANK}, Transfer every: {TRANSFER_EVERY} updates")
    print(f"Device: {DEVICE}")
    print("Last layer uses FloatingPointDevice for precise classification")
    print("=" * 70)
    
    # Load datasets.
    train_data, validation_data = load_images()

    # Create the analog model with LR-TT.
    model = create_analog_network_lrtt(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    model = model.to(DEVICE)
    
    print(f"\nModel architecture:")
    print(model)
    print("\nStarting training...")
    print("-" * 70)

    # Train the model with validation checks.
    model, avg_loss = train(model, train_data, validation_data)
    
    print("-" * 70)
    print(f"Training completed! Average loss: {avg_loss:.4f}")

    # Final evaluation on test set.
    print("\nFinal evaluation:")
    
    # Test with visible weights only (actual learned weights)
    accuracy_visible = test_evaluation(model, validation_data, use_visible_only=True)
    
    # Test with A@B injection (effective weights during training)
    accuracy_inject = test_evaluation(model, validation_data, use_visible_only=False)
    
    print("=" * 70)
    print(f"Final Test Accuracy (visible only): {accuracy_visible:.2f}%")
    print(f"Final Test Accuracy (with A@B): {accuracy_inject:.2f}%")
    print(f"Difference: {accuracy_inject - accuracy_visible:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()