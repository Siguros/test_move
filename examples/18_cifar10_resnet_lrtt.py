# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 18 with LRTT: ResNet32 CNN with CIFAR10 using Low-Rank Tiki-Taka.

CIFAR10 dataset on a ResNet inspired network with LR-TT (Low-Rank Tiki-Taka)
transfer learning for efficient analog training. Based on the paper:
https://arxiv.org/abs/1512.03385
"""
# pylint: disable=invalid-name

# Imports
import os
from datetime import datetime

# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.optim import AnalogSGD
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.rpu_base import cuda


# Device to use
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET_LRTT")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "example_18_lrtt_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 100  # Reduced for LRTT demonstration
BATCH_SIZE = 32
LEARNING_RATE = 0.1
N_CLASSES = 10

# LR-TT specific parameters
LRTT_RANK = 16  # Low-rank dimension for ResNet layers
TRANSFER_EVERY = 100  # Transfer every N updates
TRANSFER_LR = 0.5  # Transfer learning rate


def create_lrtt_config():
    """Create LR-TT configuration for analog conversion."""
    device = ConstantStepDevice(dw_min=0.005)
    
    lrtt_config = LRTTTransferCompound(
        unit_cell_devices=[device, device, device],  # fastA, fastB, visible
        rank=LRTT_RANK,
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        forward_inject=True,  # Use effective weights W_eff = W_visible + α*(A@B)
        lora_alpha=1.0,
    )
    
    # Add mapping for weight scaling
    mapping = MappingParameter(weight_scaling_omega=0.6)
    return SingleRPUConfig(device=lrtt_config, mapping=mapping)


class ResidualBlock(nn.Module):
    """Residual block of a residual network with option for the skip connection."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=stride)
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual block to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlock(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlock(hidden_ch, hidden_ch))
    return layers


def create_model():
    """Return a ResNet32 inspired model.

    Returns:
       nn.Module: model
    """
    b0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    
    # Create residual layers
    b1 = nn.Sequential(*concatenate_layer_blocks(64, 64, 5, first_layer=True))
    b2 = nn.Sequential(*concatenate_layer_blocks(64, 128, 5))
    b3 = nn.Sequential(*concatenate_layer_blocks(128, 256, 5))
    b4 = nn.Sequential(*concatenate_layer_blocks(256, 512, 5))
    
    # Global average pooling and classifier
    net = nn.Sequential(
        b0,
        b1,
        b2,
        b3,
        b4,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, N_CLASSES),
    )
    return net


def create_analog_model_lrtt():
    """Create and convert model to analog with LR-TT."""
    model = create_model()
    
    # Get LR-TT configuration
    lrtt_config = create_lrtt_config()
    
    # Convert to analog with LR-TT
    model = convert_to_analog(model, lrtt_config)
    
    return model


def train_step(train_data, model, criterion, optimizer):
    """Train a single epoch.
    
    Args:
        train_data (DataLoader): training data loader
        model (nn.Module): analog model with LR-TT
        criterion: loss function
        optimizer: analog optimizer
        
    Returns:
        float: average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_data)


def test_step(test_data, model, criterion, use_visible_only=False):
    """Test the model.
    
    For LRTT layers, we can evaluate using only visible weights (without A@B contribution)
    by temporarily disabling forward_inject during evaluation.
    
    Args:
        test_data (DataLoader): test data loader
        model (nn.Module): trained model
        criterion: loss function
        use_visible_only (bool): If True, use only visible weights for LRTT layers
        
    Returns:
        float, float: test loss and accuracy
    """
    model.eval()
    
    # Store original forward_inject states and disable for validation if requested
    original_states = []
    if use_visible_only:
        from aihwkit.nn import AnalogLinear
        for layer in model.modules():
            if isinstance(layer, AnalogLinear):
                try:
                    # Check if this is an LRTT layer
                    if hasattr(layer.analog_module, 'tile'):
                        tile = layer.analog_module.tile
                        if hasattr(tile, 'get_meta_parameters'):
                            meta_params = tile.get_meta_parameters()
                            if hasattr(meta_params, 'forward_inject'):
                                original_states.append((meta_params, meta_params.forward_inject))
                                meta_params.forward_inject = False
                except:
                    pass
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with no_grad():
        for images, labels in test_data:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            output = model(images)
            loss = criterion(output, labels)
            
            total_loss += loss.item()
            _, predicted = torch_max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Restore original forward_inject states
    for meta_params, original_state in original_states:
        meta_params.forward_inject = original_state
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_data)
    
    return avg_loss, accuracy


def main():
    """Main training function with LR-TT."""
    
    # Set seed for reproducibility
    manual_seed(SEED)
    
    print("=" * 70)
    print("CIFAR-10 ResNet32 Training with LR-TT (Low-Rank Tiki-Taka)")
    print(f"Rank: {LRTT_RANK}, Transfer every: {TRANSFER_EVERY} updates")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.CIFAR10(
        root=PATH_DATASET, train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=PATH_DATASET, train=False, download=True, transform=transform
    )
    
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create analog model with LR-TT
    model = create_analog_model_lrtt()
    model = model.to(DEVICE)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Starting training with LR-TT...")
    print("-" * 70)
    
    # Setup optimizer and loss
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(N_EPOCHS):
        epoch_start = datetime.now()
        
        # Train
        train_loss = train_step(train_data, model, criterion, optimizer)
        
        # Test every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Test with visible weights only
            test_loss_vis, test_acc_vis = test_step(test_data, model, criterion, use_visible_only=True)
            # Test with A@B injection
            test_loss_inj, test_acc_inj = test_step(test_data, model, criterion, use_visible_only=False)
            
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Test Loss (vis): {test_loss_vis:.4f} - "
                  f"Test Acc (vis): {test_acc_vis:.2f}% - "
                  f"Test Acc (inj): {test_acc_inj:.2f}% - "
                  f"Time: {(datetime.now() - epoch_start).total_seconds():.1f}s")
            
            # Save best model based on visible weights accuracy
            if test_acc_vis > best_accuracy:
                best_accuracy = test_acc_vis
                save(model.state_dict(), WEIGHT_PATH)
                print(f"  -> New best accuracy! Model saved.")
        else:
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Time: {(datetime.now() - epoch_start).total_seconds():.1f}s")
    
    # Final evaluation
    print("-" * 70)
    test_loss_vis, test_acc_vis = test_step(test_data, model, criterion, use_visible_only=True)
    test_loss_inj, test_acc_inj = test_step(test_data, model, criterion, use_visible_only=False)
    print(f"Final Test Accuracy (visible only): {test_acc_vis:.2f}%")
    print(f"Final Test Accuracy (with A@B): {test_acc_inj:.2f}%")
    print(f"Difference (inject - visible): {test_acc_inj - test_acc_vis:.2f}%")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()