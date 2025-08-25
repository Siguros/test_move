#!/usr/bin/env python3

"""
LR-TT CUDA functionality smoke test
Tests basic LR-TT functionality to ensure CUDA compilation and user modifications work correctly.
"""

import torch
import pytest
from aihwkit.simulator import rpu_base
from aihwkit.algorithms.lr_tt import LRTransferHook
from aihwkit.simulator.configs import SingleRPUConfig, TransferCompound
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.nn import AnalogLinear


def test_cuda_available():
    """Test that CUDA is available and compiled."""
    print(f"CUDA compiled: {rpu_base.cuda.is_compiled()}")
    assert rpu_base.cuda.is_compiled(), "CUDA should be compiled"


def test_lr_tt_import():
    """Test that LR-TT components can be imported."""
    from aihwkit.algorithms.lr_tt import LRTransferHook, lrtt_transfer_step, plan_lr_vectors
    assert LRTransferHook is not None
    assert lrtt_transfer_step is not None
    assert plan_lr_vectors is not None


def test_lr_tt_basic_functionality():
    """Test basic LR-TT functionality with a simple analog layer."""
    # Skip if CUDA not available
    if not rpu_base.cuda.is_compiled():
        pytest.skip("CUDA not available")
    
    # Create a simple transfer device configuration
    rpu_config = SingleRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()]
        )
    )
    
    # Create analog layer
    layer = AnalogLinear(in_features=10, out_features=5, rpu_config=rpu_config)
    if torch.cuda.is_available():
        layer = layer.cuda()
    
    # Test forward pass
    x = torch.randn(2, 10)
    if torch.cuda.is_available():
        x = x.cuda()
    
    try:
        output = layer(x)
        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"
        print("✓ Basic forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_lr_transfer_hook():
    """Test LR Transfer Hook functionality."""
    # Skip if CUDA not available  
    if not rpu_base.cuda.is_compiled():
        pytest.skip("CUDA not available")
    
    try:
        # Create a test layer first
        rpu_config = SingleRPUConfig(
            device=TransferCompound(
                unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice()]
            )
        )
        layer = AnalogLinear(in_features=10, out_features=5, rpu_config=rpu_config)
        
        # Test hook creation with required modules list
        hook = LRTransferHook(
            modules_or_tiles=[layer],  # Required list of modules/tiles
            rank=10,                   # Low-rank dimension
            transfer_every=10          # Transfer frequency (must be > 0)
        )
        assert hook is not None
        print("✓ LRTransferHook creation successful")
        
    except Exception as e:
        print(f"✗ LRTransferHook creation failed: {e}")
        raise


if __name__ == "__main__":
    print("Running LR-TT CUDA smoke tests...")
    print("=" * 50)
    
    test_cuda_available()
    test_lr_tt_import()
    test_lr_tt_basic_functionality()  
    test_lr_transfer_hook()
    
    print("=" * 50)
    print("✓ All LR-TT CUDA smoke tests passed!")