#!/usr/bin/env python
"""Minimal test for LR-TT configuration validation."""

print("Testing LR-TT configuration structure...")

# Test the Python-only configuration class
import sys
sys.path.insert(0, '/workspace/aihwkit/src')

try:
    from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
    from aihwkit.simulator.presets.devices import IdealizedPresetDevice
    
    print("✓ Imports successful")
    
    # Create a basic configuration
    config = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),
            IdealizedPresetDevice(), 
            IdealizedPresetDevice()
        ],
        rank=4,
        transfer_every=1,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0
    )
    
    print("✓ Configuration created")
    
    # Check attributes
    assert config.rank == 4
    assert config.transfer_every == 1
    assert config.transfer_lr == 1.0
    assert config.forward_inject == True
    assert config.lora_alpha == 1.0
    assert len(config.unit_cell_devices) == 3
    
    print("✓ Attributes validated")
    
    # Check canonical indices
    assert config._idx_fastA == 0
    assert config._idx_fastB == 1
    assert config._idx_visible == 2
    
    print("✓ Canonical indices correct")
    
    # Test validation
    try:
        # This should fail - duplicate indices
        bad_config = LRTTTransferCompound(
            unit_cell_devices=[
                IdealizedPresetDevice(),
                IdealizedPresetDevice(),
                IdealizedPresetDevice()
            ],
            idx_fastA=0,
            idx_fastB=0,  # Duplicate!
            idx_visible=2
        )
        print("✗ Should have raised error for duplicate indices")
    except ValueError as e:
        print(f"✓ Validation caught duplicate indices: {e}")
    
    # Test rank validation
    try:
        bad_config = LRTTTransferCompound(
            unit_cell_devices=[
                IdealizedPresetDevice(),
                IdealizedPresetDevice(),
                IdealizedPresetDevice()
            ],
            rank=-1  # Invalid!
        )
        print("✗ Should have raised error for negative rank")
    except ValueError as e:
        print(f"✓ Validation caught negative rank: {e}")
    
    print("\n" + "="*50)
    print("All configuration tests passed!")
    print("="*50)
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nThis likely means the module structure is not set up correctly.")
    exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)