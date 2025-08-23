#!/usr/bin/env python
"""Test LR-TT configuration without requiring full aihwkit installation."""

import sys
import pathlib
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

print("=" * 60)
print("LR-TT Configuration Tests (Python-only)")
print("=" * 60)

# Load the lrtt_compound module directly
here = pathlib.Path(__file__).resolve().parent
lrtt_compound_path = here / "src" / "aihwkit" / "simulator" / "configs" / "lrtt_compound.py"

if not lrtt_compound_path.exists():
    print(f"ERROR: lrtt_compound.py not found at {lrtt_compound_path}")
    sys.exit(1)

# Create minimal mock classes to satisfy imports
class MockDevice:
    """Mock device for testing configuration."""
    def __init__(self):
        pass
    
    def as_bindings(self, data_type):
        """Mock as_bindings method."""
        return None

# Inject mocks into sys.modules to avoid import errors
sys.modules['aihwkit'] = type(sys)('aihwkit')
sys.modules['aihwkit.simulator'] = type(sys)('aihwkit.simulator')
sys.modules['aihwkit.simulator.configs'] = type(sys)('aihwkit.simulator.configs')
sys.modules['aihwkit.simulator.configs.compounds'] = type(sys)('aihwkit.simulator.configs.compounds')
sys.modules['aihwkit.simulator.configs.devices'] = type(sys)('aihwkit.simulator.configs.devices')
sys.modules['aihwkit.simulator.parameters'] = type(sys)('aihwkit.simulator.parameters')
sys.modules['aihwkit.simulator.parameters.enums'] = type(sys)('aihwkit.simulator.parameters.enums')

# Add mock classes
sys.modules['aihwkit.simulator.configs.compounds'].TransferCompound = object
sys.modules['aihwkit.simulator.configs.devices'].PulsedDevice = MockDevice
sys.modules['aihwkit.simulator.parameters.enums'].RPUDataType = type('RPUDataType', (), {'FLOAT': 1})

# Load the actual LRTTTransferCompound
print("Loading lrtt_compound.py...")
exec(open(lrtt_compound_path).read(), globals())

print("✓ Module loaded successfully")

# Test 1: Basic configuration
print("\nTest 1: Basic configuration creation")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        transfer_every=2,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=2.0,
    )
    print("✓ Configuration created")
    
    # Check attributes
    assert config.rank == 4
    assert config.transfer_every == 2
    assert config.transfer_lr == 1.0
    assert config.forward_inject == True
    assert config.lora_alpha == 2.0
    assert len(config.unit_cell_devices) == 3
    print("✓ Basic attributes correct")
    
    # Check canonical indices
    assert config._idx_fastA == 0
    assert config._idx_fastB == 1
    assert config._idx_visible == 2
    print("✓ Canonical indices correct")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Validation - duplicate indices
print("\nTest 2: Validation - duplicate indices")
try:
    bad_config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        idx_fastA=0,
        idx_fastB=0,  # Duplicate!
        idx_visible=2
    )
    print("✗ Should have raised ValueError for duplicate indices")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Test 3: Validation - negative rank
print("\nTest 3: Validation - negative rank")
try:
    bad_config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=-1
    )
    print("✗ Should have raised ValueError for negative rank")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Test 4: Validation - wrong device count
print("\nTest 4: Validation - wrong device count")
try:
    bad_config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice()],  # Only 2 devices!
        rank=4
    )
    print("✗ Should have raised ValueError for wrong device count")
except ValueError as e:
    print(f"✓ Caught expected error: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Test 5: Rank chunking parameters
print("\nTest 5: Rank chunking parameters")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=8,
        rank_chunk=4,
        rank_offset=0
    )
    assert config.rank == 8
    assert config.rank_chunk == 4
    assert config.rank_offset == 0
    print("✓ Rank chunking parameters set correctly")
except Exception as e:
    print(f"✗ Test failed: {e}")

# Test 6: Step 1 - Removed fields (reset_policy, gamma)
print("\nTest 6: Removed fields should not cause errors")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4
    )
    # These attributes might exist from parent class but shouldn't be in __post_init__ validation
    print("✓ Configuration created without reset_policy/gamma issues")
except Exception as e:
    print(f"✗ Test failed: {e}")

# Test 7: Update rule is fixed to LR_TT
print("\nTest 7: Update rule fixed to LR_TT")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4
    )
    assert config.update_rule == "LR_TT"
    print("✓ update_rule property returns 'LR_TT'")
    
    # Try to set it to something else
    try:
        config.update_rule = "VANILLA_SGD"
        print("✗ Should have raised error when setting update_rule")
    except ValueError as e:
        print(f"✓ Caught expected error when setting update_rule: {e}")
    
except Exception as e:
    print(f"✗ Test failed: {e}")

print("\n" + "=" * 60)
print("Configuration tests completed!")
print("=" * 60)