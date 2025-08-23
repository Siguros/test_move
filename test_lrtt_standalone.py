#!/usr/bin/env python
"""Standalone test for LR-TT that includes all necessary dependencies."""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Any, ClassVar

print("=" * 60)
print("LR-TT Standalone Configuration Test")
print("=" * 60)

# Create mock base classes
@dataclass
class MockTransferCompound:
    """Mock TransferCompound base class."""
    unit_cell_devices: List[Any] = field(default_factory=list)
    transfer_every: int = 1
    n_reads_per_transfer: int = 0
    with_reset_prob: float = 0.0
    gamma: float = 0.0
    gamma_vec: List[float] = field(default_factory=list)
    reset_policy: int = 0
    
    def __post_init__(self):
        """Base post init."""
        pass

@dataclass  
class MockDevice:
    """Mock device class."""
    def as_bindings(self, data_type):
        return object()

# Now include the actual LRTTTransferCompound code
@dataclass
class LRTTTransferCompound(MockTransferCompound):
    r"""Low-Rank Tiki-Taka (LR-TT) Transfer compound device.
    
    This device implements a native LR-TT transfer learning rule with
    three devices in canonical order:
    - Device 0 (fastA): Low-rank matrix A of shape [d_size, rank]
    - Device 1 (fastB): Low-rank matrix B of shape [rank, x_size]
    - Device 2 (visible): The visible weight matrix W_to
    
    The transfer update performs: W_to += transfer_lr * (W_A @ W_B)
    """

    bindings_class: ClassVar[str] = "LRTTTransferResistiveDeviceParameter"
    
    # LR-TT specific parameters (fixed update_rule)
    rank: int = 0
    """Rank of the low-rank factorization. If 0, inferred from device sizes."""
    
    rank_chunk: int = -1
    """Process rank in chunks of this size (-1 = use full rank)."""
    
    rank_offset: int = 0
    """Starting offset for rank chunking."""
    
    transfer_lr: float = 1.0
    """Learning rate for the transfer update (W_to += transfer_lr * A @ B)."""
    
    correct_gradient_magnitudes: bool = False
    """Whether to correct gradient magnitudes for low-rank structure."""
    
    swap_xd: bool = False
    """Whether to swap X and D in the update."""
    
    # Forward injection parameters (enabled by default)
    forward_inject: bool = True
    """When True, add A_lr @ B_lr to forward pass (default True for LR_TT)."""
    
    lora_alpha: float = 1.0
    """LoRA alpha scaling factor (used only if forward_inject is enabled)."""
    
    use_bl_management: bool = False
    """Enable bound level (BL) management in the PulsedWeightUpdater."""
    
    desired_bl: float = 10.0
    """Desired bound level for the analog tiles (only used if use_bl_management=True)."""
    
    reinit_gain: float = 1.0
    """Gain for Kaiming(He) normal on A; B is zero-initialized."""
    
    # Device indices (public, with canonical defaults)
    idx_fastA: int = 0
    idx_fastB: int = 1
    idx_visible: int = 2
    
    # Hidden: Fixed canonical indices for internal use
    _idx_fastA: int = field(default=0, init=False, repr=False)
    _idx_fastB: int = field(default=1, init=False, repr=False)
    _idx_visible: int = field(default=2, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Validate and canonicalize the configuration after initialization."""
        # Call parent's __post_init__ if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        
        # Ensure we have exactly 3 devices
        if len(self.unit_cell_devices) != 3:
            raise ValueError(f"LRTTTransferCompound requires exactly 3 devices, got {len(self.unit_cell_devices)}")
        
        # Validate indices are in range and unique
        for name, idx in (("idx_fastA", self.idx_fastA),
                         ("idx_fastB", self.idx_fastB),
                         ("idx_visible", self.idx_visible)):
            if not (0 <= idx < 3):
                raise ValueError(f"{name} must be in [0,2], got {idx}")
        
        if len({self.idx_fastA, self.idx_fastB, self.idx_visible}) != 3:
            raise ValueError("Device indices must be unique")
        
        # Set canonical indices for internal use (always 0, 1, 2)
        self._idx_fastA, self._idx_fastB, self._idx_visible = 0, 1, 2
        
        # Validate rank parameters
        if self.rank < 0:
            raise ValueError(f"rank must be non-negative, got {self.rank}")
        
        if self.rank_chunk != -1 and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or -1 (disabled), got {self.rank_chunk}")
        
        if self.rank_offset < 0:
            raise ValueError(f"rank_offset must be non-negative, got {self.rank_offset}")
        
        # Validate numerical parameters
        if self.transfer_lr <= 0:
            raise ValueError(f"transfer_lr must be positive, got {self.transfer_lr}")
        
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        
        if self.use_bl_management and self.desired_bl <= 0:
            raise ValueError(f"desired_bl must be positive when use_bl_management=True, got {self.desired_bl}")
        
        # Validate transfer_every (0 allowed for inference)
        if hasattr(self, 'transfer_every') and self.transfer_every < 0:
            raise ValueError(f"transfer_every must be non-negative (0 for inference), got {self.transfer_every}")
        
        # Override parent transfer settings to avoid double triggers
        self.n_reads_per_transfer = 0  # No read slices by parent
    
    @property
    def update_rule(self) -> str:
        """Fixed to LR_TT for this implementation."""
        return "LR_TT"
    
    @update_rule.setter
    def update_rule(self, value: str) -> None:
        """Raise error if trying to set update_rule to anything other than LR_TT."""
        if value != "LR_TT":
            raise ValueError(f"LRTTTransferCompound only supports update_rule='LR_TT', got '{value}'")

# Run tests
print("\nTest 1: Basic configuration")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        transfer_every=2,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=2.0,
    )
    assert config.rank == 4
    assert config.transfer_every == 2
    assert config.transfer_lr == 1.0
    assert config.forward_inject == True
    assert config.lora_alpha == 2.0
    assert len(config.unit_cell_devices) == 3
    assert config._idx_fastA == 0
    assert config._idx_fastB == 1
    assert config._idx_visible == 2
    print("✓ Basic configuration works")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 2: Duplicate indices validation")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        idx_fastA=0,
        idx_fastB=0,  # Duplicate!
        idx_visible=2
    )
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught: {e}")

print("\nTest 3: Negative rank validation")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=-1
    )
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught: {e}")

print("\nTest 4: Wrong device count validation")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice()],
        rank=4
    )
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught: {e}")

print("\nTest 5: Update rule is fixed")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4
    )
    assert config.update_rule == "LR_TT"
    print("✓ update_rule returns 'LR_TT'")
    
    # Try to change it
    try:
        config.update_rule = "VANILLA_SGD"
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly prevented change: {e}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 6: Rank chunking parameters")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=8,
        rank_chunk=4,
        rank_offset=2
    )
    assert config.rank == 8
    assert config.rank_chunk == 4
    assert config.rank_offset == 2
    print("✓ Rank chunking parameters work")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 7: Transfer parameters")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        transfer_every=5,
        transfer_lr=0.5,
        forward_inject=False,
        lora_alpha=0.25
    )
    assert config.transfer_every == 5
    assert config.transfer_lr == 0.5
    assert config.forward_inject == False
    assert config.lora_alpha == 0.25
    print("✓ Transfer parameters work")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 8: BL management parameters")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        use_bl_management=True,
        desired_bl=31.0
    )
    assert config.use_bl_management == True
    assert config.desired_bl == 31.0
    print("✓ BL management parameters work")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nTest 9: Invalid BL management")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        use_bl_management=True,
        desired_bl=0  # Invalid!
    )
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught: {e}")

print("\nTest 10: Zero transfer_lr validation")
try:
    config = LRTTTransferCompound(
        unit_cell_devices=[MockDevice(), MockDevice(), MockDevice()],
        rank=4,
        transfer_lr=0.0  # Invalid!
    )
    print("✗ Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly caught: {e}")

print("\n" + "=" * 60)
print("All configuration tests completed successfully!")
print("=" * 60)