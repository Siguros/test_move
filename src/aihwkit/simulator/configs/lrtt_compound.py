# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Configuration for Low-Rank Tiki-Taka (LR-TT) Transfer Compound device."""

from dataclasses import dataclass, field
from typing import ClassVar, Any, List

from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.devices import PulsedDevice
from aihwkit.simulator.parameters.enums import RPUDataType


@dataclass
class LRTTTransferCompound(TransferCompound):
    r"""Low-Rank Tiki-Taka (LR-TT) Transfer compound device.
    
    This device implements a native LR-TT transfer learning rule with
    three devices in canonical order:
    - Device 0 (fastA): Low-rank matrix A of shape [d_size, rank]
    - Device 1 (fastB): Low-rank matrix B of shape [rank, x_size]
    - Device 2 (visible): The visible weight matrix W_to
    
    The transfer update performs: W_to += transfer_lr * (W_A @ W_B)
    
    This allows efficient low-rank adaptation while keeping the main
    weights in analog memory, following the Tiki-Taka algorithm but
    with a low-rank factorization for the fast weights.
    
    Note:
        This is an LR_TT-only implementation. The update_rule is fixed to "LR_TT"
        and cannot be changed. Device order is canonicalized to [fastA, fastB, visible].
    
    Args:
        unit_cell_devices: Must contain exactly 3 devices (order will be canonicalized)
        rank: Rank of the low-rank factorization (inferred from device sizes if 0)
        transfer_lr: Learning rate for the transfer update
        transfer_every: Transfer frequency (0 for inference, >0 for training)
        # Step 1: reset_policy and gamma removed - only reinit after transfer
        forward_inject: Enable forward injection (default: True)
        lora_alpha: LoRA alpha scaling factor for forward injection
        correct_gradient_magnitudes: Whether to correct gradient magnitudes
        swap_xd: Whether to swap X and D in the update
        desired_bl: Desired bound level for the analog tiles
        use_bl_management: Enable bound level management
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
    
    transfer_every: int = 1
    """Transfer frequency (0 for inference, >0 for training)."""
    
    # Step 1: reset_policy and gamma removed - only reinit remains
    
    correct_gradient_magnitudes: bool = False
    """Whether to correct gradient magnitudes for low-rank structure."""
    
    swap_xd: bool = False
    """Whether to swap X and D in the update."""
    
    # Forward injection parameters (enabled by default)
    forward_inject: bool = True
    """When True, add A_lr @ B_lr to forward pass (default True for LR_TT)."""
    
    lora_alpha: float = 1.0
    """LoRA alpha scaling factor (used only if forward_inject is enabled)."""
    
    # A/B training update BL management controls
    ab_use_bl_management: bool = True
    """Enable BL management for A/B training updates (default: True for backward compatibility)."""
    
    ab_use_update_management: bool = True
    """Enable update management for A/B training updates."""
    
    ab_desired_bl: float = -1.0
    """Desired bound level for A/B training updates (-1 = no override, use device default)."""
    
    # Transfer step BL management controls (A@B -> visible)
    transfer_use_bl_management: bool = False
    """Enable BL management for transfer step (default: False for linear scaling)."""
    
    transfer_use_update_management: bool = False
    """Enable update management for transfer step."""
    
    transfer_desired_bl: float = -1.0
    """Desired bound level for transfer step (-1 = no override)."""
    
    # Digital transfer removed - only pulsed stochastic updates are used
    
    # Legacy parameters (deprecated, mapped to ab_* for backward compatibility)
    use_bl_management: bool = False
    """DEPRECATED: Use ab_use_bl_management instead."""
    
    desired_bl: float = 1.0
    """DEPRECATED: Use ab_desired_bl instead."""
    
    # Step 1: Only reinit_gain remains from reinit parameters
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
        
        # Step 1: reset_policy validation removed
        
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
        
        # Validate A/B update BL settings (sentinel value -1 means no override)
        if self.ab_use_bl_management and self.ab_desired_bl != -1.0 and self.ab_desired_bl <= 0:
            raise ValueError(f"ab_desired_bl must be positive or -1 (no override) when ab_use_bl_management=True, got {self.ab_desired_bl}")
        
        # Validate transfer BL settings (sentinel value -1 means no override)
        if self.transfer_use_bl_management and self.transfer_desired_bl != -1.0 and self.transfer_desired_bl <= 0:
            raise ValueError(f"transfer_desired_bl must be positive or -1 (no override) when transfer_use_bl_management=True, got {self.transfer_desired_bl}")
        
        # Handle legacy parameters for backward compatibility
        if self.use_bl_management and not hasattr(self, '_legacy_warning_shown'):
            import warnings
            warnings.warn("use_bl_management is deprecated. Use ab_use_bl_management instead.", DeprecationWarning)
            self._legacy_warning_shown = True
            self.ab_use_bl_management = self.use_bl_management
        
        if self.desired_bl != 1.0 and not hasattr(self, '_legacy_bl_warning_shown'):
            import warnings
            warnings.warn("desired_bl is deprecated. Use ab_desired_bl instead.", DeprecationWarning)
            self._legacy_bl_warning_shown = True
            self.ab_desired_bl = self.desired_bl
        
        # Validate transfer_every (0 allowed for inference)
        if hasattr(self, 'transfer_every') and self.transfer_every < 0:
            raise ValueError(f"transfer_every must be non-negative (0 for inference), got {self.transfer_every}")
        
        # Override parent transfer settings to avoid double triggers
        self.n_reads_per_transfer = 0  # No read slices by parent
        
        # Set canonical indices
        self._idx_fastA = 0
        self._idx_fastB = 1
        self._idx_visible = 2
    
    def as_bindings(self, data_type: RPUDataType) -> Any:
        """Return a representation of this instance as a simulator bindings object.
        
        This creates the C++ LRTTTransferResistiveDeviceParameter with devices
        in canonical order [fastA, fastB, visible].
        """
        from aihwkit.simulator.parameters.helpers import parameters_to_bindings
        from aihwkit.exceptions import ConfigError
        
        if not isinstance(self.unit_cell_devices, list):
            raise ConfigError("unit_cell_devices should be a list of devices")
        
        if len(self.unit_cell_devices) != 3:
            raise ConfigError("LRTTTransferCompound requires exactly 3 devices")
        
        # Import the bindings module
        from aihwkit.simulator import rpu_base
        
        # Create the LRTTTransferResistiveDeviceParameter
        if data_type == RPUDataType.FLOAT:
            lrtt_params = rpu_base.devices.LRTTTransferResistiveDeviceParameter()
        else:
            # For double precision if supported
            lrtt_params = rpu_base.devices.LRTTTransferResistiveDeviceParameterDouble()
        
        # Add device parameters based on user-specified indices to canonical order
        # This ensures devices are added in the order the user specified via indices
        order = [self.idx_fastA, self.idx_fastB, self.idx_visible]
        for i in order:
            device_params = self.unit_cell_devices[i].as_bindings(data_type)
            if not lrtt_params.append_parameter(device_params):
                raise ConfigError(f"Could not add unit cell device parameter at index {i}")
        
        # Set canonical indices
        lrtt_params.idx_fastA = 0
        lrtt_params.idx_fastB = 1
        lrtt_params.idx_visible = 2
        
        # Set LR-TT specific parameters
        lrtt_params.rank = self.rank
        
        # Set chunk parameters if available
        if hasattr(lrtt_params, 'rank_chunk'):
            lrtt_params.rank_chunk = self.rank_chunk
        if hasattr(lrtt_params, 'rank_offset'):
            lrtt_params.rank_offset = self.rank_offset
        
        # Transfer and update parameters
        lrtt_params.transfer_lr = self.transfer_lr
        # Note: reset_policy and gamma were removed - only reinit_gain remains
        
        # Optional parameters
        if hasattr(lrtt_params, 'correct_gradient_magnitudes'):
            lrtt_params.correct_gradient_magnitudes = self.correct_gradient_magnitudes
        if hasattr(lrtt_params, 'swap_xd'):
            lrtt_params.swap_xd = self.swap_xd
            
        # Map Python config to C++ parameters
        # A/B update BL management
        if hasattr(lrtt_params, 'ab_use_bl_management'):
            lrtt_params.ab_use_bl_management = self.ab_use_bl_management
        if hasattr(lrtt_params, 'ab_use_update_management'):
            lrtt_params.ab_use_update_management = self.ab_use_update_management
        if hasattr(lrtt_params, 'ab_desired_bl'):
            lrtt_params.ab_desired_bl = self.ab_desired_bl
        
        # Transfer BL management
        if hasattr(lrtt_params, 'transfer_use_bl_management'):
            lrtt_params.transfer_use_bl_management = self.transfer_use_bl_management
        if hasattr(lrtt_params, 'transfer_use_update_management'):
            lrtt_params.transfer_use_update_management = self.transfer_use_update_management
        if hasattr(lrtt_params, 'transfer_desired_bl'):
            lrtt_params.transfer_desired_bl = self.transfer_desired_bl
        # Digital transfer removed - only pulsed stochastic updates are used
        
        # Legacy support (map to C++ legacy parameters)
        if hasattr(lrtt_params, 'use_bl_management'):
            lrtt_params.use_bl_management = self.use_bl_management
        if hasattr(lrtt_params, 'desired_BL'):
            lrtt_params.desired_BL = self.desired_bl
        
        # Forward injection parameters
        if hasattr(lrtt_params, 'forward_inject'):
            lrtt_params.forward_inject = self.forward_inject
        if hasattr(lrtt_params, 'lora_alpha'):
            lrtt_params.lora_alpha = self.lora_alpha
        
        # Fixed update rule for LR_TT
        if hasattr(lrtt_params, 'update_rule'):
            lrtt_params.update_rule = "LR_TT"
        
        # Reinit parameters if they exist
        if hasattr(lrtt_params, 'reinit_gain'):
            lrtt_params.reinit_gain = self.reinit_gain
        if hasattr(lrtt_params, 'reinit_only_lr_subspace'):
            lrtt_params.reinit_only_lr_subspace = self.reinit_only_lr_subspace
        if hasattr(lrtt_params, 'reinit_randomize_A'):
            lrtt_params.reinit_randomize_A = self.reinit_randomize_A
        if hasattr(lrtt_params, 'reinit_zero_B'):
            lrtt_params.reinit_zero_B = self.reinit_zero_B
        
        # Transfer compound parameters
        lrtt_params.transfer_every = self.transfer_every if self.transfer_every >= 0 else 0
        lrtt_params.n_reads_per_transfer = self.n_reads_per_transfer
        lrtt_params.with_reset_prob = self.with_reset_prob
        
        # Do not override LR-TT defaults from C++.
        # C++ sets gamma_vec = [1, 1, 0] for [fastA, fastB, visible].
        # Only set if truly needed and not already initialized
        if hasattr(lrtt_params, "gamma_vec"):
            if not lrtt_params.gamma_vec:
                # Set proper LR-TT convention: A and B active, visible off
                lrtt_params.gamma_vec = [1.0, 1.0, 0.0]
        
        return lrtt_params
    
    @property
    def update_rule(self) -> str:
        """Fixed to LR_TT for this implementation."""
        return "LR_TT"
    
    @update_rule.setter
    def update_rule(self, value: str) -> None:
        """Raise error if trying to set update_rule to anything other than LR_TT."""
        if value != "LR_TT":
            raise ValueError(f"LRTTTransferCompound only supports update_rule='LR_TT', got '{value}'")