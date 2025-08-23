"""Preset configurations for Low-Rank Tiki-Taka (LR-TT) Transfer learning.

This module provides factory functions for LR_TT-only configurations with
analog pulsed operation and forward injection enabled by default.
"""

from typing import Optional, Union
from dataclasses import dataclass, field

from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    InferenceRPUConfig,
    MappingParameter
)
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import (
    PulsedDevice,
    ConstantStepDevice,
    SoftBoundsDevice,
    LinearStepDevice,
    ExpStepDevice
)
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.enums import (
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    PulseType
)

# Import existing preset devices
from aihwkit.simulator.presets.devices import (
    EcRamPresetDevice,
    EcRamMOPresetDevice,
    IdealizedPresetDevice,
    CapacitorPresetDevice,
    ReRamESPresetDevice,
    ReRamSBPresetDevice,
    GokmenVlasovPresetDevice
)
from aihwkit.simulator.presets.utils import (
    PresetIOParameters,
    PresetUpdateParameters,
    StandardIOParameters
)
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles import AnalogTile


# ==================== Preset IO and Update Parameters ====================

@dataclass
class LRTTIOParameters(PresetIOParameters):
    """IO parameters optimized for LR-TT forward injection.
    
    Inherits from PresetIOParameters but can be customized for LR-TT specific needs.
    """
    # Use iterative bound management for better accuracy with forward injection
    bound_management: BoundManagementType = BoundManagementType.ITERATIVE
    noise_management: NoiseManagementType = NoiseManagementType.ABS_MAX
    
    # Can customize if needed for LR-TT
    out_bound: float = 20.0  # May need adjustment based on rank and lora_alpha


@dataclass  
class LRTTUpdateParameters(PresetUpdateParameters):
    """Update parameters optimized for LR-TT learning.
    
    Inherits from PresetUpdateParameters with LR-TT optimizations.
    """
    # Stochastic pulses work well with low-rank updates
    pulse_type: PulseType = PulseType.STOCHASTIC_COMPRESSED
    
    # Adjust pulse length for low-rank updates
    desired_bl: int = 31
    update_bl_management: bool = True
    update_management: bool = True


# ==================== Main LR-TT Configuration Function ====================

def lrtt_config(
    rank: int = 8,
    transfer_every: int = 1,
    transfer_lr: float = 1.0,
    forward_inject: bool = True,  # Always True for LR_TT-only
    lora_alpha: float = 1.0,
    # Step 1: reset_policy and gamma removed, only reinit remains
    # Device selection
    fast_device_type: str = "idealized",  # For A and B matrices  
    visible_device_type: str = "ecram",   # For visible weights
    # BL management
    use_bl_management: bool = False,
    desired_bl: float = 31.0,
    # Additional parameters
    correct_gradient_magnitudes: bool = False,
    rank_chunk: int = -1,
    rank_offset: int = 0,
    swap_xd: bool = False,
    # IO and Update parameters
    io_parameters: Optional[IOParameters] = None,
    update_parameters: Optional[UpdateParameters] = None,
    # Mapping
    mapping: Optional[MappingParameter] = None,
    **device_kwargs
) -> UnitCellRPUConfig:
    """Create an LR_TT-only configuration with canonical device order.
    
    Creates LRTTTransferCompound with devices in canonical order [fastA, fastB, visible]
    and forward_inject enabled by default for easy testing.
    
    Args:
        rank: Rank of the low-rank decomposition
        transfer_every: Transfer frequency (0 for inference)
        transfer_lr: Learning rate for transfer updates
        forward_inject: Enable forward injection (always True for LR_TT)
        lora_alpha: LoRA scaling factor for forward injection
        # Step 1: reset_policy and gamma removed - only reinit after transfer
        fast_device_type: Device type for fast A/B matrices 
            ('idealized', 'ecram', 'ecram_mo', 'capacitor', 'reram_es', 'reram_sb')
        visible_device_type: Device type for visible weights
            ('ecram', 'idealized', 'ecram_mo', 'capacitor', 'reram_es', 'reram_sb')
        use_bl_management: Enable bound level management
        desired_bl: Desired bound level (if BL management enabled)
        correct_gradient_magnitudes: Correct gradient magnitudes for low-rank
        rank_chunk: Process rank in chunks (-1 for full rank)
        rank_offset: Starting offset for rank chunking
        swap_xd: Swap X and D in update
        io_parameters: Custom IO parameters (defaults to LRTTIOParameters)
        update_parameters: Custom update parameters (defaults to LRTTUpdateParameters)
        mapping: Mapping parameters
        **device_kwargs: Additional arguments passed to device constructors
    
    Returns:
        UnitCellRPUConfig with LR-TT transfer compound device
    """
    # Enforce forward_inject=True for LR_TT-only mode
    if not forward_inject:
        print("Warning: forward_inject forced to True for LR_TT-only mode")
        forward_inject = True
    
    # Device type mapping
    device_map = {
        "idealized": IdealizedPresetDevice,
        "ecram": EcRamPresetDevice,
        "ecram_mo": EcRamMOPresetDevice,
        "capacitor": CapacitorPresetDevice,
        "reram_es": ReRamESPresetDevice,
        "reram_sb": ReRamSBPresetDevice,
        "gokmen_vlasov": GokmenVlasovPresetDevice,
        # Basic devices
        "constant_step": ConstantStepDevice,
        "soft_bounds": SoftBoundsDevice,
        "linear_step": LinearStepDevice,
        "exp_step": ExpStepDevice
    }
    
    # Create fast devices (A and B matrices)
    if fast_device_type not in device_map:
        raise ValueError(f"Unknown fast_device_type: {fast_device_type}")
    FastDeviceClass = device_map[fast_device_type]
    
    # Create visible device
    if visible_device_type not in device_map:
        raise ValueError(f"Unknown visible_device_type: {visible_device_type}")
    VisibleDeviceClass = device_map[visible_device_type]
    
    # Create three devices in canonical order: [fastA, fastB, visible]
    # Fast devices can use simpler/idealized devices since they're auxiliary
    fastA = FastDeviceClass(**device_kwargs)
    fastB = FastDeviceClass(**device_kwargs)
    # Visible device should use more realistic device
    visible = VisibleDeviceClass(**device_kwargs)
    
    # Create LR-TT compound with canonical order
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[fastA, fastB, visible],
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        forward_inject=forward_inject,  # Always True
        lora_alpha=lora_alpha,
        # Step 1: reset_policy and gamma removed
        use_bl_management=use_bl_management,
        desired_bl=desired_bl,
        correct_gradient_magnitudes=correct_gradient_magnitudes,
        rank_chunk=rank_chunk,
        rank_offset=rank_offset,
        swap_xd=swap_xd
    )
    
    # Use provided or default IO/Update parameters
    io_params = io_parameters or LRTTIOParameters()
    update_params = update_parameters or LRTTUpdateParameters()
    
    return UnitCellRPUConfig(
        device=lrtt,
        forward=io_params,
        backward=io_params,
        update=update_params,
        mapping=mapping or MappingParameter()
    )


# ==================== Specific Preset Configurations ====================

def lrtt_idealized(rank: int = 8) -> UnitCellRPUConfig:
    """LR-TT with idealized devices (perfect, no noise).
    
    Best for initial testing and debugging.
    
    Args:
        rank: Rank of the low-rank decomposition
    
    Returns:
        UnitCellRPUConfig with idealized LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="idealized",
        visible_device_type="idealized",
        transfer_every=1,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed
    )


def lrtt_ecram(rank: int = 16) -> UnitCellRPUConfig:
    """LR-TT with ECRAM devices (realistic).
    
    Uses ECRAM devices for all weights to avoid device mixing issues.
    
    Args:
        rank: Rank of the low-rank decomposition
    
    Returns:
        UnitCellRPUConfig with ECRAM LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="ecram",      # Use ECRAM for consistency
        visible_device_type="ecram",    # Use ECRAM for visible weights
        transfer_every=1,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed,
        # Step 1: gamma removed,
        use_bl_management=True,
        desired_bl=31
    )


def lrtt_ecram_mo(rank: int = 16) -> UnitCellRPUConfig:
    """LR-TT with metal-oxide ECRAM devices.
    
    Uses MO-ECRAM devices for all weights to avoid device mixing issues.
    
    Args:
        rank: Rank of the low-rank decomposition
    
    Returns:
        UnitCellRPUConfig with MO-ECRAM LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="ecram_mo",      # MO-ECRAM for consistency
        visible_device_type="ecram_mo",   # MO-ECRAM for visible weights
        transfer_every=1,
        transfer_lr=0.5,  # Lower LR for MO-ECRAM
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed,
        # Step 1: gamma removed,
        use_bl_management=True,
        desired_bl=31
    )


def lrtt_reram(rank: int = 8) -> UnitCellRPUConfig:
    """LR-TT with ReRAM devices.
    
    Uses soft-bounds ReRAM for all devices.
    
    Args:
        rank: Rank of the low-rank decomposition
    
    Returns:
        UnitCellRPUConfig with ReRAM LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="reram_sb",
        visible_device_type="reram_sb",
        transfer_every=2,  # Less frequent transfer for ReRAM
        transfer_lr=0.8,
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed,
        # Step 1: gamma removed,
        use_bl_management=True,
        desired_bl=31
    )


def lrtt_capacitor(rank: int = 8) -> UnitCellRPUConfig:
    """LR-TT with capacitor devices (with leakage).
    
    Uses capacitor devices which include leakage modeling.
    
    Args:
        rank: Rank of the low-rank decomposition
    
    Returns:
        UnitCellRPUConfig with capacitor LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="capacitor",
        visible_device_type="capacitor",
        transfer_every=1,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed,  # Capacitors discharge
        use_bl_management=False
    )


def lrtt_mixed_precision(
    rank: int = 16,
    fast_bits: int = 4,
    visible_bits: int = 8
) -> UnitCellRPUConfig:
    """LR-TT with mixed precision devices.
    
    Uses lower precision for fast weights and higher for visible.
    
    Args:
        rank: Rank of the low-rank decomposition
        fast_bits: Bit precision for fast A/B matrices (affects dw_min)
        visible_bits: Bit precision for visible weights
    
    Returns:
        UnitCellRPUConfig with mixed precision LR-TT configuration
    """
    # Calculate dw_min based on bit precision
    fast_dw_min = 2.0 / (2**fast_bits - 1)
    visible_dw_min = 2.0 / (2**visible_bits - 1)
    
    return lrtt_config(
        rank=rank,
        fast_device_type="constant_step",
        visible_device_type="constant_step",
        transfer_every=1,
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=1.0,
        # Step 1: reset_policy removed,
        # Device-specific kwargs
        dw_min=visible_dw_min  # This goes to visible device
    )


def lrtt_lora_style(
    rank: int = 8,
    alpha: float = 16.0
) -> UnitCellRPUConfig:
    """LR-TT configured to mimic LoRA behavior.
    
    Args:
        rank: Rank of the low-rank decomposition
        alpha: LoRA scaling factor
    
    Returns:
        UnitCellRPUConfig with LoRA-style LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        fast_device_type="idealized",
        visible_device_type="idealized",
        transfer_every=1,  # Update every step
        transfer_lr=1.0,
        forward_inject=True,
        lora_alpha=alpha,
        # Step 1: reset_policy removed,  # LoRA doesn't decay
        correct_gradient_magnitudes=True,  # Important for LoRA
        # Small weight range for LoRA-style updates
        w_min=-0.1,
        w_max=0.1,
        dw_min=0.0001
    )


def lrtt_inference(
    rank: int = 16,
    device_type: str = "ecram",
    perfect_forward: bool = False
) -> InferenceRPUConfig:
    """LR-TT configuration for inference only.
    
    Args:
        rank: Rank of the low-rank decomposition
        device_type: Type of device to use
        perfect_forward: Use perfect forward pass (no analog noise)
    
    Returns:
        InferenceRPUConfig with LR-TT for inference
    """
    # Get device class
    device_map = {
        "idealized": IdealizedPresetDevice,
        "ecram": EcRamPresetDevice,
        "ecram_mo": EcRamMOPresetDevice,
        "reram_sb": ReRamSBPresetDevice
    }
    
    DeviceClass = device_map.get(device_type, EcRamPresetDevice)
    
    # Create base LR-TT config with transfer disabled
    lrtt = LRTTTransferCompound(
        unit_cell_devices=[
            IdealizedPresetDevice(),  # Fast A
            IdealizedPresetDevice(),  # Fast B  
            DeviceClass()            # Visible
        ],
        rank=rank,
        transfer_every=0,  # No transfer during inference
        forward_inject=True,
        lora_alpha=1.0
    )
    
    # Wrap in inference config
    io_params = IOParameters(is_perfect=perfect_forward) if perfect_forward else LRTTIOParameters()
    
    return InferenceRPUConfig(
        device=lrtt,
        forward=io_params,
        noise_model=None  # No inference noise
    )


# ==================== Validation and Helper Functions ====================

def validate_lrtt_config(config: Union[UnitCellRPUConfig, InferenceRPUConfig]) -> bool:
    """Validate an LR-TT configuration.
    
    Checks that:
    - Device is LRTTTransferCompound
    - update_rule is LR_TT (implicitly enforced)
    - Rank > 0
    - Exactly 3 devices
    - forward_inject is True
    
    Args:
        config: Configuration to validate
    
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    device = config.device
    
    if not isinstance(device, LRTTTransferCompound):
        raise ValueError("Configuration must use LRTTTransferCompound device")
    
    # Check device count
    if len(device.unit_cell_devices) != 3:
        raise ValueError("LR-TT requires exactly 3 devices")
    
    # Check rank
    if device.rank <= 0 and device.rank != 0:  # 0 means auto-infer
        raise ValueError(f"Rank must be positive or 0 (auto), got {device.rank}")
    
    # Check forward_inject is enabled
    if not device.forward_inject:
        raise ValueError("forward_inject must be True for LR_TT-only mode")
    
    # update_rule is enforced to be LR_TT by the class
    if device.update_rule != "LR_TT":
        raise ValueError(f"update_rule must be LR_TT, got {device.update_rule}")
    
    return True


def extract_cuda_tile_from_layer(layer: AnalogLinear) -> Optional[object]:
    """Extract CUDA tile with lrtt_compose_w_eff from an AnalogLinear layer.
    
    This helper robustly fetches the underlying CUDA tile that may have
    the lrtt_compose_w_eff(alpha) method. It handles both direct CUDA tiles
    and CPU tiles that need wrapping.
    
    Args:
        layer: AnalogLinear instance
    
    Returns:
        CUDA tile object with potential lrtt_compose_w_eff method, or None
        
    Note:
        The lrtt_compose_w_eff method may not be exposed in Python bindings
        for all tile types. This is expected behavior.
    """
    # Try analog_tiles first (newer API), then analog_tile
    tile = None
    if hasattr(layer, 'analog_tiles'):
        if callable(layer.analog_tiles):
            # It's a method that returns a generator
            tiles = list(layer.analog_tiles())
            tile = tiles[0] if tiles else None
        else:
            # It's a property/attribute
            tile = layer.analog_tiles[0] if len(layer.analog_tiles) > 0 else None
    elif hasattr(layer, 'analog_tile'):
        tile = layer.analog_tile
    
    if tile is None:
        return None
    
    # Get the underlying C++ tile
    if hasattr(tile, 'tile'):
        cpp_tile = tile.tile
        
        # Check if it's already a CUDA tile
        if 'Cuda' in cpp_tile.__class__.__name__:
            return cpp_tile
        
        # Try to wrap CPU tile as CUDA tile if needed
        try:
            from aihwkit.simulator.rpu_base import tiles
            if hasattr(tiles, 'CudaAnalogTile'):
                # Note: This wrapping may not work for all configurations
                # The tile needs to be created on CUDA from the start
                return cpp_tile
        except Exception:
            pass
    
    return cpp_tile if hasattr(tile, 'tile') else None


# ==================== Convenience Aliases ====================

# Default configurations
lrtt_default = lrtt_idealized  # Use idealized as default for testing

# Realistic configurations  
lrtt_ecram_ideal = lrtt_ecram  # Alias for backward compatibility
lrtt_ecram_realistic = lrtt_ecram_mo  # MO-ECRAM is more realistic

# Chunked configuration
def lrtt_chunked(
    rank: int = 64,
    chunk_size: int = 8,
    device_type: str = "ecram"
) -> UnitCellRPUConfig:
    """LR-TT with rank chunking for memory efficiency.
    
    Args:
        rank: Total rank of the decomposition
        chunk_size: Size of each rank chunk
        device_type: Device type to use
    
    Returns:
        UnitCellRPUConfig with chunked LR-TT configuration
    """
    return lrtt_config(
        rank=rank,
        rank_chunk=chunk_size,
        fast_device_type="idealized",
        visible_device_type=device_type,
        # Step 1: reset_policy removed,
        # Step 1: gamma removed,
        transfer_lr=0.1,
        transfer_every=chunk_size,  # Transfer after processing all chunks
        forward_inject=True,
        lora_alpha=1.0
    )


# Export key functions
__all__ = [
    # Main config function
    'lrtt_config',
    # Preset configurations
    'lrtt_idealized',
    'lrtt_ecram',
    'lrtt_ecram_mo',
    'lrtt_reram',
    'lrtt_capacitor',
    'lrtt_mixed_precision',
    'lrtt_lora_style',
    'lrtt_chunked',
    'lrtt_inference',
    # Aliases
    'lrtt_default',
    'lrtt_ecram_ideal',
    'lrtt_ecram_realistic',
    # Utilities
    'validate_lrtt_config',
    'extract_cuda_tile_from_layer',
    # Parameter classes
    'LRTTIOParameters',
    'LRTTUpdateParameters'
]