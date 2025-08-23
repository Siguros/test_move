# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Linear layer with forward injection support."""

import torch
from typing import Optional
import warnings
from aihwkit.nn.modules.linear import AnalogLinear


class LRTTForwardInjectLinear(AnalogLinear):
    """AnalogLinear layer with LR-TT forward injection support.
    
    This layer extends AnalogLinear to support efficient forward pass
    using composed effective weights (W_eff = W_visible + α·A·B) when
    forward_inject is enabled and the underlying tile supports it.
    
    During training, it falls back to standard analog forward.
    During inference with forward_inject=True, it attempts to use
    the composed weights for faster digital computation.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        rpu_config,
        bias: bool = True, 
        tile_module_class: Optional[type] = None,
        inject_during_train: bool = True
    ):
        """Initialize LRTTForwardInjectLinear.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features  
            rpu_config: RPU configuration with LRTTTransferCompound device
            bias: Whether to use bias
            tile_module_class: Optional custom tile module class
            inject_during_train: Whether to use forward injection during training
        """
        super().__init__(
            in_features, 
            out_features, 
            bias=bias,
            rpu_config=rpu_config, 
            tile_module_class=tile_module_class
        )
        
        # Extract forward injection settings from config
        dev = getattr(rpu_config, "device", None)
        self._fi = bool(getattr(dev, "forward_inject", False)) if dev else False
        self._alpha = float(getattr(dev, "lora_alpha", 1.0)) if dev else 1.0
        self._inject_during_train = bool(inject_during_train)
        
        # Validate that this is an LRTT device if forward_inject is enabled
        if self._fi and dev and "LRTT" not in type(dev).__name__:
            raise ValueError(
                "forward_inject=True requires an LRTTTransferCompound device. "
                f"Got {type(dev).__name__}"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional forward injection.
        
        During training or when forward_inject is disabled, uses standard
        analog forward. During inference with forward_inject=True, attempts
        to use composed effective weights for faster computation.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Check if we should use forward injection
        use_injection = self._fi and (not self.training or self._inject_during_train)
        
        # Use standard analog forward if injection disabled
        if not use_injection:
            return super().forward(x)
        
        # Try to use composed weights for inference
        try:
            # Extract CUDA tile
            tile = None
            if hasattr(self, "analog_tiles") and callable(self.analog_tiles):
                tiles = list(self.analog_tiles())
                tile = tiles[0] if tiles else None
            elif hasattr(self, "analog_tile"):
                tile = self.analog_tile
            
            # Get underlying C++ tile
            cuda_tile = getattr(tile, "tile", None) if tile else None
            
            # Check if compose method is available
            if cuda_tile is None or not hasattr(cuda_tile, "lrtt_compose_w_eff"):
                # Fall back to analog forward if compose not available
                return super().forward(x)
            
            # Get composed effective weights
            w_eff = cuda_tile.lrtt_compose_w_eff(self._alpha)
            
            # Ensure weights are on same device as input
            if x.device != w_eff.device:
                w_eff = w_eff.to(x.device, non_blocking=True)
            
            # Perform digital matrix multiplication
            y = x.matmul(w_eff.t())
            
            # Add bias if present
            if self.bias is not None:
                y = y + self.bias
            
            return y
            
        except Exception as e:
            # Fall back to analog forward on any error
            warnings.warn(
                f"[LRTTForwardInjectLinear] Falling back to analog forward: {e}",
                RuntimeWarning
            )
            return super().forward(x)


def create_lrtt_linear_layer(
    in_features: int,
    out_features: int,
    rpu_config,
    bias: bool = True,
    forward_inject: bool = True,
    lora_alpha: float = 1.0
) -> AnalogLinear:
    """Factory function to create LRTT linear layer with optional forward injection.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        rpu_config: RPU configuration (should contain LRTTTransferCompound device)
        bias: Whether to use bias
        forward_inject: Whether to enable forward injection (inference optimization)
        lora_alpha: LoRA scaling factor for forward injection
        
    Returns:
        LRTTForwardInjectLinear if forward_inject=True, else standard AnalogLinear
    """
    # Set forward injection parameters on the config
    if hasattr(rpu_config, "device"):
        rpu_config.device.forward_inject = bool(forward_inject)
        rpu_config.device.lora_alpha = float(lora_alpha)
    
    # Return appropriate layer type
    if forward_inject:
        return LRTTForwardInjectLinear(
            in_features, 
            out_features, 
            rpu_config, 
            bias=bias
        )
    else:
        return AnalogLinear(
            in_features, 
            out_features,
            bias=bias,
            rpu_config=rpu_config
        )