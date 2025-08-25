#!/usr/bin/env python
"""Test that all LRTT parameters are correctly passed from Python to CUDA."""

import os
os.environ["AIHWKIT_DEBUG_LRTT"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.lrtt_compound import LRTTTransferCompound
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import AnalogLinear

def test_all_lrtt_parameters():
    """Test all LRTT parameters are correctly passed to CUDA."""
    
    print("=" * 80)
    print("Testing ALL LRTT Parameter Passing")
    print("=" * 80)
    
    # Test various parameter combinations
    test_configs = [
        {
            "name": "Test 1: transfer_lr=0.01, transfer_every=5",
            "transfer_lr": 0.01,
            "transfer_every": 5,
            "rank": 16,
            "ab_use_bl_management": True,
            "ab_desired_bl": 2.0,
            "transfer_use_bl_management": False,
            "transfer_desired_bl": -1.0,
            "rank_chunk": 8,
            "rank_offset": 0,
            "correct_gradient_magnitudes": True,
            "swap_xd": False,
            "forward_inject": True,
            "lora_alpha": 0.5,
            "reinit_gain": 1.5,
        },
        {
            "name": "Test 2: transfer_lr=0.5, bl management enabled",
            "transfer_lr": 0.5,
            "transfer_every": 10,
            "rank": 8,
            "ab_use_bl_management": False,
            "ab_desired_bl": -1.0,
            "transfer_use_bl_management": True,
            "transfer_desired_bl": 3.0,
            "rank_chunk": -1,  # disabled
            "rank_offset": 4,
            "correct_gradient_magnitudes": False,
            "swap_xd": True,
            "forward_inject": False,
            "lora_alpha": 2.0,
            "reinit_gain": 0.5,
        }
    ]
    
    for config in test_configs:
        print("\n" + "=" * 60)
        print(config["name"])
        print("=" * 60)
        
        # Create the configuration
        rpu_config = InferenceRPUConfig()
        rpu_config.mapping.max_input_size = 512
        rpu_config.mapping.max_output_size = 512
        
        rpu_config.device = LRTTTransferCompound(
            unit_cell_devices=[
                ConstantStepDevice(),  # fastA
                ConstantStepDevice(),  # fastB  
                ConstantStepDevice(),  # visible
            ],
            transfer_lr=config["transfer_lr"],
            transfer_every=config["transfer_every"],
            rank=config["rank"],
            ab_use_bl_management=config["ab_use_bl_management"],
            ab_desired_bl=config["ab_desired_bl"],
            transfer_use_bl_management=config["transfer_use_bl_management"],
            transfer_desired_bl=config["transfer_desired_bl"],
            rank_chunk=config["rank_chunk"],
            rank_offset=config["rank_offset"],
            correct_gradient_magnitudes=config["correct_gradient_magnitudes"],
            swap_xd=config["swap_xd"],
            forward_inject=config["forward_inject"],
            lora_alpha=config["lora_alpha"],
            reinit_gain=config["reinit_gain"],
        )
        
        # Print expected values
        print(f"\nExpected Python values:")
        print(f"  transfer_lr: {config['transfer_lr']}")
        print(f"  transfer_every: {config['transfer_every']}")
        print(f"  rank: {config['rank']}")
        print(f"  ab_use_bl_management: {config['ab_use_bl_management']}")
        print(f"  ab_desired_bl: {config['ab_desired_bl']}")
        print(f"  transfer_use_bl_management: {config['transfer_use_bl_management']}")
        print(f"  transfer_desired_bl: {config['transfer_desired_bl']}")
        print(f"  rank_chunk: {config['rank_chunk']}")
        print(f"  rank_offset: {config['rank_offset']}")
        print(f"  correct_gradient_magnitudes: {config['correct_gradient_magnitudes']}")
        print(f"  swap_xd: {config['swap_xd']}")
        print(f"  forward_inject: {config['forward_inject']}")
        print(f"  lora_alpha: {config['lora_alpha']}")
        print(f"  reinit_gain: {config['reinit_gain']}")
        
        # Create analog layer
        layer = AnalogLinear(128, 64, rpu_config=rpu_config)
        layer = layer.cuda()
        
        # Trigger a forward pass (will show CUDA debug output)
        print("\nTriggering forward pass (check CUDA debug output)...")
        x = torch.randn(32, 128).cuda()
        y = layer(x)
        
        # Do an update to trigger transfer
        print("\nTriggering update (will check transfer parameters)...")
        grad = torch.randn_like(y)
        y.backward(grad)
        
        # Do multiple updates to trigger transfer_every
        for i in range(config["transfer_every"] + 1):
            x = torch.randn(32, 128).cuda()
            y = layer(x)
            grad = torch.randn_like(y)
            y.backward(grad)
        
        print("\nTest completed for this configuration")

if __name__ == "__main__":
    test_all_lrtt_parameters()
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED - Check debug output above")
    print("=" * 80)