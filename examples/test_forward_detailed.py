#!/usr/bin/env python3
import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

# Configuration
device = ConstantStepDevice(dw_min=0.001, dw_min_dtod=0.0, up_down_dtod=0.0)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=10,
    transfer_lr=1,
    forward_inject=True,
    lora_alpha=1.0,
    units_in_mbatch=False,
)
rpu_config = UnitCellRPUConfig(device=lrtt_config)
rpu_config.reinit_gain = 0.5

# Create model
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Prepare input
x = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])

if cuda.is_compiled():
    x = x.cuda()
    model = model.cuda()

print("=" * 50)
print("Testing forward with forward_inject=True")
print("=" * 50)

# Get initial weights
tile = next(model.analog_tiles())
# For LRTT, get_weights returns the visible weights
visible_weights = tile.get_weights()[0]

print(f"\nVisible weights shape: {visible_weights.shape}")
print(f"Visible weights:\n{visible_weights}")
print(f"Visible weights norm: {visible_weights.norm():.6f}")

# Forward pass
print("\n" + "=" * 50)
print("Running forward pass...")
print("=" * 50)
print(f"\nInput shape: {x.shape}")
print(f"Input:\n{x}")

output = model(x)
print(f"\nOutput shape: {output.shape}")
print(f"Output:\n{output}")
print(f"Output norm: {output.norm():.6f}")

# Manual computation
print("\n" + "=" * 50)
print("Manual computation check:")
print("=" * 50)
manual_output = x @ visible_weights.T
print(f"Manual output (x @ W_visible.T):\n{manual_output}")
print(f"Manual output norm: {manual_output.norm():.6f}")

# Check if output is zero
if output.norm() < 1e-6:
    print("\n⚠️ WARNING: Output is effectively zero!")
else:
    print(f"\n✓ Output is non-zero (norm: {output.norm():.6f})")