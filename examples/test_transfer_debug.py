#!/usr/bin/env python3
import torch
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda
import os

os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

# Prepare the datasets
x = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = torch.tensor([[1.0, 0.5], [0.7, 0.3]])

# Test with original aggressive parameters
device = ConstantStepDevice(dw_min=0.001, dw_min_dtod=0.0, up_down_dtod=0.0)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=5,  # Very frequent transfers to see the effect
    transfer_lr=1.0,   # Original high transfer LR
    forward_inject=True,
    lora_alpha=1.0,    # Original high alpha
    units_in_mbatch=False,
)
rpu_config = UnitCellRPUConfig(device=lrtt_config)
rpu_config.reinit_gain = 0.5

model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

opt = AnalogSGD(model.parameters(), lr=0.1)  # Original LR
opt.regroup_param_groups(model)

print("Testing transfer mechanism with aggressive parameters")
print("-" * 50)

# Get initial weights
tile = next(model.analog_tiles())
initial_weights = tile.get_weights()[0].clone()
print(f"Initial weights norm: {initial_weights.norm():.6f}")

# Train for just a few epochs around transfer
for epoch in range(12):
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    
    # Check weights before step
    weights_before = tile.get_weights()[0].clone()
    
    opt.step()
    
    # Check weights after step
    weights_after = tile.get_weights()[0].clone()
    weight_change = (weights_after - weights_before).norm()
    
    print(f"Epoch {epoch:2d} - Loss: {loss:.6f}, Weight change: {weight_change:.6f}")
    
    if (epoch + 1) % 5 == 0:
        print(f"  --> TRANSFER at epoch {epoch + 1}")
        print(f"      Weights norm after transfer: {weights_after.norm():.6f}")

print("\n" + "=" * 50)
print("Weight analysis:")
final_weights = tile.get_weights()[0]
print(f"Initial weights norm: {initial_weights.norm():.6f}")
print(f"Final weights norm: {final_weights.norm():.6f}")
print(f"Total change norm: {(final_weights - initial_weights).norm():.6f}")