import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda
import os

# Enable debug
os.environ['AIHWKIT_DEBUG_LRTT'] = '1'

# Prepare the datasets
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

print("Debug forward_inject operation")
print("="*60)

# Define LRTT config with forward_inject=True
device = ConstantStepDevice(dw_min=0.001, dw_min_dtod=0.0, up_down_dtod=0.0)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=10,
    transfer_lr=1,
    forward_inject=True,  # Using forward injection
    lora_alpha=1.0,
    units_in_mbatch=False,
)
rpu_config = UnitCellRPUConfig(device=lrtt_config)
rpu_config.reinit_gain = 0.5
model_inject = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

# Also create one without forward_inject for comparison
lrtt_config_no_inject = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=10,
    transfer_lr=1,
    forward_inject=False,  # NOT using forward injection
    lora_alpha=1.0,
    units_in_mbatch=False,
)
rpu_config_no_inject = UnitCellRPUConfig(device=lrtt_config_no_inject)
rpu_config_no_inject.reinit_gain = 0.5
model_no_inject = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_no_inject)

if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model_inject = model_inject.cuda()
    model_no_inject = model_no_inject.cuda()

print("Model WITH forward_inject=True:")
print("-" * 40)
weights_inject = model_inject.get_weights()[0]
print(f"Visible weights norm: {torch.norm(weights_inject):.6f}")
print(f"Visible weights:\n{weights_inject}")

with torch.no_grad():
    pred_inject = model_inject(x)
    print(f"Prediction: {pred_inject}")
    loss_inject = mse_loss(pred_inject, y)
    print(f"Loss: {loss_inject:.8f}")

print("\nModel WITHOUT forward_inject=False:")
print("-" * 40)
weights_no_inject = model_no_inject.get_weights()[0]
print(f"Visible weights norm: {torch.norm(weights_no_inject):.6f}")
print(f"Visible weights:\n{weights_no_inject}")

with torch.no_grad():
    pred_no_inject = model_no_inject(x)
    print(f"Prediction: {pred_no_inject}")
    loss_no_inject = mse_loss(pred_no_inject, y)
    print(f"Loss: {loss_no_inject:.8f}")

print("\n" + "="*60)
print("ANALYSIS:")
if torch.allclose(pred_inject, torch.zeros_like(pred_inject)):
    print("✗ forward_inject=True produces ZERO output")
    print("  This means forward injection is not working correctly")
else:
    print("✓ forward_inject=True produces non-zero output")

if torch.allclose(pred_no_inject, torch.zeros_like(pred_no_inject)):
    print("✗ forward_inject=False also produces ZERO output")
else:
    print("✓ forward_inject=False produces non-zero output")

# Now let's try training with forward_inject=False to see if that works
print("\n" + "="*60)
print("Training with forward_inject=False:")
opt = AnalogSGD(model_no_inject.parameters(), lr=0.1)
opt.regroup_param_groups(model_no_inject)

losses = []
for i in range(20):
    opt.zero_grad()
    pred = model_no_inject(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    
    if i % 5 == 0:
        print(f"Step {i:2d}: Loss = {loss:.8f}, pred[0,0] = {pred[0,0].item():.6f}")

if len(set(losses[:10])) > 1:
    print(f"✓ Training works with forward_inject=False (loss changes)")
else:
    print(f"✗ Training does NOT work even with forward_inject=False")