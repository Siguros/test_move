#!/usr/bin/env python3
import torch
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets
x = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = torch.tensor([[1.0, 0.5], [0.7, 0.3]])

# Test configuration
device = ConstantStepDevice(dw_min=0.001, dw_min_dtod=0.0, up_down_dtod=0.0)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=10,  # Small value to see transfer effects
    transfer_lr=0.1,    # Smaller transfer LR
    forward_inject=True,
    lora_alpha=0.1,     # Smaller alpha to reduce instability
    units_in_mbatch=False,
)
rpu_config = UnitCellRPUConfig(device=lrtt_config)
rpu_config.reinit_gain = 0.1  # Smaller init for stability

model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

opt = AnalogSGD(model.parameters(), lr=0.01)  # Smaller learning rate
opt.regroup_param_groups(model)

print("Testing LRTT training dynamics")
print("-" * 50)

# Track loss history
losses = []
for epoch in range(100):
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} - Loss: {loss:.8f}")
        
    # Check for transfer events
    if (epoch + 1) % 10 == 0:
        print(f"  --> Transfer event at epoch {epoch + 1}")

# Analyze loss trend
print("\n" + "=" * 50)
print("Loss analysis:")
initial_loss = losses[0]
final_loss = losses[-1]
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss: {final_loss:.6f}")
print(f"Change: {final_loss - initial_loss:.6f}")

# Check for instability
max_loss = max(losses)
min_loss = min(losses)
print(f"Max loss: {max_loss:.6f}")
print(f"Min loss: {min_loss:.6f}")
print(f"Range: {max_loss - min_loss:.6f}")

if final_loss > initial_loss:
    print("⚠️ WARNING: Loss increased during training!")
else:
    print("✓ Loss decreased during training")