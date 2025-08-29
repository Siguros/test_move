import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

print("Testing simple forward_inject")
print("="*60)

# Test data
x = Tensor([[1.0, 1.0, 1.0, 1.0]])
y = Tensor([[1.0, 1.0]])

# Configure LRTT
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
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Check initial weights
weights = model.get_weights()[0]
print(f"Initial weights shape: {weights.shape}")
print(f"Initial weights:\n{weights}")
print(f"Weights norm: {torch.norm(weights):.6f}")

# Test forward
print("\n--- Testing forward pass ---")
with torch.no_grad():
    pred = model(x)
    print(f"Input: {x}")
    print(f"Output: {pred}")
    print(f"Expected (approx): {weights @ x.T}")
    
# Train one step
print("\n--- Training one step ---")
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

opt.zero_grad()
pred = model(x)
print(f"Forward output: {pred}")
loss = mse_loss(pred, y)
print(f"Loss: {loss:.8f}")
loss.backward()
opt.step()

# Check after one step
print("\n--- After one training step ---")
with torch.no_grad():
    pred = model(x)
    print(f"Output: {pred}")
    loss = mse_loss(pred, y)
    print(f"Loss: {loss:.8f}")

# Check if training changes anything
print("\n--- Training for 10 more steps ---")
losses = []
for i in range(10):
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"Loss trajectory: {losses}")
if len(set(losses)) > 1:
    print("✓ Loss is changing")
else:
    print("✗ Loss is NOT changing")