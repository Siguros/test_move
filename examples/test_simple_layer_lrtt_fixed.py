import torch
from torch import Tensor
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

print("Testing LRTT with random initial visible weights")
print("="*60)

# Define LRTT config
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
# Set reinit_gain to initialize visible weights with random values
rpu_config.reinit_gain = 0.5  # This should give random initial visible weights

model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Get initial state
print("Initial weights:")
weights = model.get_weights()[0]
print(f"Visible weights norm: {torch.norm(weights):.6f}")
print(f"Visible weights:\n{weights}")

# Test initial forward pass
print("\n--- Initial forward pass ---")
with torch.no_grad():
    pred = model(x)
    print(f"Prediction: {pred}")
    loss = mse_loss(pred, y)
    print(f"Initial loss: {loss:.8f}")

opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

print("\n--- Training for 50 steps ---")
losses = []
for i in range(50):
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    
    if i % 10 == 0:
        current_weights = model.get_weights()[0]
        print(f"Step {i:2d}: Loss = {loss:.8f}, pred[0,0] = {pred[0,0].item():.6f}")
        print(f"         Weight norm = {torch.norm(current_weights):.6f}")

# Check if loss is changing
print(f"\nLoss trajectory first 10 steps: {losses[:10]}")
unique_losses = len(set(losses[:10]))
if unique_losses > 1:
    print(f"✓ Loss is changing ({unique_losses} unique values in first 10 steps)")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  After 10 steps: {losses[9]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
else:
    print(f"✗ Loss is NOT changing (stuck at {losses[0]:.6f})")

print("\n--- Final state ---")
final_weights = model.get_weights()[0]
print(f"Final weights norm: {torch.norm(final_weights):.6f}")
print(f"Final weights:\n{final_weights}")