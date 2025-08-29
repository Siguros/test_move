import torch
from torch import Tensor
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice, UnitCellRPUConfig
from aihwkit.simulator.rpu_base import cuda

print("Testing syncVisibleWithAggregated call")
print("="*60)

# Simple LRTT config
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

print("Creating model...")
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

if cuda.is_compiled():
    model = model.cuda()

print("Model created.")
weights = model.get_weights()[0]
print(f"Weights shape: {weights.shape}")
print(f"Weight values: {weights}")

# Try setting weights explicitly
print("\nSetting weights explicitly...")
new_weights = torch.randn(2, 4)
if cuda.is_compiled():
    new_weights = new_weights.cuda()
    
model.set_weights(new_weights)
print(f"New weights set: {new_weights}")

# Now test forward pass
x = Tensor([[0.1, 0.2, 0.4, 0.3]])
if cuda.is_compiled():
    x = x.cuda()
    
print("\nTesting forward pass...")
with torch.no_grad():
    pred = model(x)
    print(f"Prediction: {pred}")
    print(f"Expected (approx): {new_weights @ x.T}")