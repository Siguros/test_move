import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.configs import LRTTTransferCompound
from aihwkit.simulator.tiles import AnalogTile

# Create LRTT layer with ConstantStepDevice
device = ConstantStepDevice(dw_min=0.00001)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=3,
    transfer_lr=1.0,
    forward_inject=True,
    lora_alpha=1.0
)

# Use SingleRPUConfig wrapper
config = SingleRPUConfig(device=lrtt_config)
layer = AnalogLinear(4, 3, rpu_config=config)
layer = layer.cuda()

# Set initial weights
W = torch.randn(3, 4).cuda() * 0.1
print(f"Setting weights with norm: {torch.norm(W).item():.6f}")
layer.set_weights(W)

tiles = list(layer.analog_tiles())
cpp_tile = tiles[0].tile

print("\n=== Weight getters comparison ===")
print("Testing different methods to get weights from LRTT device\n")

# 1. layer.get_weights() - PyTorch layer level
w1 = layer.get_weights()
print(f"1. layer.get_weights() norm: {torch.norm(w1).item():.6f}")

# 2. tile.get_weights() - Tile wrapper level
w2 = tiles[0].get_weights()
print(f"2. tile.get_weights() norm: {torch.norm(w2).item():.6f}")

# 3. cpp_tile.get_weights() - C++ tile level
w3 = cpp_tile.get_weights()
print(f"3. cpp_tile.get_weights() norm: {torch.norm(w3).item():.6f}")

# Check if LRTT methods exist
if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
    print("\nLRTT-specific methods found!")
    
    # 4. LRTT visible weights
    w_vis = cpp_tile.lrtt_get_visible_weights()
    print(f"4. lrtt_get_visible_weights() norm: {torch.norm(w_vis).item():.6f}")
    
    # 5. LRTT composed effective weights
    if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
        w_eff = cpp_tile.lrtt_compose_w_eff()
        print(f"5. lrtt_compose_w_eff() norm: {torch.norm(w_eff).item():.6f}")
    
    # Check A and B
    a_lr = cpp_tile.lrtt_get_A_lr()
    b_lr = cpp_tile.lrtt_get_B_lr()
    print(f"\nA_lr norm: {torch.norm(a_lr).item():.6f}")
    print(f"B_lr norm: {torch.norm(b_lr).item():.6f}")
    
    # Compare them
    print("\n=== Comparing values ===")
    print(f"get_weights vs visible: max diff = {torch.max(torch.abs(w1 - w_vis)).item():.6e}")
    if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
        print(f"get_weights vs compose_eff: max diff = {torch.max(torch.abs(w1 - w_eff)).item():.6e}")
    
    # Print first few values
    print("\nFirst 4 values comparison:")
    print(f"get_weights:    {w1.flatten()[:4].tolist()}")
    print(f"visible:        {w_vis.flatten()[:4].tolist()}")
    if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
        print(f"compose_eff:    {w_eff.flatten()[:4].tolist()}")
else:
    print("\nNo LRTT methods found - might not be an LRTT tile")

# Do forward pass and check again
print("\n=== After forward pass ===")
x = torch.randn(2, 4).cuda()
with torch.no_grad():
    y = layer(x)

w_after = layer.get_weights()
print(f"layer.get_weights() norm: {torch.norm(w_after).item():.6f}")

if hasattr(cpp_tile, 'lrtt_get_visible_weights'):
    w_vis_after = cpp_tile.lrtt_get_visible_weights()
    print(f"lrtt_get_visible_weights() norm: {torch.norm(w_vis_after).item():.6f}")
    print(f"Difference: {torch.max(torch.abs(w_after - w_vis_after)).item():.6e}")

# Set A and B manually and check
print("\n=== After setting A and B manually ===")
if hasattr(cpp_tile, 'lrtt_set_A_lr'):
    A = torch.ones(3, 4).cuda() * 0.05
    B = torch.ones(3, 4).cuda() * 0.1
    cpp_tile.lrtt_set_A_lr(A)
    cpp_tile.lrtt_set_B_lr(B)
    
    w_with_ab = layer.get_weights()
    w_vis_with_ab = cpp_tile.lrtt_get_visible_weights()
    
    print(f"layer.get_weights() norm: {torch.norm(w_with_ab).item():.6f}")
    print(f"lrtt_get_visible_weights() norm: {torch.norm(w_vis_with_ab).item():.6f}")
    
    if hasattr(cpp_tile, 'lrtt_compose_w_eff'):
        w_eff_with_ab = cpp_tile.lrtt_compose_w_eff()
        print(f"lrtt_compose_w_eff() norm: {torch.norm(w_eff_with_ab).item():.6f}")
