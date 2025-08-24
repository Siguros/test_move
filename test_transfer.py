import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import LRTTTransferCompound, ConstantStepDevice
from aihwkit.simulator.tiles import AnalogTile

# Create LRTT layer
device = ConstantStepDevice(dw_min=0.00001)
lrtt_config = LRTTTransferCompound(
    unit_cell_devices=[device, device, device],
    rank=2,
    transfer_every=3,
    transfer_lr=1.0,  # High value to see clear transfer
    forward_inject=True,
    lora_alpha=1.0
)

layer = AnalogLinear(4, 3, rpu_config=lrtt_config, tile_module_class=AnalogTile)
layer = layer.cuda()

# Set initial weights
W = torch.randn(3, 4).cuda() * 0.1
layer.set_weights(W)

tiles = list(layer.analog_tiles())
cpp_tile = tiles[0].tile

print("=== Testing Transfer vs Reinit ===")
print(f"Initial visible weights norm: {torch.norm(cpp_tile.lrtt_get_visible_weights()).item():.6f}")

# Manually set A and B to known values
A = torch.ones(3, 4).cuda() * 0.1
B = torch.ones(3, 4).cuda() * 0.2
cpp_tile.lrtt_set_A_lr(A)
cpp_tile.lrtt_set_B_lr(B)

print(f"Set A_lr norm: {torch.norm(A).item():.6f}")
print(f"Set B_lr norm: {torch.norm(B).item():.6f}")

# Expected transfer amount: A[:,:2] @ B[:2,:]
expected_ab = A[:, :2] @ B[:2, :]
print(f"Expected A@B product norm: {torch.norm(expected_ab).item():.6f}")

# Track visible weights through updates
x = torch.randn(2, 4).cuda()
grad = torch.randn(2, 3).cuda() * 0.1

print("\nTracking through updates (transfer_every=3):")
for i in range(5):
    vis_before = cpp_tile.lrtt_get_visible_weights().clone()
    a_before = cpp_tile.lrtt_get_A_lr().clone()
    b_before = cpp_tile.lrtt_get_B_lr().clone()
    
    # Do update
    tiles[0].update(x, grad, learning_rate=0.01)
    
    vis_after = cpp_tile.lrtt_get_visible_weights()
    a_after = cpp_tile.lrtt_get_A_lr()
    b_after = cpp_tile.lrtt_get_B_lr()
    
    vis_change = torch.norm(vis_after - vis_before).item()
    a_norm_before = torch.norm(a_before).item()
    a_norm_after = torch.norm(a_after).item()
    b_norm_before = torch.norm(b_before).item()
    b_norm_after = torch.norm(b_after).item()
    
    print(f"\nStep {i+1}:")
    print(f"  Visible: norm={torch.norm(vis_after).item():.6f}, change={vis_change:.6f}")
    print(f"  A_lr: {a_norm_before:.6f} -> {a_norm_after:.6f}")
    print(f"  B_lr: {b_norm_before:.6f} -> {b_norm_after:.6f}")
    
    if i == 2:  # Step 3 - transfer should happen
        if vis_change > 0.001:
            print(f"  >>> TRANSFER OCCURRED: Visible changed by {vis_change:.6f}")
        if abs(a_norm_after) < 0.001:
            print(f"  >>> REINIT OCCURRED: A zeroed")
        if abs(b_norm_after - b_norm_before) > 0.5:
            print(f"  >>> REINIT OCCURRED: B reinitialized")

print("\nFinal state:")
print(f"  Visible norm: {torch.norm(cpp_tile.lrtt_get_visible_weights()).item():.6f}")
print(f"  A_lr norm: {torch.norm(cpp_tile.lrtt_get_A_lr()).item():.6f}")
print(f"  B_lr norm: {torch.norm(cpp_tile.lrtt_get_B_lr()).item():.6f}")
