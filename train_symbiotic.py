import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from core.physics import generate_training_data
from core.architecture import SynapseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training on: {device}")

model = SynapseNet().to(device)

# Load best weights from previous run — don't throw away prior learning
try:
    model.load_state_dict(torch.load("synapse_brain_best.pth"), strict=False)
    print("✅ Loaded weights from synapse_brain_best.pth")
except:
    print("⚠️  Starting from scratch (no checkpoint found)")

# Fresh LR — previous run had collapsed to 1e-5, we reset to continue learning
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=20, factor=0.5, min_lr=1e-5
)

criterion_mod   = nn.CrossEntropyLoss()
criterion_phase = nn.MSELoss()
criterion_bits  = nn.BCELoss()

EPOCHS = 400
best_bit_loss = float('inf')

print("🛠️ Starting Two-Stage Phase Refinement Training...")

for epoch in range(EPOCHS):
    model.train()

    if epoch < 100:
        snr_range = (20, 35)
        weights = (1.0, 8.0, 4.0, 5.0)   # (mod, coarse_phase, residual_phase, bits)
    elif epoch < 250:
        snr_range = (12, 25)
        weights = (1.0, 4.0, 6.0, 15.0)
    else:
        snr_range = (8, 20)
        weights = (1.0, 2.0, 4.0, 25.0)

    # More data per epoch = cleaner gradients
    X, Y_bits, Y_mod, Y_phase = generate_training_data(2000, snr_range)

    X      = torch.FloatTensor(X).to(device)
    Y_bits = torch.FloatTensor(Y_bits).to(device)
    Y_mod  = torch.LongTensor(Y_mod).to(device)

    Y_sincos = np.stack([np.sin(Y_phase), np.cos(Y_phase)], axis=1)
    Y_sincos = torch.FloatTensor(Y_sincos).to(device)

    # Residual target is zero — after coarse correction, nothing should remain
    Y_residual_zero = torch.zeros_like(Y_sincos)
    Y_residual_zero[:, 1] = 1.0  # cos(0)=1, sin(0)=0 means no residual rotation

    optimizer.zero_grad()
    pred_mod, pred_coarse, pred_residual, pred_bits = model(X)

    l_mod      = criterion_mod(pred_mod, Y_mod)
    l_coarse   = criterion_phase(pred_coarse, Y_sincos)
    l_residual = criterion_phase(pred_residual, Y_residual_zero)
    l_bits     = criterion_bits(pred_bits, Y_bits)

    loss = (weights[0]*l_mod + weights[1]*l_coarse +
            weights[2]*l_residual + weights[3]*l_bits)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step(l_bits.item())

    if l_bits.item() < best_bit_loss:
        best_bit_loss = l_bits.item()
        torch.save(model.state_dict(), "synapse_brain_best.pth")

    if (epoch + 1) % 20 == 0:
        lr = optimizer.param_groups[0]['lr']
        pred_angle = torch.atan2(pred_coarse[:, 0], pred_coarse[:, 1])
        true_angle = torch.atan2(Y_sincos[:, 0], Y_sincos[:, 1])
        coarse_mae = torch.mean(torch.abs(pred_angle - true_angle)).item()

        res_angle  = torch.atan2(pred_residual[:, 0], pred_residual[:, 1])
        residual_mae = torch.mean(torch.abs(res_angle)).item()

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Bit Loss: {l_bits.item():.4f} | "
              f"Coarse MAE: {coarse_mae:.4f} | Residual MAE: {residual_mae:.4f} | "
              f"LR: {lr:.2e} | Best: {best_bit_loss:.4f}")

torch.save(model.state_dict(), "synapse_brain.pth")
print(f"\n✅ Training Complete! Best Bit Loss: {best_bit_loss:.4f}")