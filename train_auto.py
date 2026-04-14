import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from core.physics import generate_training_data
from core.architecture import SynapseNet

TARGET_BIT_LOSS    = 0.01
MAX_CYCLES         = 20
EPOCHS_PER_CYCLE   = 250
PATIENCE           = 40
CHECKPOINT         = "synapse_brain_best.pth"
BASE_LR            = 3e-4
LR_DECAY_PER_CYCLE = 0.85
MIN_LR_FLOOR       = 5e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
print(f"🎯 Target Bit Loss: {TARGET_BIT_LOSS}")
print("=" * 65)

criterion_phase = nn.MSELoss()
criterion_bits  = nn.BCELoss()

global_best_loss = float('inf')
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
    print("🗑️  Deleted old checkpoint — labels were corrupted by bad LDPC encoding")
print("🆕 Fresh start with clean random-bit labels\n")


def compute_ber(pred_bits, Y_bits):
    """Hard-decision BER — the real metric that matters."""
    decisions = (pred_bits > 0.5).float()
    return (decisions != Y_bits).float().mean().item()


def run_cycle(cycle_num, current_best_loss):
    model = SynapseNet().to(device)

    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT))
        print(f"   ✅ Loaded checkpoint (best BCE: {current_best_loss:.4f})")
    else:
        print("   🆕 Fresh weights")

    cycle_lr = max(BASE_LR * (LR_DECAY_PER_CYCLE ** cycle_num), MIN_LR_FLOOR)
    print(f"   📈 Cycle LR: {cycle_lr:.2e}")

    optimizer = optim.Adam(model.parameters(), lr=cycle_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=12, factor=0.6, min_lr=MIN_LR_FLOOR
    )

    # SNR curriculum tightens as model improves
    if current_best_loss > 0.10:
        snr_easy, snr_hard = (20, 35), (15, 28)
    elif current_best_loss > 0.05:
        snr_easy, snr_hard = (15, 28), (10, 22)
    elif current_best_loss > 0.02:
        snr_easy, snr_hard = (12, 25), (8, 20)
    else:
        snr_easy, snr_hard = (8, 22), (5, 18)

    print(f"   📡 SNR: {snr_easy} → {snr_hard}")
    print(f"{'='*65}")

    cycle_best = current_best_loss
    no_improve = 0

    for epoch in range(EPOCHS_PER_CYCLE):
        model.train()

        # Phase-first in early epochs, bit-focused later
        if epoch < EPOCHS_PER_CYCLE // 3:
            snr_range = snr_easy
            w_c, w_r, w_b = 8.0, 5.0, 10.0
        elif epoch < (2 * EPOCHS_PER_CYCLE) // 3:
            snr_range = snr_hard
            w_c, w_r, w_b = 4.0, 3.0, 25.0
        else:
            snr_range = snr_hard
            w_c, w_r, w_b = 2.0, 2.0, 40.0

        X, Y_bits, Y_phase = generate_training_data(2000, snr_range)

        X      = torch.FloatTensor(X).to(device)
        Y_bits = torch.FloatTensor(Y_bits).to(device)

        Y_sincos           = np.stack([np.sin(Y_phase), np.cos(Y_phase)], axis=1)
        Y_sincos           = torch.FloatTensor(Y_sincos).to(device)
        Y_res_zero         = torch.zeros_like(Y_sincos)
        Y_res_zero[:, 1]   = 1.0

        optimizer.zero_grad()
        p_coarse, p_res, pred_bits = model(X)

        l_coarse   = criterion_phase(p_coarse,  Y_sincos)
        l_residual = criterion_phase(p_res,     Y_res_zero)
        l_bits     = criterion_bits(pred_bits,  Y_bits)

        loss = w_c*l_coarse + w_r*l_residual + w_b*l_bits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(l_bits.item())

        cur = l_bits.item()

        if cur < cycle_best:
            cycle_best = cur
            no_improve = 0
            if cur < current_best_loss:
                current_best_loss = cur
                torch.save(model.state_dict(), CHECKPOINT)
        else:
            no_improve += 1

        if (epoch + 1) % 25 == 0:
            lr_now   = optimizer.param_groups[0]['lr']
            ber_now  = compute_ber(pred_bits, Y_bits)
            p_ang    = torch.atan2(p_coarse[:, 0], p_coarse[:, 1])
            t_ang    = torch.atan2(Y_sincos[:, 0], Y_sincos[:, 1])
            c_deg    = torch.mean(torch.abs(p_ang - t_ang)).item() * (180/np.pi)
            r_ang    = torch.atan2(p_res[:, 0], p_res[:, 1])
            r_deg    = torch.mean(torch.abs(r_ang)).item() * (180/np.pi)

            print(f"  Ep [{epoch+1:3d}/{EPOCHS_PER_CYCLE}] | "
                  f"BCE: {cur:.4f} | BER: {ber_now:.4f} | "
                  f"Coarse: {c_deg:.1f}° | Res: {r_deg:.1f}° | "
                  f"LR: {lr_now:.2e} | Stall: {no_improve}/{PATIENCE} | "
                  f"Best: {current_best_loss:.4f}")

        if no_improve >= PATIENCE:
            print(f"\n  ⏹  Early stop at ep {epoch+1}")
            break

        if current_best_loss <= TARGET_BIT_LOSS:
            print(f"\n  🎯 TARGET HIT at ep {epoch+1}!")
            return current_best_loss, True

    return current_best_loss, False


# Main loop
target_hit = False
for cycle in range(MAX_CYCLES):
    print(f"\n{'='*65}")
    print(f"🔄 CYCLE {cycle+1}/{MAX_CYCLES} | Global Best: {global_best_loss:.4f}")
    global_best_loss, target_hit = run_cycle(cycle, global_best_loss)
    print(f"  📊 Cycle {cycle+1} done | Global Best: {global_best_loss:.4f}")
    if target_hit:
        break

print("\n" + "="*65)
if target_hit:
    print(f"🏆 TARGET ACHIEVED! Best Loss: {global_best_loss:.4f}")
else:
    print(f"✅ Done. Best Loss: {global_best_loss:.4f}")
print(f"   Saved → {CHECKPOINT}")
print("="*65)