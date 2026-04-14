import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import erfc
from core.physics import generate_training_data, SymbioticChannel
from core.architecture import SynapseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SynapseNet().to(device)
model.load_state_dict(torch.load("synapse_brain_best.pth", map_location=device))
model.eval()

snr_values = np.arange(0, 30, 2)

ber_ai      = []
ber_naive   = []
ber_theory  = []

def theoretical_qpsk_ber(snr_db):
    return 0.5 * erfc(np.sqrt(10**(snr_db / 10)))

def naive_decode(X_np, Y_bits_np):
    """
    Baseline: no phase correction, just hard-decision on raw I/Q.
    This is what any dumb receiver does — shows the value of our AI.
    QPSK decision: real>0 → bit0=0, imag>0 → bit1=0
    """
    total_errors = 0
    total_bits   = 0
    for i in range(len(X_np)):
        real_ch = X_np[i, 0, :]   # shape (64,)
        imag_ch = X_np[i, 1, :]
        # Interleave real and imag decisions → 128 bits
        bit0 = (real_ch < 0).astype(int)   # real < 0 → bit=1
        bit1 = (imag_ch < 0).astype(int)
        naive_bits = np.empty(128, dtype=int)
        naive_bits[0::2] = bit0
        naive_bits[1::2] = bit1
        total_errors += np.sum(naive_bits != Y_bits_np[i])
        total_bits   += 128
    return total_errors / total_bits

print("📊 Running Final Performance Evaluation...")
print(f"{'SNR':>5} | {'AI BER':>12} | {'Naive BER':>12} | {'Theory BER':>12} | {'AI Gain':>10}")
print("-" * 60)

for snr in snr_values:
    num_packets = 3000 if snr < 10 else 6000

    X, Y_bits, _ = generate_training_data(
        num_packets=num_packets, snr_range=(snr, snr)
    )
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        _, _, pred_bits = model(X_tensor)

    # AI decoder BER
    decisions = (pred_bits > 0.5).float().cpu().numpy()
    ai_ber    = np.mean(decisions != Y_bits)

    # Naive decoder BER
    naive_ber = naive_decode(X, Y_bits)

    # Theoretical limit
    th_ber = theoretical_qpsk_ber(snr)

    # Gain of AI over naive (how much the phase correction helps)
    gain_db = 10 * np.log10(naive_ber / max(ai_ber, 1e-9)) if ai_ber > 0 else float('inf')

    ber_ai.append(max(ai_ber,   1e-7))
    ber_naive.append(max(naive_ber, 1e-7))
    ber_theory.append(max(th_ber,  1e-7))

    print(f"{snr:>5} | {ai_ber:>12.6f} | {naive_ber:>12.6f} | "
          f"{th_ber:>12.6f} | {gain_db:>+9.1f} dB")

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("SYNAPSE-RAN: Final Performance Analysis", fontsize=14, fontweight='bold')

# Left: Waterfall plot
ax = axes[0]
ax.semilogy(snr_values, ber_naive,  'r--o', linewidth=2, label='Naive Receiver (No AI)')
ax.semilogy(snr_values, ber_ai,     'b-o',  linewidth=2, label='SYNAPSE-RAN AI Decoder')
ax.semilogy(snr_values, ber_theory, 'g:',   linewidth=1.5, alpha=0.7, label='Theoretical QPSK Limit')

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Bit Error Rate (Log Scale)', fontsize=12)
ax.set_title('BER vs SNR — Waterfall Curve', fontsize=11)
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.set_ylim(1e-7, 1.0)
ax.set_xlim(0, 28)
ax.legend(fontsize=10)

# Shade the "AI improvement zone"
ax.fill_between(snr_values, ber_naive, ber_ai,
                alpha=0.1, color='blue', label='AI Improvement Zone')

# Right: SNR gain bar chart (how many dB advantage AI gives at each SNR)
ax2 = axes[1]
gains = []
for a, n in zip(ber_ai, ber_naive):
    if a > 0 and n > 0:
        gains.append(10 * np.log10(n / a))
    else:
        gains.append(0)

colors = ['#2ecc71' if g > 3 else '#f39c12' if g > 1 else '#e74c3c' for g in gains]
bars = ax2.bar(snr_values, gains, color=colors, edgecolor='black', linewidth=0.5, width=1.5)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('AI Gain over Naive (dB)', fontsize=12)
ax2.set_title('AI Correction Advantage', fontsize=11)
ax2.axhline(0, color='black', linewidth=1)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

# Legend for bar colors
patches = [
    mpatches.Patch(color='#2ecc71', label='Strong gain (>3 dB)'),
    mpatches.Patch(color='#f39c12', label='Moderate gain (1-3 dB)'),
    mpatches.Patch(color='#e74c3c', label='Marginal gain (<1 dB)')
]
ax2.legend(handles=patches, fontsize=9)

plt.tight_layout()
plt.savefig("synapse_ran_final_analysis.png", dpi=150, bbox_inches='tight')
print("\n✅ Saved → synapse_ran_final_analysis.png")
plt.show()