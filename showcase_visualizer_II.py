"""
SYNAPSE-RAN: Real-Time Signal Showcase
Demonstrates live inference using the INT8 quantized ONNX model.
Visualizes the two-stage neural phase correction process.
"""

import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from core.physics import generate_training_data, SymbioticChannel

# ── Configuration ─────────────────────────────────────────────
SNR_TEST   = 8.0   # dB — representative cell-edge scenario
MODEL_PATH = "synapse_brain_int8.onnx"

# ── Load INT8 Quantized Engine ─────────────────────────────────
print(f"🚀 Loading INT8 Quantized Engine: {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH)
print(f"   Input  : {session.get_inputs()[0].name}  "
      f"{session.get_inputs()[0].shape}")
print(f"   Outputs: phase_coarse, phase_residual, decoded_bits")

# ── Generate Test Signal ───────────────────────────────────────
X, Y_bits, Y_phase = generate_training_data(
    num_packets=200, snr_range=(SNR_TEST, SNR_TEST)
)

# ── Run Inference ──────────────────────────────────────────────
inputs  = {session.get_inputs()[0].name: X.astype(np.float32)}
outputs = session.run(None, inputs)

pred_coarse   = outputs[0]   # shape (200, 2) — [sin, cos]
pred_residual = outputs[1]   # shape (200, 2)
pred_bits_raw = outputs[2]   # shape (200, 128)

# Hard-decision decoding
pred_bits = (pred_bits_raw > 0.5).astype(int)

# ── Metrics ────────────────────────────────────────────────────
ber_total = np.mean(pred_bits != Y_bits)

# Phase error per packet (degrees)
pred_angle = np.arctan2(pred_coarse[:, 0], pred_coarse[:, 1])
true_angle = Y_phase
phase_mae_deg = np.mean(np.abs(pred_angle - true_angle)) * (180 / np.pi)

res_angle    = np.arctan2(pred_residual[:, 0], pred_residual[:, 1])
residual_mae = np.mean(np.abs(res_angle)) * (180 / np.pi)

# ── Pick a representative single packet for constellation ──────
packet_idx = np.argmin(np.abs(Y_phase - np.median(Y_phase)))
X_pkt      = X[packet_idx]         # (2, 64)
phi_true   = Y_phase[packet_idx]
phi_pred   = pred_angle[packet_idx]

# Stage 1: Apply coarse correction
cos_c = pred_coarse[packet_idx, 1]
sin_c = pred_coarse[packet_idx, 0]
real_c =  X_pkt[0, :] * cos_c + X_pkt[1, :] * sin_c
imag_c = -X_pkt[0, :] * sin_c + X_pkt[1, :] * cos_c

# Stage 2: Apply residual correction
cos_r = pred_residual[packet_idx, 1]
sin_r = pred_residual[packet_idx, 0]
real_r =  real_c * cos_r + imag_c * sin_r
imag_r = -real_c * sin_r + imag_c * cos_r

# Ground truth (oracle correction — shows theoretical max)
real_gt =  X_pkt[0, :] * np.cos(-phi_true) - X_pkt[1, :] * np.sin(-phi_true)
imag_gt =  X_pkt[0, :] * np.sin(-phi_true) + X_pkt[1, :] * np.cos(-phi_true)

# ── Plot ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#0a0a0f')

gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

DARK   = '#0a0a0f'
MID    = '#12121a'
ACCENT = '#00d4ff'
GREEN  = '#00ff88'
RED    = '#ff4466'
ORANGE = '#ffaa00'
WHITE  = '#e8e8f0'

def style_ax(ax, title):
    ax.set_facecolor(MID)
    ax.set_title(title, color=ACCENT, fontsize=9, fontweight='bold', pad=8)
    ax.tick_params(colors=WHITE, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#2a2a3a')
    ax.grid(True, color='#1e1e2e', linewidth=0.5)

# ── Panel 1: Raw Noisy Signal ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X_pkt[0, :], X_pkt[1, :],
            c=RED, alpha=0.7, s=25, edgecolors='none')
ax1.axhline(0, color='#2a2a3a', lw=0.8)
ax1.axvline(0, color='#2a2a3a', lw=0.8)
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
style_ax(ax1, f"① RAW INPUT  (SNR: {SNR_TEST}dB)")
ax1.set_xlabel("In-Phase (I)", color=WHITE, fontsize=7)
ax1.set_ylabel("Quadrature (Q)", color=WHITE, fontsize=7)
ax1.text(0.05, 0.95, f"φ_true = {np.degrees(phi_true):.1f}°",
         transform=ax1.transAxes, color=RED, fontsize=7, va='top')

# ── Panel 2: After Coarse Correction ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(real_c, imag_c,
            c=ORANGE, alpha=0.7, s=25, edgecolors='none')
ax2.axhline(0, color='#2a2a3a', lw=0.8)
ax2.axvline(0, color='#2a2a3a', lw=0.8)
ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
style_ax(ax2, "② AFTER COARSE PLL")
ax2.set_xlabel("In-Phase (I)", color=WHITE, fontsize=7)
ax2.text(0.05, 0.95,
         f"φ_est = {np.degrees(phi_pred):.1f}°\n"
         f"err = {abs(np.degrees(phi_pred - phi_true)):.1f}°",
         transform=ax2.transAxes, color=ORANGE, fontsize=7, va='top')

# ── Panel 3: After Residual Correction ────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(real_r, imag_r,
            c=ACCENT, alpha=0.7, s=25, edgecolors='none')
ax3.axhline(0, color='#2a2a3a', lw=0.8)
ax3.axvline(0, color='#2a2a3a', lw=0.8)
ax3.set_xlim(-2.2, 2.2); ax3.set_ylim(-2.2, 2.2)
style_ax(ax3, "③ AFTER RESIDUAL REFINEMENT")
ax3.set_xlabel("In-Phase (I)", color=WHITE, fontsize=7)
ax3.text(0.05, 0.95,
         f"residual err = {residual_mae:.1f}°",
         transform=ax3.transAxes, color=ACCENT, fontsize=7, va='top')

# ── Panel 4: Oracle (Ground Truth) ────────────────────────────
ax4 = fig.add_subplot(gs[0, 3])
ax4.scatter(real_gt, imag_gt,
            c=GREEN, alpha=0.7, s=25, edgecolors='none')
ax4.axhline(0, color='#2a2a3a', lw=0.8)
ax4.axvline(0, color='#2a2a3a', lw=0.8)
# Mark ideal QPSK points
for ix, iy in [(0.707, 0.707), (-0.707, 0.707),
               (-0.707, -0.707), (0.707, -0.707)]:
    ax4.scatter(ix, iy, c='white', s=80, marker='+', linewidths=1.5, zorder=5)
ax4.set_xlim(-2.2, 2.2); ax4.set_ylim(-2.2, 2.2)
style_ax(ax4, "④ ORACLE (True φ Applied)")
ax4.set_xlabel("In-Phase (I)", color=WHITE, fontsize=7)
ax4.text(0.05, 0.95, "Theoretical best\n(+ = ideal QPSK)",
         transform=ax4.transAxes, color=GREEN, fontsize=7, va='top')

# ── Panel 5: BER per packet distribution ──────────────────────
ax5 = fig.add_subplot(gs[1, 0:2])
per_pkt_ber = np.mean(pred_bits != Y_bits, axis=1)
ax5.hist(per_pkt_ber, bins=40, color=ACCENT, alpha=0.8, edgecolor=MID)
ax5.axvline(ber_total, color=RED, lw=2, linestyle='--',
            label=f'Mean BER = {ber_total:.4f}')
ax5.axvline(0.5, color='#444', lw=1, linestyle=':',
            label='Random guess (0.5)')
style_ax(ax5, "PER-PACKET BER DISTRIBUTION  (200 packets)")
ax5.set_xlabel("Bit Error Rate", color=WHITE, fontsize=8)
ax5.set_ylabel("Count", color=WHITE, fontsize=8)
ax5.legend(fontsize=8, facecolor=MID, labelcolor=WHITE)

# ── Panel 6: Phase Error Distribution ─────────────────────────
ax6 = fig.add_subplot(gs[1, 2:4])
phase_errors_deg = np.abs(pred_angle - true_angle) * (180 / np.pi)
ax6.hist(phase_errors_deg, bins=40, color=ORANGE, alpha=0.8, edgecolor=MID)
ax6.axvline(phase_mae_deg, color=GREEN, lw=2, linestyle='--',
            label=f'Mean MAE = {phase_mae_deg:.2f}°')
ax6.axvline(45, color='#444', lw=1, linestyle=':',
            label='Random guess (45°)')
style_ax(ax6, "PHASE ESTIMATION ERROR DISTRIBUTION  (200 packets)")
ax6.set_xlabel("Phase Error (degrees)", color=WHITE, fontsize=8)
ax6.set_ylabel("Count", color=WHITE, fontsize=8)
ax6.legend(fontsize=8, facecolor=MID, labelcolor=WHITE)

# ── Title & Stats Bar ──────────────────────────────────────────
stats_text = (
    f"SYNAPSE-RAN  |  INT8 Quantized ONNX  |  SNR: {SNR_TEST} dB  |  "
    f"BER: {ber_total:.4f} ({(1-ber_total)*100:.2f}% accuracy)  |  "
    f"Phase MAE: {phase_mae_deg:.2f}°  |  Residual MAE: {residual_mae:.2f}°  |  "
    f"Packets: 200"
)
fig.text(0.5, 0.97, "SYNAPSE-RAN: Two-Stage Neural PLL — Live Inference",
         ha='center', color=ACCENT, fontsize=13, fontweight='bold')
fig.text(0.5, 0.935, stats_text,
         ha='center', color=WHITE, fontsize=8, alpha=0.8)

plt.savefig("synapse_ran_showcase.png", dpi=150,
            bbox_inches='tight', facecolor=DARK)
print(f"\n📊 Results:")
print(f"   BER         : {ber_total:.4f}  ({(1-ber_total)*100:.2f}% bit accuracy)")
print(f"   Phase MAE   : {phase_mae_deg:.2f}°")
print(f"   Residual MAE: {residual_mae:.2f}°")
print(f"\n✅ Saved → synapse_ran_showcase.png")
plt.show()
