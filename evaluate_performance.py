import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from core.physics import generate_training_data
from core.architecture import SynapseNet

# 1. Setup & Hardware Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SynapseNet().to(device)
model.load_state_dict(torch.load("synapse_brain_best.pth", map_location=device))
model.eval()

# 2. Performance Parameters
# Sweeping from Cell-Edge (0dB) to Laboratory-Perfect (30dB)
snr_values = np.arange(0, 32, 2) 
ber_results = []

def theoretical_qpsk_ber(snr_db):
    """Calculates textbook QPSK BER in AWGN channel."""
    snr_linear = 10**(snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_linear))

print(f"📊 Starting Deep Performance Sweep (0dB to 30dB)...")
print("-" * 45)

for snr in snr_values:
    # We increase packets as SNR increases to catch tiny error rates
    num_packets = 5000 if snr < 15 else 10000
    
    X, Y_bits, _ = generate_training_data(num_packets=num_packets, snr_range=(snr, snr))
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        _, _, pred_bits = model(X_tensor)
        
    # Decision Logic
    decoded_bits = (pred_bits > 0.5).float().cpu().numpy()
    
    # Calculate BER
    errors = np.sum(decoded_bits != Y_bits)
    total_bits = Y_bits.size
    ber = errors / total_bits
    ber_results.append(ber)
    
    # Prevent log(0) in plotting
    display_ber = max(ber, 1e-7)
    print(f"SNR: {snr:2d} dB | AI BER: {display_ber:.6f} | Total Errors: {errors}")

# 3. Generating the Waterfall Plot
theoretical_ber = [theoretical_qpsk_ber(s) for s in snr_values]

plt.figure(figsize=(10, 7))
plt.semilogy(snr_values, ber_results, 'b-o', linewidth=2, label='Synapse-RAN AI Decoder')
plt.semilogy(snr_values, theoretical_ber, 'r--', alpha=0.7, label='Theoretical QPSK (Limit)')

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.ylim(1e-6, 1)
plt.xlim(0, 30)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (Log Scale)')
plt.title('Phase 4: Comparative Superiority Analysis')
plt.legend()

# Save the proof for your portfolio
plt.savefig("performance_waterfall_final.png")
print(f"\n✅ Sweep Complete. Plot saved as 'performance_waterfall_final.png'")
plt.show()