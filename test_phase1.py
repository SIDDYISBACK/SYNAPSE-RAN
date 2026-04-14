from core.physics import generate_training_data
import matplotlib.pyplot as plt
import numpy as np

print("📡 Testing SYNAPSE-RAN Physics Engine...")

# Generate 5 test packets
inputs, bits, mods, phases = generate_training_data(num_packets=5)

# Pick the first packet for visualization
packet_idx = 0
iq_data = inputs[packet_idx]
mod_name = "16-QAM" if mods[packet_idx] == 1 else "QPSK"

print(f"✅ Packet Metadata: Mod={mod_name}, Phase Offset={phases[packet_idx]:.4f} rad")

# Visualize
plt.figure(figsize=(7, 7))
plt.scatter(iq_data[0, :], iq_data[1, :], alpha=0.6, c='crimson', edgecolors='k')
plt.axhline(0, color='black', lw=1, alpha=0.3)
plt.axvline(0, color='black', lw=1, alpha=0.3)
plt.title(f"Phase 1 Verification: Noisy {mod_name} Signal")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("\n🚀 SUCCESS: If the plot showed messy clusters, your Physics Engine is ready.")