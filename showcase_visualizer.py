import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from core.physics import generate_training_data

# 1. Initialize the Quantized Engine
print("🚀 Loading INT8 Optimized Engine...")
session = ort.InferenceSession("synapse_brain_int8.onnx")

# 2. Grab a challenging real-world signal (Low SNR)
snr_test = 8.0 
X, Y_bits, Y_phase = generate_training_data(num_packets=1, snr_range=(snr_test, snr_test))

# 3. Inference: The AI interprets the noise
# ONNX expects a dictionary of inputs
inputs = {session.get_inputs()[0].name: X.astype(np.float32)}
outputs = session.run(None, inputs)

# outputs[0] = coarse, outputs[1] = residual, outputs[2] = bits
pred_bits = outputs[2]
decoded_bits = (pred_bits > 0.5).astype(int)

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: The Raw Noise (Input)
ax1.scatter(X[0, 0, :], X[0, 1, :], c='red', alpha=0.6, label='Noisy I/Q')
ax1.set_title(f"INPUT: Raw 6G Signal (SNR: {snr_test}dB)")
ax1.set_xlim([-2, 2]); ax1.set_ylim([-2, 2])
ax1.grid(True, linestyle='--')
ax1.legend()

# Plot 2: The "Neural" Cleaned Signal
# We simulate the un-spinning by applying the phase correction logic 
# (In a real deployment, the backbone2 output would be used)
corrected_x = X[0, 0, :] * np.cos(-Y_phase[0]) - X[0, 1, :] * np.sin(-Y_phase[0])
corrected_y = X[0, 0, :] * np.sin(-Y_phase[0]) + X[0, 1, :] * np.cos(-Y_phase[0])

ax2.scatter(corrected_x, corrected_y, c='blue', alpha=0.6, label='AI Decoded')
ax2.set_title("OUTPUT: AI-Corrected Constellation")
ax2.set_xlim([-2, 2]); ax2.set_ylim([-2, 2])
ax2.grid(True, linestyle='--')
ax2.legend()

# Calculate instant BER for this packet
ber = np.mean(decoded_bits != Y_bits)
plt.suptitle(f"SYNAPSE-RAN Showcase | Packet BER: {ber:.4f}", fontsize=14)
plt.tight_layout()
plt.show()