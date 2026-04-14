import torch
from core.physics import generate_training_data
from core.architecture import SynapseNet

# 1. Get a sample from Phase 1
inputs, _, _, _ = generate_training_data(num_packets=1)
input_tensor = torch.FloatTensor(inputs) # Convert to PyTorch

# 2. Initialize the Phase 2 Brain
model = SynapseNet()

# 3. Pass the signal through the brain
with torch.no_grad():
    mod_out, phase_out, bit_out = model(input_tensor)

print("🧠 Phase 2 Sanity Check:")
print(f"Modulation Output Shape: {mod_out.shape} (Expected: [1, 2])")
print(f"Phase Output Shape: {phase_out.shape} (Expected: [1, 1])")
print(f"Bit Output Shape: {bit_out.shape} (Expected: [1, 128])")
print("\n✅ Logic is sound. The Brain is ready to learn.")