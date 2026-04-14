import torch
import torch.onnx
from core.architecture import SynapseNet
import onnx
from onnxsim import simplify

# 1. Load the "Superior" Brain
device = torch.device("cpu")
model = SynapseNet()
model.load_state_dict(torch.load("synapse_brain_best.pth", map_location=device))
model.eval()

# 2. Create Dummy Input
dummy_input = torch.randn(1, 2, 64)

# 3. Export to ONNX (Using Opset 18 for Residual Support)
onnx_path = "synapse_brain.onnx"
print(f"📦 Exporting to ONNX (Opset 18)...")

torch.onnx.export(model, 
                  dummy_input, 
                  onnx_path, 
                  export_params=True, 
                  opset_version=18,  # Force modern opset
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['phase_coarse', 'phase_residual', 'decoded_bits'],
                  dynamic_axes={'input': {0: 'batch_size'}}) # Support variable traffic

# 4. SIMPLIFICATION (The Silicon Polish)
# This fixes the "8192 vs 512" error by pre-calculating constant shapes
print("✨ Simplifying ONNX Graph...")
onnx_model = onnx.load(onnx_path)
model_simp, check = simplify(onnx_model)
assert check, "❌ Simplification failed"
onnx.save(model_simp, onnx_path)

print(f"✅ Success! Static graph saved as {onnx_path}")