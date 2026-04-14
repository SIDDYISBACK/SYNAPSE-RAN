import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "synapse_brain.onnx"
model_int8 = "synapse_brain_int8.onnx"

print("🔨 Starting INT8 Hardware Quantization...")

# Standard Dynamic Quantization for Edge/DSP deployment
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

# 📊 Compression Report
size_fp32 = os.path.getsize(model_fp32) / 1024
size_int8 = os.path.getsize(model_int8) / 1024

print("-" * 45)
print(f"📦 FP32 Size: {size_fp32:.2f} KB")
print(f"📦 INT8 Size: {size_int8:.2f} KB")
print(f"🚀 Saved: {((size_fp32 - size_int8) / size_fp32) * 100:.1f}% of memory space")
print(f"✅ Deployment artifact ready: {model_int8}")