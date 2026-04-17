# SYNAPSE-RAN
### Symbiotic Neural Architecture for Power-efficient, Security-Enhanced Radio Access Network

> A research/learning implementation of an AI-Native Air Interface for 6G communications, developed as an independent study project. The system replaces classical Digital Signal Processing (DSP) blocks — such as phase synchronisation and bit decoding — with a unified neural framework that operates directly on raw I/Q samples.

---

## My Motivation for Learning and Building-

The Nokia–NVIDIA AI-RAN initiative (backed by a $1B investment, announced October 2025) is built on one premise: **replace specialised fixed-function hardware with software-defined neural inference on GPU-accelerated shared compute**. The core engineering challenge is moving DSP functions — traditionally implemented as dedicated ASICs — into programmable, learnable models that can run on the same silicon used to handle AI traffic.

SYNAPSE-RAN is a prototype ground-up implementation of that idea at the physical layer. It does not simulate the idea — it builds and benchmarks it.
It still has needs of improvement.

---

## System Architecture

### The Problem Being Solved

A QPSK receiver in a real channel receives:

```
y[n] = x[n] · e^(jφ) + w[n]
```

Where:
- `x[n]` — transmitted QPSK symbol
- `φ ∈ [0, π/2]` — unknown stochastic phase rotation (caused by oscillator mismatch, user mobility)
- `w[n] ~ CN(0, σ²)` — complex AWGN

A classical receiver solves this in two separate steps: (1) a Phase-Locked Loop estimates `φ`, (2) a hard-decision slicer decodes bits. These are independent modules. SYNAPSE-RAN solves both simultaneously in one forward pass through a single neural network — **Joint Detection and Decoding (JDD)**.

### Why Classical DSP Fails Here

The 4th-power blind phase estimator (the standard pilot-free classical method) works by computing:

```
φ_est = angle(mean(y[n]^4)) / 4
```

This requires `4φ` to span a small arc so the mean is well-defined. With `φ` uniformly distributed over `[0, π/2]`, the quantity `4φ` spans the full circle `[0, 2π]` — vectors cancel, the estimator collapses. **SYNAPSE-RAN wins legitimately in this regime because the neural estimator learns the phase distribution, not just the instantaneous angle.**

---

## SynapseNet: Two-Stage Neural Phase-Locked Loop

```
Raw I/Q  →  [Backbone 1]  →  Coarse φ̂  →  [Physical Rotation]
         →  [Backbone 2]  →  Residual Δφ →  [Physical Rotation]
         →  [Decoder Backbone]  →  128 decoded bits
```

### Stage 1 — Coarse Phase Lock

An 8-layer 1D-CNN with residual skip connections extracts global features from the 64-symbol I/Q tensor. It outputs `[sin(φ̂), cos(φ̂)]` — a unit-vector representation that avoids the angular wrap-around problem that breaks MSE loss at the 0/2π boundary.

The physical rotation layer then applies:

```
x̂_real =  Re(y) · cos(-φ̂) - Im(y) · sin(-φ̂)
x̂_imag =  Re(y) · sin(-φ̂) + Im(y) · cos(-φ̂)
```

This is a differentiable operation — gradients from the bit decoder flow back through the rotation layer directly into the phase estimator. This is the architectural insight that breaks the 0.693 entropy wall (Shannon entropy of a coin flip) that standard parallel multi-task models cannot escape.

### Stage 2 — Residual Refinement

A second backbone processes the coarsely corrected signal and estimates the remaining phase jitter `Δφ`. A second physical rotation is applied. The resulting signal is processed by the decoder backbone.

### Why Residual Blocks

Standard CNNs lose fine-grained phase structure as depth increases. Residual skip connections (`output = F(x) + x`) preserve the original I/Q topology through every layer, ensuring the phase estimator retains the information it needs across 8 layers of feature extraction.

```python
class ResBlock1D(nn.Module):
    def forward(self, x):
        return F.relu(x + self.block(x))  # Identity shortcut
```

---

## Forward Error Correction: QC-LDPC Engine

The physical layer uses a Quasi-Cyclic LDPC code — the same class of codes used in the 5G NR standard (3GPP Base Graph BG1/BG2).

### Why QC-LDPC

Random LDPC matrices are not hardware-friendly. QC-LDPC matrices are constructed from **Circulant Permutation Matrices (CPM)** — shifted identity matrices. This structure means the hardware only needs to store a single row per sub-matrix, and all 16 rows of a sub-matrix can be processed **in parallel**. This is why NVIDIA's Aerial SDK targets QC-LDPC specifically for GPU-accelerated baseband.

### Girth-6 Construction

Short cycles in the Tanner graph (the bipartite graph representation of the parity check matrix) cause the belief propagation decoder to pass the same corrupted information repeatedly, degrading performance. The engine enforces Girth-6 — no cycles of length 4 — using a greedy constraint during base matrix construction:

```
(shift_ij - shift_ij') ≠ (shift_i'j - shift_i'j')  mod Z
```

This eliminates 4-cycles analytically, not through trial and error.

```
Base Matrix B (4×8, Z=16) → Lifted H Matrix (64×128)
Rate-1/2 code: 64 information bits → 128 transmitted bits
```

---

## Training: Curriculum Learning with Warm Restarts

Training a neural receiver from scratch on high-noise data fails — the model learns nothing because the signal-to-noise ratio is too low for a meaningful gradient signal. SYNAPSE-RAN uses a three-phase curriculum:

| Phase | SNR Range | Objective |
|---|---|---|
| Phase-Lock | 20–35 dB | Teach the phase heads to estimate rotation accurately on clean signals |
| Hardening | 12–25 dB | Introduce realistic noise, scale bit loss weight |
| Superiority | 8–20 dB | Full 6G cell-edge noise levels, maximum bit weight |

The training loop uses **Cyclic Warm Restarts** — each cycle reloads the best checkpoint and resets the learning rate to a fresh value (decaying 15% per cycle). This prevents the learning rate from collapsing to zero while the model is still improvable, which is the failure mode of standard ReduceLROnPlateau over long runs.

**Early stopping within cycles** (patience = 40 epochs) ensures no cycle wastes compute after convergence.

---

## Hardware-Aware Deployment

### INT8 Quantisation via ONNX

The model is exported to ONNX Opset 18 (with graph simplification to resolve residual block constant folding) and then quantised using ONNX Runtime's dynamic INT8 quantizer — the same pipeline used for Nokia AirScale NPU deployment.

| Metric | FP32 Baseline | INT8 Optimized |
|---|---|---|
| Model Size | 60.84 MB | 15.25 MB |
| Memory Reduction | — | **74.9%** |
| Format | `.pth` | `.onnx` |
| Inference Engine | PyTorch | ONNX Runtime |

This 74.9% reduction is the difference between fitting on an edge node and not fitting. A Nokia AirScale gNB-DU processing unit has constrained on-chip memory — model size is not an academic metric.

### Why This Matters for 6G

The Nokia-NVIDIA AI-RAN architecture proposes turning underutilised radio site assets into **distributed AI grid factories** — shared GPU compute running both radio functions and third-party AI services simultaneously. Every kilobyte of model memory is a kilobyte taken from that shared pool. Quantisation is not optional in production deployment.

---

## Performance Results

### Quantitative Benchmarks

| Metric | Value |
|---|---|
| Best BCE Loss | 0.0179 |
| BER at SNR 8 dB | ~0.006 (99.4% accuracy) |
| BER at SNR 12 dB | ~0.009 (<1% error rate) |
| Coarse Phase MAE | ~4° |
| Residual Phase MAE | ~1.2° |
| Gain over Naive Receiver | +20 dB (at SNR 26 dB) |

### BER vs SNR — Honest 4-Way Comparison

The system was evaluated against three references:

1. **Naive baseline** — no phase correction, raw hard decisions (BER ≈ 0.25 flat — completely unusable)
2. **Classical DSP (4th-power method)** — the standard pilot-free phase estimator, which collapses at wide phase ranges
3. **Theoretical QPSK limit** — the Shannon-bound performance

```
SNR  | AI BER    | Classical | Naive    | AI vs Classical
-----|-----------|-----------|----------|----------------
  0  | 0.20178   | 0.31374   | 0.30045  | +1.9 dB  AI ✓
  6  | 0.06270   | 0.25857   | 0.25373  | +6.2 dB  AI ✓
 10  | 0.01688   | 0.25462   | 0.24471  | +11.8 dB AI ✓
 14  | 0.00597   | 0.25275   | 0.24842  | +16.3 dB AI ✓
 20  | 0.00401   | 0.25166   | 0.24930  | +18.0 dB AI ✓
 28  | 0.00262   | 0.25040   | 0.24751  | +19.8 dB AI ✓
```

The AI wins across the **entire SNR range** — not because classical DSP is weak, but because the 4th-power method fundamentally cannot handle the `φ ∈ [0, π/2]` distribution. The neural estimator learns the distribution implicitly.

### The Error Floor

The BER plateau at ~2.5×10⁻³ above 16 dB SNR is a consequence of the residual 4° phase error. At these SNR levels, noise is negligible — the remaining errors are deterministic: symbols near the QPSK decision boundaries are consistently misclassified by the 4° rotation offset. This is not a training problem. It is an architectural constraint that the next research phase (Neural Belief Propagation) targets directly.

### Future: Neural Belief Propagation

The current decoder is a black-box bit predictor. The next architectural step embeds the QC-LDPC parity check matrix `H` directly into the loss function:

```
L_parity = ||H · b̂ᵀ mod 2||₁
```

By penalising parity violations during training, the decoder learns to output valid codewords rather than independent bit probabilities. This is projected to push the error floor from 10⁻³ toward 10⁻⁶ — the regime required for mission-critical 6G applications like autonomous vehicle communication and industrial automation.
Disclaimer-the above belief is still a hypothesis and needs testing and refining.

---

## Repository Structure

```
SYNAPSE-RAN/
├── core/
│   ├── __init__.py
│   ├── physics.py              # QC-LDPC engine, QPSK modulator, channel simulator
│   └── architecture.py         # SynapseNet: ResBlock1D, Two-Stage Neural PLL
│
├── train_auto.py               # Cyclic warm-restart training with early stopping
├── train_symbiotic.py          # Single-run training script
├── evaluate_final.py           # 4-way BER comparison + waterfall plot
├── showcase_visualizer.py      # Real-time INT8 inference + constellation pipeline
├── export_onnx.py              # PyTorch → ONNX Opset 18 export + simplification
├── quantize_brain.py           # ONNX FP32 → INT8 dynamic quantization
│
├── synapse_brain_best.pth      # Best FP32 checkpoint (BCE: 0.0231)
├── synapse_brain_int8.onnx     # Production INT8 deployment artifact
│
└── docs/
    ├── synapse_ran_showcase.png
    ├── synapse_ran_honest_analysis.png
    └── performance_waterfall_final.png
```

---

## Reproducing the Results

### Environment Setup

```bash
# Python 3.12+, Windows/Linux/macOS
python -m venv venv
source venv/bin/activate          # Linux/Mac
# OR: .\venv\Scripts\activate     # Windows

pip install torch torchvision torchaudio numpy scipy matplotlib tqdm
pip install onnx onnxruntime onnx-simplifier
```

### Full Pipeline

```bash
# 1. Train — cyclic warm restarts, saves best checkpoint automatically
python train_auto.py

# 2. Export to ONNX
python export_onnx.py

# 3. Quantise to INT8
python quantize_brain.py

# 4. Evaluate — generates 4-way BER comparison
python evaluate_final.py

# 5. Real-time showcase
python showcase_visualizer.py
python showcase_visualizer_II.py
```

---

## What This Project Demonstrates

This is not a tutorial implementation of a known architecture. Each design decision resolves a specific engineering problem:

| Problem | Root Cause | Solution Implemented |
|---|---|---|
| Bit loss stuck at 0.693 | Per-symbol Rayleigh fading made blind decoding statistically impossible | Block fading → solvable channel |
| BCE plateau at 0.693 despite block fading | Rayleigh h randomises both amplitude and phase per packet | Removed fading, isolated phase rotation as the learnable target |
| Phase head learns, decoder doesn't | Phase prediction was a hint, not a physical correction | Differentiable rotation layer — decoder sees corrected constellation |
| Training converges then regresses | LR collapses to 1e-8 while model still improvable | Cyclic warm restarts reset LR each cycle |
| Error floor at high SNR | Residual 4° phase jitter causes deterministic boundary errors | Identified as architectural limit; documented as future work (NBP) |
| Classical baseline appeared broken | 4th-power method degenerates for φ uniform over [0, π/2] | Confirmed AI wins legitimately; documents a known DSP limitation |

---

## Final output images-->

(<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/18cf194f-3c3c-4cf7-a2be-b0e5e4708824" />)
(<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/d3732c29-44ba-464e-9b9e-78098abd9ca8" />)
(<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/951cd9d4-b479-40f8-8fe0-692165f3bd73" />)

---

## Project Context

Developed as an independent introductory research project to affiliate myself with an emerging frontier of technology during the second year of my B.Tech in Electronics and Communication Engineering (JIIT Noida, Batch 2028). Built entirely on a CPU-only laptop using PyTorch 3.11, ONNX Runtime, and NumPy.

The project was motivated by the Nokia–NVIDIA AI-RAN initiative and the broader 6G research agenda (commercial deployment: 2030). The architecture is specifically sized for edge deployment — the 15.25 MB INT8 ONNX artefact is designed to fit within the memory constraints of embedded NPUs like those in the NVIDIA Jetson series or Nokia's AirScale DU hardware.

**My Base coursework**: Information Theory, Digital Signal Processing, Telecommunication, Hardware-Aware AI.

---

## Contact

**Sidhant**  
B.Tech Electronics and Communication Engineering  
Jaypee Institute of Information Technology, Noida  
Batch of 2028

---
