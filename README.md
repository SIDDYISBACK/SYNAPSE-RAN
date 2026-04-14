# SYNAPSE-RAN: Symbiotic Neural Architecture for Power-efficient Security Enhanced-Radio Access Netowrk

## 1. Abstract and Technical Definition
SYNAPSE-RAN represents a standalone research entity focused on the development of an **AI-Native Air Interface** for 6G communications. The system transitions traditional Digital Signal Processing (DSP) functions—specifically phase synchronization and error correction—into a unified, symbiotic neural framework. By integrating physical constraints directly into a deep learning backbone, the architecture provides a hardware-efficient solution for software-defined edge-RAN deployment.

### 1.1 Technical Etymology
* **Symbiotic**: Refers to the joint optimization of modulation classification, phase estimation, and bit recovery within a single inference pass.
* **Neural Architecture**: Utilizes deep 1D Convolutional Neural Networks (CNN) with residual skip-connections to process complex I/Q data.
* **Power-efficient**: Achieved through **INT8 Dynamic Quantization**, reducing computational overhead and memory footprint for edge-tier deployment.
* **SE-RAN**: Software-defined, Edge-oriented Radio Access Network.

---

## 2. Mathematical System Model
The project models a complex baseband transmission over an Additive White Gaussian Noise (AWGN) channel characterized by stochastic phase rotation.

### 2.1 Signal Equation
The received complex signal $y[n]$ is defined by the following equation:
$$y[n] = x[n] \cdot e^{j\phi} + w[n]$$
Where:
* $x[n]$ is the QPSK modulated information symbol.
* $\phi$ is the stochastic phase offset, where $\phi \in [0, \pi/2]$.
* $w[n]$ represents the complex AWGN component, $w[n] \sim \mathcal{CN}(0, \sigma^2)$.

### 2.2 Forward Error Correction (FEC) Framework
Redundancy is introduced through a **Quasi-Cyclic Low-Density Parity-Check (QC-LDPC)** engine. The system utilizes a rate-1/2 systematic code based on a (128, 64) block structure. The encoding process maps 64 information bits $\mathbf{b}$ to a 128-bit codeword $\mathbf{c}$ using a generator matrix $\mathbf{G}$:
$$\mathbf{c} = \mathbf{b} \cdot \mathbf{G} \pmod 2$$
The engine generates high-girth base matrices to ensure parity-check stability, facilitating successful bit recovery beyond the 0.693 Shannon entropy barrier.

---

## 3. Component Architecture: SynapseNet
The receiver utilizes a multi-stage residual architecture to perform joint detection and decoding (JDD).

### 3.1 Two-Stage Residual Neural PLL
Traditional Phase-Locked Loops (PLL) are replaced by a two-stage neural estimator:
1. **Stage 1: Coarse Estimation**: An 8-layer convolutional backbone extracts global features from the I/Q tensor to estimate the primary rotation vector $[\sin\hat{\phi}, \cos\hat{\phi}]$.
2. **Physical Rotation Layer**: A differentiable layer that mathematically "un-spins" the received symbols using trigonometric correction:
   $$\hat{x}[n] = \text{Re}(y[n])\cos(-\hat{\phi}) - \text{Im}(y[n])\sin(-\hat{\phi})$$
3. **Stage 2: Residual Refinement**: A secondary deep backbone identifies and corrects fine-grained phase jitter $\Delta\phi$ to minimize the residual error floor.

### 3.2 Visual Verification: Phase 1 Data Generation
Initial signal reception is characterized by high entropy and phase-induced displacement, as visualized in the noisy QPSK constellation below:

![Phase 1 Verification: Noisy QPSK Signal](https://github.com/user-attachments/assets/56720cab-2bb0-4b5c-9477-42a056b98ba7)

---

## 4. Performance Metrics and Analysis

### 4.1 Comparative Waterfall Analysis
The model's Bit Error Rate (BER) was benchmarked against the theoretical QPSK limit, demonstrating the AI's ability to track physical limits until hitting a residual floor.

![Phase 4: Waterfall Curve (AI vs Physics)](https://github.com/user-attachments/assets/f08ceadf-4599-491a-b1dd-4494e56d16a0)

### 4.2 Statistical Results
* **Observed Bit Error Rate (BER)**: 0.0078 at 8.0dB SNR.
* **Optimal Bit Loss**: 0.0170 achieved during the hardening phase.
* **Phase Error Resolution**: Sub-6 degree Mean Absolute Error (MAE) achieved through 100+ epochs of neural training.

---

## 5. Final Performance Evaluation
The system was validated using a tri-modal comparison involving the **SYNAPSE-RAN AI Decoder**, a **Naive Baseline** (zero phase correction), and the **Theoretical QPSK Limit**.

### 5.1 Comparative Mitigation
The AI-native receiver successfully mitigates the catastrophic failure of the Naive Baseline ($BER \approx 0.25$), restoring signal integrity across the sweep range.

| SNR (dB) | AI BER (Observed) | AI Gain over Naive (dB) | Status |
| :--- | :--- | :--- | :--- |
| 4.0 | 0.1042 | +3.9 dB | Strong Gain |
| 10.0 | 0.0165 | +11.8 dB | Neural Lock |
| 28.0 | 0.0022 | +20.5 dB | Superiority Ceiling |

### 5.2 The AI Correction Advantage
* **Maximum Observed Gain**: +20.5 dB over non-AI receivers.
* **Irreducible Floor**: $10^{-3}$ identified as a consequence of residual phase jitter in the neural estimator.
* **Optimization Impact**: The INT8 Quantized model preserves the observed AI gain while achieving significant architectural compression.

![SYNAPSE-RAN Final Performance Analysis](![final performance analysis](https://github.com/user-attachments/assets/37213879-df2e-4b39-8e09-f499ffacd424)
)

---

## 6. Hardware-Aware Optimization
To facilitate deployment on 6G edge nodes (e.g., NVIDIA Aerial or Nokia AirScale), the model underwent rigorous optimization.

### 6.1 INT8 Dynamic Quantization
The model was exported via **ONNX Opset 18** and simplified to resolve internal graph metadata.

| Metric | FP32 Baseline | INT8 Optimized |
| :--- | :--- | :--- |
| **Model Size** | 60.84 MB | 15.25 MB |
| **Memory Savings** | 0.0% | **74.9% reduction** |
| **Deployment Format** | .pth | .onnx (INT8) |

---

## 7. Real-Time Showcase
The final implementation provides a real-time visualization of the symbiotic decoding process, reconstructing the four-quadrant constellation with 99.2% accuracy at 8 dB SNR.

![SYNAPSE-RAN Showcase | Packet BER: 0.0078](https://github.com/user-attachments/assets/e0c7f0e1-b165-4979-bb6d-6268301190be)

---

## 8. Applicability and Future Research

### 8.1 Industrial Application
* **6G Radio Access Networks**: Replaces traditional, high-latency hardware blocks with software-defined neural entities.
* **Low-Power Edge-AI**: The 74.9% memory reduction allows deployment on hardware-constrained IoT gateways and small-cell base stations.

### 8.2 Future Research: Neural Belief Propagation
Current research is focused on integrating **Neural Belief Propagation (NBP)**. By embedding the Parity-Check Matrix $H$ into the neural decoder's loss function, the system will move toward an algebraic solution to the error floor:
$$H \cdot \hat{b}^T = \mathbf{0} \pmod 2$$
This transition is projected to drive the $10^{-3}$ BER toward $10^{-6}$ for mission-critical 6G applications.

---

## 9. Directory Structure
* **/core**: Contains the ResNet backbones, `SynapseNet` architecture, and the `QCLDPCEngine`.
* **/scripts**: Training cycles, INT8 quantization pipelines, and final performance evaluators.
* **/models**: Optimized INT8 ONNX artifacts and FP32 research checkpoints.
* **/docs**: Technical visualization, waterfall curves, and final analysis assets.
