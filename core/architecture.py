import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    """
    Residual Block for 1D Signals. 
    Prevents vanishing gradients and preserves signal features through depth.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class SynapseNet(nn.Module):
    def __init__(self):
        super(SynapseNet, self).__init__()

        # --- Stage 1: The "Eye" (Coarse Feature Extraction) ---
        self.backbone = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResBlock1D(64), # Preserves initial pulse shapes
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResBlock1D(128),
            nn.Flatten()   # → 128 * 64 = 8192
        )

        self.phase_head_coarse = nn.Sequential(
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # Outputs [sin(phi), cos(phi)]
        )

        # --- Stage 2: The "Refiner" (Residual Feature Extraction) ---
        self.backbone2 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResBlock1D(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        self.phase_head_residual = nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        # --- Decoder Backbone: Processes the physical corrected signal ---
        self.decoder_backbone = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResBlock1D(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1), # Hardware-aware regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Sigmoid()
        )

    def _rotate(self, x, sincos):
        """
        Superior Rotation: Physically un-spins the signal.
        Includes L2 Normalization to prevent signal attenuation/gain errors.
        """
        # CRITICAL FIX: Normalize output to unit vector to keep signal amplitude stable
        sincos = F.normalize(sincos, p=2, dim=1)
        
        sin_phi = sincos[:, 0:1].unsqueeze(-1)
        cos_phi = sincos[:, 1:2].unsqueeze(-1)
        
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        
        # Standard complex rotation formula: e^(j*phi) correction
        corrected_real =  real * cos_phi + imag * sin_phi
        corrected_imag = -real * sin_phi + imag * cos_phi
        
        return torch.cat([corrected_real, corrected_imag], dim=1)

    def forward(self, x):
        # 1. Coarse Lock
        feat1 = self.backbone(x)
        p_coarse = self.phase_head_coarse(feat1)
        x_coarse = self._rotate(x, p_coarse)

        # 2. Residual Refinement
        feat2 = self.backbone2(x_coarse)
        p_res = self.phase_head_residual(feat2)
        x_refined = self._rotate(x_coarse, p_res)

        # 3. Final Neural Decoding
        dec_feat = self.decoder_backbone(x_refined)
        decoded_bits = self.decoder(dec_feat)

        return p_coarse, p_res, decoded_bits