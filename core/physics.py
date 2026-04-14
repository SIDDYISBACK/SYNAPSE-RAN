import numpy as np

class QCLDPCEngine:
    """
    QC-LDPC engine kept for its H-matrix (used in project documentation
    and future belief-propagation decoder integration).
    Encoding removed — generator matrix extraction was producing invalid
    codewords due to non-systematic H structure, corrupting training labels.
    """
    def __init__(self, base_rows=4, base_cols=8, Z=16):
        self.Z = Z
        self.base_rows = base_rows
        self.base_cols = base_cols
        self.B = self.generate_high_girth_base()
        self.H = self.lift_matrix()

    def generate_high_girth_base(self):
        B = -np.ones((self.base_rows, self.base_cols), dtype=int)
        for i in range(self.base_rows):
            for j in range(self.base_cols):
                for shift in range(self.Z):
                    valid = True
                    for i_prev in range(i):
                        for j_prev in range(j):
                            if (B[i_prev, j] != -1 and
                                    B[i, j_prev] != -1 and
                                    B[i_prev, j_prev] != -1):
                                if ((shift - B[i, j_prev]) % self.Z ==
                                        (B[i_prev, j] - B[i_prev, j_prev]) % self.Z):
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        B[i, j] = shift
                        break
        return B

    def lift_matrix(self):
        H = np.zeros((self.base_rows * self.Z, self.base_cols * self.Z), dtype=int)
        for i in range(self.base_rows):
            for j in range(self.base_cols):
                shift = self.B[i, j]
                if shift != -1:
                    H[i*self.Z:(i+1)*self.Z,
                      j*self.Z:(j+1)*self.Z] = np.roll(np.eye(self.Z), shift, axis=1)
        return H


class SymbioticChannel:
    @staticmethod
    def modulate_qpsk(bits):
        # 128 bits → 64 symbols, every symbol carries real data
        bits = bits.reshape(-1, 2)
        symbols = (1 - 2*bits[:, 0]) + 1j*(1 - 2*bits[:, 1])
        return symbols / np.sqrt(2)

    def apply_channel(self, symbols, snr_db, phase_offset):
        # Phase rotation + AWGN — clean, solvable channel
        rotated  = symbols * np.exp(1j * phase_offset)
        noise_std = np.sqrt(1 / (2 * 10**(snr_db / 10)))
        noise = noise_std * (
            np.random.randn(len(symbols)) +
            1j * np.random.randn(len(symbols))
        )
        return rotated + noise


def generate_training_data(num_packets=1000, snr_range=(5, 20)):
    chan = SymbioticChannel()

    inputs, labels_bits, labels_phase = [], [], []

    for _ in range(num_packets):
        snr   = np.random.uniform(*snr_range)
        phase = np.random.uniform(0, np.pi / 2)

        # Pure random bits — clean labels, no encoding corruption
        # The neural decoder learns the direct noisy-IQ → bits mapping
        raw_bits = np.random.randint(0, 2, 128)
        symbols  = chan.modulate_qpsk(raw_bits)
        received = chan.apply_channel(symbols, snr, phase)

        iq_data = np.stack([received.real, received.imag], axis=0)

        inputs.append(iq_data)
        labels_bits.append(raw_bits)
        labels_phase.append(phase)

    return (
        np.array(inputs),
        np.array(labels_bits),
        np.array(labels_phase)
    )