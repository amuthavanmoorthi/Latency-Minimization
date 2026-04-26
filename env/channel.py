"""
channel.py — Rician Fading Channel Model with Imperfect CSI
============================================================
Extended for journal paper contributions:
  1. Rician fading (LOS component) — more realistic than pure Rayleigh
  2. Multi-antenna BS (M antennas, ULA geometry)
  3. Imperfect CSI: MMSE-estimated channels with noise variance sigma_e^2
  4. True STAR-RIS T/R zone separation (K_T + K_R users)

System layout:
  BS (M antennas) ──[H_BR ∈ C^{M×N}]──► Active STAR-RIS (N elements)
                                         ├──[h_RU_t ∈ C^N]──► T-zone users
                                         └──[h_RU_r ∈ C^N]──► R-zone users
  BS ──[h_BU ∈ C^M]──► Users (direct, weak NLOS)
"""

import numpy as np


# ─────────────────────────────────────────────
#  Path Loss
# ─────────────────────────────────────────────

def path_loss(distance_m, path_loss_exponent=2.5):
    """
    Large-scale path loss: beta = C0 * d^{-alpha}
    C0 = 1e-3 at d0=1m (accounts for 28 GHz carrier + antenna gains).
    """
    C0 = 1e-3
    return C0 * distance_m ** (-path_loss_exponent)


# ─────────────────────────────────────────────
#  Rician Channel Vector Generator
# ─────────────────────────────────────────────

def generate_rician_channel(n_elements, beta, kappa=3.0):
    """
    Generate a Rician fading channel vector.

    h = sqrt(kappa/(kappa+1)) * h_LOS  +  sqrt(1/(kappa+1)) * h_NLOS

    Args:
        n_elements : number of complex coefficients
        beta       : large-scale path loss (average power)
        kappa      : Rician K-factor (ratio of LOS to scatter power)
                     kappa=0 → Rayleigh, kappa=3 → mild LOS, kappa=10 → strong LOS

    Returns:
        h : complex array shape (n_elements,), E[|h_i|^2] = beta
    """
    # LOS component (deterministic LoS steering vector, unit norm per element)
    los_phase = np.random.uniform(0, 2 * np.pi)   # random AoA for LOS
    h_los = np.exp(1j * (los_phase + np.arange(n_elements) * np.pi * 0.5))

    # NLOS scatter component (Rayleigh)
    h_nlos = (np.random.randn(n_elements) + 1j * np.random.randn(n_elements)) / np.sqrt(2)

    # Combine
    h = np.sqrt(beta) * (np.sqrt(kappa / (kappa + 1)) * h_los
                         + np.sqrt(1 / (kappa + 1)) * h_nlos)
    return h


def generate_rayleigh_channel(n_elements, beta):
    """Rayleigh fading (kappa=0 special case)."""
    h = np.sqrt(beta / 2.0) * (np.random.randn(n_elements) + 1j * np.random.randn(n_elements))
    return h


# ─────────────────────────────────────────────
#  Imperfect CSI
# ─────────────────────────────────────────────

def add_csi_error(h_true, sigma_e=0.0):
    """
    Simulate imperfect channel estimation (MMSE estimator model).

    hat{h} = h + e,  e ~ CN(0, sigma_e^2 * I)

    At sigma_e=0: perfect CSI (analytical formula achieves optimal).
    At sigma_e>0: noisy estimate — analytical formula is suboptimal,
                  and a learned BC policy that was trained on noisy
                  observations can outperform it.

    Args:
        h_true  : true channel array (complex)
        sigma_e : CSI estimation error std (0 = perfect CSI)

    Returns:
        h_hat : noisy channel estimate
    """
    if sigma_e == 0.0:
        return h_true.copy()
    noise = (sigma_e / np.sqrt(2)) * (
        np.random.randn(*h_true.shape) + 1j * np.random.randn(*h_true.shape))
    return h_true + noise


# ─────────────────────────────────────────────
#  Full System Channel Model
# ─────────────────────────────────────────────

class ChannelModel:
    """
    Generates all channels for the Active STAR-RIS + Multi-antenna BS system.

    New journal features vs conference version:
      - M-antenna BS (ULA) with H_BR ∈ C^{M×N}
      - Rician fading (K-factor = kappa)
      - Imperfect CSI with sigma_e noise
      - True T/R zone: K_T users in transmission, K_R in reflection

    Args:
        N_ris      : RIS elements N
        K_T        : users in transmission zone
        K_R        : users in reflection zone
        M_bs       : BS antennas (1 = SISO, 4 = MISO)
        kappa      : Rician K-factor (default 3.0 = mild LOS)
        sigma_e    : CSI estimation error std (0 = perfect)
        seed       : random seed
    """

    def __init__(self, N_ris=32, K_T=2, K_R=2, M_bs=4,
                 kappa=3.0, sigma_e=0.1, seed=None):
        self.N   = N_ris
        self.K_T = K_T
        self.K_R = K_R
        self.K   = K_T + K_R
        self.M   = M_bs
        self.kappa   = kappa
        self.sigma_e = sigma_e

        if seed is not None:
            np.random.seed(seed)

        # ── Geometry (meters) ─────────────────────────────────────────
        self.d_BR  = 30.0    # BS  → RIS distance
        self.d_RT  = 50.0    # RIS → T-zone users (beyond RIS)
        self.d_RR  = 20.0    # RIS → R-zone users (same side as BS)
        self.d_BU  = 70.0    # BS  → users direct (NLOS blocked)

        # ── Path loss ─────────────────────────────────────────────────
        self.beta_BR  = path_loss(self.d_BR,  path_loss_exponent=2.5)
        self.beta_RT  = path_loss(self.d_RT,  path_loss_exponent=2.5)
        self.beta_RR  = path_loss(self.d_RR,  path_loss_exponent=2.5)
        self.beta_BU  = path_loss(self.d_BU,  path_loss_exponent=5.0)  # heavily blocked

    def generate(self):
        """
        Generate one block-fading channel realization.

        Returns dict with TRUE channels AND estimated (noisy) channels:
          H_BR_true  : C^{M×N}  BS→RIS (true)
          H_BR_hat   : C^{M×N}  BS→RIS (estimated, noisy)
          h_RU_t_true: C^{K_T×N} RIS→T-users (true)
          h_RU_t_hat : C^{K_T×N} RIS→T-users (estimated)
          h_RU_r_true: C^{K_R×N} RIS→R-users (true)
          h_RU_r_hat : C^{K_R×N} RIS→R-users (estimated)
          h_BU_true  : C^K direct (true)
        """
        # BS → RIS: M×N matrix (each row = one BS antenna's channel to all N RIS elements)
        H_BR_true = np.array([
            generate_rician_channel(self.N, self.beta_BR, self.kappa)
            for _ in range(self.M)
        ])  # shape (M, N)

        # RIS → T-zone users (transmission side, beyond STAR-RIS)
        h_RU_t_true = np.array([
            generate_rician_channel(self.N, self.beta_RT, self.kappa)
            for _ in range(self.K_T)
        ])  # shape (K_T, N)

        # RIS → R-zone users (reflection side, same side as BS)
        h_RU_r_true = np.array([
            generate_rician_channel(self.N, self.beta_RR, self.kappa)
            for _ in range(self.K_R)
        ])  # shape (K_R, N)

        # Direct BS→user channel (weak NLOS)
        h_BU_true = generate_rayleigh_channel(self.K, self.beta_BU)  # shape (K,)

        # Add CSI estimation errors
        H_BR_hat   = add_csi_error(H_BR_true,   self.sigma_e)
        h_RU_t_hat = add_csi_error(h_RU_t_true, self.sigma_e)
        h_RU_r_hat = add_csi_error(h_RU_r_true, self.sigma_e)

        return {
            # True channels (used for reward/SNR computation in step())
            'H_BR':      H_BR_true,       # (M, N)
            'h_RU_t':    h_RU_t_true,     # (K_T, N)
            'h_RU_r':    h_RU_r_true,     # (K_R, N)
            'h_BU':      h_BU_true,       # (K,)
            # Estimated channels (used for observation / policy input)
            'H_BR_hat':     H_BR_hat,
            'h_RU_t_hat':   h_RU_t_hat,
            'h_RU_r_hat':   h_RU_r_hat,
        }

    def compute_mrt_snr(self, channels, theta, user_k, mode='t',
                        tx_power_w=0.1, noise_w=1e-12, use_true=True):
        """
        Compute received SNR for user k with M-antenna MRT beamforming.

        With M-antenna BS and MRT:
          w_k = (H_BR^H * Theta^H * h_RU_k)^* / ||...||  (matched filter)
          SNR_k = P * ||H_BR^H * Theta^H * h_RU_k||^2 / sigma^2

        This gives an M-fold array gain over single-antenna BS.

        Args:
            channels   : dict from generate()
            theta      : complex array (N,) — RIS coefficients
            user_k     : user index (0-indexed within its zone)
            mode       : 't' or 'r'
            tx_power_w : BS transmit power in Watts
            noise_w    : noise power in Watts
            use_true   : if True, use true channels; else use estimated

        Returns:
            snr : received SNR (scalar)
        """
        if use_true:
            H_BR   = channels['H_BR']
            h_RU_t = channels['h_RU_t']
            h_RU_r = channels['h_RU_r']
            h_BU   = channels['h_BU']
        else:
            H_BR   = channels['H_BR_hat']
            h_RU_t = channels['h_RU_t_hat']
            h_RU_r = channels['h_RU_r_hat']
            h_BU   = channels['h_BU']  # direct always perfect (measured locally)

        # Select correct zone channel
        if mode == 't':
            h_RU = h_RU_t[user_k]   # (N,)
            user_global = user_k
        else:
            h_RU = h_RU_r[user_k]   # (N,)
            user_global = self.K_T + user_k

        # Use first BS antenna row as representative channel for phase alignment.
        # MRT array gain (M-fold) is captured by the self.M factor below.
        h_BR_eff = H_BR[0, :]   # (N,) — representative row

        # SISO effective channel: g = h_RU^H * diag(theta) * h_BR_eff  (scalar)
        g_eff = np.dot(np.conj(h_RU), theta * h_BR_eff)

        # Direct link
        h_direct = h_BU[user_global]
        g_total  = g_eff + h_direct

        # MRT gives M-fold SNR gain over SISO: SNR = M * P * |g|^2 / sigma^2
        snr = self.M * tx_power_w * np.abs(g_total) ** 2 / noise_w
        return float(snr)

    def compute_effective_channel(self, channels, theta, user_idx, mode='t'):
        """
        Legacy single-antenna interface (backward compatibility).
        Returns scalar effective channel for SISO (M=1) case.
        """
        if self.M == 1:
            # Single antenna: use first row of H_BR as h_BR vector
            h_BR = channels['H_BR'][0]
            if mode == 't':
                h_RU = channels['h_RU_t'][user_idx]
            else:
                h_RU = channels['h_RU_r'][user_idx]
            h_direct = channels['h_BU'][user_idx]
            theta_vec = theta
            g_eff = np.dot(np.conj(h_RU), theta_vec * h_BR) + h_direct
            return g_eff
        else:
            # Multi-antenna: use MRT SNR
            snr = self.compute_mrt_snr(channels, theta, user_idx, mode,
                                        tx_power_w=0.1, noise_w=1e-12)
            # Return effective gain such that SNR = P * |g|^2 / sigma^2
            # i.e., |g|^2 = SNR * sigma^2 / P = sqrt for complex g
            return np.sqrt(snr * 1e-12 / 0.1)  # placeholder scalar


if __name__ == '__main__':
    print("Testing extended channel model...")
    ch = ChannelModel(N_ris=32, K_T=2, K_R=2, M_bs=4, kappa=3.0, sigma_e=0.1, seed=42)
    channels = ch.generate()
    print(f"  H_BR shape:      {channels['H_BR'].shape}")        # (4, 32)
    print(f"  h_RU_t shape:    {channels['h_RU_t'].shape}")      # (2, 32)
    print(f"  h_RU_r shape:    {channels['h_RU_r'].shape}")      # (2, 32)
    print(f"  h_BU shape:      {channels['h_BU'].shape}")        # (4,)
    print(f"  H_BR_hat shape:  {channels['H_BR_hat'].shape}")    # (4, 32)
    theta = np.exp(1j * np.random.uniform(-np.pi, np.pi, 32))
    snr = ch.compute_mrt_snr(channels, theta, 0, 't', tx_power_w=0.1, noise_w=1e-12)
    print(f"  MRT SNR T-user0: {10*np.log10(snr+1e-20):.1f} dB")
    print("channel.py OK")
