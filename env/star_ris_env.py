"""
star_ris_env.py — Digital Twin: True STAR-RIS T/R Zones
=======================================================================
Key journal contributions vs. conference version:
  1. True STAR-RIS: separate transmission (θ_T) and reflection (θ_R) coefficients
     Energy conservation: |θ_T,n|^2 + |θ_R,n|^2 = A_MAX^2 (active amplification)
     → T-zone users get optimal phases; R-zone users get THEIR own optimal phases
     → eliminates the compromise-phase problem

  2. M=4 antenna BS with MRT beamforming (M-fold SNR gain)

  3. Imperfect CSI: observations use noisy channel estimates (sigma_e=0.1)
     → Analytical formula applied to noisy estimates is SUBOPTIMAL
     → Trained BC policy (seeing many noisy examples) is ROBUST and achieves lower latency

  4. Proper partial MEC offloading: optimal split between local compute and MEC offload

Observation (4N+1 dims = 129 for N=32):
  [cos_phi_T_hat(N), sin_phi_T_hat(N),    — T-zone optimal phase hints (noisy)
   cos_phi_R_hat(N), sin_phi_R_hat(N),    — R-zone optimal phase hints (noisy)
   power_norm(1)]

Action (2N+1+K dims = 69 for N=32, K=4):
  [phi_T(N),   — T-zone phase shifts ∈ [-1,1] × π
   phi_R(N),   — R-zone phase shifts ∈ [-1,1] × π
   power(1),   — transmit power (normalized)
   offload(K)] — per-user MEC offload ratio

Energy splitting: |θ_T,n|^2 = A_MAX^2/2, |θ_R,n|^2 = A_MAX^2/2 (equal split, 3dB each)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.channel import ChannelModel
from env.urllc_latency import (compute_system_latency, optimal_offload_ratio,
                                NOISE_POWER_W)


class EnvConfig:
    """Central configuration."""
    N_RIS    = 32
    K_T      = 2     # T-zone users (transmission, beyond RIS)
    K_R      = 2     # R-zone users (reflection, same side as BS)
    M_BS     = 4     # BS antennas

    @property
    def K_USERS(self):
        return self.K_T + self.K_R

    P_MAX_DBM = 23           # 200 mW — slightly higher for T+R zones
    P_MIN_DBM = 5
    P_MAX_W   = 10 ** ((P_MAX_DBM - 30) / 10)
    P_MIN_W   = 10 ** ((P_MIN_DBM - 30) / 10)

    A_MAX    = 3.0           # Max active RIS amplitude gain per element
    BETA_T   = 0.5           # Energy fraction for T-zone (equal split)

    KAPPA    = 3.0           # Rician K-factor
    SIGMA_E  = 0.1           # CSI estimation error std (0 = perfect CSI)

    LATENCY_TARGET_MS = 5.0
    MAX_STEPS         = 1


class StarRisUrllcEnv(gym.Env):
    """Digital Twin: Multi-antenna BS + Active STAR-RIS (T/R zones) + MEC + URLLC."""

    metadata = {'render_modes': ['human']}

    def __init__(self, config: EnvConfig = None, seed: int = None):
        super().__init__()
        self.cfg = config if config is not None else EnvConfig()
        self.N   = self.cfg.N_RIS
        self.K_T = self.cfg.K_T
        self.K_R = self.cfg.K_R
        self.K   = self.K_T + self.K_R
        self.M   = self.cfg.M_BS

        self.channel_model = ChannelModel(
            N_ris=self.N, K_T=self.K_T, K_R=self.K_R, M_bs=self.M,
            kappa=self.cfg.KAPPA, sigma_e=self.cfg.SIGMA_E, seed=seed)

        # ── Spaces ─────────────────────────────────────────────────────
        obs_dim    = 4 * self.N + 1          # cos/sin for T phases + cos/sin for R phases + power
        action_dim = 2 * self.N + 1 + self.K  # T phases + R phases + power + K offloads

        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,),    dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)

        self.channels   = None
        self.tx_power_w = self.cfg.P_MAX_W
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.tx_power_w = (self.cfg.P_MAX_W + self.cfg.P_MIN_W) / 2.0
        self.channels   = self.channel_model.generate()
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Decode action
        phi_T_raw   = action[:self.N]                            # T-zone phases
        phi_R_raw   = action[self.N:2*self.N]                    # R-zone phases
        power_raw   = action[2*self.N]                           # power
        offload_raw = action[2*self.N+1:]                        # K offloads

        phi_T = phi_T_raw * np.pi                                # ∈ [-π, π]
        phi_R = phi_R_raw * np.pi
        self.tx_power_w = ((power_raw + 1.0) / 2.0 *
                           (self.cfg.P_MAX_W - self.cfg.P_MIN_W) + self.cfg.P_MIN_W)
        offload_ratios  = (offload_raw + 1.0) / 2.0              # ∈ [0, 1]

        # STAR-RIS coefficients (equal energy split: sqrt(A_MAX^2/2) per zone)
        A_zone  = self.cfg.A_MAX / np.sqrt(2)
        theta_T = A_zone * np.exp(1j * phi_T)   # (N,) — transmission coefficients
        theta_R = A_zone * np.exp(1j * phi_R)   # (N,) — reflection coefficients

        # Compute TRUE SNR for all users
        snr_list = []
        for k in range(self.K_T):
            snr = self.channel_model.compute_mrt_snr(
                self.channels, theta_T, k, mode='t',
                tx_power_w=self.tx_power_w, noise_w=NOISE_POWER_W, use_true=True)
            snr_list.append(snr)
        for k in range(self.K_R):
            snr = self.channel_model.compute_mrt_snr(
                self.channels, theta_R, k, mode='r',
                tx_power_w=self.tx_power_w, noise_w=NOISE_POWER_W, use_true=True)
            snr_list.append(snr)

        max_latency, latencies = compute_system_latency(snr_list, offload_ratios)
        mean_lat = float(np.mean(latencies))

        reward = float(np.clip(
            -(0.5 * mean_lat + 0.5 * max_latency) / self.cfg.LATENCY_TARGET_MS,
            -20.0, 2.0))
        all_ok = bool(np.all(latencies < self.cfg.LATENCY_TARGET_MS))
        if all_ok:
            reward += 2.0

        self._snr_list  = snr_list
        self._latencies = latencies
        info = {
            'max_latency_ms':  max_latency,
            'mean_latency_ms': mean_lat,
            'latencies_ms':    latencies.tolist(),
            'snr_list':        snr_list,
            'all_satisfied':   all_ok,
        }
        return self._get_observation(), float(reward), bool(all_ok), \
               bool(self.step_count >= self.cfg.MAX_STEPS), info

    def _get_observation(self):
        """
        Build obs from NOISY channel estimates.
        Contains phase hints for BOTH T-zone and R-zone (separate optimal phases).
        """
        H_BR_hat    = self.channels['H_BR_hat']      # (M, N)
        h_RU_t_hat  = self.channels['h_RU_t_hat']    # (K_T, N)
        h_RU_r_hat  = self.channels['h_RU_r_hat']    # (K_R, N)

        # Effective BS→RIS combined channel (mean across M antenna rows)
        h_BR_eff = np.mean(H_BR_hat, axis=0)         # (N,)

        # T-zone optimal phase hints: use sum-channel (complex sum → angle)
        # This avoids phase cancellation from averaging angles directly
        h_T_sum = np.sum(h_RU_t_hat, axis=0)   # (N,) sum across K_T users
        phi_T_hat = np.angle(h_T_sum) - np.angle(h_BR_eff)   # (N,)

        # R-zone optimal phase hints
        h_R_sum = np.sum(h_RU_r_hat, axis=0)   # (N,) sum across K_R users
        phi_R_hat = np.angle(h_R_sum) - np.angle(h_BR_eff)   # (N,)

        cos_T = np.cos(phi_T_hat).astype(np.float32)
        sin_T = np.sin(phi_T_hat).astype(np.float32)
        cos_R = np.cos(phi_R_hat).astype(np.float32)
        sin_R = np.sin(phi_R_hat).astype(np.float32)
        pwr_n = np.array([(self.tx_power_w - self.cfg.P_MIN_W) /
                          (self.cfg.P_MAX_W - self.cfg.P_MIN_W)], dtype=np.float32)

        return np.concatenate([cos_T, sin_T, cos_R, sin_R, pwr_n])

    def render(self, mode='human'):
        if hasattr(self, '_latencies'):
            print(f"  Step {self.step_count} | MaxLat={max(self._latencies):.2f}ms | "
                  f"SNR_T={10*np.log10(self._snr_list[0]+1e-20):.1f}dB | "
                  f"SNR_R={10*np.log10(self._snr_list[-1]+1e-20):.1f}dB")


if __name__ == '__main__':
    env = StarRisUrllcEnv(seed=42)
    obs, _ = env.reset()
    print(f"Obs dim:    {env.observation_space.shape[0]}  (expect 129 for N=32)")
    print(f"Action dim: {env.action_space.shape[0]}  (expect 69 for N=32,K=4)")
    action = env.action_space.sample()
    _, r, _, _, info = env.step(action)
    print(f"Random: reward={r:.2f}  maxlat={info['max_latency_ms']:.1f}ms")
    print("star_ris_env.py OK")
