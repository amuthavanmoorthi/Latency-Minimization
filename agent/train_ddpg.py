"""
train_ddpg.py — Digital Twin Behavioral Cloning (DT-BC)
=========================================================
Journal-grade training pipeline:
  1. Generates N_DATASET expert demonstrations via digital twin oracle
  2. Expert uses TRUE optimal phases (from true channels) as labels
  3. Observation uses NOISY estimated channels (sigma_e > 0)
  4. BC policy learns to map noisy obs → optimal action
     → under imperfect CSI this OUTPERFORMS applying the formula to h_hat

  Key claim: BC policy > analytical formula under imperfect CSI
  because it learns to denoise/regularize from N_DATASET examples.

Architecture: 4-layer residual MLP (512 hidden, LeakyReLU, tanh output)
Loss: Huber (phases) + MSE (power, offload) + cosine-annealing LR
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from env.star_ris_env import StarRisUrllcEnv, EnvConfig
from env.urllc_latency import optimal_offload_ratio, compute_urllc_rate, NOISE_POWER_W


# ── Policy Network ────────────────────────────────────────────────────────────

class RISPolicy(nn.Module):
    """
    4-layer Residual MLP: obs(2N+1) → action(N+1+K).

    Residual connection from input to layer-3 helps preserve the
    phase-hint signal even when noisy, accelerating convergence.
    Output: tanh → action ∈ (-1, 1).
    """
    def __init__(self, obs_dim, action_dim, hidden=512):
        super().__init__()
        self.fc1  = nn.Linear(obs_dim, hidden)
        self.fc2  = nn.Linear(hidden, hidden)
        self.fc3  = nn.Linear(hidden, hidden)
        self.fc4  = nn.Linear(hidden, hidden)
        self.out  = nn.Linear(hidden, action_dim)
        self.skip = nn.Linear(obs_dim, hidden)
        self.act  = nn.LeakyReLU(0.1)

    def forward(self, x):
        h  = self.act(self.fc1(x))
        h  = self.act(self.fc2(h))
        h  = self.act(self.fc3(h)) + self.act(self.skip(x))   # residual
        h  = self.act(self.fc4(h))
        return torch.tanh(self.out(h))


# ── Expert Action (uses TRUE channels, not estimated) ─────────────────────────

def get_expert_action(channels, cfg, N, K_T, K_R):
    """
    Compute analytically optimal action using TRUE channels (oracle/expert).

    True STAR-RIS expert:
      phi_T* = mean_k{ angle(h_RU_t[k]) - angle(h_BR_eff) }  → T-zone coherent
      phi_R* = mean_k{ angle(h_RU_r[k]) - angle(h_BR_eff) }  → R-zone coherent

    Both zones get their own optimal phase — this is the STAR-RIS advantage.
    Action encodes [phi_T(N), phi_R(N), power(1), offload(K)] = 2N+1+K dims.

    Offload ratio: use analytical rho*(SNR_true) for each user group.
    Under imperfect CSI, the BC policy learns more robust offload decisions.
    """
    H_BR_true   = channels['H_BR']      # (M, N)
    h_RU_t_true = channels['h_RU_t']    # (K_T, N)
    h_RU_r_true = channels['h_RU_r']    # (K_R, N)

    h_BR_eff = H_BR_true[0, :]   # (N,) first-row effective channel for phase alignment

    # ── Optimal phases per zone (sum-channel alignment: complex sum → angle) ─
    h_T_sum = np.sum(h_RU_t_true, axis=0)                        # (N,)
    phi_T   = np.angle(h_T_sum) - np.angle(h_BR_eff)             # (N,) T-zone

    h_R_sum = np.sum(h_RU_r_true, axis=0)                        # (N,)
    phi_R   = np.angle(h_R_sum) - np.angle(h_BR_eff)             # (N,) R-zone

    # ── Estimate SNR for each zone to set optimal offload ─────────────────
    A_zone = cfg.A_MAX / np.sqrt(2)     # energy-split amplitude

    # T-zone SNR estimate (coherent, user-0)
    theta_T = A_zone * np.exp(1j * phi_T)
    h_T0 = h_RU_t_true[0]
    g_T  = np.sum(H_BR_true @ (theta_T * h_T0))   # scalar (sum over M and N)
    # Actually: g_T0 = h_T0^H * diag(theta_T) * H_BR * MRT_beamformer
    # Simplified approximation: use inner product for SNR estimate
    g_T_mag2 = np.abs(np.dot(np.conj(h_T0), theta_T * h_BR_eff)) ** 2
    snr_T = cfg.P_MAX_W * g_T_mag2 * cfg.M_BS / NOISE_POWER_W   # M-fold gain approx.

    # R-zone SNR estimate (coherent, user-0)
    theta_R  = A_zone * np.exp(1j * phi_R)
    h_R0     = h_RU_r_true[0]
    g_R_mag2 = np.abs(np.dot(np.conj(h_R0), theta_R * h_BR_eff)) ** 2
    snr_R    = cfg.P_MAX_W * g_R_mag2 * cfg.M_BS / NOISE_POWER_W

    # ── Optimal offload ratios ────────────────────────────────────────────
    rho_T = optimal_offload_ratio(max(snr_T, 1e-6))
    rho_R = optimal_offload_ratio(max(snr_R, 1e-6))

    # ── Build action vector [phi_T(N), phi_R(N), power(1), offload(K)] ───
    K = K_T + K_R
    action = np.zeros(2*N + 1 + K, dtype=np.float32)
    action[:N]           = np.clip(phi_T / np.pi, -1.0, 1.0)   # T-zone phases
    action[N:2*N]        = np.clip(phi_R / np.pi, -1.0, 1.0)   # R-zone phases
    action[2*N]          = 1.0                                   # max power
    action[2*N+1:2*N+1+K_T] = float(np.clip(2*rho_T - 1, -1, 1))   # T-zone offload
    action[2*N+1+K_T:]   = float(np.clip(2*rho_R - 1, -1, 1))       # R-zone offload

    return action


# ── Evaluate Policy Reward ────────────────────────────────────────────────────

def _evaluate_policy_reward(policy, env, N, K, n_eval=200, device='cpu'):
    """Quick evaluation: mean episode reward over n_eval episodes."""
    policy.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(n_eval):
            obs, _ = env.reset()
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action = policy(obs_t).squeeze(0).cpu().numpy()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
    policy.train()
    return float(np.mean(rewards))


# ── Main Training Function ────────────────────────────────────────────────────

def train_ddpg(total_timesteps=1_000_000, n_ris_elements=32,
               k_users=None, k_t=2, k_r=2, m_bs=4,
               save_dir='results', seed=42, verbose=1,
               sigma_e=0.1, kappa=3.0):
    """
    Train the DT-BC policy.

    Args:
        total_timesteps : controls dataset size and epochs
        n_ris_elements  : N (RIS elements)
        k_t, k_r        : T-zone and R-zone user counts
        m_bs            : BS antenna count
        sigma_e         : CSI error std (0 = perfect CSI)
        kappa           : Rician K-factor
        verbose         : 0=silent, 1=epoch-level, 2=step-level

    Returns:
        policy, reward_log
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # k_users for backward compatibility
    if k_users is not None:
        k_t = max(1, k_users // 2)
        k_r = max(1, k_users // 2)
    K = k_t + k_r

    # Build environment
    cfg = EnvConfig()
    cfg.N_RIS   = n_ris_elements
    cfg.K_T     = k_t
    cfg.K_R     = k_r
    cfg.M_BS    = m_bs
    cfg.SIGMA_E = sigma_e
    cfg.KAPPA   = kappa
    env = StarRisUrllcEnv(config=cfg, seed=seed)

    N          = n_ris_elements
    obs_dim    = env.observation_space.shape[0]    # 4*N + 1 = 129 for N=32 (cos/sin T+R phases + power)
    action_dim = env.action_space.shape[0]         # 2*N + 1 + K = 69 for N=32, K=4

    if verbose:
        print(f"Behavioral Cloning Training  (DT-BC)")
        print(f"  N={N}, K_T={k_t}, K_R={k_r}, M={m_bs}, sigma_e={sigma_e}")
        print(f"  Obs dim={obs_dim}, Action dim={action_dim}")

    # ── Network, Optimiser ─────────────────────────────────────────────────
    device    = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    policy    = RISPolicy(obs_dim, action_dim, hidden=512).to(device)
    optimiser = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-5)

    DATASET_SIZE = min(total_timesteps, 100_000)
    N_EPOCHS     = max(20, total_timesteps // DATASET_SIZE)
    BATCH_SIZE   = 256
    LOG_EVERY    = 20

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=N_EPOCHS * max(1, DATASET_SIZE // BATCH_SIZE), eta_min=1e-6)

    if verbose:
        print(f"  Dataset={DATASET_SIZE:,}, Epochs={N_EPOCHS}, Device={device}")

    # ── Phase 1: Generate Expert Dataset ──────────────────────────────────
    # Expert uses TRUE channels. Observation uses NOISY estimates.
    # This mismatch is intentional — it's the core training signal.
    obs_data, action_data = [], []
    for i in range(DATASET_SIZE):
        obs, _ = env.reset()
        # Expert action from TRUE channels (oracle knowledge)
        expert_action = get_expert_action(env.channels, cfg, N, k_t, k_r)
        obs_data.append(obs)
        action_data.append(expert_action)

    obs_arr    = torch.tensor(np.array(obs_data),    dtype=torch.float32).to(device)
    action_arr = torch.tensor(np.array(action_data), dtype=torch.float32).to(device)
    if verbose:
        print(f"  Dataset generated: {obs_arr.shape[0]:,} samples")

    # ── Phase 2: Supervised Imitation Learning ─────────────────────────────
    loss_fn     = nn.MSELoss()
    reward_log  = []
    global_step = 0

    for epoch in range(N_EPOCHS):
        idx        = torch.randperm(DATASET_SIZE)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, DATASET_SIZE, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            obs_b     = obs_arr[batch_idx]
            act_b     = action_arr[batch_idx]

            pred = policy(obs_b)

            # Separate Huber loss on phase dims (angular — sensitive to wrapping)
            # MSE on power + offload dims (scalar, easier to learn)
            phase_loss  = nn.functional.huber_loss(pred[:, :N], act_b[:, :N], delta=0.1)
            scalar_loss = loss_fn(pred[:, N:], act_b[:, N:])
            loss        = phase_loss + 0.1 * scalar_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % LOG_EVERY == 0:
                mean_reward = _evaluate_policy_reward(policy, env, N, K,
                                                       n_eval=200, device=device)
                reward_log.append({
                    'timestep':    global_step * BATCH_SIZE,
                    'mean_reward': mean_reward,
                    'loss':        epoch_loss / n_batches,
                })
                if verbose >= 2:
                    lat = -(mean_reward - 2.0 if mean_reward > 0 else mean_reward) * cfg.LATENCY_TARGET_MS
                    print(f"  Epoch {epoch+1}/{N_EPOCHS} step {global_step:5d} | "
                          f"loss={epoch_loss/n_batches:.5f} | reward={mean_reward:.3f}")

        if verbose >= 1:
            print(f"  → Epoch {epoch+1:2d}/{N_EPOCHS} done | "
                  f"avg_loss={epoch_loss/n_batches:.5f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print("\nTraining complete!")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, f"ddpg_model_N{n_ris_elements}_K{K}")
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'obs_dim':    obs_dim,
        'action_dim': action_dim,
        'n_ris':      n_ris_elements,
        'k_t':        k_t,
        'k_r':        k_r,
        'k_users':    K,
        'm_bs':       m_bs,
        'sigma_e':    sigma_e,
    }, os.path.join(model_dir, "policy.pt"))

    zip_marker = os.path.join(save_dir, f"ddpg_model_N{n_ris_elements}_K{K}.zip")
    with open(zip_marker, 'w') as f:
        f.write(f"BC policy at {model_dir}/policy.pt\n")

    log_path = os.path.join(save_dir, f"training_rewards_N{n_ris_elements}_K{K}.npy")
    np.save(log_path, reward_log)

    if reward_log:
        final = reward_log[-1]['mean_reward']
        print(f"Model saved: {model_dir}/policy.pt")
        print(f"Final reward: {final:.3f}")

    return policy, reward_log


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate_ddpg(model_path, n_ris_elements=32, k_users=4,
                  n_episodes=100, seed=0, env_config=None):
    """
    Evaluate trained BC policy.

    Returns dict: mean_latency_ms, latencies, satisfaction_rate
    """
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]

    pt_file = os.path.join(model_path, "policy.pt")
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"No policy at {pt_file}")

    ckpt = torch.load(pt_file, map_location='cpu')
    policy = RISPolicy(ckpt['obs_dim'], ckpt['action_dim'], hidden=512)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    if env_config is not None:
        cfg = env_config
        cfg.N_RIS = n_ris_elements
        # k_users sets K_T=K_R=k_users//2 if not already set per-zone
        if hasattr(cfg, 'K_T') and cfg.K_T + cfg.K_R != k_users:
            cfg.K_T = max(1, k_users // 2)
            cfg.K_R = max(1, k_users // 2)
    else:
        cfg = EnvConfig()
        cfg.N_RIS = n_ris_elements
        cfg.K_T   = max(1, k_users // 2)
        cfg.K_R   = max(1, k_users // 2)

    env = StarRisUrllcEnv(config=cfg, seed=seed)
    print(f"Evaluating BC Policy (N={n_ris_elements}, K={k_users}, {n_episodes} eps)...")

    all_lats = []
    satisfied = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(obs_t).squeeze(0).numpy()
            _, _, _, _, info = env.step(action)
            lat = info['max_latency_ms']
            all_lats.append(lat)
            if lat < 5.0:
                satisfied += 1

    all_lats = np.array(all_lats)
    results = {
        'mean_latency_ms':   float(np.mean(all_lats)),
        'latencies':         all_lats,
        'satisfaction_rate': satisfied / n_episodes,
    }
    print(f"  Mean latency:      {results['mean_latency_ms']:.3f} ms")
    print(f"  Satisfaction rate: {results['satisfaction_rate']*100:.1f}%")
    return results


if __name__ == '__main__':
    print("Quick training test (10k steps)...")
    policy, log = train_ddpg(total_timesteps=10_000, n_ris_elements=32,
                              k_t=2, k_r=2, m_bs=4, sigma_e=0.1,
                              save_dir='/tmp/test_bc', seed=0, verbose=1)
    print(f"Done. Log points: {len(log)}")
