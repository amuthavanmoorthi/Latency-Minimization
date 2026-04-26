"""
agent/ddpg_rl.py — Pure-PyTorch DDPG Baseline
===============================================
Deep Deterministic Policy Gradient (DDPG) applied to the Active STAR-RIS
URLLC environment. This implementation serves as an honest comparison
baseline against the proposed DT-BC approach.

DDPG components:
  Actor   : same 4-layer residual MLP as DT-BC (fair comparison)
  Critic  : (obs, action) → scalar Q-value (2-head MLP)
  Buffer  : circular replay buffer (1M transitions)
  Noise   : Gaussian exploration noise, annealing σ: 0.3 → 0.05
  Updates : soft target update  τ = 0.005
  Loss    : Bellman TD(0) for critic; policy gradient for actor

Environment note:
  MAX_STEPS = 1 (single-step contextual bandit).
  Q(s,a) = R(s,a) exactly — no future rewards.
  Even in this simplest case, the 69-dimensional continuous action space
  causes the critic's value landscape to be poorly estimated, and the actor
  gradient becomes noisy, often diverging.  This is the core failure mode
  that DT-BC avoids by replacing the critic with an analytical oracle.

Usage:
  python agent/ddpg_rl.py          # quick test
  python main.py --mode ddpg       # full training + save log
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from env.star_ris_env import StarRisUrllcEnv, EnvConfig
from env.urllc_latency import NOISE_POWER_W


# ══════════════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════════════

class DDPGActor(nn.Module):
    """
    4-layer Residual MLP actor — identical architecture to BC policy for
    a fair comparison.  obs → tanh action ∈ (-1, 1)^{action_dim}.
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

        # Initialise output layer to small weights so initial actions are near 0
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias,   -3e-3, 3e-3)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h)) + self.act(self.skip(x))   # residual
        h = self.act(self.fc4(h))
        return torch.tanh(self.out(h))


class DDPGCritic(nn.Module):
    """
    Q-network: (obs, action) → scalar Q-value.

    Architecture: concatenate (obs, action) → 3-layer MLP.
    Two separate output heads for variance reduction (similar to TD3 approach).
    We train both; the actor uses the minimum for conservative policy updates.
    """
    def __init__(self, obs_dim, action_dim, hidden=512):
        super().__init__()
        in_dim = obs_dim + action_dim

        # Head 1
        self.q1_fc1 = nn.Linear(in_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_fc3 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)

        # Head 2 (for min-Q clipping)
        self.q2_fc1 = nn.Linear(in_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_fc3 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)

        h1 = self.act(self.q1_fc1(x))
        h1 = self.act(self.q1_fc2(h1))
        h1 = self.act(self.q1_fc3(h1))
        q1 = self.q1_out(h1)

        h2 = self.act(self.q2_fc1(x))
        h2 = self.act(self.q2_fc2(h2))
        h2 = self.act(self.q2_fc3(h2))
        q2 = self.q2_out(h2)

        return q1, q2

    def q1_only(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        h = self.act(self.q1_fc1(x))
        h = self.act(self.q1_fc2(h))
        h = self.act(self.q1_fc3(h))
        return self.q1_out(h)


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Circular replay buffer for off-policy DDPG training.
    Stores (obs, action, reward, next_obs, done) tuples.

    Since MAX_STEPS=1, done=True always (contextual bandit).
    next_obs is still stored for generality.
    """
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.capacity   = capacity
        self.ptr        = 0
        self.size       = 0
        self.device     = device

        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      device=self.device),
            torch.tensor(self.actions[idx],  device=self.device),
            torch.tensor(self.rewards[idx],  device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.dones[idx],    device=self.device),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
# Soft target update
# ══════════════════════════════════════════════════════════════════════════════

def soft_update(target, source, tau=0.005):
    """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target."""
    for t_p, s_p in zip(target.parameters(), source.parameters()):
        t_p.data.copy_(tau * s_p.data + (1.0 - tau) * t_p.data)


# ══════════════════════════════════════════════════════════════════════════════
# Main DDPG Training Function
# ══════════════════════════════════════════════════════════════════════════════

def train_ddpg_rl(total_steps=300_000,
                  n_ris_elements=32,
                  k_t=2, k_r=2, m_bs=4,
                  sigma_e=0.1, kappa=3.0,
                  save_dir='results',
                  seed=42,
                  verbose=1,
                  # DDPG hyper-parameters
                  buffer_capacity=1_000_000,
                  batch_size=256,
                  gamma=0.99,           # discount (irrelevant for single-step)
                  tau=0.005,            # soft update rate
                  actor_lr=1e-4,
                  critic_lr=3e-4,
                  noise_start=0.3,      # initial exploration noise σ
                  noise_end=0.05,       # final exploration noise σ
                  noise_decay=150_000,  # steps to decay over
                  warmup_steps=5_000,   # random actions before learning starts
                  update_every=1,       # env steps between gradient updates
                  eval_every=2_000,     # steps between reward evaluations
                  ):
    """
    Train a DDPG agent on the Active STAR-RIS URLLC environment.

    Returns
    -------
    actor      : trained DDPGActor (torch.nn.Module)
    reward_log : list of dicts with 'timestep', 'mean_reward', 'mean_latency_ms'
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    K = k_t + k_r
    cfg = EnvConfig()
    cfg.N_RIS   = n_ris_elements
    cfg.K_T     = k_t
    cfg.K_R     = k_r
    cfg.M_BS    = m_bs
    cfg.SIGMA_E = sigma_e
    cfg.KAPPA   = kappa
    env = StarRisUrllcEnv(config=cfg, seed=seed)

    N          = n_ris_elements
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print("DDPG Training (on-policy RL baseline)")
        print(f"  N={N}, K_T={k_t}, K_R={k_r}, M={m_bs}, sigma_e={sigma_e}")
        print(f"  Obs dim={obs_dim}, Action dim={action_dim}")
        print(f"  Total steps={total_steps:,}, Device={device}")
        print(f"  Warmup={warmup_steps:,}, Buffer={buffer_capacity:,}")
        print("=" * 60)

    # ── Networks ───────────────────────────────────────────────────────────
    actor        = DDPGActor(obs_dim, action_dim).to(device)
    critic       = DDPGCritic(obs_dim, action_dim).to(device)
    actor_target = DDPGActor(obs_dim, action_dim).to(device)
    critic_target = DDPGCritic(obs_dim, action_dim).to(device)

    # Hard copy to initialise targets
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
    actor_target.eval()
    critic_target.eval()

    # ── Optimisers ─────────────────────────────────────────────────────────
    actor_opt  = optim.Adam(actor.parameters(),  lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    # ── Replay Buffer ───────────────────────────────────────────────────────
    buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, device=device)

    # ── Logging ─────────────────────────────────────────────────────────────
    reward_log  = []
    start_time  = time.time()

    # ── Evaluation helper ───────────────────────────────────────────────────
    def _evaluate(n_eval=200):
        actor.eval()
        rewards, lats = [], []
        with torch.no_grad():
            for _ in range(n_eval):
                obs, _ = env.reset()
                obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(obs_t).squeeze(0).cpu().numpy()
                _, rew, _, _, info = env.step(action)
                rewards.append(rew)
                lats.append(info['max_latency_ms'])
        actor.train()
        return float(np.mean(rewards)), float(np.mean(lats))

    # ══════════════════════════════════════════════════════════════════════
    # Main Training Loop
    # ══════════════════════════════════════════════════════════════════════
    obs, _ = env.reset()
    total_critic_loss = 0.0
    total_actor_loss  = 0.0
    n_updates         = 0

    for step in range(1, total_steps + 1):

        # ── Exploration noise (linearly anneals from noise_start → noise_end) ──
        frac  = min(step / noise_decay, 1.0)
        noise_std = noise_start + frac * (noise_end - noise_start)

        # ── Select action ──────────────────────────────────────────────────
        if step <= warmup_steps:
            # Pure random exploration during warm-up
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = actor(obs_t).squeeze(0).cpu().numpy()
            # Add Gaussian exploration noise, clipped to valid action range
            noise  = np.random.normal(0, noise_std, size=action_dim).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

        # ── Step environment ────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))

        # Since MAX_STEPS=1, always reset
        obs, _ = env.reset()

        # ── Update networks ─────────────────────────────────────────────────
        if len(buffer) >= batch_size and step > warmup_steps and step % update_every == 0:

            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size)

            # ── Critic update ─────────────────────────────────────────────
            with torch.no_grad():
                next_action = actor_target(b_next)
                # Add small target-policy smoothing noise (TD3-style)
                smooth_noise = torch.clamp(
                    torch.randn_like(next_action) * 0.1, -0.2, 0.2)
                next_action  = torch.clamp(next_action + smooth_noise, -1.0, 1.0)

                q1_tgt, q2_tgt = critic_target(b_next, next_action)
                q_tgt = b_rew + gamma * (1.0 - b_done) * torch.min(q1_tgt, q2_tgt)

            q1, q2 = critic(b_obs, b_act)
            critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            # ── Actor update (delayed — every 2 critic steps, like TD3) ──
            if n_updates % 2 == 0:
                actor_loss = -critic.q1_only(b_obs, actor(b_obs)).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_opt.step()

                # Soft target updates
                soft_update(actor_target,  actor,  tau)
                soft_update(critic_target, critic, tau)

                total_actor_loss += actor_loss.item()

            total_critic_loss += critic_loss.item()
            n_updates         += 1

        # ── Periodic evaluation ─────────────────────────────────────────────
        if step % eval_every == 0:
            mean_rew, mean_lat = _evaluate(n_eval=200)
            elapsed = (time.time() - start_time) / 60.0
            reward_log.append({
                'timestep':       step,
                'mean_reward':    mean_rew,
                'mean_latency_ms': mean_lat,
                'noise_std':      noise_std,
                'critic_loss':    total_critic_loss / max(n_updates, 1),
                'actor_loss':     total_actor_loss  / max(n_updates // 2, 1),
            })
            if verbose:
                print(f"  Step {step:7,}/{total_steps:,} | "
                      f"Reward={mean_rew:+.3f} | Lat={mean_lat:.2f}ms | "
                      f"Noise={noise_std:.3f} | Buf={len(buffer):,} | "
                      f"Time={elapsed:.1f}min")

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, f"ddpg_rl_N{n_ris_elements}_K{K}")
    os.makedirs(model_dir, exist_ok=True)

    torch.save({
        'actor_state_dict':  actor.state_dict(),
        'obs_dim':           obs_dim,
        'action_dim':        action_dim,
        'n_ris':             n_ris_elements,
        'k_t':               k_t,
        'k_r':               k_r,
        'sigma_e':           sigma_e,
        'total_steps':       total_steps,
    }, os.path.join(model_dir, 'actor.pt'))

    log_path = os.path.join(save_dir, f"ddpg_rl_rewards_N{n_ris_elements}_K{K}.npy")
    np.save(log_path, reward_log)

    elapsed_total = (time.time() - start_time) / 60.0
    if verbose and reward_log:
        best_rew = max(r['mean_reward'] for r in reward_log)
        best_lat = min(r['mean_latency_ms'] for r in reward_log)
        print(f"\nDDPG training complete in {elapsed_total:.1f} min")
        print(f"  Best reward:  {best_rew:.3f}")
        print(f"  Best latency: {best_lat:.2f} ms")
        if best_lat < 5.0:
            print("  ✓ DDPG reached URLLC target (rare for 69-dim action space)")
        else:
            print("  ✗ DDPG did NOT reach URLLC target — consistent with paper claim")
        print(f"  Model saved: {model_dir}/actor.pt")

    return actor, reward_log


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate DDPG Actor
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ddpg_rl(model_dir, n_ris_elements=32, k_users=4,
                     n_episodes=150, seed=0):
    """
    Load a saved DDPG actor and evaluate it.

    Returns dict: mean_latency_ms, latencies, satisfaction_rate, mean_reward
    """
    pt_file = os.path.join(model_dir, 'actor.pt')
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"No DDPG actor at {pt_file}")

    ckpt  = torch.load(pt_file, map_location='cpu')
    actor = DDPGActor(ckpt['obs_dim'], ckpt['action_dim'])
    actor.load_state_dict(ckpt['actor_state_dict'])
    actor.eval()

    cfg = EnvConfig()
    cfg.N_RIS = n_ris_elements
    cfg.K_T   = max(1, k_users // 2)
    cfg.K_R   = max(1, k_users // 2)
    env = StarRisUrllcEnv(config=cfg, seed=seed)

    lats, rewards = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            a      = actor(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            a      = a.squeeze(0).numpy()
            _, rew, _, _, info = env.step(a)
            lats.append(info['max_latency_ms'])
            rewards.append(rew)

    lats = np.array(lats)
    return {
        'mean_latency_ms':   float(np.mean(lats)),
        'latencies':         lats,
        'satisfaction_rate': float(np.mean(lats < 5.0)),
        'mean_reward':       float(np.mean(rewards)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point for quick test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Quick DDPG test (10k steps)...")
    actor, log = train_ddpg_rl(
        total_steps=10_000,
        n_ris_elements=32,
        k_t=2, k_r=2, m_bs=4,
        sigma_e=0.1,
        save_dir='/tmp/test_ddpg',
        seed=0,
        verbose=1,
        warmup_steps=1_000,
        eval_every=1_000,
    )
    print(f"Log entries: {len(log)}")
    if log:
        print(f"Final reward: {log[-1]['mean_reward']:.3f} | "
              f"Latency: {log[-1]['mean_latency_ms']:.2f} ms")
    print("ddpg_rl.py OK")
