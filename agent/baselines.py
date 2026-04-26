"""
agent/baselines.py — Baseline Comparison Algorithms
====================================================
What this file does:
  Implements 3 baseline policies that we compare against DDPG.
  In every IEEE paper, you MUST compare your method against baselines
  to prove it is actually better. Without baselines, reviewers reject the paper.

Our 4 methods (plotted as 4 curves in each result figure):
  1. DDPG (Ours)    — our proposed method (trained in train_ddpg.py)
  2. Random Policy  — completely random actions → worst case lower bound
  3. Fixed Policy   — fixed max-power + equal phase shifts (naive baseline)
  4. No-RIS         — RIS disabled (theta = 0) → shows the VALUE of having RIS

Why these specific baselines?
  - Random: shows the floor — how bad is "no intelligence"
  - Fixed:  shows a simple heuristic (like manual configuration)
  - No-RIS: directly shows contribution of the Active STAR-RIS component
  If your DDPG beats all 3, the paper proves the RIS + DRL combination works.

Each baseline function takes the same StarRisUrllcEnv and returns the same
result format as evaluate_ddpg() in train_ddpg.py, so they are directly comparable.
"""

import os
import sys
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from env.star_ris_env import StarRisUrllcEnv, EnvConfig
from env.urllc_latency import compute_snr, compute_system_latency, NOISE_POWER_W


# ── Helper: Run one baseline policy for N episodes ───────────────────────────

def run_policy(env, get_action_fn, n_episodes=100, label="Baseline"):
    """
    Run any policy for n_episodes and collect latency results.

    Args:
        env           : StarRisUrllcEnv instance
        get_action_fn : function(obs) → action array
                        Takes the current observation, returns action vector
        n_episodes    : number of episodes to run
        label         : name for printing progress

    Returns:
        dict with keys:
          'mean_latency_ms'  : average worst-case latency across all episodes
          'latencies'        : array of best latencies per episode (for CDF)
          'satisfaction_rate': fraction of episodes meeting < 5ms URLLC target
    """
    print(f"Evaluating: {label} ({n_episodes} episodes)...")

    all_latencies = []
    satisfied_count = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_latencies = []

        done = False
        while not done:
            action = get_action_fn(obs)               # Policy decides action
            obs, reward, term, trunc, info = env.step(action)
            episode_latencies.append(info['max_latency_ms'])
            done = term or trunc

        # Mean of last 20% of steps = converged steady-state performance.
        # Avoids giving random policies credit for lucky single-step hits.
        n_tail = max(1, len(episode_latencies) // 5)
        converged_latency = float(np.mean(episode_latencies[-n_tail:]))
        all_latencies.append(converged_latency)

        if converged_latency < 5.0:
            satisfied_count += 1

    all_latencies = np.array(all_latencies)
    results = {
        'mean_latency_ms': np.mean(all_latencies),
        'latencies': all_latencies,
        'satisfaction_rate': satisfied_count / n_episodes,
        'label': label
    }

    print(f"  Mean latency: {results['mean_latency_ms']:.3f} ms | "
          f"Satisfied: {results['satisfaction_rate']*100:.1f}%")
    return results


# ── Baseline 1: Random Policy ─────────────────────────────────────────────────

def evaluate_random_policy(n_ris_elements=32, k_users=2, n_episodes=100, seed=1):
    """
    Random policy: sample a completely random action every step.

    This is the WORST expected performance — no intelligence at all.
    If DDPG is not better than random, the model failed to learn.

    In the paper: shown as "Random Benchmark" curve (dashed, gray).
    """
    config = EnvConfig()
    config.N_RIS = n_ris_elements
    config.K_T = k_users // 2; config.K_R = k_users // 2
    config.MAX_STEPS = 50

    env = StarRisUrllcEnv(config=config, seed=seed)

    def random_action(obs):
        # action_space.sample() gives a uniformly random action in [-1, 1]
        return env.action_space.sample()

    return run_policy(env, random_action, n_episodes, label="Random Policy")


# ── Baseline 2: Fixed Policy (Max Power + Equal Phase Shifts) ─────────────────

def evaluate_fixed_policy(n_ris_elements=32, k_users=2, n_episodes=100, seed=2):
    """
    Fixed policy: always use maximum transmit power and equal (zero) phase shifts.

    This represents a naive "set-and-forget" RIS configuration:
      - Phase shifts: all set to 0 (no beamforming)
      - Amplitudes:   all set to maximum (A_MAX)
      - Power:        maximum (P_MAX)
      - Offload:      50% for all users

    In the paper: shown as "Fixed Configuration" curve.
    This baseline shows that simply having an active RIS at max power
    is not enough — intelligent optimization (DDPG) is needed.
    """
    config = EnvConfig()
    config.N_RIS = n_ris_elements
    config.K_T = k_users // 2; config.K_R = k_users // 2
    config.MAX_STEPS = 50

    env = StarRisUrllcEnv(config=config, seed=seed)

    # Build the fixed action vector:
    # Phase shifts = 0 → maps to action value of -1 (since 0 = (a+1)/2 * 2pi → a=-1)
    # Amplitudes = A_MAX → maps to action value of +1
    # Power = P_MAX → maps to +1
    # Offload = 0.5 → maps to 0

    action_dim = env.action_space.shape[0]
    N = n_ris_elements
    K = k_users

    # Action vector is now: [N phases, 1 power, K offloads] (amplitudes removed)
    fixed_action = np.zeros(action_dim)

    # Phase shifts: all -1 → phase = 0 (no phase rotation, incoherent combining)
    fixed_action[0:N] = -1.0

    # Power: +1 → P_MAX (maximum transmit power)
    fixed_action[N] = 1.0

    # Offload ratios: 0 → 50% offload for each user
    fixed_action[N+1: N+1+K] = 0.0

    def fixed_action_fn(obs):
        return fixed_action.copy()   # Return same action every step

    return run_policy(env, fixed_action_fn, n_episodes, label="Fixed Policy")


# ── Baseline 3: No-RIS Policy ─────────────────────────────────────────────────

def evaluate_no_ris_policy(n_ris_elements=32, k_users=2, n_episodes=100, seed=3):
    """
    No-RIS baseline: set all RIS amplitudes to 0 (RIS completely disabled).

    This shows what the system performance would be WITHOUT any RIS.
    It directly quantifies the GAIN provided by the Active STAR-RIS.

    In the paper: shown as "Without RIS" curve.
    The gap between "Without RIS" and "DDPG (Ours)" = the RIS contribution.

    Note: In our model, setting amplitudes to A_MIN=1.0 (not 0) is closest
    to passive RIS (no amplification). For "no RIS", we set power to max
    but use random phase shifts (RIS provides no benefit when uncontrolled).
    """
    config = EnvConfig()
    config.N_RIS = n_ris_elements
    config.K_T = k_users // 2; config.K_R = k_users // 2
    config.MAX_STEPS = 50

    env = StarRisUrllcEnv(config=config, seed=seed)

    action_dim = env.action_space.shape[0]
    N = n_ris_elements
    K = k_users

    def no_ris_action(obs):
        """
        Action vector: [N phases, 1 power, K offloads] (amplitudes removed from action).
        NOTE: amplitude is fixed at A_MAX in the environment.
        This baseline models the case where phases are NOT optimized (random),
        simulating a passive/uncontrolled RIS without intelligent beamforming.
        """
        action = np.zeros(action_dim)

        # Phase shifts: random (no coherent beamforming — RIS not optimized)
        action[0:N] = np.random.uniform(-1, 1, N)

        # Power: max
        action[N] = 1.0

        # Offload: 50%
        action[N+1: N+1+K] = 0.0

        return action

    return run_policy(env, no_ris_action, n_episodes, label="No Active RIS (Passive)")


# ── Run all baselines for one N value ────────────────────────────────────────

def run_all_baselines(n_ris_elements=32, k_users=2, n_episodes=100):
    """
    Run all 3 baselines for given N and K, return results dict.

    Used by main.py and plot_results.py to collect all comparison data.

    Returns:
        dict: {
            'random': result_dict,
            'fixed':  result_dict,
            'no_ris': result_dict
        }
    """
    print(f"\n{'='*50}")
    print(f"Running all baselines: N={n_ris_elements}, K={k_users}")
    print(f"{'='*50}")

    results = {}
    results['random'] = evaluate_random_policy(n_ris_elements, k_users, n_episodes)
    results['fixed']  = evaluate_fixed_policy(n_ris_elements, k_users, n_episodes)
    results['no_ris'] = evaluate_no_ris_policy(n_ris_elements, k_users, n_episodes)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing baselines with N=32, K=4...")

    results = run_all_baselines(n_ris_elements=32, k_users=2, n_episodes=20)

    print("\nSummary:")
    for name, r in results.items():
        print(f"  {r['label']:30s}: {r['mean_latency_ms']:.3f} ms "
              f"({r['satisfaction_rate']*100:.1f}% URLLC satisfied)")

    print("\nbaselines.py OK")
