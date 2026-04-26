"""
urllc_latency.py — End-to-End URLLC Latency with Proper Partial MEC Offloading
================================================================================
Journal-grade model: each user JOINTLY decides how much to offload to MEC
vs. compute locally. The optimal split depends on channel quality (SNR),
making it a genuine learning target.

E2E latency model (parallel MEC + local):
  Offload path  : L_off = rho * D / R  (tx) + rho * D * C / f_mec  (MEC compute)
  Local path    : L_loc = (1-rho) * D * C_loc / f_device            (device compute)
  E2E           : L_e2e = max(L_off, L_loc)

  With typical parameters:
    rho = 0   → L_loc = 50000 * 1000 / 2e9 = 25ms  (device too slow alone)
    rho = 1   → L_off = D/R + 0.5ms                (dominated by wireless)
    rho = opt → balance offload latency with local latency

  The optimal rho* = f(SNR) — this is what the BC policy learns to predict!
  Under imperfect CSI, the analytical formula for rho* is inaccurate,
  giving the learning policy a genuine advantage over the closed-form solution.
"""

import numpy as np
from scipy.stats import norm as scipy_norm

# ─────────────────────────────────────────────
#  System Constants
# ─────────────────────────────────────────────

BANDWIDTH_HZ          = 10e6       # 10 MHz
NOISE_POWER_DBM       = -90
NOISE_POWER_W         = 10 ** ((NOISE_POWER_DBM - 30) / 10)  # 1e-12 W

MEC_CPU_FREQ_HZ       = 10e9       # MEC server: 10 GHz
DEVICE_CPU_FREQ_HZ    = 2e9        # IoT device CPU: 2 GHz (constrained device)
CYCLES_PER_BIT_MEC    = 100        # MEC CPU cycles/bit
CYCLES_PER_BIT_LOCAL  = 1000       # Device cycles/bit (10× heavier — IoT CPU is slow)

TASK_BITS             = 50e3       # 50 kbits per device
PACKET_ERROR_RATE     = 1e-5       # URLLC reliability target
BLOCKLENGTH           = 300        # FBL blocklength (symbols)
MAX_LATENCY_MS        = 100.0      # Numerical cap

ACTIVE_RIS_NOISE_W = 10 ** ((-80 - 30) / 10)  # Active element amplifier noise


# ─────────────────────────────────────────────
#  FBL Rate
# ─────────────────────────────────────────────

def compute_urllc_rate(snr, n_symbols=BLOCKLENGTH, epsilon=PACKET_ERROR_RATE,
                       bandwidth_hz=BANDWIDTH_HZ):
    """
    Polyanskiy (2010) finite-blocklength achievable rate:

        R_fbl = B * [log2(1+SNR) - sqrt(V/n) * Q^{-1}(eps) / ln2]^+

    Channel dispersion V = 1 - (1+SNR)^{-2} captures the extra cost
    of finite-length codes compared to Shannon capacity.
    """
    snr = max(snr, 1e-10)
    V   = 1.0 - (1.0 + snr) ** (-2)
    q_inv = scipy_norm.ppf(1.0 - epsilon)   # Q^{-1}(eps) ≈ 4.265 for eps=1e-5

    correction = np.sqrt(V / n_symbols) * q_inv / np.log(2) if n_symbols > 0 else 0.0
    shannon    = np.log2(1.0 + snr)
    min_rate   = TASK_BITS / (MAX_LATENCY_MS / 1000.0) / bandwidth_hz

    fbl_rate = max(shannon - correction, min_rate)
    return bandwidth_hz * fbl_rate   # bps


# ─────────────────────────────────────────────
#  Partial Offload E2E Latency
# ─────────────────────────────────────────────

def compute_e2e_latency(snr, offload_ratio,
                        task_bits=TASK_BITS,
                        mec_cpu=MEC_CPU_FREQ_HZ,
                        dev_cpu=DEVICE_CPU_FREQ_HZ):
    """
    Compute E2E latency for ONE user with partial MEC offloading.

    Parallel execution model:
      • Offload path: transmit rho*D bits  →  MEC computes rho*D bits
        L_off = rho*D/R_fbl + rho*D*C_mec/f_mec
      • Local path  : device computes (1-rho)*D bits locally
        L_loc = (1-rho)*D*C_loc/f_device
      • Both run in PARALLEL (pipelined):
        L_e2e = max(L_off, L_loc)

    Why this model matters for learning:
      - At low SNR: R_fbl small → L_off large → prefer local (rho ↓)
      - At high SNR: R_fbl large → L_off small → prefer MEC (rho ↑)
      - Under imperfect CSI: agent doesn't know true SNR exactly,
        so it learns a robust rho*(SNR_hat) mapping from demonstrations.

    Args:
        snr           : true received SNR (linear)
        offload_ratio : rho ∈ [0, 1]

    Returns:
        e2e_ms  : total E2E latency (ms)
        off_ms  : offload path latency (ms)
        loc_ms  : local computation latency (ms)
    """
    rho = float(np.clip(offload_ratio, 0.0, 1.0))

    # Finite-blocklength rate (bps)
    rate = compute_urllc_rate(snr)

    # Offload path (wireless tx + MEC compute)
    if rho > 0:
        tx_ms  = min((rho * task_bits) / rate * 1000.0, MAX_LATENCY_MS)
        mec_ms = (rho * task_bits * CYCLES_PER_BIT_MEC) / mec_cpu * 1000.0
        off_ms = tx_ms + mec_ms
    else:
        off_ms = 0.0

    # Local path (device CPU)
    loc_ms = ((1 - rho) * task_bits * CYCLES_PER_BIT_LOCAL) / dev_cpu * 1000.0

    # Parallel execution: total = max of both paths
    e2e_ms = max(off_ms, loc_ms)
    e2e_ms = min(e2e_ms, MAX_LATENCY_MS)

    return e2e_ms, off_ms, loc_ms


def optimal_offload_ratio(snr, task_bits=TASK_BITS):
    """
    Analytical optimal offload ratio given PERFECT SNR knowledge.

    Solve: L_off(rho) = L_loc(rho)
      rho*D/R + rho*D*C_mec/f_mec  =  (1-rho)*D*C_loc/f_dev
      rho * (1/R + C_mec/f_mec) = (1-rho) * C_loc/f_dev
      rho * [1/R + C_mec/f_mec + C_loc/f_dev] = C_loc/f_dev
      rho* = (C_loc/f_dev) / (1/R + C_mec/f_mec + C_loc/f_dev)

    Under IMPERFECT CSI (R estimated from SNR_hat ≠ SNR_true),
    this formula gives a suboptimal rho. The BC policy, trained on many
    noisy observations, learns a more robust mapping.
    """
    rate = compute_urllc_rate(snr)
    a = 1.0 / rate + CYCLES_PER_BIT_MEC / MEC_CPU_FREQ_HZ
    b = CYCLES_PER_BIT_LOCAL / DEVICE_CPU_FREQ_HZ
    rho_star = b / (a + b)
    return float(np.clip(rho_star, 0.0, 1.0))


def compute_snr(channel_gain_sq, tx_power_w, noise_power_w=NOISE_POWER_W):
    """SNR = P |g|^2 / sigma^2."""
    return tx_power_w * channel_gain_sq / noise_power_w


def compute_system_latency(snr_list, offload_ratios):
    """
    Multi-user latency: return max (worst-case) and individual latencies.
    URLLC requires ALL users to meet the target simultaneously.
    """
    latencies = []
    for snr, rho in zip(snr_list, offload_ratios):
        e2e, _, _ = compute_e2e_latency(snr, rho)
        latencies.append(e2e)
    latencies = np.array(latencies)
    return float(np.max(latencies)), latencies


if __name__ == '__main__':
    print("Testing latency model with partial offloading...")
    print(f"{'SNR(dB)':>8} {'rho*':>6} {'L_e2e':>8} {'L_off':>8} {'L_loc':>8}")
    for snr_db in [5, 10, 15, 20, 25]:
        snr = 10 ** (snr_db / 10)
        rho = optimal_offload_ratio(snr)
        e2e, off, loc = compute_e2e_latency(snr, rho)
        tag = 'PASS' if e2e < 5 else 'FAIL'
        print(f"{snr_db:>8} {rho:>6.3f} {e2e:>7.2f}ms {off:>7.2f}ms {loc:>7.2f}ms  {tag}")
    print("urllc_latency.py OK")
