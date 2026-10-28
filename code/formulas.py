"""Non-Gaussian experiment-size approximations (Appendix D), for the cell estimator with a
known baseline. Cells make effects discrete, so the effect density g is a Gaussian-kernel
smoothing of the cell effects with bandwidth BW standard deviations of beta."""
import numpy as np
from model import group_means, baseline_curve, best_curve

BW = 0.3


def noise_factor(pop):
    """w = theta1 (1 - theta1): per-cell variance factor of the treated mean."""
    return pop['theta1'] * (1 - pop['theta1'])


def n_local(pop, sign):
    """Fraction-specific size n*_local(u) = K w_u g(q_u) / G(u), at u = j/K, j = 1..K-1."""
    b, K = pop['beta'], pop['K']
    w = noise_factor(pop)
    shortfall = best_curve(pop) - baseline_curve(pop, sign * pop['theta'])   # G(u)
    effects = np.sort(b)[::-1]
    h = BW * b.std()
    out = []
    for j in range(K - 1):
        q = 0.5 * (effects[j] + effects[j + 1])                               # effect cutoff q_u
        k = np.exp(-0.5 * ((b - q) / h) ** 2) / np.sqrt(2 * np.pi) / h
        g, w_u = k.mean(), (k * w).sum() / k.sum()
        out.append(K * w_u * g / shortfall[j] if shortfall[j] > 1e-15 else np.inf)
    return np.array(out)


def mean_abs_difference(x):
    x = np.asarray(x); return np.abs(x[:, None] - x[None, :]).mean()


def n_bar(pop):
    """Average-performance size n_bar* = K E[w g(beta)] / G_bar, G_bar = (Delta(beta) - Delta(m_theta)) / 4."""
    b, K = pop['beta'], pop['K']
    G_bar = (mean_abs_difference(b) - mean_abs_difference(group_means(pop))) / 4
    h = BW * b.std()
    g_at = (np.exp(-0.5 * ((b[:, None] - b[None, :]) / h) ** 2) / np.sqrt(2 * np.pi) / h).mean(1)
    return K * np.mean(noise_factor(pop) * g_at) / G_bar
