"""Check Proposition 8 without cells (Appendix E) and save results/continuous.json.

A continuous rare population: 400,000 people with indices uniform between baselines of 3%
and 20%, expected shift 0.5, and unpredictable shifts eta ~ N(0, (0.5 r)^2). The effect
estimate is beta + sigma sqrt(w) e, with w the treated-outcome variance (known baseline).
At each fraction u, bisection finds the noise level sigma^2 at which CATE prioritization ties
baseline prioritization; the proposition predicts Var(beta | mu_u) / E[w | mu_u].
"""
import json
import numpy as np
from model import F, logit, EXPECTED_SHIFT as D

N, REPS = 400_000, 4
FRACTIONS = np.arange(0.05, 0.951, 0.05)
rng = np.random.default_rng(5)


def tie_noise(beta, w, k, target, E):
    """Noise variance at which the estimate's top-k value equals `target` (bisection on a log scale)."""
    def value(sig):
        tot = 0
        for r in range(REPS):
            idx = np.argpartition(-(beta + sig * np.sqrt(w) * E[r]), k)[:k]
            tot += beta[idx].sum() / N
        return tot / REPS
    lo, hi = 1e-4, 1.0
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if value(mid) >= target else (lo, mid)
    return lo * hi                       # = (geometric midpoint)^2


def predicted_noise(mu_c, s):
    """Var(beta | mu) / E[w | mu] at the cutoff index, by Gauss-Hermite quadrature over eta."""
    z, wt = np.polynomial.hermite_e.hermegauss(60); wt = wt / wt.sum()
    t1 = F(mu_c + D + s * z)
    return ((wt * t1 ** 2).sum() - (wt * t1).sum() ** 2) / (wt * t1 * (1 - t1)).sum()


def run(r):
    s = r * D
    mu = rng.uniform(logit(0.03), logit(0.20), N)
    eta = rng.normal(0, 1, N) * s
    th, th1 = F(mu), F(mu + D + eta)
    beta, w = th1 - th, th1 * (1 - th1)
    baseline = np.cumsum(beta[np.argsort(-mu)]) / N          # rare side: highest baseline first
    E = rng.normal(0, 1, (REPS, N))
    cutoff = np.sort(mu)[::-1]
    rows = []
    for u in FRACTIONS:
        k = int(u * N)
        rows.append((round(float(u), 2), tie_noise(beta, w, k, baseline[k - 1], E), predicted_noise(cutoff[k - 1], s)))
    ratio = np.array([p / a for _, a, p in rows]); mid = [i for i, row in enumerate(rows) if 0.2 <= row[0] <= 0.8]
    print(f'r={r}: predicted/simulated noise at fractions 20-80%: {ratio[mid].min():.2f}-{ratio[mid].max():.2f}', flush=True)
    return rows


if __name__ == '__main__':
    json.dump({f'rare_{r}': run(r) for r in [0.1, 0.05, 0.025]}, open('results/continuous.json', 'w'))
