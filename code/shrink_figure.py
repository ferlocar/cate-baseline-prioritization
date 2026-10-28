"""Coordinates for Figure 5 (Section 5.3): share of the available improvement captured by
beta_hat = m + a (beta - m + sigma sqrt(w) e) at u = 0.2, 0.5, 0.8, in the continuous rare population
(r = 0.05, noise set so that forgone heterogeneity at the median baseline is half the noise).
Leading order (Corollary 3): share = a (2 - a / a_u), with a_u = H_u / (H_u + N_u) the signal share at the margin.
    python shrink_figure.py   -> results/shrink_figure.txt
"""
import numpy as np
from check_shrinkage import setup, predicted, value, N, D
from model import F

r, ratio = 0.05, 0.5
rng = np.random.default_rng(8)
mu, beta, w, mvals, m, s = setup(r)
order = np.argsort(-mu); cut = mu[order]
base = np.cumsum(beta[order]) / N
best = np.cumsum(np.sort(beta)[::-1]) / N
E = rng.normal(0, 1, (8, N))
_, H0, N0 = predicted(np.median(mu), s, m, 1.0, 1.0, 1.0)
sigma2 = H0 / (ratio * N0)
out = []
for u in [0.2, 0.5, 0.8]:
    k = int(u * N)
    _, H, Nn = predicted(cut[k - 1], s, m, sigma2, 1.0, 1.0); au = H / (H + Nn)
    G = best[k - 1] - base[k - 1]
    pts = []
    for a in np.round(np.arange(0.1, 1.01, 0.1), 2):
        sims = [value(mvals + a * (beta - mvals + np.sqrt(sigma2 * w) * E[i]), beta, k) - base[k - 1] for i in range(len(E))]
        pts.append((a, np.mean(sims) / G))
    out.append(f'u={u}: a_u={au:.3f}  simulated share: ' + ' '.join(f'({a:.1f},{v:.3f})' for a, v in pts))
    print(out[-1], flush=True)
open('results/shrink_figure.txt', 'w').write('\n'.join(out) + '\n')
