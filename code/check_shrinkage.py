"""Numerical check of the margin result for shrunk estimates (continuous rare population, as in
run_continuous.py). The estimate is the BATE plus a shrunk departure:
    beta_hat = m(mu) + a * (beta - m(mu) + sigma * sqrt(w) * e),
so its error is (a - 1) (beta - m) + a sigma sqrt(w) e. The predicted gain over BATE prioritization at fraction u is
    (g_m(c_u) / 2) * { (2 E[a] - E[a^2]) H_u - E[a^2] N_u },   H_u = Var(beta | mu_u),  N_u = sigma^2 E[w | mu_u],
with a either fixed or random (independent of everything else).
"""
import numpy as np
from model import F, f, logit, EXPECTED_SHIFT as D
N = 400_000; rng = np.random.default_rng(8)
lo, hi = logit(0.03), logit(0.20)
z, wt = np.polynomial.hermite_e.hermegauss(60); wt = wt / wt.sum()

def setup(r):
    s = r * D
    mu = rng.uniform(lo, hi, N); eta = rng.normal(0, 1, N) * s
    th1 = F(mu + D + eta); beta = th1 - F(mu); w = th1 * (1 - th1)
    m = lambda x: (wt[None, :] * F(x[:, None] + D + s * z[None, :])).sum(1) - F(x)     # BATE as a function of mu
    return mu, beta, w, m(mu), m, s

def value(score, beta, k):
    idx = np.argpartition(-score, k)[:k]; return beta[idx].sum() / N

def predicted(mu_c, s, m, sigma2, Ea, Ea2):
    t1 = F(mu_c + D + s * z)
    H = (wt * t1 ** 2).sum() - (wt * t1).sum() ** 2
    Nn = sigma2 * (wt * t1 * (1 - t1)).sum()
    eps = 1e-4; slope = abs((m(np.array([mu_c + eps])) - m(np.array([mu_c - eps])))[0] / (2 * eps))
    g = 1 / ((hi - lo) * slope)
    return g / 2 * ((2 * Ea - Ea2) * H - Ea2 * Nn), H, Nn

def main():
    global rng
    for r in [0.1, 0.05]:
        rng = np.random.default_rng(8)
        mu, beta, w, mvals, m, s = setup(r)
        order = np.argsort(-mu); cut = mu[order]
        base = np.cumsum(beta[order]) / N                       # BATE prioritization (rare side: highest baseline first)
        E = rng.normal(0, 1, (4, N))
        _, H0, N0 = predicted(np.median(mu), s, m, 1.0, 1.0, 1.0)  # H and E[w] at the median index
        for ratio in [2.0, 0.5, 0.2]:                             # heterogeneity-to-noise ratio H/N at the median
            sigma2 = H0 / (ratio * N0)
            print(f'\nr={r}, H/N at the median = {ratio}:  gain over BATE prioritization x 1e7 (simulated / predicted)')
            for label, draw in [('a=1 (unpooled)', lambda: 1.0), ('a=0.5', lambda: 0.5), ('a=0.25', lambda: 0.25),
                                ('a random U(0,0.6)', lambda: rng.uniform(0, 0.6))]:
                row = []
                for u in [0.2, 0.5, 0.8]:
                    k = int(u * N); sims = []; As = []
                    for rep in range(4):
                        a = draw(); As.append(a)
                        sc = mvals + a * (beta - mvals + np.sqrt(sigma2 * w) * E[rep])
                        sims.append(value(sc, beta, k) - base[k - 1])
                    # predicted gain for each draw of a, averaged (the result is linear in E[a] and E[a^2])
                    p = np.mean([predicted(cut[k - 1], s, m, sigma2, a, a ** 2)[0] for a in As])
                    _, H, Nn = predicted(cut[k - 1], s, m, sigma2, 1.0, 1.0)
                    row.append(f'u={u}: {1e7 * np.mean(sims):+8.2f} / {1e7 * p:+8.2f}')
                print(f'  {label:18s} ' + ' | '.join(row) + f'   (H/N at u=0.5: {H / Nn:.2f})')

    # Price of looking (Proposition 9): moments of the moment-estimated factor A = max(0, 1 - K/Q) under no heterogeneity
    rng = np.random.default_rng(1); K = 50; R = 400_000
    e = rng.normal(size=(R, K))
    for label, d in [('departures from zero (Q ~ chi2_K)', e), ('departures from an estimated common shift (Q ~ chi2_{K-1})', e - e.mean(1, keepdims=True))]:
        A = np.maximum(0, 1 - K / (d ** 2).sum(1))
        dep = ((A ** 2)[:, None] * d ** 2).mean() / (d ** 2).mean()
        print(f'\nPrice of looking, K={K}, {label}: E[A] = {A.mean():.4f}, K E[A^2] = {K * (A ** 2).mean():.3f}, '
              f'loss ratio 1/E[A^2] = {1 / (A ** 2).mean():.0f}, with dependence on own noise 1/E[A^2 d^2]/E[d^2] = {1 / dep:.0f}')


if __name__ == '__main__':
    main()
