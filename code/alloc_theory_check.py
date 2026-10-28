"""Check the predicted best treated share for estimated BATE, 1 / (1 + sqrt(4K + 1)),
in the straddling population with a common shift, for K = 20, 50, 200 cells (10 baseline
groups of M = 2, 5, 20 cells). Budget n = 200 K, so the per-cell budget is the same."""
import numpy as np
from model import make_population, normalized_qini, curve, best_curve, logit
from experiments_equal import one_parameter_logit_est
rng = np.random.default_rng(17); R = 6000
shares = np.array([0.01, 0.015, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.5])
for M in [2, 5, 20]:
    pop = make_population(0.10, 0.80, 0.0, M=M); b, K = pop['beta'], pop['K']
    best = best_curve(pop).mean(); n = 200 * K
    loss = []
    for sh in shares:
        mT = int(round(n * sh / K)); mC = int(round(n * (1 - sh) / K))
        tot = []
        for _ in range(3):
            y1 = rng.binomial(mT, pop['theta1'], (R // 3, K)); y0 = rng.binomial(mC, pop['theta'], (R // 3, K))
            mu = logit((y0 + .5) / (mC + 1)); s = one_parameter_logit_est(y1, mT, mu)
            tot.append(curve(b, -np.abs(mu + s[:, None] / 2), rng).mean(0).mean())
        loss.append(100 - normalized_qini(np.mean(tot) / best, pop))
    loss = np.array(loss)
    # fit loss = A / (1 - sh) + B / sh  (baseline noise through controls, shift noise through treated)
    X = np.column_stack([1 / (1 - shares), 1 / shares]); A, B = np.linalg.lstsq(X, loss, rcond=None)[0]
    fitted = 1 / (1 + np.sqrt(A / B)); pred = 1 / (1 + np.sqrt(4 * K + 1))
    i = int(np.argmin(loss))
    print(f'K={K:3d} n={n:6,}: loss by share ' + ' '.join(f'{v:.2f}' for v in loss))
    print(f'        empirical argmin {shares[i]:.3f}; fitted best share {fitted:.3f}; predicted 1/(1+sqrt(4K+1)) = {pred:.3f}; '
          f'A/B fitted {A / B:.0f} vs 4K+1 = {4 * K + 1}; loss balanced / best = {loss[-1] / loss.min():.2f}')
