"""Simulated experiments in the running example (Section 3.3 and Appendix D).

An experiment of n people puts m = n // (2K) in each arm of each cell and draws binomial
outcomes. Each estimated ranking's prioritization curve is averaged over R replications.
A crossover is the first tested size from which a ranking matches or beats its comparator
at every larger tested size.
"""
import numpy as np
from model import F, curve, baseline_curve, best_curve, group_means

# Experiment sizes: 120 values evenly spaced on a log scale from about 300 to 30 million.
SIZES = np.unique(np.round(np.logspace(2.5, 7.5, 120), -1)).astype(int)


def simulated_size(n):
    """Size actually simulated: each arm of each of the 50 cells gets n // 100 people."""
    return None if n is None else 100 * (int(n) // 100)


def first_from(sizes, wins):
    """First size from which `wins` holds at every larger size (None if never)."""
    for i in range(len(sizes)):
        if wins[i:].all():
            return int(sizes[i])
    return None


def draw(pop, n, R, rng):
    """Treated and control outcome counts, shape (R, K), and the per-arm size m."""
    K = pop['K']; m = max(int(n // (2 * K)), 1)
    y1 = rng.binomial(m, pop['theta1'], size=(R, K))
    y0 = rng.binomial(m, pop['theta'], size=(R, K))
    return y1, y0, m


def one_parameter_logit(y1, m, mu):
    """MLE of a common shift with the baseline index known: solve sum_c (ybar1_c - F(mu_c + s)) = 0."""
    lo = np.full(y1.shape[0], -6.0); hi = np.full(y1.shape[0], 6.0)
    target = y1.sum(1) / m
    for _ in range(50):
        mid = (lo + hi) / 2
        below = F(mu[None, :] + mid[:, None]).sum(1) < target
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return (lo + hi) / 2


def run_one_sided(pop, sign, R=3000, seed=11, sizes=SIZES):
    """Rare and common populations: CATE prioritization (cell estimator) against
    baseline prioritization in the matching direction, which here is also BATE prioritization.

    Returns crossovers on average, at every fraction, and fraction by fraction, with the
    baseline known; and on average with the baseline estimated from the same experiment.
    """
    rng = np.random.default_rng(seed)
    b, K = pop['beta'], pop['K']
    baseline_known = baseline_curve(pop, sign * pop['theta'])
    gap_known, gap_same = [], []
    for n in sizes:
        y1, y0, m = draw(pop, n, R, rng)
        cate_same = curve(b, (y1 - y0) / m, rng).mean(0)                    # difference in means
        cate_known = curve(b, y1 / m - pop['theta'][None, :], rng).mean(0)  # treated mean - baseline
        base_same = curve(b, y0 / m, rng).mean(0)                           # rank by control mean
        if sign < 0:                     # falling side: rank by the control mean from lowest up,
            y1_, y0_, m_ = draw(pop, n, R, rng)                  # using a fresh draw (as simulated)
            base_same = curve(b, -y0_ / m_, rng).mean(0)
        gap_known.append(cate_known - baseline_known)
        gap_same.append(cate_same - base_same)
    gap_known, gap_same = np.array(gap_known), np.array(gap_same)
    return dict(n_avg_known=first_from(sizes, gap_known.mean(1) >= 0),
                n_avg_same=first_from(sizes, gap_same.mean(1) >= 0),
                n_all_known=first_from(sizes, (gap_known[:, :K - 1] >= 0).all(1)),
                win_by_u=[first_from(sizes, gap_known[:, j] >= 0) for j in range(K - 1)])


def run_straddling(pop, R=3000, seed=13, sizes=SIZES):
    """Straddling population (Figure 4): estimated BATE prioritization (one-parameter logit)
    and CATE prioritization, as shares of the best ranking's average performance, plus the
    rankings that need no experimental data."""
    rng = np.random.default_rng(seed)
    b, mu, K = pop['beta'], pop['mu'], pop['K']
    best = best_curve(pop).mean()
    share = lambda c: float(c.mean() / best)
    out = dict(logit={}, cate={})
    for n in sizes:
        y1, y0, m = draw(pop, n, R, rng)
        curve(b, (y1 - y0) / m, rng)                              # unused; kept so the random stream matches
        cate_known = curve(b, y1 / m - pop['theta'][None, :], rng).mean(0)
        curve(b, y0 / m, rng)                                     # unused; kept so the random stream matches
        s = one_parameter_logit(y1, m, mu)
        logit = curve(b, -np.abs(mu[None, :] + s[:, None] / 2), rng).mean(0)
        curve(b, -y0 / m, rng)                                    # unused; kept so the random stream matches
        out['logit'][int(n)] = share(logit); out['cate'][int(n)] = share(cate_known)
    up, down = baseline_curve(pop, mu), baseline_curve(pop, -mu)
    out.update(up=share(up), down=share(down),
               threshold=share(baseline_curve(pop, -np.abs(pop['theta'] - 0.5))),
               bate=share(baseline_curve(pop, group_means(pop))))
    out['baseline'] = max(out['up'], out['down'])                 # better direction
    gaps = np.array([out['cate'][int(n)] - out['logit'][int(n)] for n in sizes])
    out['n_cate_beats_logit_avg'] = first_from(sizes, gaps >= 0)
    return out
