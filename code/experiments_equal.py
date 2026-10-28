"""Equal-n design for the running example: every ranking is estimated, and every ranking
gets the same total sample size n.

  * Baseline and threshold prioritization: an untreated sample of n people, 2m per cell.
  * Estimated BATE and CATE prioritization: one balanced experiment of n people, m per arm
    per cell (m = n // (2K)), the same experiment for both.
      - BATE: baseline from the experiment's control arm (one feature) and a common shift
        from its treated arm (one-parameter logit); ranks by -|mu_hat + s_hat / 2|.
      - CATE: the cell estimator, treated mean minus control mean.

The untreated sample is coupled to the experiment: it consists of the experiment's m
controls per cell plus m further untreated people per cell. This leaves each ranking's
marginal distribution unchanged and makes the comparisons between rankings less noisy.

Nothing is known here; the main design (run_simulations.py) subtracts the known baseline from
the treated mean and ranks baseline rankings by the true baseline.
"""
import numpy as np
from model import F, logit, curve, best_curve, group_means
from experiments import SIZES, first_from


def draw_equal(pop, n, R, rng):
    """Experiment counts y1, y0 (m per arm per cell) and untreated counts yu (2m per cell)."""
    K = pop['K']; m = max(int(n // (2 * K)), 1)
    y1 = rng.binomial(m, pop['theta1'], size=(R, K))
    y0 = rng.binomial(m, pop['theta'], size=(R, K))
    yu = y0 + rng.binomial(m, pop['theta'], size=(R, K))
    return y1, y0, yu, m


def one_parameter_logit_est(y1, m, mu_hat):
    """Common shift with an estimated baseline index per replication (rows of mu_hat):
    solve sum_c (ybar1_c - F(mu_hat_c + s)) = 0 by bisection."""
    lo = np.full(y1.shape[0], -6.0); hi = np.full(y1.shape[0], 6.0)
    target = y1.sum(1) / m
    for _ in range(50):
        mid = (lo + hi) / 2
        below = F(mu_hat + mid[:, None]).sum(1) < target
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return (lo + hi) / 2


def bate_share(K):
    """Treated share for estimated BATE prioritization: 1 / (1 + sqrt(4K + 1)), 6.59% for K = 50.
    The e-companion's formula, which accounts for the covariance between the cell baselines and the
    estimated shift, is 1 / (1 + sqrt(4K - 3)), 6.65% for K = 50; the loss is flat near its minimum
    (robustness.py), so the simulations use this slightly smaller share."""
    return 1 / (1 + np.sqrt(4 * K + 1))


def rankings(pop, sign, y1, y0, yu, m, rng, share=None):
    """Average prioritization curves (over replications) of every estimated ranking.

    share=None: BATE uses the balanced experiment. Otherwise BATE uses its own experiment
    of the same size with this treated share, coupled to the others: its controls are a
    random subset of the untreated sample, its treated a random subset of the treated arm."""
    b = pop['beta']; K = pop['K']
    th_u = yu / (2 * m)                                   # baseline from n untreated
    if share is None:
        mC, mT, y0b, y1b = m, m, y0, y1
    else:
        mT = max(int(round(2 * m * share)), 1); mC = 2 * m - mT
        y0b = rng.hypergeometric(yu, 2 * m - yu, mC)      # mC of the 2m untreated people
        y1b = rng.hypergeometric(y1, m - y1, mT)          # mT of the m treated people
    mu_c = logit((y0b + 0.5) / (mC + 1))                  # control arm, continuity-corrected
    s = one_parameter_logit_est(y1b, mT, mu_c)
    out = dict(
        cate=curve(b, (y1 - y0) / m, rng).mean(0),
        bate=curve(b, -np.abs(mu_c + s[:, None] / 2), rng).mean(0),
        threshold=curve(b, -np.abs(th_u - 0.5), rng).mean(0),
    )
    if sign == 0:                                         # straddling: both directions
        out['base_up'] = curve(b, th_u, rng).mean(0)
        out['base_down'] = curve(b, -th_u, rng).mean(0)
    else:
        out['baseline'] = curve(b, sign * th_u, rng).mean(0)
    return out, float(np.mean(s)), float(np.std(s))


def nstar_equal(pop):
    """Gaussian size at which CATE prioritization (difference in means, n/2 per arm)
    dominates baseline prioritization from n untreated: sigma_theta^2 = K vbar / n,
    sigma_beta^2 = 2K (vbar + vbar1) / n, so n >= (2K / U) {P (vbar + vbar1) - kappa vbar / 2}."""
    b, th, th1, K = pop['beta'], pop['theta'], pop['theta1'], pop['K']
    V = b.var(); U = V - group_means(pop).var(); P = 1 - U / V
    vbar = np.mean(th * (1 - th)); vbar1 = np.mean(th1 * (1 - th1)); kappa = V / th.var()
    return (2 * K / U) * (P * (vbar + vbar1) - kappa * vbar / 2)


def run(pop, sign, R=3000, seed=21, sizes=SIZES, share=None):
    """All estimated rankings at every size. Returns crossovers and learning curves."""
    rng = np.random.default_rng(seed)
    K = pop['K']; best = best_curve(pop)
    curves = {}; shifts = []
    for n in sizes:
        y1, y0, yu, m = draw_equal(pop, n, R, rng)
        c, s_mean, s_sd = rankings(pop, sign, y1, y0, yu, m, rng, share)
        if sign == 0:
            c['baseline'] = c['base_up']                  # highest first is the better direction at d = 0.5
        for k, v in c.items():
            curves.setdefault(k, []).append(v)
        shifts.append((s_mean, s_sd))
    curves = {k: np.array(v) for k, v in curves.items()}  # (sizes, K)
    share = {k: (v.mean(1) / best.mean()).tolist() for k, v in curves.items()}
    gap_b = curves['cate'] - curves['baseline']
    gap_bate = curves['cate'] - curves['bate']
    return dict(
        n_avg_vs_baseline=first_from(sizes, gap_b.mean(1) >= 0),
        n_avg_vs_bate=first_from(sizes, gap_bate.mean(1) >= 0),
        n_all_vs_baseline=first_from(sizes, (gap_b[:, :K - 1] >= 0).all(1)),
        win_by_u_vs_baseline=[first_from(sizes, gap_b[:, j] >= 0) for j in range(K - 1)],
        win_by_u_vs_bate=[first_from(sizes, gap_bate[:, j] >= 0) for j in range(K - 1)],
        n_bate_matches_baseline=first_from(sizes, curves['bate'].mean(1) >= curves['baseline'].mean(1)),
        share=share, shift_mean=[s for s, _ in shifts], shift_sd=[s for _, s in shifts],
    )
