"""The running example: populations, prioritization curves, and summary quantities.

Threshold model with logistic noise (Assumption 1). A population is K = G*M cells of equal
mass: G baseline groups of M cells each. Cells in a group share the systematic index mu,
so the baseline cannot tell them apart; their shifts are delta = expected shift + eta,
where eta is spread evenly around zero within the group (the unpredictable part).
"""
import numpy as np

# ---------------------------------------------------------------- logistic noise
F = lambda z: 1 / (1 + np.exp(-z))          # CDF
f = lambda z: F(z) * (1 - F(z))             # density
logit = lambda p: np.log(p / (1 - p))       # F^{-1}

EXPECTED_SHIFT = 0.5                        # common expected shift in the running example
POPULATIONS = {                             # baseline range and the direction effects move
    'rare':       dict(lo=0.03, hi=0.20, sign=+1),   # rising side: rank highest baseline first
    'common':     dict(lo=0.60, hi=0.95, sign=-1),   # falling side: rank lowest baseline first
    'straddling': dict(lo=0.10, hi=0.80, sign=0),    # spans the effect peak
}


def make_population(lo, hi, shift_sd, d=EXPECTED_SHIFT, G=10, M=5):
    """Cells with baselines from `lo` to `hi` (G groups, evenly spaced in mu).

    shift_sd is the standard deviation of the unpredictable shift eta within each group:
    a number (the same for every group) or an array with one entry per group.
    """
    mu_g = np.linspace(logit(lo), logit(hi), G)
    shift_sd = np.broadcast_to(np.asarray(shift_sd, float), (G,))
    spread = np.linspace(-1, 1, M); spread = spread / spread.std()   # mean 0, sd 1
    mu = np.repeat(mu_g, M)
    delta = d + np.concatenate([spread * s for s in shift_sd])
    theta, theta1 = F(mu), F(mu + delta)
    return dict(mu=mu, delta=delta, theta=theta, theta1=theta1, beta=theta1 - theta,
                group=np.repeat(np.arange(G), M), K=G * M, d=d)


def reversed_shift_sd(lo, hi, d=EXPECTED_SHIFT, G=10):
    """Reversed variant: shift spread largest at the bottom of the ranking, proportional to
    (max sensitivity / group sensitivity)^1.5 and averaging 0.4 d, as with r = 0.4."""
    w = f(np.linspace(logit(lo), logit(hi), G) + d)
    sd = (w.max() / w) ** 1.5
    return sd / sd.mean() * 0.4 * d


def scenario(key):
    """Population and ranking direction for a scenario key such as 'rare_0.4' or 'reversed'."""
    if key == 'reversed':
        p = POPULATIONS['rare']
        return make_population(p['lo'], p['hi'], reversed_shift_sd(p['lo'], p['hi'])), +1
    name, r = key.split('_'); p = POPULATIONS[name]
    return make_population(p['lo'], p['hi'], float(r) * EXPECTED_SHIFT), p['sign']


def group_means(pop):
    """BATE of each cell: the mean effect in its baseline group."""
    b, g = pop['beta'], pop['group']
    return np.array([b[g == k].mean() for k in g])


def summaries(pop):
    """Potential, forgone heterogeneity, and the Gaussian experiment sizes n* (Cor. 2, App. A.5)."""
    b, th, th1, K = pop['beta'], pop['theta'], pop['theta1'], pop['K']
    V = b.var(); S = group_means(pop).var(); U = V - S; P = S / V
    vbar = np.mean(th * (1 - th)); vbar1 = np.mean(th1 * (1 - th1)); kappa = V / th.var()
    return dict(V=V, S=S, U=U, P=P, kappa=kappa, forgone_share=U / V, vbar1=vbar1,
                nstar_known=2 * K * vbar1 * P / U,                   # baseline known
                nstar_same=(2 * K / U) * (P * (vbar + vbar1) - kappa * vbar))   # baseline from the same experiment


# ---------------------------------------------------------------- prioritization curves
def curve(beta, scores, rng):
    """V(u; S) at u = j/K for each row of `scores`; exact ties are broken at random."""
    K = beta.shape[-1]
    jitter = rng.random(scores.shape) * 1e-9 * (np.abs(scores).max() + 1e-12)
    order = np.argsort(-(scores + jitter), axis=-1)
    ranked = np.take_along_axis(np.broadcast_to(beta, scores.shape), order, axis=-1)
    return np.cumsum(ranked, axis=-1) / K


def baseline_curve(pop, key):
    """Expected curve of a ranking that only sees the baseline group (ties within groups
    are broken at random, so each treated cell contributes its group's mean effect)."""
    order = np.argsort(-key, kind='stable')
    return np.cumsum(group_means(pop)[order]) / pop['K']


def best_curve(pop):
    return np.cumsum(np.sort(pop['beta'])[::-1]) / pop['K']


def random_share(pop):
    """Average performance of random targeting relative to the best ranking, over u = j/K."""
    K, b = pop['K'], pop['beta']
    return np.mean(np.arange(1, K + 1) / K * b.mean()) / best_curve(pop).mean()


def normalized_qini(share, pop):
    """Convert A(S)/A(best) into normalized Qini (0 = random targeting, 100 = best), in %."""
    a0 = random_share(pop)
    return 100 * (share - a0) / (1 - a0)
