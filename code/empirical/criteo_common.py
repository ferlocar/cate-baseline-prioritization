"""Shared pieces of the Criteo illustration (Section 6).

Data: Criteo uplift v2.1 (Diemert et al. 2018), 14 million users from a randomized
advertising experiment (85% treated). Outcome: visit. We use treatment assignment
(intention to treat) and ignore exposure.

Splits (fixed seed):
  * evaluation: a random 30% of users, both arms, used only to evaluate rankings;
  * training controls: all remaining control users, fixed throughout; they stand in for
    the untreated data an organization already holds and train the baseline model;
  * treated pool: all remaining treated users; an experiment of size n draws n of them.
"""
import numpy as np, lightgbm as lgb
from scipy.special import expit, logit

DATA = '../../data/criteo.npz'               # written by prepare_data.py
FRACTIONS = np.arange(1, 101) / 100           # treatment fractions u = 1%, ..., 100%
LGB = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
           verbose=-1, num_threads=2)


def load(outcome='visit', seed=0):
    d = np.load(DATA)
    X, t, y = d['X'], d['t'].astype(bool), d[outcome].astype(np.int8)
    rng = np.random.default_rng(seed)
    ev = rng.random(len(t)) < 0.3
    return dict(X=X, t=t, y=y, eval=ev, train_control=np.where(~ev & ~t)[0],
                treated_pool=np.where(~ev & t)[0])


def fit_lgb(X, y, rng, init=None, rounds=2000):
    """Gradient boosting with early stopping on a random 20% of the data."""
    v = rng.random(len(y)) < 0.2
    dtr = lgb.Dataset(X[~v], y[~v], init_score=None if init is None else init[~v])
    dva = lgb.Dataset(X[v], y[v], init_score=None if init is None else init[v])
    return lgb.train(LGB, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def uplift_curve(score, t, y, rng):
    """V(u; S) on the randomized evaluation data: treat the top fraction u by score and
    estimate the total effect per member of the population as u * (treated mean - control
    mean) among them. Ties are broken at random."""
    order = np.lexsort((rng.random(len(score)), -score))
    tt, yy = t[order], y[order]
    n1, n0 = np.cumsum(tt), np.cumsum(~tt)
    s1, s0 = np.cumsum(yy * tt), np.cumsum(yy * ~tt)
    k = np.maximum((FRACTIONS * len(score)).astype(int), 1) - 1
    return FRACTIONS * (s1[k] / n1[k] - s0[k] / np.maximum(n0[k], 1))


def one_parameter_logit(y1, lt):
    """Common shift s with the baseline log-odds lt as an offset: solve sum(y1 - F(lt + s)) = 0."""
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(lt + mid).sum() < y1.sum() else (lo, mid)
    return (lo + hi) / 2


def spline_basis(z, knots):
    """Natural cubic spline basis in z (linear beyond the boundary knots)."""
    k = np.asarray(knots); K = len(k)
    d = lambda j: (np.maximum(z - k[j], 0) ** 3 - np.maximum(z - k[-1], 0) ** 3) / (k[-1] - k[j])
    return np.column_stack([z] + [d(j) - d(K - 2) for j in range(K - 2)])


def spline_logit(y1, lt, knots, iters=50):
    """Treated outcome on a natural spline of the baseline log-odds, with the baseline
    log-odds as an offset (so a constant term is the one-parameter logit). Newton's method,
    with a small ridge for stability. Returns coefficients (intercept first)."""
    B = np.column_stack([np.ones_like(lt), spline_basis(lt, knots)])
    w = np.zeros(B.shape[1]); ridge = 1e-6 * len(lt)
    for _ in range(iters):
        p = expit(lt + B @ w)
        g = B.T @ (y1 - p) - ridge * np.r_[0, w[1:]]
        H = (B * (p * (1 - p))[:, None]).T @ B + ridge * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8: break
    return w
