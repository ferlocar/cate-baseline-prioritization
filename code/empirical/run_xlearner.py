"""Section 6: CATE prioritization with the X-learner of Kunzel et al. (2019), designed for
experiments whose arms differ greatly in size, as here (fixed controls, growing treated sample).

  * mu0: the baseline model (fixed controls); mu1: boosted model of treated outcomes;
  * tau1: boosted regression of Y - mu0(X) on the treated sample;
  * tau0: boosted regression of mu1(X) - Y on the training controls;
  * tau = g * tau0 + (1 - g) * tau1, with g the share of treated users in the training data.

    python run_xlearner.py      -> results/criteo_xlearner.npz
"""
import time
import numpy as np, lightgbm as lgb
from scipy.special import logit
from criteo_common import load, fit_lgb, uplift_curve, one_parameter_logit, LGB

from run_criteo import SIZES, REPS        # same experiment sizes and replications
REG = dict(LGB, objective='regression')


def fit_reg(X, y, rng, rounds=2000):
    v = rng.random(len(y)) < 0.2
    return lgb.train(REG, lgb.Dataset(X[~v], y[~v]), rounds, valid_sets=[lgb.Dataset(X[v], y[v])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main():
    D = load(); X, t, y, ev = D['X'], D['t'], D['y'], D['eval']
    rng = np.random.default_rng(1)
    base = fit_lgb(X[D['train_control']], y[D['train_control']], rng)          # same baseline as run_criteo.py
    theta = np.clip(base.predict(X), 1e-6, 1 - 1e-6); lt = logit(theta)
    rng = np.random.default_rng(2)
    Xe, te, ye, lte = X[ev], t[ev], y[ev], lt[ev]
    ctrl = D['train_control']; tic = time.time(); rows = []
    for n in SIZES:
        for rep in range(REPS[n]):
            idx = D['treated_pool'] if n is None else rng.choice(D['treated_pool'], n, replace=False)
            s = one_parameter_logit(y[idx], lt[idx])
            m1 = fit_lgb(X[idx], y[idx], rng)
            tau1 = fit_reg(X[idx], y[idx] - theta[idx], rng)
            tau0 = fit_reg(X[ctrl], m1.predict(X[ctrl]) - y[ctrl], rng)
            g = len(idx) / (len(idx) + len(ctrl))
            score = g * tau0.predict(Xe) + (1 - g) * tau1.predict(Xe)
            rows.append(dict(n=len(idx), rep=rep, bate=uplift_curve(-np.abs(lte + s / 2), te, ye, rng),
                             x=uplift_curve(score, te, ye, rng)))
        print(f'n={len(idx):>9,}: {REPS[n]} reps ({time.time() - tic:.0f}s)', flush=True)
        save(rows)


def save(rows):
    np.savez('results/criteo_xlearner.npz', n=np.array([r['n'] for r in rows]), rep=np.array([r['rep'] for r in rows]),
             bate=np.array([r['bate'] for r in rows]), x=np.array([r['x'] for r in rows]))


if __name__ == '__main__':
    main()
