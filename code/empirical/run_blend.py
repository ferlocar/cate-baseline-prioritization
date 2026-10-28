"""Section 6 extension: blending estimated BATE prioritization with the X-learner.

For each experiment, fit the X-learner on the whole experiment and on two independent halves
(treated users and training controls each split in two). On the evaluation users, with
d = X-learner estimate - estimated BATE (one-parameter logit), the halves' departures d1, d2 give
  signal  S = mean(d1 * d2)          (their noise is independent)
  noise   N = mean((d1 - d2)^2) / 4  (noise of the whole-experiment estimate, half that of a half)
and the blend weight S / (S + N), either one weight for everyone ('blend_const') or one per
baseline bin (20 quantile bins of the baseline, 'blend_base'). Rankings:
  bate        one-parameter logit
  x           X-learner alone
  blend_const estimated BATE + w (x - estimated BATE), one weight
  blend_base  same, weight by baseline bin

    python run_blend.py      -> results/criteo_blend.npz
"""
import time
import numpy as np, lightgbm as lgb
from scipy.special import logit, expit
from criteo_common import load, fit_lgb, uplift_curve, one_parameter_logit, LGB

REG = dict(LGB, objective='regression')
SIZES = [10_000, 100_000, 1_000_000]
REPS = {10_000: 4, 100_000: 3, 1_000_000: 2, None: 1}
NBINS = 20


def fit_reg(X, y, rng, rounds=2000):
    v = rng.random(len(y)) < 0.2
    return lgb.train(REG, lgb.Dataset(X[~v], y[~v]), rounds, valid_sets=[lgb.Dataset(X[v], y[v])],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def xlearner(X, y, theta, idx, ctrl, Xe, rng):
    m1 = fit_lgb(X[idx], y[idx], rng)
    tau1 = fit_reg(X[idx], y[idx] - theta[idx], rng)
    tau0 = fit_reg(X[ctrl], m1.predict(X[ctrl]) - y[ctrl], rng)
    g = len(idx) / (len(idx) + len(ctrl))
    return g * tau0.predict(Xe) + (1 - g) * tau1.predict(Xe)


def main():
    D = load(); X, t, y, ev = D['X'], D['t'], D['y'], D['eval']
    rng = np.random.default_rng(1)
    base = fit_lgb(X[D['train_control']], y[D['train_control']], rng)
    theta = np.clip(base.predict(X), 1e-6, 1 - 1e-6); lt = logit(theta)
    rng = np.random.default_rng(5)
    Xe, te, ye, lte, the = X[ev], t[ev], y[ev], lt[ev], theta[ev]
    edges = np.quantile(lte, np.linspace(0, 1, NBINS + 1)[1:-1]); bins = np.searchsorted(edges, lte)
    ctrl = D['train_control']; tic = time.time(); rows = []
    curve = lambda s: uplift_curve(s, te, ye, rng)
    for n in SIZES:
        for rep in range(REPS[n]):
            idx = D['treated_pool'] if n is None else rng.choice(D['treated_pool'], n, replace=False)
            s = one_parameter_logit(y[idx], lt[idx])
            mhat = expit(lte + s) - the
            tau = xlearner(X, y, theta, idx, ctrl, Xe, rng)
            pi = rng.permutation(idx); pc = rng.permutation(ctrl)
            t1 = xlearner(X, y, theta, pi[: len(pi) // 2], pc[: len(pc) // 2], Xe, rng)
            t2 = xlearner(X, y, theta, pi[len(pi) // 2:], pc[len(pc) // 2:], Xe, rng)
            d, d1, d2 = tau - mhat, t1 - mhat, t2 - mhat
            S = (d1 * d2); Nn = (d1 - d2) ** 2 / 4
            wc = max(S.mean(), 0) / (max(S.mean(), 0) + Nn.mean())
            cnt = np.maximum(np.bincount(bins, None, NBINS), 1)
            Sb = np.maximum(np.bincount(bins, S, NBINS) / cnt, 0)
            Nb = np.bincount(bins, Nn, NBINS) / cnt
            wb = np.nan_to_num(Sb / np.maximum(Sb + Nb, 1e-30))
            scores = dict(bate=-np.abs(lte + s / 2), x=tau,
                          blend_const=mhat + wc * d, blend_base=mhat + wb[bins] * d)
            rows.append(dict(n=len(idx), rep=rep, wc=wc, wb=wb, curves=np.array([curve(scores[k]) for k in scores])))
            print(f'n={len(idx):>9,} rep {rep}: w_const={wc:.3f}, w_bins={np.round(wb[[0, 5, 10, 15, 19]], 2)} ({time.time() - tic:.0f}s)', flush=True)
            np.savez('results/criteo_blend.npz', names=list(scores), n=np.array([r['n'] for r in rows]),
                     rep=np.array([r['rep'] for r in rows]), wc=np.array([r['wc'] for r in rows]),
                     wb=np.array([r['wb'] for r in rows]), curves=np.array([r['curves'] for r in rows]),
                     edges=edges, ate=float(ye[te].mean() - ye[~te].mean()))


if __name__ == '__main__':
    main()
