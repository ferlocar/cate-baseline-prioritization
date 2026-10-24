"""Section 6: the four rankings on the Criteo experiment, as the treated sample grows.

The training controls stay fixed; an experiment of size n adds n treated users drawn from
the treated pool. For each size and replication we fit
  * estimated BATE prioritization, two ways: the one-parameter logit (a common shift on the
    baseline log-odds) and a spline logit (a smooth function of the baseline log-odds);
  * CATE prioritization, two ways: a two-model learner (boosted model of treated outcomes
    minus the baseline) and a boosted model of treated outcomes started from the baseline
    log-odds (it learns departures from the baseline);
and evaluate every ranking's prioritization curve on the randomized evaluation data.

    python run_criteo.py            # full run -> results/criteo_curves.npz
    python run_criteo.py --quick    # small grid for testing
"""
import argparse, time
import numpy as np
from scipy.special import expit, logit
from criteo_common import load, fit_lgb, uplift_curve, one_parameter_logit, spline_basis, spline_logit

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, None]   # None = whole pool
REPS = {1_000: 20, 3_000: 20, 10_000: 20, 30_000: 20, 100_000: 10, 300_000: 10, 1_000_000: 5, 3_000_000: 3, None: 1}


def main(quick):
    D = load(); X, t, y, ev = D['X'], D['t'], D['y'], D['eval']
    rng = np.random.default_rng(1)
    tic = time.time()
    base = fit_lgb(X[D['train_control']], y[D['train_control']], rng)
    theta = np.clip(base.predict(X), 1e-6, 1 - 1e-6); lt = logit(theta)
    knots = np.percentile(lt[D['train_control']], [50, 90, 97, 99, 99.8])
    print(f'baseline model: {base.best_iteration} trees ({time.time() - tic:.0f}s)', flush=True)

    Xe, te, ye, the, lte = X[ev], t[ev], y[ev], theta[ev], lt[ev]
    curve = lambda s: uplift_curve(s, te, ye, rng)
    fixed = dict(baseline=curve(the), threshold=curve(-np.abs(the - 0.5)))
    out = dict(fixed_names=list(fixed), fixed=np.array(list(fixed.values())),
               ate=float(ye[te].mean() - ye[~te].mean()), knots=knots)

    pool = D['treated_pool']; sizes = SIZES[:5] if quick else SIZES
    names = ['bate_logit', 'bate_spline', 'cate_two_model', 'cate_from_baseline']
    rows = []
    for n in sizes:
        for rep in range(2 if quick else REPS[n]):
            idx = pool if n is None else rng.choice(pool, n, replace=False)
            y1, l1 = y[idx], lt[idx]
            s = one_parameter_logit(y1, l1)
            w = spline_logit(y1, l1, knots)
            mu_spline = expit(lte + np.column_stack([np.ones_like(lte), spline_basis(lte, knots)]) @ w)
            m2 = fit_lgb(X[idx], y1, rng)
            mb = fit_lgb(X[idx], y1, rng, init=l1)
            scores = dict(bate_logit=-np.abs(lte + s / 2),
                          bate_spline=mu_spline - the,
                          cate_two_model=m2.predict(Xe) - the,
                          cate_from_baseline=expit(lte + mb.predict(Xe, raw_score=True)) - the)
            rows.append(dict(n=len(idx), rep=rep, shift=s, curves=np.array([curve(scores[k]) for k in names])))
        print(f'n={len(idx):>9,}: {rep + 1} reps ({time.time() - tic:.0f}s)', flush=True)
    out.update(names=names, n=np.array([r['n'] for r in rows]), rep=np.array([r['rep'] for r in rows]),
               shift=np.array([r['shift'] for r in rows]), curves=np.array([r['curves'] for r in rows]))
    np.savez('results/criteo_curves' + ('_quick' if quick else '') + '.npz', **out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    main(ap.parse_args().quick)
