"""Section 6, supporting numbers:
  1. effects by baseline group on the randomized evaluation data (Figure 6A);
  2. evaluation noise: for rankings trained on one experiment of 1,000,000 treated users
     (T-learner, offset learner, and X-learner),
     the spread of each ranking's value, and of its difference from estimated BATE
     prioritization, across 20 random folds of the evaluation data.

    python criteo_effects.py      -> results/criteo_effects.json
"""
import json
import numpy as np
from scipy.special import expit, logit
from criteo_common import load, fit_lgb, uplift_curve, one_parameter_logit, FRACTIONS
from run_xlearner import fit_reg

CUTS = [0, .001, .002, .005, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, 1]
N_EVAL_FOLDS, N_TREATED = 20, 1_000_000


def main():
    D = load(); X, t, y, ev = D['X'], D['t'], D['y'], D['eval']
    rng = np.random.default_rng(1)
    base = fit_lgb(X[D['train_control']], y[D['train_control']], rng)     # same draws as run_criteo.py
    theta = np.clip(base.predict(X), 1e-6, 1 - 1e-6); lt = logit(theta)
    te, ye, the, lte = t[ev], y[ev], theta[ev], lt[ev]
    out = dict(groups=[])
    print('effect by baseline group: range, share, mean baseline, effect, s.e.')
    g = np.digitize(the, CUTS[1:-1])
    for k in range(len(CUTS) - 1):
        s = g == k; a, b = ye[s & te], ye[s & ~te]
        eff, se = a.mean() - b.mean(), np.sqrt(a.var() / len(a) + b.var() / len(b))
        out['groups'].append(dict(lo=CUTS[k], hi=CUTS[k + 1], share=float(s.mean()), mean_baseline=float(the[s].mean()),
                                  effect=float(eff), se=float(se), n_treated=int(len(a)), n_control=int(len(b))))
        print(f'{CUTS[k]:.3f}-{CUTS[k + 1]:.3f}  {s.mean():7.3%}  {the[s].mean():.4f}  {eff:+.4f}  {se:.4f}')
    out['share_above'] = {str(c): float((the > c).mean()) for c in [.3, .4, .5, .6]}
    print('share with baseline above', out['share_above'])

    idx = rng.choice(D['treated_pool'], N_TREATED, replace=False); y1, l1 = y[idx], lt[idx]
    sh = one_parameter_logit(y1, l1)
    mb = fit_lgb(X[idx], y1, rng, init=l1); m2 = fit_lgb(X[idx], y1, rng)
    ctrl = D['train_control']
    tau1 = fit_reg(X[idx], y1 - theta[idx], rng); tau0 = fit_reg(X[ctrl], m2.predict(X[ctrl]) - y[ctrl], rng)
    g = N_TREATED / (N_TREATED + len(ctrl))
    scores = dict(baseline=the, threshold=-np.abs(the - 0.5), bate_logit=-np.abs(lte + sh / 2),
                  cate_two_model=m2.predict(X[ev]) - the,
                  cate_from_baseline=expit(lte + mb.predict(X[ev], raw_score=True)) - the,
                  cate_x=g * tau0.predict(X[ev]) + (1 - g) * tau1.predict(X[ev]))
    out['shift'], out['peak_baseline'] = float(sh), float(expit(-sh / 2))
    print(f'shift {sh:.3f}; implied peak baseline {expit(-sh / 2):.3f}')
    folds = rng.integers(0, N_EVAL_FOLDS, len(te))
    cols = ['qini', 'top1', 'top5', 'top10', 'top25', 'top50', 'top80']
    pick = lambda c, ate: [np.mean(c - FRACTIONS * ate), c[0], c[4], c[9], c[24], c[49], c[79]]
    per = {k: [] for k in scores}
    for f in range(N_EVAL_FOLDS):
        m = folds == f; ate = ye[m & te].mean() - ye[m & ~te].mean()
        for k, sc in scores.items():
            per[k].append(pick(uplift_curve(sc[m], te[m], ye[m], rng), ate))
    per = {k: 1000 * np.array(v) for k, v in per.items()}
    se = lambda a: a.std(0, ddof=1) / np.sqrt(N_EVAL_FOLDS)
    out['eval'] = dict(columns=cols, n_treated=N_TREATED,
                       value={k: dict(mean=a.mean(0).tolist(), se=se(a).tolist()) for k, a in per.items()},
                       diff_vs_bate={k: dict(mean=(a - per['bate_logit']).mean(0).tolist(), se=se(a - per['bate_logit']).tolist())
                                     for k, a in per.items()})
    print('\nvisits per 1,000 people, mean ± s.e. over evaluation folds:', ' '.join(cols))
    for k, a in per.items():
        d = a - per['bate_logit']
        print(f'{k:20s}', '  '.join(f'{m:6.3f}±{e:.3f}' for m, e in zip(a.mean(0), se(a))))
        print(f'{"  minus BATE":20s}', '  '.join(f'{m:+6.3f}±{e:.3f}' for m, e in zip(d.mean(0), se(d))))
    json.dump(out, open('results/criteo_effects.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
