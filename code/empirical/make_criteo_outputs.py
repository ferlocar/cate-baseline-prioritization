"""Section 6 outputs from the saved results: Tables 4 and 8, Figure 5 coordinates, and the
numbers quoted in the text.

    python make_criteo_outputs.py
"""
import json
import numpy as np
from scipy.special import expit, logit

R = np.load('results/criteo_curves.npz', allow_pickle=True)
XL = np.load('results/criteo_xlearner.npz')
E = json.load(open('results/criteo_effects.json'))
F = np.arange(1, 101) / 100
ATE = float(R['ate'])
SIZES = np.unique(R['n'])
NAMES = list(R['names'])
LABELS = dict(baseline='Baseline', threshold='Threshold', bate_logit='BATE, one-parameter logit',
              cate_two_model='CATE, T-learner', cate_from_baseline='CATE, offset learner',
              cate_x='CATE, X-learner')


def mean_curves(n):
    """Prioritization curves, in incremental visits per 1,000 people, averaged over replications."""
    c = {k: 1000 * v for k, v in zip(R['fixed_names'], R['fixed'])}
    c.update({k: 1000 * v for k, v in zip(NAMES, R['curves'][R['n'] == n].mean(0))})
    c['cate_x'] = 1000 * XL['x'][XL['n'] == n].mean(0)
    return c


def qini(c):
    """Mean over u of the curve minus random targeting's, u * ATE (per 1,000 people)."""
    return np.mean(c - 1000 * F * ATE)


def first_from(ok):
    """First tested size from which the condition holds at every larger tested size."""
    for i in range(len(ok)):
        if ok[i:].all():
            return int(SIZES[i])
    return None


def fmt(n):
    return 'all' if n == SIZES[-1] else (f'{n // 1000}k' if n < 1_000_000 else f'{n // 1_000_000}M')


def main():
    C = {n: mean_curves(n) for n in SIZES}
    print(f'ATE on the evaluation data: {100 * ATE:.2f} points; treated pool {SIZES[-1]:,}')

    print('\nTABLE 4: incremental visits per 1,000 people, top 1% and top 50%')
    cols = [1_000, 10_000, 100_000, 1_000_000, SIZES[-1]]
    print(f'{"":26s}' + ''.join(f'{fmt(n):>7s}' for n in cols) * 2)
    for k in LABELS:
        print(f'{LABELS[k]:26s}' + ''.join(f'{C[n][k][0]:7.2f}' for n in cols) + ''.join(f'{C[n][k][49]:7.2f}' for n in cols))

    print('\nTABLE 8: Qini (mean over fractions of the curve minus random targeting), every size')
    print(f'{"":26s}' + ''.join(f'{fmt(n):>7s}' for n in SIZES))
    for k in LABELS:
        print(f'{LABELS[k]:26s}' + ''.join(f'{qini(C[n][k]):7.3f}' for n in SIZES))
    for u in [1, 5, 10, 25, 50, 80]:
        print(f'\nAPPENDIX TABLE: top {u}%')
        for k in LABELS:
            print(f'{LABELS[k]:26s}' + ''.join(f'{C[n][k][u - 1]:7.2f}' for n in SIZES))
    print('\nAPPENDIX F: BATE prioritization with the spline logit (Qini, top 1%, top 50%)')
    for n in SIZES:
        c = C[n]['bate_spline']; print(f'{fmt(n):>5s} {qini(c):6.3f} {c[0]:5.2f} {c[49]:6.2f}')
    print('\nfitted common shift by size (mean, sd):',
          [(fmt(n), round(R['shift'][R['n'] == n].mean(), 3), round(R['shift'][R['n'] == n].std(), 3)) for n in SIZES])

    print('\nCROSSOVERS vs estimated BATE prioritization (one-parameter logit), by treatment fraction')
    cross = {}
    for k in ['cate_two_model', 'cate_from_baseline', 'cate_x']:
        d = np.array([C[n][k] - C[n]['bate_logit'] for n in SIZES])
        cross[k] = [first_from(d[:, j] > 0) for j in range(100)]
        last = max(j + 1 for j in range(100) if cross[k][j] is not None)
        print(f'{k}: ' + ', '.join(f'{u}%: {cross[k][u - 1]}' for u in [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80]),
              f'| last fraction with a crossover: {last}% | on average: {first_from(d.mean(1) > 0)}')

    print('\nFIGURE 5A: (mean baseline, effect in points, 95% half-width)')
    print(' '.join(f'({g["mean_baseline"]:.4f},{100 * g["effect"]:.2f}) +- (0,{196 * g["se"]:.2f})' for g in E['groups']))
    s = float(R['shift'][-1])
    th = np.exp(np.linspace(np.log(5e-4), np.log(0.85), 60))
    print(f'common shift {s:.3f} (whole pool), implied peak {expit(-s / 2):.3f}:')
    print(' '.join(f'({t:.4f},{100 * (expit(logit(t) + s) - t):.2f})' for t in th))
    print('shares above:', E['share_above'])

    print('\nFIGURE 5B: gain of the X-learner over estimated BATE prioritization (u, visits per 1,000 users)')
    for n in [3_000, 30_000, 300_000, int(SIZES[-1])]:
        d = C[n]['cate_x'] - C[n]['bate_logit']
        print(f'{fmt(n)}:', ' '.join(f'({u / 100:.2f},{d[u - 1]:.3f})' for u in range(1, 101)))

    print('\nEVALUATION NOISE (rankings from one experiment of 1M treated users; mean ± s.e. over 20 folds)')
    ev = E['eval']
    print('columns', ev['columns'])
    for k, v in ev['diff_vs_bate'].items():
        print(f'{k:20s} minus BATE: ' + '  '.join(f'{m:+.3f}±{e:.3f}' for m, e in zip(v['mean'], v['se'])))


if __name__ == '__main__':
    main()
