"""Summaries of the pooled learner (results/pooled.json) (Sections 4.1 and 5.3, Appendix E)."""
import json, numpy as np
from model import make_population, normalized_qini, scenario, best_curve, baseline_curve, EXPECTED_SHIFT as D
from run_simulations import R_VALUES
P = json.load(open('results/pooled.json'))
tex = lambda n: '--' if n is None else f'{int(n):,}'.replace(',', '{,}')
ROWS = [f'{n}_{r}' for n in ['rare', 'common'] for r in R_VALUES] + ['reversed']

def stays(t, ok):
    for i in range(len(t)):
        if ok[i:].all(): return int(t[i])
    return None

print('% TABLE pooled: scenario, attainable gain, half-gain size cell / pooled, win by >=1 point cell / pooled, largest loss pooled')
for k in ROWS:
    x = P[k]; pop, sign = scenario(k); t = np.array(x['treated']); q = lambda v: normalized_qini(v, pop)
    base = q(x['base_share']); gain = 100 - base
    c = np.array([q(v) - base for v in x['share']['cate']]); pl = np.array([q(v) - base for v in x['share']['pooled']])
    print(f"{k:11s} gain {gain:5.1f} | half: cell {tex(stays(t, c >= gain / 2))} pooled {tex(stays(t, pl >= gain / 2))} | "
          f"win>=1pt: cell {tex(stays(t, c >= 1))} pooled {tex(stays(t, pl >= 1))} | pooled worst {pl.min():+.1f}")
print('% fraction-specific, matches-or-beats (by_u) and beats by 1% of the best value at that fraction (by_u_tol): top 10/50/80')
for k in ROWS:
    x = P[k]; pop, sign = scenario(k); t = np.array(x['treated'])
    best = best_curve(pop); base = baseline_curve(pop, sign * pop['theta'])
    K = pop['K']
    # recover curves at fractions from shares is impossible; use stored crossovers (gap >= 0) as reported
    f = lambda v: tex(v)
    print(f"{k:11s} >=0: cell {[f(x['cate_by_u'][j]) for j in (4, 24, 39)]} pooled {[f(x['pooled_by_u'][j]) for j in (4, 24, 39)]} | 1% margin: cell {[f(x['cate_by_u_tol'][j]) for j in (4, 24, 39)]} pooled {[f(x['pooled_by_u_tol'][j]) for j in (4, 24, 39)]}")
for r in [0.0, 0.4]:
    x = P[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D); t = np.array(x['treated']); q = lambda v: normalized_qini(v, pop)
    vals = {k: [q(v) for v in x['share'][k]] for k in ['bate', 'cate', 'pooled']}
    pts = ' | '.join(f"{t[i]:,}: B {vals['bate'][i]:.1f} C {vals['cate'][i]:.1f} P {vals['pooled'][i]:.1f}" for i in [int(np.argmin(np.abs(t - n))) for n in [150, 500, 1500, 5000, 15000, 50000]])
    print(f'% straddling r={r}: {pts}')
    print(f"%   pooled overtakes BATE (avg): {tex(x['pooled_vs_bate_avg'])}; cell: {tex(x['cate_vs_bate_avg'])}; pooled reaches 99: {tex(stays(t, np.array(vals['pooled']) >= 99))}; cell reaches 99: {tex(stays(t, np.array(vals['cate']) >= 99))}")
    if r == 0.0:   # price of looking (Proposition 9): cell vs pooled shortfall from the best ranking, and the pooled factor
        cell, pool = 100 - np.array(vals['cate']), 100 - np.array(vals['pooled'])
        ok = t >= 12000
        ratio = cell[ok] / pool[ok]
        print(f"%   shortfall ratio cell/pooled from 12,000 treated: min {ratio.min():.0f} median {np.median(ratio):.0f} max {ratio.max():.0f}; at 12,250/47,600/103,300/223,950: "
              + ' '.join(f"{cell[i] / pool[i]:.0f}" for i in [int(np.argmin(np.abs(t - n))) for n in [12250, 47600, 103300, 223950]])
              + f"; pooled factor (weight on cells) at largest size {x['weight_on_cell'][-1]:.3f}")
        for ub in [500000, 2000000]:
            sel = ok & (t <= ub); rr = cell[sel] / pool[sel]
            print(f"%   ratio between 12,000 and {ub:,}: min {rr.min():.0f} max {rr.max():.0f}")
    for k in ['bate', 'cate', 'pooled']:
        merged = {}
        for n, v in zip(t, vals[k]):
            merged.setdefault(int(n), []).append(v)
        print(f'%   FIG4 {k} r={r}: ' + ' '.join(f'({n},{np.mean(v):.2f})' for n, v in sorted(merged.items())))

# How often the pooled factor is exactly zero under a common shift (Section 5.3): fresh replications
from run_pooled import pooled_scores
_p = make_population(0.10, 0.80, 0.0); _rng = np.random.default_rng(5)
for _t in [1600, 12250, 500000]:
    _m = _t // 50; _y1 = _rng.binomial(_m, _p['theta1'], size=(20000, 50))
    _, _, _w = pooled_scores(_y1, _m, _p['mu'])
    print(f'%   straddling r=0, {_t:,} treated: pooled factor exactly zero in {100 * (_w == 0).mean():.1f}% of 20,000 experiments; mean factor {_w.mean():.3f}')
