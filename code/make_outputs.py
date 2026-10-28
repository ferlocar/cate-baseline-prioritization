"""Running-example tables and figure coordinates in treated people: every size from
results/simulations.json is halved, since a simulated experiment of n people has n/2 treated
(and, with a known baseline, an untreated arm the estimators do not use)."""
import json
import numpy as np
from model import make_population, normalized_qini, scenario
from experiments import simulated_size as sim
from formulas import n_local, n_bar
from run_simulations import R_VALUES

O = json.load(open('results/simulations.json'))
tex = lambda n: '--' if n is None else f'{int(n):,}'.replace(',', '{,}')
half = lambda n: None if n is None else sim(n) // 2
r50 = lambda x: 'any' if x < 0 else tex(int(round(x / 2, -2)) if x / 2 >= 1000 else int(round(x / 2, -1)))
ROWS = [(name, r) for name in ['rare', 'common'] for r in R_VALUES]
coords = lambda pts: ' '.join(f'({a},{b})' for a, b in pts)

print('% TABLE 2 (known: n*, sim | same: n*, sim)')
for name, r in ROWS:
    x = O[f'{name}_{r}']
    print(f'{name.capitalize():6s} & {r} & {100 * x["forgone_share"]:.0f}\\% & {r50(x["nstar_known"])} & {tex(half(x["n_avg_known"]))} & '
          f'{r50(x["nstar_same"])} & {tex(half(x["n_avg_same"]))} \\\\')
print('% TABLE 3')
for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
    x = O[key]; w = x['win_by_u']
    lead = 'Reversed & ' if key == 'reversed' else f'{key.split("_")[0].capitalize():6s} & {key.split("_")[1]}'
    print(f'{lead} & {100 * x["forgone_share"]:.0f}\\% & {r50(x["nstar_known"])} & ' +
          ' & '.join(tex(half(v)) for v in [w[4], w[24], w[39], x['n_all_known']]) + ' \\\\')
for r in [0.0, 0.4]:
    x = O[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * 0.5); q = lambda v: normalized_qini(v, pop)
    for ranking in ['logit', 'cate']:
        merged = {}
        for n, v in x[ranking].items():
            merged.setdefault(sim(int(n)) // 2, []).append(v)
        print(f'% FIG4 r={r} {ranking}: ' + coords((n, f'{q(np.mean(v)):.2f}') for n, v in sorted(merged.items())))
for key in ['rare_0.4', 'rare_0.2']:
    x = O[key]; pop, sign = scenario(key)
    print(f'% FIG7 {key} nstar {x["nstar_known"] / 2:.6f}')
    print(f'% FIG7 {key} sim: ' + coords((f'{(j + 1) / 50:.2f}', half(v)) for j, v in enumerate(x['win_by_u'][:45]) if v is not None))
    print(f'% FIG7 {key} formula: ' + coords((f'{(j + 1) / 50:.2f}', f'{v / 2:.0f}') for j, v in enumerate(n_local(pop, sign)[:45])))
print('% APPENDIX D n_bar* (halved): ' + ', '.join(f'{k}: {tex(int(round(n_bar(scenario(k)[0]) / 2, -2)))}' for k in [f'{n}_{r}' for n, r in ROWS] + ['reversed']))
print('% reversed: nstar', r50(O['reversed']['nstar_known']), 'avg', tex(half(O['reversed']['n_avg_known'])))
