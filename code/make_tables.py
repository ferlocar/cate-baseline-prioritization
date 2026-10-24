"""Print Tables 2 and 3 and the numbers quoted in the text and Appendix D, from
results/simulations.json and results/continuous.json. Reported sizes are those actually
simulated, 100 * floor(n / 100)."""
import json
import numpy as np
from model import EXPECTED_SHIFT as D, make_population, normalized_qini, baseline_curve, best_curve, scenario
from experiments import simulated_size as sim
from formulas import n_local, n_bar
from run_simulations import R_VALUES

fmt = lambda n: '--' if n is None else f'{int(n):,}'
r100 = lambda x: int(round(x, -2))
pct = lambda x: f'{100 * x:.0f}%'
ROWS = [(name, r) for name in ['rare', 'common'] for r in R_VALUES]


def main():
    S = json.load(open('results/simulations.json'))
    print('TABLE 2: population, r, forgone share | baseline known: n*, simulated | same experiment: n*, simulated')
    for name, r in ROWS:
        x = S[f'{name}_{r}']
        same = 'any' if x['nstar_same'] < 0 else fmt(r100(x['nstar_same']))
        print(f'{name:7s} {r:<4} {pct(x["forgone_share"]):>4} | {fmt(r100(x["nstar_known"])):>8} {fmt(sim(x["n_avg_known"])):>8} | {same:>8} {fmt(sim(x["n_avg_same"])):>8}')

    print('\nTABLE 3: population, r, forgone share, n*, top 10%, top 50%, top 80%, everywhere')
    for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
        x = S[key]; w = x['win_by_u']
        print(f'{key:11s} {pct(x["forgone_share"]):>4} {fmt(r100(x["nstar_known"])):>8} ' +
              ' '.join(f'{fmt(sim(v)):>10}' for v in [w[4], w[24], w[39], x['n_all_known']]))

    print('\nAPPENDIX D: accuracy of the approximations (baseline known)')
    for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
        x = S[key]; pop, sign = scenario(key)
        simulated = sim(x['n_avg_known']); nb = n_bar(pop)
        by_u = np.array([np.nan if v is None else sim(v) for v in x['win_by_u'][:45]], float)
        pred = n_local(pop, sign)[:45]
        ratio = (pred / by_u)[9:45]; ratio = ratio[np.isfinite(ratio)]           # fractions 20%-90%
        print(f'{key:11s} n*/sim {x["nstar_known"] / simulated:.2f}  n_bar* {fmt(r100(nb)):>8} (n_bar*/sim {nb / simulated:.2f})'
              f'  fraction-specific formula / sim, median over 20-90%: {np.median(ratio):.2f}')
        if key == 'reversed':
            print(f'            average crossover {fmt(sim(x["n_avg_known"]))}; top 90%: {fmt(sim(x["win_by_u"][44]))}; '
                  f'formula at top 10% and 90%: {fmt(r100(pred[4]))}, {fmt(r100(pred[44]))}')

    print('\nSECTION 4: straddling population (normalized Qini, %)')
    for r in [0.0, 0.4]:
        x = S[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D)
        q = lambda v: normalized_qini(v, pop)
        print(f'r={r}: baseline (highest first) {q(x["up"]):.1f}, lowest first {q(x["down"]):.1f}, threshold {q(x["threshold"]):.1f}, '
              f'BATE {q(x["bate"]):.1f}; CATE overtakes estimated BATE at {fmt(sim(x["n_cate_beats_logit_avg"]))}')
    pop = make_population(0.10, 0.80, 0.0, d=2.0); best = best_curve(pop).mean()       # common shift of 2
    share = lambda key: baseline_curve(pop, key).mean() / best
    print(f'shift 2: lowest first {normalized_qini(share(-pop["mu"]), pop):.1f}, '
          f'threshold {normalized_qini(share(-np.abs(pop["theta"] - 0.5)), pop):.1f}')

    C = json.load(open('results/continuous.json'))
    print('\nAPPENDIX D: continuous population, predicted/simulated noise at fractions 20-80%')
    for key, rows in C.items():
        ratio = [p / a for u, a, p in rows if 0.2 <= u <= 0.8]
        print(f'{key}: {min(ratio):.2f}-{max(ratio):.2f}')


if __name__ == '__main__':
    main()
