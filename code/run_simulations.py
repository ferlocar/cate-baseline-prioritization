"""Run every simulated experiment in the paper and save results/simulations.json.

    python run_simulations.py              # full run, 3,000 replications (slow)
    python run_simulations.py --reps 100   # quick check with fewer replications

Scenarios: rare and common populations with r = 0.8, 0.4, 0.2, 0.1 (Tables 2 and 3, Figure 7);
the reversed rare population (last row of Table 3); and the straddling population with a
common shift and with r = 0.4 (Figure 4). The random seeds reproduce the paper's numbers.
"""
import argparse, json, time
import numpy as np
from model import POPULATIONS, EXPECTED_SHIFT as D, make_population, summaries, reversed_shift_sd
from experiments import run_one_sided, run_straddling, SIZES

R_VALUES = [0.8, 0.4, 0.2, 0.1]


def main(R):
    out = {'sizes': SIZES.tolist(), 'replications': R}; t0 = time.time()
    for name in ['rare', 'common']:
        p = POPULATIONS[name]
        for r in R_VALUES:
            pop = make_population(p['lo'], p['hi'], r * D)
            res = run_one_sided(pop, p['sign'], R, seed=11)
            out[f'{name}_{r}'] = dict(r=r, **{k: float(v) for k, v in summaries(pop).items()}, **res)
            print(f'{name} r={r}: average crossover {res["n_avg_known"]}  ({time.time() - t0:.0f}s)', flush=True)
    p = POPULATIONS['rare']; sd = reversed_shift_sd(p['lo'], p['hi'])
    pop = make_population(p['lo'], p['hi'], sd)
    out['reversed'] = dict(shift_sd_by_group=sd.tolist(), **{k: float(v) for k, v in summaries(pop).items()},
                           **run_one_sided(pop, +1, R, seed=12))
    print(f'reversed done ({time.time() - t0:.0f}s)', flush=True)
    p = POPULATIONS['straddling']
    for r in [0.0, 0.4]:
        pop = make_population(p['lo'], p['hi'], r * D)
        out[f'straddling_{r}'] = dict(r=r, forgone_share=float(summaries(pop)['forgone_share']),
                                      **run_straddling(pop, R, seed=13))
        print(f'straddling r={r} done ({time.time() - t0:.0f}s)', flush=True)
    json.dump(out, open('results/simulations.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=3000)
    main(ap.parse_args().reps)
