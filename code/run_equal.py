"""Run the equal-n design for every scenario of the running example -> results/equal.json.

    python run_equal.py              # 3,000 replications
    python run_equal.py --reps 200   # quick check
    python run_equal.py --own        # BATE at its own allocation -> results/equal_own.json
"""
import argparse, json, time
import numpy as np
from model import POPULATIONS, EXPECTED_SHIFT as D, make_population, summaries, reversed_shift_sd
from experiments import SIZES
from experiments_equal import run, nstar_equal, bate_share
from run_simulations import R_VALUES


def main(R, own):
    out = {'sizes': SIZES.tolist(), 'replications': R}; t0 = time.time()
    jobs = [(f'{name}_{r}', make_population(POPULATIONS[name]['lo'], POPULATIONS[name]['hi'], r * D), POPULATIONS[name]['sign'])
            for name in ['rare', 'common'] for r in R_VALUES]
    p = POPULATIONS['rare']
    jobs.append(('reversed', make_population(p['lo'], p['hi'], reversed_shift_sd(p['lo'], p['hi'])), +1))
    p = POPULATIONS['straddling']
    jobs += [(f'straddling_{r}', make_population(p['lo'], p['hi'], r * D), 0) for r in [0.0, 0.4]]
    for key, pop, sign in jobs:
        res = run(pop, sign, R, share=bate_share(pop['K']) if own else None)
        summ = {k: float(v) for k, v in summaries(pop).items()}
        out[key] = dict(**summ, nstar_equal=float(nstar_equal(pop)) if summ['U'] > 1e-12 else None, **res)
        print(f'{key}: CATE vs baseline {res["n_avg_vs_baseline"]}, vs BATE {res["n_avg_vs_bate"]} ({time.time() - t0:.0f}s)', flush=True)
    out['bate_share'] = bate_share(50) if own else 0.5
    json.dump(out, open('results/equal_own.json' if own else 'results/equal.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=3000)
    ap.add_argument('--own', action='store_true', help='BATE uses its own control-heavy allocation')
    a = ap.parse_args(); main(a.reps, a.own)
