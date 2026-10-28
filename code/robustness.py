"""Robustness checks for the equal-n design (are we fooling ourselves?).

1. BATE treated share: is 6.6% cherry-picked? CATE-vs-BATE crossovers and BATE value across shares.
2. CATE allocation: does a balanced experiment handicap CATE? Crossovers vs baseline at 40/50/60% treated.
3. Continuity correction in BATE's estimated baseline.
4. Coupled vs independent samples.
5. Seed variability of crossovers, original (known baseline) and equal-n designs.
6. Allocation formula beyond the case it was checked on: shift heterogeneity, larger shift.
"""
import numpy as np
from model import make_population, normalized_qini, curve, best_curve, baseline_curve, logit, F, POPULATIONS, EXPECTED_SHIFT as D
from experiments import SIZES, first_from, run_one_sided
from experiments_equal import one_parameter_logit_est, run as run_equal, bate_share

SUB = SIZES[(SIZES >= 300) & (SIZES <= 2_000_000)][::2]       # every other grid size, up to 2 million


def pop_of(name, r, d=D, M=5):
    p = POPULATIONS[name]
    return make_population(p['lo'], p['hi'], r * d, d=d, M=M), p['sign']


def bate_curve(pop, n, share, R, rng, cc=(0.5, 1.0)):
    K = pop['K']; mT = max(int(round(n * share / K)), 1); mC = max(int(round(n * (1 - share) / K)), 1)
    y1 = rng.binomial(mT, pop['theta1'], (R, K)); y0 = rng.binomial(mC, pop['theta'], (R, K))
    mu = logit((y0 + cc[0]) / (mC + cc[1])); s = one_parameter_logit_est(y1, mT, mu)
    return curve(pop['beta'], -np.abs(mu + s[:, None] / 2), rng).mean(0)


def cate_curve(pop, n, share, R, rng):
    K = pop['K']; mT = max(int(round(n * share / K)), 1); mC = max(int(round(n * (1 - share) / K)), 1)
    y1 = rng.binomial(mT, pop['theta1'], (R, K)); y0 = rng.binomial(mC, pop['theta'], (R, K))
    return curve(pop['beta'], y1 / mT - y0 / mC, rng).mean(0)


def base_curve(pop, sign, n, R, rng):
    K = pop['K']; m = max(n // K, 1)
    th = rng.binomial(m, pop['theta'], (R, K)) / m
    return curve(pop['beta'], (sign or 1) * th, rng).mean(0)


def crossover(curve_a, curve_b, sizes):
    """First size from which mean(curve_a) >= mean(curve_b) at every larger size."""
    return first_from(sizes, np.array([a.mean() >= b.mean() for a, b in zip(curve_a, curve_b)]))


def check1(R=2000):
    print('1. BATE treated share (straddling, K=50)')
    rng = np.random.default_rng(101)
    for r in [0.0, 0.4]:
        pop, _ = pop_of('straddling', r); best = best_curve(pop).mean(); q = lambda c: normalized_qini(c.mean() / best, pop)
        cates = [cate_curve(pop, n, 0.5, R, rng) for n in SUB]
        line = []
        for sh in [0.03, 0.066, 0.10, 0.20, 0.50]:
            bates = [bate_curve(pop, n, sh, R, rng) for n in SUB]
            vals = {n: q(b) for n, b in zip(SUB, bates)}
            v3k = vals[min(SUB, key=lambda x: abs(x - 3000))]; v10k = vals[min(SUB, key=lambda x: abs(x - 10000))]
            line.append(f'{100 * sh:4.1f}%: BATE@3k {v3k:5.1f} @10k {v10k:5.1f}' + (f', CATE overtakes at {crossover(cates, bates, SUB)}' if r > 0 else ''))
        print(f'  r={r}: ' + ' | '.join(line))


def check2(R=2000):
    print('2. CATE treated share vs baseline prioritization (crossover on average)')
    rng = np.random.default_rng(202)
    for name, r in [('rare', 0.4), ('rare', 0.2), ('common', 0.4)]:
        pop, sign = pop_of(name, r)
        bases = [base_curve(pop, sign, n, R, rng) for n in SUB]
        v, v1 = np.mean(pop['theta'] * (1 - pop['theta'])), np.mean(pop['theta1'] * (1 - pop['theta1']))
        neyman = np.sqrt(v1) / (np.sqrt(v) + np.sqrt(v1))
        res = []
        for sh in [0.4, 0.5, 0.6, neyman]:
            cates = [cate_curve(pop, n, sh, R, rng) for n in SUB]
            res.append(f'{100 * sh:.1f}%: {crossover(cates, bases, SUB)}')
        print(f'  {name} r={r} (Neyman share {100 * neyman:.1f}%): ' + ', '.join(res))


def check3(R=3000):
    print('3. Continuity correction in BATE baseline (straddling r=0, 6.6% treated), normalized Qini at n = 1k / 3k / 10k')
    rng = np.random.default_rng(303); pop, _ = pop_of('straddling', 0.0); best = best_curve(pop).mean()
    for cc in [(0.5, 1.0), (1.0, 2.0), (0.25, 0.5)]:
        vals = [normalized_qini(bate_curve(pop, n, 0.066, R, rng, cc).mean() / best, pop) for n in [1000, 3000, 10000]]
        print(f'  (y+{cc[0]})/(m+{cc[1]}): ' + ' / '.join(f'{v:.1f}' for v in vals))


def check4(R=2000):
    print('4. Coupled (paper) vs independent samples: CATE crossovers')
    rng = np.random.default_rng(404)
    for name, r in [('rare', 0.4), ('common', 0.4)]:
        pop, sign = pop_of(name, r)
        bases = [base_curve(pop, sign, n, R, rng) for n in SUB]; cates = [cate_curve(pop, n, 0.5, R, rng) for n in SUB]
        res = run_equal(pop, sign, R=R, seed=405, sizes=SUB, share=bate_share(50))
        print(f'  {name} r={r}: CATE vs baseline, independent {crossover(cates, bases, SUB)}, coupled {res["n_avg_vs_baseline"]}')
    pop, _ = pop_of('straddling', 0.4)
    bates = [bate_curve(pop, n, 0.066, R, rng) for n in SUB]; cates = [cate_curve(pop, n, 0.5, R, rng) for n in SUB]
    res = run_equal(pop, 0, R=R, seed=406, sizes=SUB, share=bate_share(50))
    print(f'  straddling r=0.4: CATE vs BATE, independent {crossover(cates, bates, SUB)}, coupled {res["n_avg_vs_bate"]}')


def check5(R=3000):
    print('5. Seed variability of average crossovers (CATE vs baseline), three seeds each')
    for name, r in [('rare', 0.4), ('rare', 0.2), ('common', 0.4), ('common', 0.2)]:
        pop, sign = pop_of(name, r)
        known = [run_one_sided(pop, sign, R, seed=s, sizes=SUB)['n_avg_known'] for s in [1, 2, 3]]
        equal = [run_equal(pop, sign, R=R, seed=s, sizes=SUB, share=bate_share(50))['n_avg_vs_baseline'] for s in [1, 2, 3]]
        print(f'  {name} r={r}: known {known}  equal-n {equal}  ratio of means {np.mean(equal) / np.mean(known):.2f}')


def check6(R=4000):
    print('6. Best BATE treated share beyond straddling r=0 (K=50, predicted 6.6%)')
    rng = np.random.default_rng(606)
    shares = np.array([0.02, 0.03, 0.045, 0.066, 0.1, 0.15, 0.22, 0.3, 0.5])
    for label, (name, r, d) in {'straddling r=0.4': ('straddling', 0.4, D), 'straddling, shift 1.0': ('straddling', 0.0, 1.0),
                                'straddling, shift 0.25': ('straddling', 0.0, 0.25)}.items():
        pop, _ = pop_of(name, r, d=d); best = best_curve(pop).mean(); n = 10000
        target = baseline_curve(pop, pop['mu'] * 0 + (-np.abs(pop['mu'] + d / 2))).mean()   # population BATE ranking under the expected shift
        loss = np.array([(target - bate_curve(pop, n, sh, R, rng).mean()) / best * 100 for sh in shares])
        X = np.column_stack([1 / (1 - shares), 1 / shares]); A, B = np.linalg.lstsq(X, loss, rcond=None)[0]
        fitted = 1 / (1 + np.sqrt(max(A / B, 0))) if B > 0 else float('nan')
        print(f'  {label:22s} loss (% of best) ' + ' '.join(f'{v:.2f}' for v in loss) +
              f' | argmin {shares[int(np.argmin(loss))]:.3f}, fitted {fitted:.3f}, balanced/best {loss[-1] / loss.min():.2f}')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
    for w in which:
        globals()[f'check{w}']()
