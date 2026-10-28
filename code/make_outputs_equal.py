"""LaTeX rows, figure coordinates, and quoted numbers for the equal-n running example,
from results/equal_own.json (BATE at its own allocation) and results/simulations.json."""
import json
import numpy as np
from model import make_population, normalized_qini, scenario, best_curve, baseline_curve, EXPECTED_SHIFT as D
from experiments import simulated_size as sim
from formulas import BW
from run_simulations import R_VALUES

W = json.load(open('results/equal_own.json')); O = json.load(open('results/simulations.json'))
sizes = np.array(W['sizes'])
tex = lambda n: '--' if n is None else f'{int(n):,}'.replace(',', '{,}')
r100 = lambda x: 'any' if x < 0 else tex(int(round(x, -2)))
ROWS = [(name, r) for name in ['rare', 'common'] for r in R_VALUES]


def n_local_equal(pop, sign):
    """Fraction-specific size for the difference in means from a balanced experiment of n:
    the known-baseline formula of formulas.n_local with w = theta1(1-theta1) replaced by
    theta(1-theta) + theta1(1-theta1); noise in the comparator's baseline is ignored."""
    b, K = pop['beta'], pop['K']
    w = pop['theta'] * (1 - pop['theta']) + pop['theta1'] * (1 - pop['theta1'])
    shortfall = best_curve(pop) - baseline_curve(pop, sign * pop['theta'])
    effects = np.sort(b)[::-1]; h = BW * b.std(); out = []
    for j in range(K - 1):
        q = 0.5 * (effects[j] + effects[j + 1])
        k = np.exp(-0.5 * ((b - q) / h) ** 2) / np.sqrt(2 * np.pi) / h
        g, w_u = k.mean(), (k * w).sum() / k.sum()
        out.append(K * w_u * g / shortfall[j] if shortfall[j] > 1e-15 else np.inf)
    return np.array(out)


def table3():
    print('% Learned baseline, average crossovers (Appendix F): population, r, forgone, n*, simulated, known baseline (simulated)')
    for name, r in ROWS:
        x = W[f'{name}_{r}']; o = O[f'{name}_{r}']
        print(f'{name.capitalize():6s} & {r} & {100 * x["forgone_share"]:.0f}\\% & {r100(x["nstar_equal"])} & '
              f'{tex(sim(x["n_avg_vs_baseline"]))} & {tex(sim(o["n_avg_known"]))} \\\\')


def table4():
    print('% Learned baseline, crossovers by fraction (Appendix F)')
    for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
        x = W[key]; w = x['win_by_u_vs_baseline']
        lead = 'Reversed & ' if key == 'reversed' else f'{key.split("_")[0].capitalize():6s} & {key.split("_")[1]}'
        print(f'{lead} & {100 * x["forgone_share"]:.0f}\\% & {r100(x["nstar_equal"])} & ' +
              ' & '.join(tex(sim(v)) for v in [w[4], w[24], w[39], x['n_all_vs_baseline']]) + ' \\\\')


def coords(pairs):
    return ' '.join(f'({a},{b})' for a, b in pairs)


def figure4():
    for r in [0.0, 0.4]:
        x = W[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D)
        q = lambda v: normalized_qini(v, pop)
        print(f'% FIGURE 8 r={r}')
        for k in ['bate', 'cate', 'threshold', 'baseline']:
            merged = {}
            for n, v in zip(sizes, x['share'][k]):
                merged.setdefault(sim(n), []).append(v)
            print(f'%   {k}: ' + coords((n, f'{q(np.mean(v)):.2f}') for n, v in sorted(merged.items())))


def figure5():
    for key in ['rare_0.4', 'rare_0.2']:
        x = W[key]; pop, sign = scenario(key)
        print(f'% Learned baseline, profile by fraction {key}: n* = {x["nstar_equal"]:.6f}')
        print('%   simulated: ' + coords((f'{(j + 1) / 50:.2f}', sim(v)) for j, v in enumerate(x['win_by_u_vs_baseline'][:45]) if v is not None))
        print('%   formula:   ' + coords((f'{(j + 1) / 50:.2f}', f'{v:.0f}') for j, v in enumerate(n_local_equal(pop, sign)[:45])))


def numbers():
    print('% QUOTED NUMBERS')
    for r in [0.0, 0.4]:
        x = W[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D); q = lambda v: normalized_qini(v, pop)
        for k in ['bate', 'cate', 'threshold', 'baseline']:
            vals = [q(v) for v in x['share'][k]]
            at = {n: vals[int(np.argmin(np.abs(sizes - n)))] for n in [300, 1000, 3000, 10000, 30000]}
            print(f'%   straddling r={r} {k}: ' + ', '.join(f'{tex(sim(sizes[int(np.argmin(np.abs(sizes - n)))]))}: {v:.1f}' for n, v in at.items()))
        if r == 0.0:
            bate_best = max(q(v) for v in x['share']['bate'])
            for target in [95, 99]:
                cate = [q(v) for v in x['share']['cate']]
                n_c = next((int(sizes[i]) for i in range(len(sizes)) if all(c >= target for c in cate[i:])), None)
                bate = [q(v) for v in x['share']['bate']]
                n_b = next((int(sizes[i]) for i in range(len(sizes)) if all(c >= target for c in bate[i:])), None)
                print(f'%   r=0: reach {target}%: BATE {tex(sim(n_b))}, CATE {tex(sim(n_c))}')
        print(f'%   r={r}: CATE overtakes estimated BATE at {tex(sim(x["n_avg_vs_bate"]))}')
    acc = []
    for name, r in ROWS:
        x = W[f'{name}_{r}']
        if x['nstar_equal'] > 0:
            acc.append(x['nstar_equal'] / sim(x['n_avg_vs_baseline']))
    print(f'%   n*/simulated across scenarios: {min(acc):.2f}-{max(acc):.2f}')
    ratios = [sim(W[f"{n}_{r}"]["n_avg_vs_baseline"]) / sim(O[f"{n}_{r}"]["n_avg_known"]) for n, r in ROWS if (n, r) != ('rare', 0.8)]
    print(f'%   equal-n / known-baseline simulated sizes (excluding rare 0.8): {min(ratios):.2f}-{max(ratios):.2f}')
    print(f'%   BATE treated share: {100 * W["bate_share"]:.1f}%')


if __name__ == '__main__':
    table3(); table4(); figure4(); figure5(); numbers()


def n_bar_equal(pop):
    """Average-performance size with w = theta(1-theta) + theta1(1-theta1) (cf. formulas.n_bar)."""
    from formulas import mean_abs_difference
    from model import group_means
    b, K = pop['beta'], pop['K']
    w = pop['theta'] * (1 - pop['theta']) + pop['theta1'] * (1 - pop['theta1'])
    G_bar = (mean_abs_difference(b) - mean_abs_difference(group_means(pop))) / 4
    h = BW * b.std()
    g_at = (np.exp(-0.5 * ((b[:, None] - b[None, :]) / h) ** 2) / np.sqrt(2 * np.pi) / h).mean(1)
    return K * np.mean(w * g_at) / G_bar


def appendix_d():
    print('% APPENDIX D (equal-n)')
    for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
        x = W[key]; pop, sign = scenario(key)
        s = sim(x['n_avg_vs_baseline']); nb = n_bar_equal(pop)
        by_u = np.array([np.nan if v is None else sim(v) for v in x['win_by_u_vs_baseline'][:45]], float)
        pred = n_local_equal(pop, sign)[:45]
        ratio = (pred / by_u)[9:45]; ratio = ratio[np.isfinite(ratio)]
        print(f'%   {key:11s} n*/sim {x["nstar_equal"] / s:.2f}  n_bar* {tex(int(round(nb, -2)))} (ratio {nb / s:.2f})  '
              f'median fraction-specific/sim 20-90%: {np.median(ratio):.2f}' +
              (f'; avg crossover {tex(s)}; top 90%: {tex(sim(x["win_by_u_vs_baseline"][44]))}; formula top10/90: {tex(int(round(pred[4], -2)))}, {tex(int(round(pred[44], -2)))}' if key == 'reversed' else ''))


if __name__ == '__main__':
    appendix_d()


def pooled_learned():
    """Pooled estimator with a learned baseline (results/equal_pooled.json; all rankings from that run)."""
    Pq = json.load(open('results/equal_pooled.json'))
    print('% POOLED, LEARNED BASELINE: half-gain size vs true-baseline BATE prioritization (cell / pooled), avg crossover vs baseline and vs BATE, 1% margin top 10/50/80')
    for key in [f'{n}_{r}' for n, r in ROWS] + ['reversed']:
        x = Pq[key]; pop, sign = scenario(key); t = np.array([sim(n) for n in x['sizes']]); q = lambda v: normalized_qini(v, pop)
        base = np.array([q(v) for v in x['share']['baseline']])
        # benchmark: BATE prioritization with the true baseline (as G(u) in the theory), not the estimated baseline ranking
        bate_true = q(baseline_curve(pop, sign * pop['theta']).mean() / best_curve(pop).mean()); best_gain = 100 - bate_true
        half = {}
        for k in ['cate', 'pooled']:
            gap = np.array([q(v) for v in x['share'][k]]) - bate_true
            ok = gap >= best_gain / 2
            half[k] = next((int(t[i]) for i in range(len(t)) if ok[i:].all()), None)
        f = lambda v: tex(sim(v))
        print(f"%   {key:11s} half: cell {tex(half['cate'])} pooled {tex(half['pooled'])} | avg vs baseline: cell {f(x['cate_avg'])} pooled {f(x['pooled_avg'])}"
              f" | vs BATE: cell {f(x['cate_vs_bate_avg'])} pooled {f(x['pooled_vs_bate_avg'])}"
              f" | 1%: cell {[f(x['cate_by_u_tol'][j]) for j in (4, 24, 39)]} pooled {[f(x['pooled_by_u_tol'][j]) for j in (4, 24, 39)]}")
    for r in [0.0, 0.4]:
        x = Pq[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D); q = lambda v: normalized_qini(v, pop); t = np.array([sim(n) for n in x['sizes']])
        for k in ['bate', 'cate', 'pooled']:
            vals = [q(v) for v in x['share'][k]]
            print(f'%   straddling r={r} {k}: ' + ', '.join(f'{tex(t[i])}: {vals[i]:.1f}' for i in [int(np.argmin(np.abs(t - n))) for n in [300, 1000, 3000, 10000, 30000, 100000]]))
        print(f"%   straddling r={r}: overtakes estimated BATE on average: cell {tex(sim(x['cate_vs_bate_avg']))} pooled {tex(sim(x['pooled_vs_bate_avg']))}; pooled factor at largest size {x['weight_on_cell'][-1]:.3f}")


if __name__ == '__main__' and 'pooled' in __import__('sys').argv:
    pooled_learned()
