"""Print the plotted coordinates of Figures 3B, 4, and 7 (pgfplots format)."""
import json
import numpy as np
from model import F, f, logit, make_population, normalized_qini, scenario, EXPECTED_SHIFT as D
from experiments import simulated_size as _sim
# Sizes in treated people: each simulated experiment of 100m people has 50m treated (the paper's units).
sim = lambda n: None if n is None else _sim(n) // 2
from formulas import n_local

S = json.load(open('results/simulations.json'))
coords = lambda pts: ' '.join(f'({a},{b})' for a, b in pts)


def figure_3b():
    """Continuous straddling population: loss in normalized Qini from a peak error eps, and
    the bound of Proposition 6, g_bar * K_delta * eps^2, in the same units."""
    c = -D / 2; lo, hi = logit(0.10), logit(0.80); N = 200001
    mu = np.linspace(lo, hi, N); beta = F(mu + D) - F(mu)
    avg = lambda score: (np.cumsum(beta[np.argsort(-score)]) / N).mean()
    best = avg(-np.abs(mu - c)); gain = best - beta.mean() / 2
    loss = lambda e: 100 * (best - avg(-np.abs(mu - (c + e)))) / gain
    z = np.linspace(-20, 20, 400001); K = np.max(np.abs(f(z + D) - f(z)))
    print(f'FIGURE 3B  bound: {100 * K / (hi - lo) / gain:.6f}*x^2;  dot at 0.25: {loss(0.25):.4f}')
    print('  loss:', coords((f'{e:.2f}', f'{loss(e):.4f}') for e in np.round(np.arange(0, 0.501, 0.01), 2)))


def figure_4():
    for r, panel in [(0.0, 'A'), (0.4, 'B')]:
        x = S[f'straddling_{r}']; pop = make_population(0.10, 0.80, r * D)
        q = lambda v: normalized_qini(v, pop)
        print(f'FIGURE 4{panel}  baseline {q(x["baseline"]):.2f}, threshold {q(x["threshold"]):.2f}')
        for ranking in ['logit', 'cate']:
            merged = {}
            for n, v in x[ranking].items():
                merged.setdefault(sim(n), []).append(v)       # sizes that round to the same simulated size
            print(f'  {ranking}:', coords((n, f'{q(np.mean(v)):.2f}') for n, v in sorted(merged.items())))


def figure_5():
    for key in ['rare_0.4', 'rare_0.2']:
        x = S[key]; pop, sign = scenario(key)
        print(f'FIGURE 7 {key}  n*: {x["nstar_known"] / 2:.0f}')
        print('  simulated:', coords((f'{(j + 1) / 50:.2f}', sim(v)) for j, v in enumerate(x['win_by_u'][:45]) if v is not None))
        print('  formula:  ', coords((f'{(j + 1) / 50:.2f}', f'{v / 2:.0f}') for j, v in enumerate(n_local(pop, sign)[:45])))


if __name__ == '__main__':
    figure_3b(); figure_4(); figure_5()
