"""Pooled CATE learner in the running example, main design (baseline known, sizes in
treated people) -> results/pooled.json.

The pooled learner estimates each cell's shift on the log-odds scale, s_c = F^{-1}(treated mean)
- mu_c (continuity-corrected), and shrinks it toward the one-parameter logit's common shift by
empirical Bayes: s_c' = s + tau^2 / (tau^2 + v_c) (s_c - s), with v_c the delta-method variance of
s_c and tau^2 = max(0, mean[(s_c - s)^2 - v_c]). With tau^2 = 0 it is estimated BATE
prioritization; with tau^2 large, the cell estimator on the log-odds scale. It ranks by
F(mu_c + s_c') - F(mu_c).

Every estimated ranking is computed from the same draws, so the saturated cell estimator is
recomputed here too (it matches results/simulations.json up to Monte Carlo and grid noise).

    python run_pooled.py              # 3,000 replications, about 2 minutes
"""
import argparse, json, time
import numpy as np
from model import F, logit, POPULATIONS, EXPECTED_SHIFT as D, make_population, reversed_shift_sd, curve, baseline_curve, best_curve, group_means
from experiments import SIZES, first_from, one_parameter_logit
from run_simulations import R_VALUES


def pooled_scores(y1, m, mu):
    """Empirical-Bayes shrinkage of cell log-odds shifts toward the common shift (baseline known)."""
    s = one_parameter_logit(y1, m, mu)                                   # common shift, per replication
    th1 = (y1 + 0.5) / (m + 1)
    s_cell = logit(th1) - mu[None, :]
    v = 1 / (m * th1 * (1 - th1))
    tau2 = np.maximum(((s_cell - s[:, None]) ** 2 - v).mean(1), 0)
    w = tau2[:, None] / (tau2[:, None] + v)
    s_eb = s[:, None] + w * (s_cell - s[:, None])
    return F(mu[None, :] + s_eb) - F(mu)[None, :], s, w.mean(1)


def run(pop, sign, R, seed, sizes=SIZES):
    rng = np.random.default_rng(seed)
    b, mu, K = pop['beta'], pop['mu'], pop['K']
    base = baseline_curve(pop, sign * pop['theta']) if sign != 0 else baseline_curve(pop, mu)
    best = best_curve(pop)
    cur = {'cate': [], 'pooled': [], 'bate': []}; wbar = []
    for n in sizes:
        m = max(int(n // (2 * K)), 1)                                   # m treated per cell: n/2 = 50m treated people
        y1 = rng.binomial(m, pop['theta1'], size=(R, K))
        cur['cate'].append(curve(b, y1 / m - pop['theta'][None, :], rng).mean(0))
        sc, s, w = pooled_scores(y1, m, mu)
        cur['pooled'].append(curve(b, sc, rng).mean(0))
        cur['bate'].append(curve(b, -np.abs(mu[None, :] + s[:, None] / 2), rng).mean(0))
        wbar.append(float(w.mean()))
    cur = {k: np.array(v) for k, v in cur.items()}
    out = dict(treated=[int(50 * max(int(n // (2 * K)), 1)) for n in sizes], weight_on_cell=wbar,
               share={k: (v.mean(1) / best.mean()).tolist() for k, v in cur.items()},
               base_share=float(base.mean() / best.mean()), bate_pop_share=float(baseline_curve(pop, group_means(pop)).mean() / best.mean()))
    tre = np.array(out['treated'])
    for k in ['cate', 'pooled']:
        gap = cur[k] - base[None, :]
        out[f'{k}_avg'] = first_from(tre, gap.mean(1) >= 0)
        out[f'{k}_all'] = first_from(tre, (gap[:, :K - 1] >= 0).all(1))
        out[f'{k}_by_u'] = [first_from(tre, gap[:, j] >= 0) for j in range(K - 1)]
        out[f'{k}_by_u_tol'] = [first_from(tre, gap[:, j] >= 0.01 * best[j]) for j in range(K - 1)]   # beat by 1% of the best value at u
        out[f'{k}_vs_bate_avg'] = first_from(tre, (cur[k] - cur['bate']).mean(1) >= 0)
    return out


def main(R):
    out = {}; t0 = time.time()
    jobs = [(f'{nm}_{r}', make_population(POPULATIONS[nm]['lo'], POPULATIONS[nm]['hi'], r * D), POPULATIONS[nm]['sign'])
            for nm in ['rare', 'common'] for r in R_VALUES]
    p = POPULATIONS['rare']; jobs.append(('reversed', make_population(p['lo'], p['hi'], reversed_shift_sd(p['lo'], p['hi'])), +1))
    p = POPULATIONS['straddling']; jobs += [(f'straddling_{r}', make_population(p['lo'], p['hi'], r * D), 0) for r in [0.0, 0.4]]
    for key, pop, sign in jobs:
        out[key] = run(pop, sign, R, seed=31)
        x = out[key]
        print(f'{key}: avg crossover vs baseline, cell {x["cate_avg"]}, pooled {x["pooled_avg"]}; vs BATE: cell {x["cate_vs_bate_avg"]}, pooled {x["pooled_vs_bate_avg"]} ({time.time() - t0:.0f}s)', flush=True)
    json.dump(out, open('results/pooled.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=3000)
    main(ap.parse_args().reps)
