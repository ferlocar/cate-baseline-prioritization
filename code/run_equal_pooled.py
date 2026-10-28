"""Pooled estimator when the baseline must be learned (Appendix F) -> results/equal_pooled.json.

Same design as `run_equal.py --own`: every ranking gets n observations. Baseline and threshold
prioritization use n untreated people; estimated BATE prioritization its own control-heavy
experiment; the cell estimator and the pooled estimator one balanced experiment (m per arm per
cell). The pooled estimator estimates each cell's log-odds shift as the difference of the
continuity-corrected arm log-odds, with delta-method variance v_c, and shrinks it by empirical
Bayes toward the one-parameter logit's common shift fitted with the control-arm baselines. It ranks
by F(mu_hat_c + s_c') - F(mu_hat_c). All rankings come from the same draws in this run.

    python run_equal_pooled.py              # 3,000 replications
"""
import argparse, json, time
import numpy as np
from model import F, logit, POPULATIONS, EXPECTED_SHIFT as D, make_population, reversed_shift_sd, curve, best_curve
from experiments import SIZES, first_from
from experiments_equal import draw_equal, rankings, bate_share, one_parameter_logit_est
from run_simulations import R_VALUES


def pooled_scores_est(y1, y0, m):
    th1 = (y1 + 0.5) / (m + 1); th0 = (y0 + 0.5) / (m + 1)
    mu = logit(th0)
    s = one_parameter_logit_est(y1, m, mu)
    s_cell = logit(th1) - mu
    v = 1 / (m * th1 * (1 - th1)) + 1 / (m * th0 * (1 - th0))
    tau2 = np.maximum(((s_cell - s[:, None]) ** 2 - v).mean(1), 0)
    w = tau2[:, None] / (tau2[:, None] + v)
    return F(mu + s[:, None] + w * (s_cell - s[:, None])) - F(mu), w.mean(1)


def run(pop, sign, R, seed=41, sizes=SIZES):
    rng = np.random.default_rng(seed); K = pop['K']; best = best_curve(pop); b = pop['beta']
    curves = {}; wbar = []
    for n in sizes:
        y1, y0, yu, m = draw_equal(pop, n, R, rng)
        c, _, _ = rankings(pop, sign, y1, y0, yu, m, rng, share=bate_share(K))
        if sign == 0:
            c['baseline'] = c['base_up']
        sc, w = pooled_scores_est(y1, y0, m)
        c['pooled'] = curve(b, sc, rng).mean(0); wbar.append(float(w.mean()))
        for k in ['baseline', 'threshold', 'bate', 'cate', 'pooled']:
            curves.setdefault(k, []).append(c[k])
    curves = {k: np.array(v) for k, v in curves.items()}
    out = dict(sizes=[int(x) for x in sizes], weight_on_cell=wbar,
               share={k: (v.mean(1) / best.mean()).tolist() for k, v in curves.items()})
    for k in ['cate', 'pooled']:
        gap = curves[k] - curves['baseline']
        out[f'{k}_avg'] = first_from(sizes, gap.mean(1) >= 0)
        out[f'{k}_vs_bate_avg'] = first_from(sizes, (curves[k] - curves['bate']).mean(1) >= 0)
        out[f'{k}_by_u'] = [first_from(sizes, gap[:, j] >= 0) for j in range(K - 1)]
        out[f'{k}_by_u_tol'] = [first_from(sizes, gap[:, j] >= 0.01 * best[j]) for j in range(K - 1)]
    return out


def main(R):
    out = {}; t0 = time.time()
    jobs = [(f'{nm}_{r}', make_population(POPULATIONS[nm]['lo'], POPULATIONS[nm]['hi'], r * D), POPULATIONS[nm]['sign'])
            for nm in ['rare', 'common'] for r in R_VALUES]
    p = POPULATIONS['rare']; jobs.append(('reversed', make_population(p['lo'], p['hi'], reversed_shift_sd(p['lo'], p['hi'])), +1))
    p = POPULATIONS['straddling']; jobs += [(f'straddling_{r}', make_population(p['lo'], p['hi'], r * D), 0) for r in [0.0, 0.4]]
    for key, pop, sign in jobs:
        out[key] = x = run(pop, sign, R)
        print(f"{key}: vs baseline cell {x['cate_avg']} pooled {x['pooled_avg']}; vs BATE cell {x['cate_vs_bate_avg']} pooled {x['pooled_vs_bate_avg']} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open('results/equal_pooled.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=3000)
    main(ap.parse_args().reps)
