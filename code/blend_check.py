"""Check of baseline-dependent blending (Section 5.3): constant vs baseline-dependent weights.

Baseline known; m treated people per cell (n = 50 m treated). Every ranking is computed from the
same draws. Anchor for all blends: the one-parameter logit's BATE, F(mu+s)-F(mu) (as the pooled
estimator), except the oracle, which uses the true BATE and true signal shares.

  cell       treated mean - baseline (CATE estimate alone)
  bate       one-parameter logit (estimated BATE prioritization)
  const      one weight for everyone, moment estimate on the effect scale (EB with a common prior variance)
  pooled     the pooled estimator (log-odds EB, weights fall with each cell's noise)
  local      weight s(theta) from each baseline group's own moment estimate (5 cells)
  local3     same, pooling each group with its neighbors (15 cells)
  model      threshold-model weight: U(theta) = v f(mu+s)^2 with one v estimated from all cells
  oracle     true BATE and true signal share at each baseline
  oracle_const  true BATE and one weight, the overall signal share

    python blend_check.py          # 2,000 replications, about 10 seconds -> results/blend_check.json (Appendix F)
"""
import json, sys
import numpy as np
from model import F, logit, f, POPULATIONS, EXPECTED_SHIFT as D, make_population, reversed_shift_sd, curve, baseline_curve, best_curve, group_means
from experiments import one_parameter_logit
from run_pooled import pooled_scores

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MS = [4, 10, 20, 40, 100, 200, 400, 1000, 2000, 4000]          # treated per cell -> n = 50 m

def scenarios():
    out = []
    for nm in ['rare', 'common']:
        for r in [0.8, 0.4, 0.2, 0.1]:
            p = POPULATIONS[nm]; out.append((f'{nm}_{r}', make_population(p['lo'], p['hi'], r * D), p['sign']))
    p = POPULATIONS['rare']; out.append(('reversed', make_population(p['lo'], p['hi'], reversed_shift_sd(p['lo'], p['hi'])), +1))
    return out

def blend(m_hat, b_hat, w): return m_hat + w * (b_hat - m_hat)

def run(pop, sign, seed=7):
    rng = np.random.default_rng(seed)
    b, mu, th, th1, g, K = pop['beta'], pop['mu'], pop['theta'], pop['theta1'], pop['group'], pop['K']
    G = g.max() + 1
    base = baseline_curve(pop, sign * th); best = best_curve(pop)
    bate_true = group_means(pop)
    U_true = np.array([b[g == k].var() for k in g])          # forgone heterogeneity at each baseline
    res = {}
    for m in MS:
        y1 = rng.binomial(m, th1, size=(R, K))
        bh = y1 / m - th[None, :]                              # CATE estimate
        s = one_parameter_logit(y1, m, mu)
        mh = F(mu[None, :] + s[:, None]) - th[None, :]         # estimated BATE
        th1h = np.clip(th[None, :] + mh, 1e-6, 1 - 1e-6)
        N = th1h * (1 - th1h) / m                              # noise from the baseline and the BATE
        d = bh - mh
        sc = {}
        sc['cell'] = bh
        sc['bate'] = mh                                        # same ranking as the logit within one-sided pops
        Uc = np.maximum((d ** 2 - N).mean(1, keepdims=True), 0)
        sc['const'] = blend(mh, bh, Uc / (Uc + N.mean(1, keepdims=True)))
        sc['pooled'] = pooled_scores(y1, m, mu)[0]
        # local moments by baseline group
        dd = d ** 2 - N
        Ug = np.stack([dd[:, g == k].mean(1) for k in range(G)], 1)        # R x G
        Ul = np.maximum(Ug[:, g], 0)
        sc['local'] = blend(mh, bh, Ul / (Ul + N))
        Ug3 = np.stack([Ug[:, max(k - 1, 0):k + 2].mean(1) for k in range(G)], 1)
        U3 = np.maximum(Ug3[:, g], 0)
        sc['local3'] = blend(mh, bh, U3 / (U3 + N))
        sens2 = f(mu[None, :] + s[:, None]) ** 2
        v = np.maximum((d ** 2 - N).sum(1, keepdims=True) / sens2.sum(1, keepdims=True), 0)
        Um = v * sens2
        sc['model'] = blend(mh, bh, Um / (Um + N))
        Nt = th1 * (1 - th1) / m
        sc['oracle'] = blend(bate_true[None, :], bh, (U_true / (U_true + Nt))[None, :])
        sc['oracle_const'] = blend(bate_true[None, :], bh, np.full((1, K), U_true.mean() / (U_true.mean() + Nt.mean())))
        out = {}
        for k, S in sc.items():
            c = curve(b, S, rng).mean(0)
            gain = (c - base) / np.maximum(best - base, 1e-15)          # normalized gain over baseline prioritization
            out[k] = dict(avg=float((c - base).mean() / (best - base).mean()),
                          u10=float(gain[4]), u50=float(gain[24]), u80=float(gain[39]))
        res[50 * m] = out
    return res

if __name__ == '__main__':
    allres = {}
    for key, pop, sign in scenarios():
        allres[key] = run(pop, sign)
        print(key, flush=True)
    json.dump(allres, open('results/blend_check.json', 'w'), indent=1)
