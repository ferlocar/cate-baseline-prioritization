"""Blend outputs (Section 6): values at selected fractions and Qini, by experiment size.

    python make_blend_outputs.py
"""
import numpy as np
B = np.load('results/criteo_blend.npz', allow_pickle=True)
F = np.arange(1, 101) / 100; ATE = float(B['ate']); NAMES = list(B['names'])
LAB = dict(bate='BATE, one-parameter logit', x='CATE, X-learner', blend_const='Blend, one weight', blend_base='Blend, weights by baseline')
for n in np.unique(B['n']):
    sel = B['n'] == n; C = 1000 * B['curves'][sel].mean(0)
    print(f'\nn = {n:,} ({sel.sum()} reps); weight (one): {B["wc"][sel].mean():.2f}; weights by baseline bin (mean): {np.round(B["wb"][sel].mean(0), 2)}')
    print(f'{"":28s}{"1%":>7s}{"5%":>7s}{"10%":>7s}{"25%":>7s}{"50%":>7s}{"80%":>7s}{"Qini":>8s}')
    for i, k in enumerate(NAMES):
        c = C[i]
        print(f'{LAB[k]:28s}' + ''.join(f'{c[j]:7.2f}' for j in [0, 4, 9, 24, 49, 79]) + f'{np.mean(c - 1000 * F * ATE):8.3f}')
