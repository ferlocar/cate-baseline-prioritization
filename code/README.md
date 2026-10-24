# Code for "From Outcome Predictions to Effect Estimates"

Reproduces every simulated number, table, and figure in the paper. Requires Python 3 and numpy.
Run all commands from this folder.

The Criteo illustration (Section 6 and Appendix E) is in `empirical/`, with its own README; it also needs pandas and lightgbm.

## Quick start

```
python make_tables.py      # Tables 2 and 3, and the numbers quoted in Section 4 and Appendix D
python make_figures.py     # plotted coordinates of Figures 3B, 4, and 5
```

These read the saved results in `results/` and take about a second.

To regenerate the results from scratch:

```
python run_simulations.py  # all simulated experiments, about 1.5 minutes -> results/simulations.json
python run_continuous.py   # continuous-population check of Proposition 10, a few minutes -> results/continuous.json
```

The random seeds are fixed, so a full run reproduces the saved results exactly.
`python run_simulations.py --reps 100` gives a quick, noisier check.

## Files

| File | Contents |
|---|---|
| `model.py` | The running example: logistic threshold model, the 50-cell populations (rare, common, straddling, reversed), prioritization curves, potential, forgone heterogeneity, the Gaussian sizes $n^*$, and normalized Qini |
| `experiments.py` | Simulated experiments: binomial outcomes per cell, the cell estimator, the one-parameter logit, and crossover sizes |
| `formulas.py` | The non-Gaussian approximations of Appendix C: the fraction-specific size and $\bar n^*$ |
| `run_simulations.py` | Runs every scenario and saves `results/simulations.json` |
| `run_continuous.py` | Runs the continuous population and saves `results/continuous.json` |
| `make_tables.py` | Prints the tables and quoted numbers |
| `make_figures.py` | Prints figure coordinates |

## Where each result comes from

| In the paper | Script and output section |
|---|---|
| Table 2 | `make_tables.py`, TABLE 2 |
| Table 3 | `make_tables.py`, TABLE 3 |
| Figure 3B | `make_figures.py`, FIGURE 3B (computed directly, no simulation) |
| Figure 4 and the Section 4 percentages | `make_figures.py`, FIGURE 4; `make_tables.py`, SECTION 4 |
| Figure 5 | `make_figures.py`, FIGURE 5 |
| Appendix D accuracy summaries and the reversed population | `make_tables.py`, APPENDIX D |

## Conventions

- An experiment of size $n$ gives $\lfloor n/100 \rfloor$ people to each arm of each of the 50 cells; reported sizes are $100\lfloor n/100\rfloor$.
- A crossover is the first tested size from which a ranking matches or beats its comparator at every larger tested size, using curves averaged over 3,000 replications.
- Normalized Qini is $(A(S)-A_{\text{random}})/(A_{\text{best}}-A_{\text{random}})$, with $A$ the mean of the prioritization curve over $u=j/50$.
