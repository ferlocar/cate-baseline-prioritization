# Code for "When Do Experiments Improve Treatment Prioritization?"

Reproduces every simulated number, table, and figure in the paper. Requires Python 3 and numpy.
Run all commands from this folder. The random seeds are fixed, so a full run reproduces the saved results exactly.

The Criteo illustration (Section 6 and Appendix F) is in `empirical/`, with its own README; it also needs pandas and lightgbm.

Section, table, and figure numbers refer to the working-paper version of the paper. In the submitted manuscript, Appendices A–F are e-companion sections EC.1–EC.6; Tables 6, 7, and 8 are Tables EC.1, EC.2, and EC.3; Figures 6 and 7 are Figures EC.1 and EC.2; Propositions 9 and 10 are Propositions EC.1 and EC.2; and Corollaries 3, 4, and 5 are Corollaries EC.1, EC.2, and EC.3.

## Quick start

These read the saved results in `results/` and take seconds:

```
python make_outputs.py          # Tables 2 and 3 -> results/outputs.tex
python make_outputs_pooled.py   # pooled-estimator numbers (Appendix D) and Figure 4 coordinates -> results/outputs_pooled.tex
python make_outputs_equal.py    # learned-baseline numbers (Appendix E) and Figure 7 coordinates -> results/outputs_equal.tex
python make_outputs_equal.py pooled   # pooled estimator with a learned baseline (Appendix E) -> results/outputs_equal_pooled.tex
python make_tables.py           # Tables 2 and 3 and the accuracy summaries of Appendix C
python make_figures.py          # Figure 3B coordinates
python shrink_figure.py         # blending figure coordinates (Appendix D) -> results/shrink_figure.txt
python check_shrinkage.py       # continuous-population check of the blending corollary and the moments of a weight learned when nothing is missed (Appendices C and D) -> results/shrinkage_check.txt
```

## Regenerating the saved results

```
python run_simulations.py       # cell estimator and one-parameter logit, baseline known -> results/simulations.json (about 1.5 minutes)
python run_pooled.py            # pooled estimator, with the cell estimator and logit on the same draws -> results/pooled.json (about 2 minutes)
python run_continuous.py        # continuous-population check of Proposition 8 -> results/continuous.json (a few minutes)
python blend_check.py           # blends of the BATE and the cell estimator with known and learned weights (Appendix D) -> results/blend_check.json (about 10 seconds)
python run_equal.py --own       # every ranking estimated from n observations (Appendix E) -> results/equal_own.json
python run_equal_pooled.py      # pooled estimator when the baseline must be learned -> results/equal_pooled.json (about 5 minutes)
python alloc_theory_check.py    # best treated share for estimated BATE prioritization (Appendix E) -> results/allocation_theory_check.txt
python robustness.py            # robustness of that share (Appendix E) -> results/robustness.txt
```

`--reps 100` gives a quick, noisier check for the `run_*.py` scripts that accept it.

## Units

Each simulated experiment puts m people in each arm of each of the 50 cells (100m in total). With the baseline known, the cell estimator and the one-parameter logit use only the treated arm, so the paper reports sizes as treated people, 50m; the output scripts halve the simulated size and every sample-size formula accordingly. When the baseline must be learned (Appendix F), sizes count all observations.

## Files

| File | Contents |
|---|---|
| `model.py` | The running example: logistic threshold model, the 50-cell populations (rare, common, straddling, reversed), prioritization curves, potential, forgone heterogeneity, the Gaussian size $n^*$, and normalized Qini |
| `experiments.py` | Simulated experiments with a known baseline: binomial outcomes per cell, the cell estimator, the one-parameter logit, and crossover sizes |
| `experiments_equal.py` | Simulated experiments when every ranking is estimated from the same number of observations |
| `formulas.py` | Fraction-specific and average-fraction size approximations (an earlier version of the paper; no longer reported) |
| `run_*.py` | Run the simulations and save `results/*.json` |
| `make_*.py` | Print tables, figure coordinates, and quoted numbers from the saved results |
| `shrink_figure.py`, `check_shrinkage.py` | Continuous-population checks of the blending results (Appendices C and D) |
| `alloc_theory_check.py`, `robustness.py` | The treated share for estimated BATE prioritization (Appendix E) |

## Where each result comes from

| In the paper | Source |
|---|---|
| Table 3 (narrow campaigns) | `make_outputs.py`, TABLE 3 block (or `make_tables.py`, TABLE 3) |
| Figure 3B | `make_figures.py`, FIGURE 3B (computed directly, no simulation) |
| Figure 4 | `make_outputs_pooled.py`, FIG4 lines (all estimators from the same run) |
| Figure 6 (Appendix D) | `shrink_figure.py` |
| Table 6 (Appendix D) | `blend_check.py` |
| Pooled estimator (Appendix D): half-gain sizes, narrow-campaign sizes, and how often the prior variance is zero | `make_outputs_pooled.py` (TABLE pooled block and "exactly zero" lines) |
| Table 2 and the size comparisons of Section 5.1 (Appendix B) | `make_outputs.py`, TABLE 2 block (or `make_tables.py`, TABLE 2) |
| Accuracy summaries, reversed population, pooled estimator (Appendix C) | `make_tables.py`; `make_outputs.py`; `make_outputs_pooled.py` |
| Checks of the blending corollary and of the 1/K cost of learning a weight (Appendices C and D) | `check_shrinkage.py` |
| Figure 7 and Appendix E | `make_outputs_equal.py`; `make_outputs_equal.py pooled`; `alloc_theory_check.py`; `robustness.py` |

## Conventions

- A crossover is the first tested size from which a ranking matches or beats its comparator at every larger tested size, using curves averaged over 3,000 replications.
- Normalized Qini is $(A(S)-A_{\text{random}})/(A_{\text{best}}-A_{\text{random}})$, with $A$ the mean of the prioritization curve over $u=j/50$.
