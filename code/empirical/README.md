# Criteo illustration (Section 6 and Appendix E)

Requires Python 3 with numpy, scipy, pandas, and lightgbm. Run all commands from this folder.

## Data

Download `criteo-research-uplift-v2.1.csv.gz` (see `../../data/README.md`) into `../../data/`, then

```
python prepare_data.py          # -> ../../data/criteo.npz (about 30 seconds)
```

## Quick start

```
python make_criteo_outputs.py   # Tables 4 and 8, Figure 5 coordinates, and numbers quoted in the text
```

This reads the saved results in `results/` and takes a second. To regenerate them:

```
python run_criteo.py            # BATE, T-learner, and offset learner for every experiment, about 40 minutes on two cores -> results/criteo_curves.npz
python run_xlearner.py          # the X-learner for the same experiment sizes, about an hour -> results/criteo_xlearner.npz
python run_blend.py             # blends of estimated BATE and the X-learner, weights from split halves (not in the paper), about 15 minutes -> results/criteo_blend.npz
python make_blend_outputs.py     # prints the blending table (not in the paper)
python criteo_effects.py        # effects by baseline group and evaluation noise, a few minutes -> results/criteo_effects.json
```

Seeds are fixed. `python run_criteo.py --quick` runs a small grid for testing.

## Design

- A random 30% of users (both arms) is held out to evaluate rankings.
- The remaining control users train the baseline and stay fixed.
- An experiment of size n draws n users from the remaining treated users; sizes count treated users only.
- Outcome: visit. Treatment assignment is used throughout (intention to treat).

## Files

| File | Contents |
|---|---|
| `criteo_common.py` | Data splits, boosted models, prioritization curves, the one-parameter logit, and the spline logit |
| `prepare_data.py` | Converts the CSV to `../../data/criteo.npz` |
| `run_criteo.py` | Fits estimated BATE prioritization (one-parameter logit, and a spline logit not reported in the paper) and the T- and offset learners for every experiment size and replication, and saves their curves |
| `run_xlearner.py` | The X-learner, which also learns from the training controls |
| `criteo_effects.py` | Effects by baseline group (Figure 5A) and evaluation noise from 20 folds of the held-out data |
| `make_criteo_outputs.py` | Prints the tables, figure coordinates, crossovers, and quoted numbers |
