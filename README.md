# From Outcome Predictions to Effect Estimates

Code for the paper **"From Outcome Predictions to Effect Estimates: How Much Experimental Data Does Treatment Prioritization Need?"** by Carlos Fernández-Loría, Yanfang Hou, and Jorge Loría.

Organizations often decide whom to treat by ranking people on a prediction of what would happen without treatment. Ranking by estimated treatment effects instead requires experimental data. The paper studies when an experiment improves prioritization and how much data it needs, for binary outcomes in a threshold model. This repository reproduces every simulated and empirical number, table, and figure in the paper.

## Contents

```
code/                 simulations of the running example (Sections 3-5, Appendices C-D)
  results/            saved simulation results
  empirical/          the Criteo advertising experiment (Section 6, Appendix E)
    results/          saved empirical results
data/                 where to put the Criteo data (not included; see data/README.md)
```

## Setup

Python 3.10 or later.

```
git clone https://github.com/ferlocar/cate-baseline-prioritization.git
cd cate-baseline-prioritization
pip install -r requirements.txt
```

The simulations need only numpy. The empirical analysis also needs scipy, pandas, and lightgbm.

## Reproducing the results

Every table, figure coordinate, and quoted number can be printed in seconds from the saved results:

```
cd code
python make_tables.py              # Tables 2 and 3, numbers in Section 4 and Appendix D
python make_figures.py             # coordinates of Figures 3B, 4, and 5
cd empirical
python make_criteo_outputs.py      # Tables 4 and 6, Figure 6, numbers in Section 6 and Appendix E
```

To regenerate the saved results from scratch (seeds are fixed):

```
cd code
python run_simulations.py          # about 1.5 minutes
python run_continuous.py           # a few minutes
cd empirical
python prepare_data.py             # after downloading the data (see data/README.md)
python run_criteo.py               # about 40 minutes on two cores
python run_xlearner.py             # about an hour on two cores
python criteo_effects.py           # a few minutes
```

`code/README.md` and `code/empirical/README.md` describe each file and map every result in the paper to the script that produces it. Appendix letters refer to the working-paper version of the paper.

The saved results were produced with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, and lightgbm 4.7. The simulations reproduce exactly under any numpy version with the same random generator; the empirical results may differ slightly under other lightgbm versions.

## Data

The empirical analysis uses the Criteo uplift dataset, version 2.1 (Diemert et al. 2018, 2021), released under a CC BY-NC-SA 4.0 license. It is not redistributed here; `data/README.md` explains where to download it.

## Citation

If you use this code, please cite the paper:

```
@misc{fernandezloria2026outcome,
  author = {Fern{\'a}ndez-Lor{\'i}a, Carlos and Hou, Yanfang and Lor{\'i}a, Jorge},
  title  = {From Outcome Predictions to Effect Estimates: How Much Experimental Data Does Treatment Prioritization Need?},
  year   = {2026},
  note   = {Working paper}
}
```

## License

Code: MIT (see `LICENSE`). The Criteo data keep their own license.
