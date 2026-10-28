# When Do Experiments Improve Treatment Prioritization?

Code for the paper **"When Do Experiments Improve Treatment Prioritization?"** by Carlos Fernández-Loría, Yanfang Hou, and Jorge Loría.

Organizations often decide whom to treat by ranking people on a prediction of what would happen without treatment. Such predictions can contain everything needed for a good allocation yet put the wrong people first. The paper asks when experiments improve prioritization beyond these predictions, for binary outcomes in a threshold model: learning how to use what predictions contain can be cheap, while learning what they miss pays only near the treatment cutoff, so its value depends on how many people can be treated. This repository reproduces every simulated and empirical number, table, and figure in the paper.

## Contents

```
code/                 simulations of the running example (Sections 3-5, Appendices B-E)
  results/            saved simulation results and printed outputs
  empirical/          the Criteo advertising experiment (Section 6, Appendices D and F)
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
python make_outputs.py             # Tables 2 and 3
python make_outputs_pooled.py      # Figure 4 and the pooled estimator (Appendix D)
python make_outputs_equal.py       # learned baseline (Appendix E), Figure 7
python make_figures.py             # Figure 3B
python shrink_figure.py            # Figure 6
python blend_check.py              # Table 6: reruns the blends in about 10 seconds -> results/blend_check.json
cd empirical
python make_criteo_outputs.py      # Tables 4 and 8, Figure 5, numbers in Section 6 and Appendix F
python make_blend_outputs.py       # Table 7
```

`code/README.md` lists the scripts that regenerate the saved results from scratch (seeds are fixed) and maps every result in the paper to the script that produces it; `code/empirical/README.md` does the same for the Criteo analysis, whose full run takes about two hours on two cores. Section, table, and figure numbers refer to the working-paper version of the paper.

The saved results were produced with Python 3.13, numpy 2.5, scipy 1.18, pandas 3.0, and lightgbm 4.7. The simulations reproduce exactly under any numpy version with the same random generator; the empirical results may differ slightly under other lightgbm versions.

## Data

The empirical analysis uses the Criteo uplift dataset, version 2.1 (Diemert et al. 2018, 2021), released under a CC BY-NC-SA 4.0 license. It is not redistributed here; `data/README.md` explains where to download it.

## Citation

If you use this code, please cite the paper:

```
@misc{fernandezloria2026experiments,
  author = {Fern{\'a}ndez-Lor{\'i}a, Carlos and Hou, Yanfang and Lor{\'i}a, Jorge},
  title  = {When Do Experiments Improve Treatment Prioritization?},
  year   = {2026},
  note   = {Working paper}
}
```

## License

Code: MIT (see `LICENSE`). The Criteo data keep their own license.
