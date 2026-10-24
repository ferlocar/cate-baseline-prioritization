# Data

The empirical analysis (Section 6) uses the Criteo uplift dataset, version 2.1: about 14 million users from a randomized advertising experiment.

1. Download `criteo-research-uplift-v2.1.csv.gz` (311 MB) from https://huggingface.co/datasets/criteo/criteo-uplift ("Files and versions" tab) and place it in this folder.
2. From `code/empirical`, run `python prepare_data.py`. It writes `criteo.npz` here, which the other scripts read.

Columns:
- `f0`–`f11`: anonymized, randomly projected features
- `treatment`: 1 = assigned to treatment, 0 = control
- `visit`, `conversion`: binary outcomes (the paper uses `visit`)
- `exposure`: whether a treated user was actually shown an ad (not used; the analysis is by assignment)

License: CC BY-NC-SA 4.0. The data are not redistributed in this repository.

Citation:
- Diemert, E., Betlei, A., Renaudin, C., and Amini, M.-R. (2018). A Large Scale Benchmark for Uplift Modeling. AdKDD and TargetAd Workshop, KDD.
- Diemert, E., Betlei, A., Renaudin, C., Amini, M.-R., Gregoir, T., and Rahier, T. (2021). A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling. arXiv:2111.10106.
