"""Convert the downloaded Criteo CSV into a compact NumPy file read by the other scripts.

    python prepare_data.py     # ../../data/criteo-research-uplift-v2.1.csv.gz -> ../../data/criteo.npz
"""
import sys
import numpy as np, pandas as pd

SRC = '../../data/criteo-research-uplift-v2.1.csv.gz'
OUT = '../../data/criteo.npz'


def main(src=SRC, out=OUT):
    d = pd.read_csv(src)
    np.savez(out, X=d[[f'f{j}' for j in range(12)]].to_numpy(np.float32),
             t=d['treatment'].to_numpy(np.int8), visit=d['visit'].to_numpy(np.int8),
             conversion=d['conversion'].to_numpy(np.int8), exposure=d['exposure'].to_numpy(np.int8))
    print(f'{len(d):,} rows -> {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
