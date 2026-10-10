import os
import sys, numpy as np, pandas as pd
from pathlib import Path
start, end = map(int, sys.argv[1:3])
D = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'derived'
raw = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'raw_traj_uncompressed.pkl'
N = 51122257
sec = np.memmap(D / 'gps_sec_i32.bin', np.int32, 'r', shape=(N,))
LAT_OFF = 1473052904 + 9
LNG_OFF = LAT_OFF + N * 8
lat = np.memmap(raw, dtype='<f8', mode='r', offset=LAT_OFF, shape=(N,))
lng = np.memmap(raw, dtype='<f8', mode='r', offset=LNG_OFF, shape=(N,))
ids = pd.read_csv(D / 'trajectory_courier_canonical_map.csv')
city_by = np.full(len(ids), -1, np.int8)
cm = {'cq': 0, 'hz': 1, 'jl': 2, 'sh': 3, 'yt': 4}
for c, v in cm.items():
    city_by[ids.loc[ids.city == c, 'courier_canon'].astype(int).to_numpy()] = v
runs = pd.read_csv(D / 'courier_day_run_index.csv')
items = list(runs.groupby(['courier_canon', 'ds'], sort=False).indices.items())[start:end]
cuts = np.array([0.15721296146512032, 0.6208023428916931, 2.289337992668152], float)
A = np.zeros((5, 6), np.int64)
for (canon, ds), idxs in items:
    rr = runs.iloc[idxs]
    ss = []
    xx = []
    yy = []
    for r in rr.itertuples(index=False):
        a = int(r.start_idx)
        b = int(r.end_idx_exclusive)
        ss.append(np.asarray(sec[a:b]))
        xx.append(np.asarray(lng[a:b]))
        yy.append(np.asarray(lat[a:b]))
    s = np.concatenate(ss) if len(ss) > 1 else ss[0]
    x = np.concatenate(xx) if len(xx) > 1 else xx[0]
    y = np.concatenate(yy) if len(yy) > 1 else yy[0]
    if len(ss) > 1:
        o = np.argsort(s, kind='mergesort')
        s = s[o]
        x = x[o]
        y = y[o]
    d0 = np.diff(s.astype(np.int64))
    keep = np.r_[True, d0 > 0]
    s = s[keep]
    x = x[keep]
    y = y[keep]
    if len(s) < 3:
        continue
    dt = np.diff(s.astype(np.int64))
    rate = np.hypot(np.diff(x), np.diff(y)) / dt
    elig = dt[:-1] == 20
    if not elig.any():
        continue
    pr = rate[:-1][elig]
    nxt = dt[1:][elig]
    state = np.empty(len(pr), np.int8)
    state[pr == 0] = 0
    pos = pr > 0
    state[pos] = 1 + np.searchsorted(cuts, pr[pos], side='left')
    for k in range(5):
        m = state == k
        if not m.any():
            continue
        z = nxt[m]
        A[k, 0] += len(z)
        A[k, 1] += int((z > 20).sum())
        A[k, 2] += int((z > 40).sum())
        A[k, 3] += int((z > 60).sum())
        A[k, 4] += int((z > 120).sum())
        A[k, 5] += int((z > 300).sum())
np.save(D / f'mobility_{start:05d}_{end:05d}.npy', A)
print(start, end, A.tolist())
