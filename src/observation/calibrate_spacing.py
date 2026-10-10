import os
import numpy as np, pandas as pd, json
from pathlib import Path
D = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'derived'
N = 51122257
sec = np.memmap(D / 'gps_sec_i32.bin', np.int32, 'r', shape=(N,))
ids = pd.read_csv(D / 'trajectory_courier_canonical_map.csv')
city_by = np.full(len(ids), -1, np.int8)
cm = {'cq': 0, 'hz': 1, 'jl': 2, 'sh': 3, 'yt': 4}
for c, v in cm.items():
    city_by[ids.loc[ids.city == c, 'courier_canon'].astype(int).to_numpy()] = v
runs = pd.read_csv(D / 'courier_day_run_index.csv')
items = list(runs.groupby(['courier_canon', 'ds'], sort=False).indices.items())
agg = {k: {'trans': np.zeros((2, 2), np.int64), 'start': np.zeros(2, np.int64), 'k': np.zeros(31, np.int64), 'eligible': 0, 'boundaries': 0, 'segments': 0} for k in [-2, 0, 1, 2, 3, 4]}
for gi, ((canon, ds), idxs) in enumerate(items):
    ci = int(city_by[int(canon)])
    if ci < 0:
        continue
    rr = runs.iloc[idxs]
    ss = []
    for r in rr.itertuples(index=False):
        ss.append(np.asarray(sec[int(r.start_idx):int(r.end_idx_exclusive)]))
    s = np.concatenate(ss) if len(ss) > 1 else ss[0]
    if len(ss) > 1:
        s = np.sort(s, kind='mergesort')
    if len(s) < 2:
        continue
    d0 = np.diff(s.astype(np.int64))
    s = s[np.r_[True, d0 > 0]]
    if len(s) < 2:
        continue
    dt = np.diff(s.astype(np.int64))
    for key in (-2, ci):
        A = agg[key]
        A['boundaries'] += int((dt > 600).sum())
    ok = dt <= 600
    k = np.floor(dt / 20.0 + 0.5).astype(np.int16)
    k = np.clip(k, 1, 30)
    state = (k >= 2).astype(np.int8)
    starts = np.flatnonzero(ok & np.r_[True, ~ok[:-1]])
    for key in (-2, ci):
        A = agg[key]
        A['start'] += np.bincount(state[starts], minlength=2)
        A['segments'] += len(starts)
        A['eligible'] += int(ok.sum())
        A['k'] += np.bincount(k[ok], minlength=31)[:31]
    pair = ok[:-1] & ok[1:]
    if pair.any():
        p = state[:-1][pair]
        n = state[1:][pair]
        C = np.zeros((2, 2), np.int64)
        for a in (0, 1):
            for b in (0, 1):
                C[a, b] = int(((p == a) & (n == b)).sum())
        for key in (-2, ci):
            agg[key]['trans'] += C
    if gi % 5000 == 0:
        print('group', gi, flush=True)
names = {-2: 'pooled_linked', 0: 'cq', 1: 'hz', 2: 'jl', 3: 'sh', 4: 'yt'}
rows = []
krows = []
for key, A in agg.items():
    T = A['trans']
    rows.append({'group': names[key], 'eligible_intervals_le600': A['eligible'], 'boundaries_gt600': A['boundaries'], 'active_segments': A['segments'], 'start_regular': A['start'][0], 'start_gap': A['start'][1], 'p_start_gap': A['start'][1] / A['start'].sum() if A['start'].sum() else np.nan, 't00': T[0, 0], 't01': T[0, 1], 't10': T[1, 0], 't11': T[1, 1], 'p_gap_after_regular': T[0, 1] / T[0].sum() if T[0].sum() else np.nan, 'p_gap_after_gap': T[1, 1] / T[1].sum() if T[1].sum() else np.nan})
    for kk in range(1, 31):
        if A['k'][kk]:
            krows.append({'group': names[key], 'slot_jump_k': kk, 'count': int(A['k'][kk]), 'probability': A['k'][kk] / A['k'][1:].sum()})
pd.DataFrame(rows).to_csv(D / 'empirical_spacing_transition_parameters.csv', index=False)
pd.DataFrame(krows).to_csv(D / 'empirical_spacing_jump_distribution.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\npooled k top')
print(pd.DataFrame(krows).query("group=='pooled_linked'").sort_values('count', ascending=False).head(15).to_string(index=False))
