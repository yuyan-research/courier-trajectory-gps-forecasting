import os
import sys, numpy as np, pandas as pd, json, math
from pathlib import Path
start = int(sys.argv[1])
end = int(sys.argv[2])
tag = f'{start:05d}_{end:05d}'
D = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'derived'
raw = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'raw_traj_uncompressed.pkl'
N = 51122257
memo = np.memmap(D / 'courier_memo_i32.bin', np.int32, 'r', shape=(N,))
sec = np.memmap(D / 'gps_sec_i32.bin', np.int32, 'r', shape=(N,))
ds_arr = np.memmap(raw, dtype='<i8', mode='r', offset=200, shape=(N,))
LAT_OFF = 1473052904 + 9
LNG_OFF = LAT_OFF + N * 8
lat = np.memmap(raw, dtype='<f8', mode='r', offset=LAT_OFF, shape=(N,))
lng = np.memmap(raw, dtype='<f8', mode='r', offset=LNG_OFF, shape=(N,))
ids = pd.read_csv(D / 'trajectory_courier_canonical_map.csv')
city_by_canon = np.full(len(ids), -1, np.int8)
cmap = {'cq': 0, 'hz': 1, 'jl': 2, 'sh': 3, 'yt': 4}
for c, v in cmap.items():
    city_by_canon[ids.loc[ids.city == c, 'courier_canon'].astype(int).to_numpy()] = v
runs = pd.read_csv(D / 'courier_day_run_index.csv')
gb = runs.groupby(['courier_canon', 'ds'], sort=False).indices
items = list(gb.items())[start:end]
MAXDT = 86400
dt_counts = np.zeros(MAXDT + 1, np.int64)
dt_mod20 = np.zeros(20, np.int64)
transitions = np.zeros((2, 2), np.int64)
runlen_counts = {}
city_stats = {i: np.zeros(9, np.float64) for i in [-1, 0, 1, 2, 3, 4]}
hour_stats = {h: np.zeros(9, np.float64) for h in range(24)}
date_stats = {}
run_rows = []
rf = open(D / f'nominal20_rate_{tag}.bin', 'wb')
nominal_rate_count = 0
raw_dup = raw_dup_coord_diff = dedup_points = intervals = repeated_coord_pairs = source_speed_exceed = 0
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
        order = np.argsort(s, kind='mergesort')
        s = s[order]
        x = x[order]
        y = y[order]
    dd0 = np.diff(s.astype(np.int64))
    dup = dd0 == 0
    raw_dup += int(dup.sum())
    if dup.any():
        raw_dup_coord_diff += int((((np.diff(x) != 0) | (np.diff(y) != 0)) & dup).sum())
    keep = np.r_[True, dd0 > 0]
    s = s[keep]
    x = x[keep]
    y = y[keep]
    dedup_points += len(s)
    if len(s) < 2:
        run_rows.append([canon, ds, int(city_by_canon[int(canon)]), len(s), 0, np.nan, np.nan, np.nan, 0, 0, 0, 0, 0])
        continue
    dt = np.diff(s.astype(np.int64))
    dist = np.hypot(np.diff(x), np.diff(y))
    rate = dist / dt
    intervals += len(dt)
    repeated_coord_pairs += int((dist == 0).sum())
    source_speed_exceed += int((rate > 41.6666666667).sum())
    bc = np.bincount(dt, minlength=MAXDT + 1)
    dt_counts[:len(bc)] += bc
    dt_mod20 += np.bincount((dt % 20).astype(np.int64), minlength=20)[:20]
    nom = dt == 20
    if nom.any():
        np.asarray(rate[nom], dtype=np.float32).tofile(rf)
        nominal_rate_count += int(nom.sum())
    gap = dt > 20
    if len(gap) > 1:
        transitions[0, 0] += int((~gap[:-1] & ~gap[1:]).sum())
        transitions[0, 1] += int((~gap[:-1] & gap[1:]).sum())
        transitions[1, 0] += int((gap[:-1] & ~gap[1:]).sum())
        transitions[1, 1] += int((gap[:-1] & gap[1:]).sum())
    if gap.any():
        z = np.r_[False, gap, False].astype(np.int8)
        d = np.diff(z)
        lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        vals, cnts = np.unique(lens, return_counts=True)
        for L, nc in zip(vals, cnts):
            runlen_counts[int(L)] = runlen_counts.get(int(L), 0) + int(nc)
        maxrun = int(lens.max())
    else:
        maxrun = 0
    ci = int(city_by_canon[int(canon)])
    h = (s[:-1] // 3600).astype(int)

    def add(A, m):
        if not m.any():
            return
        z = dt[m]
        A[0] += len(z)
        A[1] += int((z > 20).sum())
        A[2] += int((z > 40).sum())
        A[3] += int((z > 60).sum())
        A[4] += int((z > 120).sum())
        A[5] += int((z > 300).sum())
        A[6] += float(z.sum())
        A[7] += float(np.maximum(z - 20, 0).sum())
        A[8] = max(A[8], float(z.max()))
    allm = np.ones(len(dt), bool)
    add(city_stats[ci], allm)
    for hh in np.unique(h):
        add(hour_stats[int(hh)], h == hh)
    if int(ds) not in date_stats:
        date_stats[int(ds)] = np.zeros(9, np.float64)
    add(date_stats[int(ds)], allm)
    run_rows.append([canon, ds, ci, len(s), len(dt), float(np.median(dt)), float(np.mean(dt)), float(dt.max()), int((dt > 20).sum()), int((dt > 60).sum()), int((dt > 120).sum()), int((dt > 300).sum()), maxrun])
rf.close()
np.savez_compressed(D / f'gap_shard_{tag}.npz', dt_counts=dt_counts, dt_mod20=dt_mod20, transitions=transitions, city_keys=np.array(list(city_stats)), city_vals=np.stack(list(city_stats.values())), hour_keys=np.array(list(hour_stats)), hour_vals=np.stack(list(hour_stats.values())), date_keys=np.array(list(date_stats)), date_vals=np.stack(list(date_stats.values())) if date_stats else np.empty((0, 9)), counters=np.array([raw_dup, raw_dup_coord_diff, dedup_points, intervals, repeated_coord_pairs, source_speed_exceed, nominal_rate_count], dtype=np.int64))
pd.DataFrame(run_rows, columns=['courier_canon', 'ds', 'city_code', 'n_unique_points', 'n_intervals', 'median_dt_sec', 'mean_dt_sec', 'max_dt_sec', 'gap_gt20', 'gap_gt60', 'gap_gt120', 'gap_gt300', 'max_consecutive_gap_intervals']).to_csv(D / f'run_rows_{tag}.csv', index=False)
json.dump(runlen_counts, open(D / f'runlen_{tag}.json', 'w'))
print(tag, 'groups', len(items), 'intervals', intervals, 'dup', raw_dup, 'nom20', nominal_rate_count)
