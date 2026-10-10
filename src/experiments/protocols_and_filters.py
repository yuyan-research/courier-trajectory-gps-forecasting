"""Courier-disjoint and temporal evaluation, Kalman filtering and aggregation.

The immutable mask generator is imported from sha256_missingness_masks.
Validation tuning is performed without access to held-out test windows.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from sha256_missingness_masks import mask_matrix as deterministic_mask_matrix
CITIES = ['cq', 'hz', 'jl', 'sh', 'yt']
SEEDS = [20260811, 20260812, 20260813, 20260814, 20260815]
HORIZONS = ['1m', '2m', '5m', 'ADE', 'FDE']
SCENARIOS = ['clean', 'sample_40s', 'sample_60s', 'sample_120s', 'sample_300s', 'random_10', 'random_30', 'random_50', 'burst_1m', 'burst_3m', 'burst_5m']
GRID_Q = [0.1, 1.0, 10.0]
GRID_R = [10.0, 25.0, 50.0]

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

def savecsv(rows, path, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False, encoding='utf-8-sig')

def load(root):
    z = np.load(root / 'data/windows_cache.npz', allow_pickle=True)
    return {k: z[k] for k in z.files if k != 'courier_categories'}

def courier_split(root, d):
    manifest = pd.read_csv(root / 'data/courier_split_manifest.csv')
    m = dict(zip(manifest.courier_code.astype(int), manifest.split.astype(str)))
    split = np.array([m[int(i)] for i in d['courier_code']])
    return split

def selection_splits(d, primary):
    """Temporal uses original fixed city-stratified courier assignments INTERSECTED with original chronological bands.
 No courier ever straddles groups. Identity date exclusivity is tested afterwards.
 """
    dates = np.sort(np.unique(d['date_key']))
    n = len(dates)
    i1 = max(1, int(n * 0.6))
    i2 = max(i1 + 1, int(n * 0.7))
    band = np.where(np.isin(d['date_key'], dates[:i1]), 'train', np.where(np.isin(d['date_key'], dates[i1:i2]), 'val', 'test'))
    temporal = {sp: np.flatnonzero((primary == sp) & (band == sp)) for sp in ['train', 'val', 'test']}
    base = {sp: np.flatnonzero(primary == sp) for sp in ['train', 'val', 'test']}
    loco = {}
    for held in range(5):
        training_pool = np.flatnonzero(d['city_code'] != held)
        couriers = np.unique(d['courier_code'][training_pool])
        rng = np.random.default_rng(20260811 + held)
        rng.shuffle(couriers)
        nv = max(1, int(len(couriers) * 0.1))
        vset = couriers[:nv]
        va = training_pool[np.isin(d['courier_code'][training_pool], vset)]
        tr = training_pool[~np.isin(d['courier_code'][training_pool], vset)]
        te = np.flatnonzero(d['city_code'] == held)
        loco[CITIES[held]] = {'train': tr, 'val': va, 'test': te}

    def assertsep(s, chron=False):
        ids = [set(d['courier_code'][s[k]].tolist()) for k in ['train', 'val', 'test']]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        if chron:
            a = [d['date_key'][s[k]] for k in ['train', 'val', 'test']]
            assert all((len(x) > 0 for x in a))
            assert a[0].max() < a[1].min() and a[1].max() < a[2].min()
    assertsep(base)
    assertsep(temporal, True)
    for ss in loco.values():
        assertsep(ss)
        a = set(d['city_code'][ss['train']])
        b = set(d['city_code'][ss['val']])
        c = set(d['city_code'][ss['test']])
        assert len(c) == 1 and (not c.intersection(a | b))
    return (base, temporal, loco, dates)

def nonoverlap(data, idx):
    d = data
    keys = d['courier_code'][idx].astype(np.int64) * 1000 + d['date_key'][idx]
    order = np.lexsort((d['cutoff_sec_of_day'][idx], keys))
    keep = []
    lastkey = -1
    lastend = -1e+20
    for i in idx[order]:
        key = int(d['courier_code'][i]) * 1000 + int(d['date_key'][i])
        start = float(d['cutoff_sec_of_day'][i]) + float(np.min(d['t_hist'][i]))
        if key != lastkey or start > lastend:
            keep.append(i)
            lastend = float(d['cutoff_sec_of_day'][i]) + float(np.max(d['target_dt'][i]))
            lastkey = key
    return np.sort(np.asarray(keep, dtype=np.int64))

@njit(cache=False)
def kf_one(t, x, y, q, r, hor):
    state = np.zeros(4, np.float64)
    state[0] = x[0]
    state[1] = y[0]
    P = np.zeros((4, 4), np.float64)
    P[0, 0] = 400.0
    P[1, 1] = 400.0
    P[2, 2] = 25.0
    P[3, 3] = 25.0
    last = float(t[0])
    rr = r * r
    for k in range(1, len(t)):
        dt = max(1.0, float(t[k] - last))
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        G = np.zeros((4, 2))
        G[0, 0] = 0.5 * dt * dt
        G[1, 1] = 0.5 * dt * dt
        G[2, 0] = dt
        G[3, 1] = dt
        state = F @ state
        P = F @ P @ F.T + q * (G @ G.T)
        s00 = P[0, 0] + rr
        s11 = P[1, 1] + rr
        s01 = P[0, 1]
        det = s00 * s11 - s01 * s01
        K = np.empty((4, 2))
        for j in range(4):
            K[j, 0] = (P[j, 0] * s11 - P[j, 1] * s01) / det
            K[j, 1] = (P[j, 1] * s00 - P[j, 0] * s01) / det
        u = x[k] - state[0]
        v = y[k] - state[1]
        state = state + K[:, 0] * u + K[:, 1] * v
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        P = (np.eye(4) - K @ H) @ P
        last = float(t[k])
    out = np.empty((3, 2))
    for i in range(3):
        dt = hor[i] - last
        out[i, 0] = state[0] + state[2] * dt
        out[i, 1] = state[1] + state[3] * dt
    return out

@njit(cache=False)
def ctrv_transition(st, dt):
    out = st.copy()
    a = st[4] * dt
    c = np.cos(a)
    s = np.sin(a)
    vx2 = c * st[2] - s * st[3]
    vy2 = s * st[2] + c * st[3]
    out[0] += dt * 0.5 * (st[2] + vx2)
    out[1] += dt * 0.5 * (st[3] + vy2)
    out[2] = vx2
    out[3] = vy2
    return out

@njit(cache=False)
def sigma_pts(st, P):
    n = 5
    pts = np.empty((11, 5))
    pts[0, :] = st
    pp = (P + P.T) * 0.5 + np.eye(5) * 1e-08
    L = np.linalg.cholesky(pp * 5.0)
    for i in range(5):
        pts[1 + i, :] = st + L[:, i]
        pts[6 + i, :] = st - L[:, i]
    return pts

@njit(cache=False)
def ukf_one(t, x, y, q, r, hor):
    st = np.zeros(5)
    st[0] = x[0]
    st[1] = y[0]
    P = np.diag(np.array([400.0, 400.0, 25.0, 25.0, 0.0004]))
    last = float(t[0])
    n = 5
    wc0 = 2.0
    for k in range(1, len(t)):
        dt = max(1.0, float(t[k] - last))
        sigma = sigma_pts(st, P)
        pp = np.empty((11, 5))
        for j in range(11):
            pp[j] = ctrv_transition(sigma[j], dt)
        mean = np.zeros(5)
        for j in range(1, 11):
            mean += pp[j] * 0.1
        cov = np.zeros((5, 5))
        for j in range(11):
            v = pp[j] - mean
            w = wc0 if j == 0 else 0.1
            cov += w * np.outer(v, v)
        G = np.zeros((5, 2))
        G[0, 0] = 0.5 * dt * dt
        G[1, 1] = 0.5 * dt * dt
        G[2, 0] = dt
        G[3, 1] = dt
        cov += q * (G @ G.T)
        cov[4, 4] += 1e-06 * dt
        ss = sigma_pts(mean, cov)
        ym = np.zeros(2)
        for j in range(1, 11):
            ym += ss[j, :2] * 0.1
        S = np.zeros((2, 2))
        cross = np.zeros((5, 2))
        for j in range(11):
            ey = ss[j, :2] - ym
            ex = ss[j] - mean
            w = wc0 if j == 0 else 0.1
            S += w * np.outer(ey, ey)
            cross += w * np.outer(ex, ey)
        S += np.eye(2) * r * r
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = cross @ Sinv
        innov = np.array([x[k], y[k]]) - ym
        st = mean + K @ innov
        P = cov - K @ S @ K.T
        P = (P + P.T) * 0.5
        last = float(t[k])
    out = np.empty((3, 2))
    for j in range(3):
        out[j] = ctrv_transition(st, hor[j] - last)[:2]
    return out

@njit(cache=False)
def batch_filter(xh, yh, th, target, indices, masks, q, r, ukf):
    n = len(indices)
    err = np.empty((n, 3), np.float64)
    horizons = np.array([60.0, 120.0, 300.0])
    for j in range(n):
        i = indices[j]
        mask = masks[j]
        t = th[i][mask]
        x = xh[i][mask]
        y = yh[i][mask]
        if ukf:
            pred = ukf_one(t, x, y, q, r, horizons)
        else:
            pred = kf_one(t, x, y, q, r, horizons)
        for h in range(3):
            err[j, h] = np.sqrt((target[i, h, 0] - pred[h, 0]) ** 2 + (target[i, h, 1] - pred[h, 1]) ** 2)
    return err

def get_masks(indices, scenario, rep=0):
    """Return the exact SHA-256-based formal evaluation masks."""
    return deterministic_mask_matrix(indices, scenario, rep)

def loss(err):
    return float(err.mean())

def get_report(err, indices, d, protocol, model, scenario, rep):
    """For balanced views, average WITHIN-GROUP mean/median/P90 (not quantile of group means)."""
    outputs = []
    city = d['city_code'][indices]
    cour = d['courier_code'][indices]
    date = d['date_key'][indices]
    if not len(err):
        return []
    E = np.column_stack((err, err.mean(axis=1), err[:, -1]))

    def stats(v):
        return (float(np.mean(v)), float(np.median(v)), float(np.quantile(v, 0.9)))
    for h, hn in enumerate(HORIZONS):
        v = E[:, h]
        for aggregation, groups in [('pooled_micro', None), ('equal_city_macro', city), ('courier_balanced', cour), ('equal_date_supplementary', date)]:
            if groups is None:
                mean, med, p90 = stats(v)
                n_groups = len(v)
            else:
                unique, inv = np.unique(groups, return_inverse=True)
                triplets = np.array([stats(v[inv == i]) for i in range(len(unique))])
                mean, med, p90 = triplets.mean(axis=0)
                n_groups = len(unique)
            outputs.append(dict(protocol=protocol, model=model, scenario=scenario, rep=rep, horizon=hn, aggregation=aggregation, n_windows=len(v), n_groups=n_groups, n_city=len(np.unique(city)), n_courier=len(np.unique(cour)), n_dates=len(np.unique(date)), mean_error_units=mean, median_error_units=med, p90_error_units=p90))
    return outputs

def run_filters(root, out, d, split, non_overlap_indices):
    """Validation ONLY tuning grid; no test access occurs until chosen configuration is written."""
    validation = split['val']
    test = split['test']
    validation_scen = ['clean', 'sample_60s', 'random_30', 'burst_3m']
    validation_rows = []
    selected = {}
    for model in ['KF_tuned', 'UKF_CTRV']:
        ukf = model.startswith('UKF')
        for q in GRID_Q:
            for r in GRID_R:
                vals = []
                for sc in validation_scen:
                    mask = get_masks(validation, sc, 0)
                    err = batch_filter(d['x_hist'], d['y_hist'], d['t_hist'], d['targets'], validation.astype(np.int64), mask, q, r, ukf)
                    vals.append(loss(err))
                validation_rows.append(dict(model=model, q=q, observation_noise_sd=r, validation_ADE_mean=float(np.mean(vals)), val_clean=vals[0], val_sample_60s=vals[1], val_random_30_rep0=vals[2], val_burst_3m=vals[3], n_val=len(validation)))
        subset = [x for x in validation_rows if x['model'] == model]
        best = min(subset, key=lambda x: (x['validation_ADE_mean'], x['q'], x['observation_noise_sd']))
        selected[model] = {'q': best['q'], 'r': best['observation_noise_sd'], 'val_ADE': best['validation_ADE_mean']}
        print('VALIDATION_SELECTED', model, selected[model], flush=True)
    savecsv(validation_rows, out / 'tables/KF_UKF_VALIDATION_GRID.csv')
    (out / 'tables/VAL_ONLY_SELECTION.json').write_text(json.dumps({'grid_q': GRID_Q, 'grid_observation_sd': GRID_R, 'fixed_selection_scenarios': validation_scen, 'fixed_reps': [0], 'selection_objective': 'unweighted_mean_of_four_scenario_ADE', 'selected': selected}, indent=2))
    filters = [('KF_original', 1.0, 25.0, False), ('KF_tuned', selected['KF_tuned']['q'], selected['KF_tuned']['r'], False), ('UKF_CTRV', selected['UKF_CTRV']['q'], selected['UKF_CTRV']['r'], True)]
    metric = []
    aggregate = []
    rows = []
    for sc in SCENARIOS:
        reps = range(30) if sc.startswith('random_') else [0]
        for rep in reps:
            masks = get_masks(test, sc, rep)
            for model, q, r, ukf in filters:
                t = time.perf_counter()
                saved = out / 'derived' / f'FILTER_{model}__{sc}__rep{rep:02d}.npz'
                if saved.exists():
                    existing = np.load(saved)
                    assert np.array_equal(existing['window_index'], test)
                    err = existing['error'].astype(np.float64)
                else:
                    err = batch_filter(d['x_hist'], d['y_hist'], d['t_hist'], d['targets'], test.astype(np.int64), masks, q, r, ukf)
                    if not np.isfinite(err).all():
                        raise RuntimeError('NaN filter errors ' + model)
                    np.savez_compressed(saved, error=err.astype(np.float32), window_index=test)
                metric += get_report(err, test, d, 'primary_courier_disjoint', model, sc, rep)
                print('FILTER', sc, rep, model, 'ADE', round(err.mean(), 3), 'seconds', round(time.perf_counter() - t, 1), flush=True)
                if rep == 0:
                    no = np.isin(test, non_overlap_indices)
                    aggregate += get_report(err[no], test[no], d, 'primary_nonoverlap', model, sc, rep)
                rows.append(dict(protocol='primary_courier_disjoint', model=model, scenario=sc, rep=rep, n_windows=len(test), ADE_mean_error_units=err.mean(), q=q, observation_noise_sd=r, selection='fixed original' if model == 'KF_original' else 'validation_only'))
    savecsv(metric, out / 'KF_UKF_VALIDATION_AND_TEST_RESULTS.csv')
    savecsv(rows, out / 'tables/FILTER_MODEL_SUMMARY.csv')
    savecsv(aggregate, out / 'tables/NONOVERLAP_FILTER_AGGREGATIONS.csv')
    return (metric, aggregate, selected)

def run_existing_networks(root, out, d, base, nonidx, device='cpu', checkpoint_dir=None):
    """Evaluate supplied primary checkpoints without retraining or re-selection."""
    import torch
    file = root / 'src/trajectory_forecasting.py'
    spec = importlib.util.spec_from_file_location('trajectory_forecasting_core', file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dat = module.load_cache(root / 'data/windows_cache.npz')
    test = base['test']
    valid = base['val']
    rows = []
    nonrows = []
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path(root) / 'checkpoints'
    for model in ['TCN_robust', 'Transformer_robust']:
        p = checkpoint_dir / f'primary_{model}.pt'
        chk = torch.load(p, map_location='cpu', weights_only=True)
        net = module.make_model(model)
        net.load_state_dict(chk['state_dict'])
        net.to(torch.device(device))
        net.eval()
        for sc in SCENARIOS:
            for rep in range(30) if sc.startswith('random_') else [0]:
                saved = out / 'derived' / f'PRIMARY_{model}__{sc}__rep{rep:02d}.npz'
                if saved.exists():
                    old = np.load(saved)
                    assert np.array_equal(old['window_index'], test)
                    e = old['error'].astype(np.float64)
                else:
                    mask = get_masks(test, sc, rep)
                    X = np.empty((len(test), 31, 5), np.float32)
                    for j, i in enumerate(test):
                        X[j], _, _ = module.prepare_input_features(d['x_hist'][i], d['y_hist'][i], d['t_hist'][i], mask[j])
                    predictions = []
                    with torch.no_grad():
                        for i in range(0, len(test), 512):
                            predictions.append(net(torch.from_numpy(X[i:i + 512]).to(device)).cpu().numpy().reshape(-1, 3, 2))
                    pred = np.concatenate(predictions)
                    target = (d['targets'][test] - np.stack([d['x_hist'][test, 0], d['y_hist'][test, 0]], axis=1)[:, None, :]) / 1000.0
                    e = np.linalg.norm(pred - target, axis=2) * 1000
                    assert np.isfinite(e).all()
                    np.savez_compressed(saved, error=e.astype(np.float32), window_index=test)
                rows += get_report(e, test, d, 'primary_courier_disjoint', model, sc, rep)
                if rep == 0:
                    no = np.isin(test, nonidx)
                    nonrows += get_report(e[no], test[no], d, 'primary_nonoverlap', model, sc, rep)
                print('PRIMARY_NET', model, sc, rep, 'ADE', round(e.mean(), 2), flush=True)
    savecsv(rows, out / 'tables/PRIMARY_NETWORK_AGGREGATIONS.csv')
    savecsv(nonrows, out / 'tables/NONOVERLAP_NETWORK_AGGREGATIONS.csv')
    return (rows, nonrows)
