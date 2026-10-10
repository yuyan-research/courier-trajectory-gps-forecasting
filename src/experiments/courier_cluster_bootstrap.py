"""Paired courier-cluster bootstrap and result aggregation.

All contrasts use the same 2,000 resampled courier clusters. Missing
model/scenario error arrays fail explicitly rather than being imputed.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from protocols_and_filters import CITIES, SCENARIOS, HORIZONS, sha, savecsv, load, courier_split, selection_splits, nonoverlap, get_report
MODELS = ['KF_original', 'KF_tuned', 'UKF_CTRV', 'TCN_robust', 'Transformer_robust']
COMPARISONS = [('KF_original', 'KF_tuned'), ('KF_tuned', 'UKF_CTRV'), ('UKF_CTRV', 'TCN_robust'), ('UKF_CTRV', 'Transformer_robust'), ('TCN_robust', 'Transformer_robust')]

def get_errors(out, model, sc):
    rep_range = range(30) if sc.startswith('random_') else [0]
    folder = 'FILTER' if model.startswith('KF') or model == 'UKF_CTRV' else 'PRIMARY'
    e = []
    for rep in rep_range:
        f = out / 'derived' / f'{folder}_{model}__{sc}__rep{rep:02d}.npz'
        if not f.is_file():
            raise FileNotFoundError(str(f))
        z = np.load(f)
        e.append(z['error'])
    return np.stack(e).mean(axis=0)

def regen_summaries(root, out):
    d = load(root)
    sp = courier_split(root, d)
    base, t, loco, dates = selection_splits(d, sp)
    test = base['test']
    non = nonoverlap(d, test)
    no = np.isin(test, non)
    row = []
    nonrows = []
    for model in MODELS:
        for sc in SCENARIOS:
            rs = range(30) if sc.startswith('random_') else [0]
            for rep in rs:
                folder = 'FILTER' if model.startswith('KF') or model == 'UKF_CTRV' else 'PRIMARY'
                f = out / 'derived' / f'{folder}_{model}__{sc}__rep{rep:02d}.npz'
                if not f.exists():
                    continue
                z = np.load(f)
                err = z['error']
                if not np.array_equal(test, z['window_index']):
                    raise ValueError('Window index alignment error ' + str(f))
                row += get_report(err, test, d, 'primary_courier_disjoint', model, sc, rep)
                nonrows += get_report(err[no], test[no], d, 'primary_nonoverlap', model, sc, rep)
    loco = out / 'STRICT_LOCO_RESULTS.csv'
    tem = out / 'STRICT_TEMPORAL_RESULTS.csv'
    for f in (loco, tem):
        if f.exists() and f.stat().st_size > 256:
            v = pd.read_csv(f)
            if len(v):
                row.extend(v.to_dict('records'))
    for scope in [f'strict_loco_{city}' for city in CITIES] + ['strict_temporal']:
        f = out / 'tables' / f'{scope}_NONOVERLAP_RESULTS.csv'
        if f.exists() and f.stat().st_size > 256:
            v = pd.read_csv(f)
            if len(v):
                nonrows.extend(v.to_dict('records'))
    history = out / 'tables/HISTORY_ONLY_FULL_RETRAIN_RESULTS.csv'
    if history.exists() and history.stat().st_size > 256:
        v = pd.read_csv(history)
        if len(v):
            nonrows.extend(v.to_dict('records'))
    savecsv(row, out / 'AGGREGATION_RESULTS.csv')
    savecsv(nonrows, out / 'NON_OVERLAP_AND_HISTORY_ONLY_SENSITIVITY.csv')

def paired_bootstrap(root, out):
    d = load(root)
    split = courier_split(root, d)
    test = np.where(split == 'test')[0]
    courier = d['courier_code'][test]
    cs, inv = np.unique(courier, return_inverse=True)
    nc = len(cs)
    counts = np.bincount(inv, minlength=nc)
    rng = np.random.default_rng(20260811)
    draws = rng.integers(0, nc, size=(2000, nc))
    weights = np.empty((2000, nc), np.int32)
    for j in range(2000):
        weights[j] = np.bincount(draws[j], minlength=nc)
    denom = weights @ counts
    rows = []
    for sc in SCENARIOS:
        errors = {m: get_errors(out, m, sc) for m in MODELS}
        for a, b in COMPARISONS:
            dlt = errors[a] - errors[b]
            horizons = [dlt[:, 0], dlt[:, 1], dlt[:, 2], dlt.mean(axis=1), dlt[:, -1]]
            for hn, vec in zip(HORIZONS, horizons):
                numerator = np.bincount(inv, weights=vec, minlength=nc)
                bootstrap = weights @ numerator / denom
                low, high = np.quantile(bootstrap, [0.025, 0.975])
                point = np.mean(vec)
                sign = 'positive' if low > 0 else 'negative' if high < 0 else 'not_significant'
                rows.append(dict(protocol='primary_courier_disjoint', scenario=sc, horizon=hn, reference=a, candidate=b, n_courier=nc, n_windows=len(test), positive_implies=f'{b} has lower error', delta_reference_minus_candidate=float(point), ci_low=float(low), ci_high=float(high), direction=sign, resamples=2000, unit='coordinate_distance_units', mask_replicates_averaged=30 if sc.startswith('random_') else 1))
    savecsv(rows, out / 'tables/PAIRED_BASELINE_BOOTSTRAP_2000.csv')
    print('BOOTSTRAP', pd.DataFrame(rows).direction.value_counts().to_dict(), flush=True)
