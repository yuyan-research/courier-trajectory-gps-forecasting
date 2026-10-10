"""History-only population construction and cohort-sensitivity analysis.

Raw input is loaded only from caller-provided trusted files. A SHA-256
input table is required for independent verification before unpickling.
"""
from __future__ import annotations
import csv
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from protocols_and_filters import CITIES, SEEDS, SCENARIOS, sha, savecsv, load, selection_splits, courier_split, get_report
from strict_protocol_training import load_can

def make_history_only_cache(root, out, raw_override=None, input_manifest=None):
    src = load_can(root)
    cfile = root / 'data/windows_cache.npz'
    original = src.load_cache(cfile)
    original_ids = np.array(original.courier_categories)
    if input_manifest is None:
        raise ValueError('Provide a SHA-256 manifest with filename, size_bytes and sha256')
    with Path(input_manifest).open(encoding='utf-8-sig', newline='') as stream:
        expected = {r['filename']: (r['size_bytes'], r['sha256']) for r in csv.DictReader(stream)}
    raw = Path(raw_override) if raw_override else root / 'data/raw'
    verified_inputs = []
    for fname in ['courier_detailed_trajectory_20s.pkl.xz', 'pickup_five_cities.pkl.xz', 'delivery_five_cities.pkl.xz']:
        f = raw / fname
        if not f.exists():
            raise FileNotFoundError(f'Required trajectory input file is missing: {f}')
        size, digest = expected[fname]
        observed_hash = sha(f)
        observed_size = f.stat().st_size
        if observed_size != int(size) or observed_hash != digest:
            raise RuntimeError('Input SHA-256 mismatch ' + str(f))
        verified_inputs.append(dict(filename=fname, size_bytes=observed_size, sha256=observed_hash))
    savecsv(verified_inputs, out / 'tables/INPUT_CHECKSUMS.csv')
    pu = pd.read_pickle(raw / 'pickup_five_cities.pkl.xz')
    de = pd.read_pickle(raw / 'delivery_five_cities.pkl.xz')
    id_city = src.build_city_map(pu, de)
    del pu, de
    gc.collect()
    tr = pd.read_pickle(raw / 'courier_detailed_trajectory_20s.pkl.xz')
    traj = pd.DataFrame({'courier_id': src.clean_id(tr['postman_id']), 'traj_time': src.parse_lade_time(tr['gps_time'], tr['ds']), 'x': pd.to_numeric(tr['lng'], errors='coerce').astype('float64'), 'y': pd.to_numeric(tr['lat'], errors='coerce').astype('float64')})
    del tr
    gc.collect()
    traj = traj.merge(id_city, on='courier_id', how='inner').dropna(subset=['courier_id', 'city', 'traj_time', 'x', 'y'])
    traj = traj.loc[np.isfinite(traj.x.to_numpy()) & np.isfinite(traj.y.to_numpy())].copy()
    traj['date_key'] = traj.traj_time.dt.month.astype(np.int16) * 100 + traj.traj_time.dt.day.astype(np.int16)
    traj['sec_of_day'] = traj.traj_time.dt.hour.astype(np.int32) * 3600 + traj.traj_time.dt.minute.astype(np.int32) * 60 + traj.traj_time.dt.second.astype(np.int32)
    traj['courier_id'] = traj['courier_id'].astype('category')
    courier_ids = traj['courier_id'].cat.categories.astype(str).to_numpy()
    if not np.array_equal(courier_ids, original_ids.astype(str)):
        raise RuntimeError('Raw courier index differs from reference window index; verify input dataset and mapping')
    traj['courier_code'] = traj['courier_id'].cat.codes.astype(np.int32)
    traj['city_code'] = traj['city'].map(src.CITY_CODE).astype(np.int8)
    cols = ['x_hist', 'y_hist', 't_hist', 'targets', 'target_dt', 'courier_code', 'city_code', 'date_key', 'cutoff_sec_of_day']
    arrs = {k: [] for k in cols}
    counts = {c: 0 for c in CITIES}
    attrition = {'hist_candidates': 0, 'hist_valid': 0, 'hist_sampled': 0, 'label_complete': 0, 'label_unavailable': 0, 'future_motion_filter_applied_in_reference': 0}
    for gi, ((cc, dk), g) in enumerate(traj.groupby(['courier_code', 'date_key'], sort=False, observed=True)):
        city_code = int(g['city_code'].iloc[0])
        city = CITIES[city_code]
        if counts[city] >= 24000:
            continue
        if len(g) < 31:
            continue
        g = g.sort_values('sec_of_day')
        tt = g.sec_of_day.to_numpy('int32')
        xx = g.x.to_numpy('float64')
        yy = g.y.to_numpy('float64')
        keep = np.r_[True, np.diff(tt) > 0]
        tt = tt[keep]
        xx = xx[keep]
        yy = yy[keep]
        if len(tt) < 31:
            continue
        dt = np.diff(tt).astype('float64')
        speed = src.planar_step_speed_kmh(xx[:-1], yy[:-1], xx[1:], yy[1:], dt)
        bad = (dt <= 0) | (dt > src.MAX_ORIGINAL_GAP_SEC) | ~np.isfinite(speed) | (speed > src.MAX_STEP_SPEED_KMH)
        csum = np.r_[0, np.cumsum(bad.astype('int32'))]
        candidates = np.arange(30, len(tt), dtype='int32')
        attrition['hist_candidates'] += len(candidates)
        if len(candidates) > 32:
            candidates = candidates[np.linspace(0, len(candidates) - 1, num=32, dtype=int)]
        valid = []
        for i in candidates:
            start = i - 30
            duration = int(tt[i] - tt[start])
            if duration < 540 or duration > 660 or csum[i] - csum[start] != 0:
                continue
            valid.append(int(i))
        attrition['hist_valid'] += len(valid)
        if len(valid) > 4:
            valid = [valid[k] for k in np.linspace(0, len(valid) - 1, num=4, dtype=int)]
        for i in valid:
            if counts[city] >= 24000:
                break
            attrition['hist_sampled'] += 1
            targets = []
            for h in src.TARGET_HORIZONS_SEC:
                j, e = src.nearest_target_index(tt, i, int(h))
                if j is None:
                    targets = []
                    break
                targets.append(int(j))
            if len(targets) != 3:
                attrition['label_unavailable'] += 1
                continue
            attrition['label_complete'] += 1
            base = float(tt[i])
            start = i - 30
            vals = [xx[start:i + 1].astype('float32'), yy[start:i + 1].astype('float32'), tt[start:i + 1].astype('float32') - base, np.column_stack([xx[targets], yy[targets]]).astype('float32'), tt[targets].astype('float32') - base, int(cc), city_code, int(dk), int(tt[i])]
            for col, v in zip(cols, vals):
                arrs[col].append(v)
            counts[city] += 1
        if gi % 5000 == 0:
            print('HISTORY_ONLY', gi, 'candidate', attrition['hist_sampled'], 'labeled', attrition['label_complete'], flush=True)
        if all((v >= 24000 for v in counts.values())):
            break
    del traj
    gc.collect()
    dat = {k: np.asarray(v, dtype='float32' if k in ['x_hist', 'y_hist', 't_hist', 'targets', 'target_dt'] else 'int8' if k == 'city_code' else 'int16' if k == 'date_key' else 'int32') for k, v in arrs.items()}
    if len(dat['x_hist']) == 0:
        raise RuntimeError('No history-only labelled windows; cannot proceed')
    dat['courier_categories'] = original_ids
    dest = out / 'derived/HISTORY_ONLY_CACHE.npz'
    np.savez_compressed(dest, **dat)
    summary = [dict(metric=k, value=v) for k, v in attrition.items()] + [dict(metric=f'windows_{city}', value=n) for city, n in counts.items()]
    savecsv(summary, out / 'tables/HISTORY_ONLY_POPULATION_FLOW.csv')
    print('HISTORY_ONLY_SAVED', len(dat['x_hist']), dest, sha(dest), flush=True)
    return dest

def run_history_only_experiment(root, out, raw_dir=None, input_manifest=None, raw_only=False):
    root = Path(root)
    out = Path(out)
    for dirname in ('tables', 'derived'):
        (out / dirname).mkdir(parents=True, exist_ok=True)
    original = load(root)
    sp = courier_split(root, original)
    splitmap = {int(i): sp[np.where(original['courier_code'] == i)[0][0]] for i in np.unique(original['courier_code'])}
    f = out / 'derived/HISTORY_ONLY_CACHE.npz'
    if not f.exists():
        f = make_history_only_cache(root, out, raw_dir, input_manifest)
    src = load_can(root)
    from validation_and_evaluation import bind_formal_inference_masks
    inference_src = bind_formal_inference_masks(load_can(root), history_only=True)
    hd = src.load_cache(f)
    hdsp = np.array([splitmap.get(int(c), 'UNKNOWN') for c in hd.courier_code])
    savecsv([dict(split=k, windows=int(np.sum(hdsp == k)), couriers=int(len(np.unique(hd.courier_code[hdsp == k])))) for k in ['train', 'val', 'test', 'UNKNOWN']], out / 'tables/HISTORY_ONLY_COHORT_SPLIT_SUMMARY.csv')
    if raw_only:
        return
    base = {name: np.where(hdsp == name)[0] for name in ('train', 'val', 'test')}
    assert all((len(base[n]) > 0 for n in ('train', 'val', 'test')))
    for name in ('train', 'val', 'test'):
        for other in ('train', 'val', 'test'):
            if other != name:
                assert not set(hd.courier_code[base[name]]) & set(hd.courier_code[base[other]])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    vr = []
    for name in ('TCN_robust', 'Transformer_robust'):
        for seed in SEEDS:
            tag = f'history_only_{name}_{seed}'
            wd = out / 'derived/history_only_train' / tag
            wd.mkdir(parents=True, exist_ok=True)
            ck = wd / 'checkpoints' / f'{tag}.pt'
            hist = wd / 'training_history.csv'
            if not (ck.exists() and hist.exists()):
                src.set_seed(seed)
                log = src.Logger(wd / 'run.log')
                model, df, _ = src.train_model(name, hd, base['train'], base['val'], device, wd, 15, 512, 'robust', tag, log)
                df.to_csv(hist, index=False)
                log.close()
            else:
                model = src.make_model(name).to(device)
                model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)['state_dict'])
            score, metrics = inference_src.validation_composite_ade(model, hd, base['val'], device, 512)
            vr.append(dict(model=name, seed=seed, validation_composite_ADE=score, n_train=len(base['train']), n_val=len(base['val']), n_test=len(base['test']), ckpt_sha256=sha(ck)))
            for sc in SCENARIOS:
                for rep in range(30) if sc.startswith('random_') else [0]:
                    err, ids = inference_src.predict_learned(model, hd, base['test'], sc, device, 512, rep)
                    d = hd.__dict__
                    outrows = get_report(err, ids, d, 'history_only_full_retrain', name, sc, rep)
                    for x in outrows:
                        x['seed'] = seed
                    rows += outrows
                    np.savez_compressed(out / 'derived' / f'HISTORY_ERR_{name}_{seed}_{sc}_{rep}.npz', error=err.astype('float32'), window_index=ids)
    savecsv(rows, out / 'tables/HISTORY_ONLY_FULL_RETRAIN_RESULTS.csv')
    savecsv(vr, out / 'tables/HISTORY_ONLY_VALIDATION_ONLY_SELECTION.csv')
