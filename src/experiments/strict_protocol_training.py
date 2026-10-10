"""Model fitting and evaluation for strict LOCO and chronological splits.

Fixed seeds and four-scenario validation selection are required before test
inference; trained weights and preprocessed inputs must be provided separately.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from protocols_and_filters import (CITIES, SEEDS, SCENARIOS, sha, savecsv,
    load, courier_split, selection_splits, nonoverlap, get_report)
NAMES = ('TCN_robust', 'Transformer_robust')

def load_can(root):
    p = Path(root) / 'src/trajectory_forecasting.py'
    if not p.is_file():
        raise FileNotFoundError('Provide the model implementation at src/trajectory_forecasting.py')
    spec = importlib.util.spec_from_file_location('trajectory_forecasting_core', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m

def run_train(module, root, out, scope, split, device):
    ckdir = out / 'derived/training_runs'
    ckdir.mkdir(parents=True, exist_ok=True)
    data = module.load_cache(root / 'data/windows_cache.npz')
    rows = []
    histories = []
    for model in NAMES:
        for seed in SEEDS:
            tag = f'{scope}__{model}__seed{seed}'
            folder = ckdir / tag
            folder.mkdir(exist_ok=True)
            ck = folder / 'checkpoints' / f'{tag}.pt'
            histfile = folder / 'training_history.csv'
            valfile = folder / 'validation_composite.json'
            if not (ck.exists() and histfile.exists() and valfile.exists()):
                module.set_seed(seed)
                logger = module.Logger(folder / 'training.log')
                model_trained, hist, actual = module.train_model(model, data, split['train'], split['val'], device, folder, epochs=15, batch_size=512, training_mode='robust', tag=tag, log=logger)
                hist.to_csv(histfile, index=False)
                logger.close()
                assert actual == ck
                vals = module.validation_composite_ade(model_trained, data, split['val'], device, 512)
                valfile.write_text(json.dumps(dict(composite_ADE=vals[0], scenario_ADE={sc: float(vals[1][i]) for i, sc in enumerate(module.ROBUST_SELECTION_SCENARIOS)}, scenarios=list(module.ROBUST_SELECTION_SCENARIOS), validated_on='training_city_only_validation', heldout_test_not_evaluated=True), indent=2), encoding='utf8')
                del model_trained
            vs = json.loads(valfile.read_text(encoding='utf8'))
            chk = torch.load(ck, map_location='cpu', weights_only=True)
            assert chk['model_name'] == model
            rows.append(dict(scope=scope, model=model, seed=seed, n_train=len(split['train']), n_val=len(split['val']), train_cities='|'.join((CITIES[c] for c in np.unique(data.city_code[split['train']]))), val_cities='|'.join((CITIES[c] for c in np.unique(data.city_code[split['val']]))), n_train_couriers=len(np.unique(data.courier_code[split['train']])), n_val_couriers=len(np.unique(data.courier_code[split['val']])), validation_composite_ADE=vs['composite_ADE'], checkpoint_sha256=sha(ck), checkpoint_relative_path=ck.relative_to(out).as_posix(), best_epoch=chk['best_epoch'], best_val_clean_loss=chk['best_val_loss']))
            print('TRAINED_OR_REUSED', tag, 'validation_composite_ADE', vs['composite_ADE'], flush=True)
    return rows

def report_results(m, root, out, scope, split, select, device):
    data = m.load_cache(root / 'data/windows_cache.npz')
    test = split['test']
    nidx = nonoverlap(data.__dict__, test)
    rows = []
    nonrows = []
    inventory = []
    for name in NAMES:
        for seed in SEEDS:
            tag = f'{scope}__{name}__seed{seed}'
            ck = out / 'derived/training_runs' / tag / 'checkpoints' / f'{tag}.pt'
            model = m.make_model(name).to(device)
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True)['state_dict'])
            model.eval()
            for sc in SCENARIOS:
                for rep in range(30) if sc.startswith('random_') else [0]:
                    fname = out / 'derived' / f'ERR_{scope}__{name}__seed{seed}__{sc}__rep{rep:02d}.npz'
                    if fname.exists():
                        cached = np.load(fname)
                        e = cached['error']
                        ii = cached['window_index']
                        assert e.shape == (len(test), 3) and np.array_equal(ii, test), ('INVALID_RESUME_CACHE', str(fname))
                    else:
                        e, ii = m.predict_learned(model, data, test, sc, device, batch_size=512, rep=rep)
                        assert np.array_equal(ii, test)
                        np.savez_compressed(fname, error=e.astype('float32'), window_index=ii.astype('int64'))
                    inventory.append(dict(scope=scope, model=name, seed=seed, scenario=sc, rep=rep, n=len(ii), path=fname.relative_to(out).as_posix(), sha256=sha(fname)))
                    r = get_report(e, ii, data.__dict__, scope, name, sc, rep)
                    for a in r:
                        a.update(seed=seed, selected_architecture=select['selection'], selected_flag=name == select['selection'], selection_basis='training_city_only_val_composite_mean_five_seeds')
                    rows.extend(r)
                    no = np.isin(ii, nidx)
                    v = get_report(e[no], ii[no], data.__dict__, scope + '_NON_OVERLAP', name, sc, rep)
                    for a in v:
                        a.update(seed=seed, selected_architecture=select['selection'], selected_flag=name == select['selection'])
                    nonrows.extend(v)
                    print('TEST', scope, name, seed, sc, rep, 'ADE', round(float(e.mean()), 3), flush=True)
            del model
    savecsv(inventory, out / 'tables' / f'{scope}_HEAVY_PREDICTION_MANIFEST.csv')
    return (rows, nonrows)

def run_strict_protocols(root, out, mode='all', train_only=False):
    """Train fixed strict folds, select on validation, then evaluate on test.

    Training uses the original augmentation RNG. Once training completes,
    validation and prediction use the fixed SHA-256 test-mask generator.
    Checkpoint and result files are local caller-owned inputs/outputs.
    """
    import hashlib
    import os
    from sha256_missingness_masks import NAMESPACE
    from validation_and_evaluation import bind_formal_inference_masks

    if mode not in ('loco', 'temporal', 'all'):
        raise ValueError('mode must be loco, temporal, or all')
    root, out = Path(root), Path(out)
    for folder in ('tables', 'derived', 'metadata'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Never patch the training module: masking during training is a different RNG.
    train_module = load_can(root)
    test_module = bind_formal_inference_masks(load_can(root))
    data = test_module.load_cache(root / 'data/windows_cache.npz')
    d = load(root)
    primary_split = courier_split(root, d)
    _, temporal, loco, _ = selection_splits(d, primary_split)
    all_folds = [('strict_loco_' + city, fold) for city, fold in loco.items()]
    if mode in ('temporal', 'all'):
        all_folds.append(('strict_temporal', temporal))
    if mode == 'temporal':
        all_folds = [('strict_temporal', temporal)]

    for scope, split in all_folds:
        train_rows = run_train(train_module, root, out, scope, split, device)
        if train_only:
            continue
        # Evaluate all five fixed seeds for each architecture on validation only.
        validation_rows = []
        for name in NAMES:
            for seed in SEEDS:
                tag = f'{scope}__{name}__seed{seed}'
                ckpt = out / 'derived/training_runs' / tag / 'checkpoints' / f'{tag}.pt'
                net = test_module.make_model(name).to(device)
                state = torch.load(ckpt, map_location=device, weights_only=True)
                net.load_state_dict(state['state_dict'])
                net.eval()
                value, scen = test_module.validation_composite_ade(
                    net, data, split['val'], device, 512)
                validation_rows.append(dict(scope=scope, model=name, seed=seed,
                    validation_composite_ADE=float(value),
                    checkpoint_sha256=sha(ckpt),
                    mask_namespace=NAMESPACE,
                    **{'ADE_' + k: float(scen[i]) for i, k in enumerate(
                        test_module.ROBUST_SELECTION_SCENARIOS)}))
                del net
        savecsv(validation_rows, out / 'tables' /
                f'{scope}_VAL_ONLY_ARCHITECTURE_SELECTION.csv')
        mean_ade = {name: float(np.mean([r['validation_composite_ADE']
            for r in validation_rows if r['model'] == name])) for name in NAMES}
        chosen = min(NAMES, key=lambda name: (mean_ade[name], name))
        selection = dict(scope=scope, selection=chosen, candidate_scores=mean_ade,
            selection_basis='mean_of_five_fixed_seed_validation_composite_ADE',
            selection_before_test=True, evaluation_mask_namespace=NAMESPACE,
            train_couriers_sha256=hashlib.sha256(np.unique(
                d['courier_code'][split['train']]).tobytes()).hexdigest(),
            val_couriers_sha256=hashlib.sha256(np.unique(
                d['courier_code'][split['val']]).tobytes()).hexdigest(),
            seed_registry=list(SEEDS))

        metadata_path = out / 'metadata' / f'{scope}_VALIDATION_SELECTION.json'
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding='utf-8'))
            if existing != selection:
                raise ValueError('Validation selection has changed; use a new output directory')
        else:
            metadata_path.write_text(json.dumps(selection, indent=2), encoding='utf-8')

        # Never silently import a result cache whose mask provenance is unknown.
        marker = out / 'metadata' / f'{scope}_EVALUATION_MASK.json'
        predictions = list((out / 'derived').glob(f'ERR_{scope}__*.npz'))
        if predictions and not marker.is_file():
            raise ValueError('Result cache without evaluation-mask metadata')
        if marker.is_file():
            previous = json.loads(marker.read_text(encoding='utf-8'))
            if previous.get('namespace') != NAMESPACE:
                raise ValueError('Result cache uses a different random-mask protocol')
        else:
            marker.write_text(json.dumps({'namespace': NAMESPACE}, indent=2), encoding='utf-8')

        results, nonoverlap_rows = report_results(
            test_module, root, out, scope, split, selection, device)
        savecsv(results, out / 'tables' / f'{scope}_RESULTS.csv')
        savecsv(nonoverlap_rows, out / 'tables' / f'{scope}_NONOVERLAP_RESULTS.csv')

    loco_outputs = [out / 'tables' / f'strict_loco_{city}_RESULTS.csv'
                    for city in CITIES]
    if all(p.exists() for p in loco_outputs):
        savecsv(pd.concat([pd.read_csv(p) for p in loco_outputs],
            ignore_index=True), out / 'STRICT_LOCO_RESULTS.csv')
    if (out / 'tables/strict_temporal_RESULTS.csv').is_file():
        savecsv(pd.read_csv(out / 'tables/strict_temporal_RESULTS.csv'),
            out / 'STRICT_TEMPORAL_RESULTS.csv')
