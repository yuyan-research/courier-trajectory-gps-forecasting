"""Validation-only selection and test-time mask binding.

The parameter selection routines accept validation arrays. Full inference
requires the corresponding trained models and preprocessed observation windows.
"""
from __future__ import annotations
import numpy as np
from protocols_and_filters import GRID_Q, GRID_R, batch_filter
from sha256_missingness_masks import NAMESPACE, HISTORY_NAMESPACE, mask_matrix, patch_module_for_evaluation
VALIDATION_SCENARIOS = ('clean', 'sample_60s', 'random_30', 'burst_3m')

def select_filter_parameters(data, validation_indices, mask_namespace=NAMESPACE):
    """Select KF/UKF grid parameters using validation data only.

    The objective is the unweighted mean ADE across four fixed missingness
    scenarios. Ties are resolved by process noise and observation noise.
    The test split is not an argument and cannot be used for selection.
    """
    val = np.asarray(validation_indices, dtype=np.int64)
    rows = []
    selected = {}
    for model in ('KF_tuned', 'UKF_CTRV'):
        ukf = model.startswith('UKF')
        for q in GRID_Q:
            for noise in GRID_R:
                losses = []
                for sc in VALIDATION_SCENARIOS:
                    mask = mask_matrix(val, sc, 0, namespace=mask_namespace)
                    errors = batch_filter(data['x_hist'], data['y_hist'], data['t_hist'], data['targets'], val, mask, q, noise, ukf)
                    losses.append(float(np.mean(errors)))
                rows.append(dict(model=model, q=float(q), observation_noise_sd=float(noise), validation_ADE_mean=float(np.mean(losses)), val_clean=losses[0], val_sample_60s=losses[1], val_random_30_rep0=losses[2], val_burst_3m=losses[3], n_val=len(val)))
        candidates = [r for r in rows if r['model'] == model]
        best = min(candidates, key=lambda x: (x['validation_ADE_mean'], x['q'], x['observation_noise_sd']))
        selected[model] = {'q': best['q'], 'r': best['observation_noise_sd'], 'val_ADE': best['validation_ADE_mean']}
    return (rows, selected)

def select_strict_architecture(validation_scores):
    """Return the lower five-seed mean validation ADE architecture.

    validation_scores maps each model name to five validation-only ADEs.
    Held-out errors are not inputs to this function.
    """
    names = ('TCN_robust', 'Transformer_robust')
    if set(validation_scores) != set(names):
        raise ValueError('Both prespecified architectures are required')
    for name in names:
        if len(validation_scores[name]) != 5:
            raise ValueError('Exactly five validation seeds are required')
        if not np.isfinite(validation_scores[name]).all():
            raise ValueError('Validation scores must be finite')
    means = {name: float(np.mean(validation_scores[name])) for name in names}
    return (min(means, key=lambda name: (means[name], name)), means)

def formal_mask_for_inference(indices, scenario, replicate=0, namespace=NAMESPACE):
    """Generate the same mask for paired models in one evaluation replicate."""
    return mask_matrix(indices, scenario, replicate, namespace=namespace)

def bind_formal_inference_masks(model_module, history_only=False):
    """Bind evaluation masks without changing the model source on disk."""
    ns = HISTORY_NAMESPACE if history_only else NAMESPACE
    return patch_module_for_evaluation(model_module, ns)
