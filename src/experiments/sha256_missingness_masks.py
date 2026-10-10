"""SHA-256-derived observation masks used for formal stochastic evaluation.

The namespace strings are part of the reported deterministic seed protocol
and must not be changed: otherwise the generated masks would differ.
Training augmentation uses a separate RNG process.
"""
import hashlib
import numpy as np
NAMESPACE = 'MEAS-MR-WP02|mask-v1'
HISTORY_NAMESPACE = 'MEAS-MR-WP03|history-index-v1|mask-v1'
RANDOM = ('random_10', 'random_30', 'random_50')

def root_seed(scenario, rep, namespace=NAMESPACE):
    msg = f'{namespace}|scenario={scenario}|rep={int(rep)}'.encode('utf-8')
    digest = hashlib.sha256(msg).digest()
    return (int.from_bytes(digest[:8], 'big'), digest.hex())

def window_seed(scenario, rep, index, namespace=NAMESPACE):
    txt = f'{namespace}|scenario={scenario}|rep={int(rep)}|window_index={int(index)}'
    dig = hashlib.sha256(txt.encode('utf-8')).digest()
    return (int.from_bytes(dig[:8], 'big'), dig.hex())

def mask_matrix(indices, scenario, rep=0, n_points=31, namespace=NAMESPACE):
    indices = np.asarray(indices, dtype=np.int64)
    m = np.ones((len(indices), n_points), dtype=np.bool_)
    if scenario == 'clean':
        return m
    if scenario.startswith('sample_'):
        sec = int(scenario.split('_')[1].replace('s', ''))
        step = max(1, round(sec / 20))
        m[:] = False
        m[:, ::step] = True
        m[:, 0] = True
        return m
    if scenario.startswith('burst_'):
        mins = int(scenario.split('_')[1].replace('m', ''))
        n_drop = min(n_points - 1, round(mins * 60 / 20))
        if n_drop:
            m[:, -n_drop:] = False
        m[:, 0] = True
        return m
    if scenario in RANDOM:
        rate = int(scenario.split('_')[1]) / 100.0
        for j, i in enumerate(indices):
            seed, _ = window_seed(scenario, rep, int(i), namespace)
            m[j] = np.random.Generator(np.random.PCG64(seed)).random(n_points) >= rate
            m[j, 0] = True
        return m
    raise ValueError('Unapproved scenario ' + scenario)

def patch_module_for_evaluation(module, namespace=NAMESPACE):

    def scenario_mask(n_points, scenario, sample_index, rep=0):
        return mask_matrix([sample_index], scenario, rep, n_points, namespace)[0]
    module.scenario_mask = scenario_mask
    return module
