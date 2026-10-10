import os
import numpy as np, pandas as pd, hashlib, json
from pathlib import Path
D = Path(os.environ['COURIER_GPS_WORK_DIR']).expanduser().resolve() / 'derived'
NWIN = 88048
SLOTS = 31
REPS = 30
VERSION = 'WP01_EMPIRICAL_RENEWAL_MARKOV_v1'
trans = pd.read_csv(D / 'empirical_spacing_transition_parameters.csv').query("group=='pooled_linked'").iloc[0]
jump = pd.read_csv(D / 'empirical_spacing_jump_distribution.csv').query("group=='pooled_linked'").copy()
counts = jump.set_index('slot_jump_k')['count']
pi_gap = float(counts[counts.index >= 2].sum() / counts.sum())
p01 = float(trans.p_gap_after_regular)
p11 = float(trans.p_gap_after_gap)
kgap = counts[counts.index >= 2].index.to_numpy(np.int16)
pgap = (counts.loc[kgap] / counts.loc[kgap].sum()).to_numpy(float)
out = np.lib.format.open_memmap(D / 'empirical_markov_masks_v1.npy', mode='w+', dtype=np.uint32, shape=(REPS, NWIN))
seed_rows = []
sum_rows = []
slots = np.arange(SLOTS, dtype=np.uint32)
for rep in range(REPS):
    label = f'{VERSION}|rep={rep:02d}|windows={NWIN}|slots={SLOTS}'
    digest = hashlib.sha256(label.encode()).digest()
    seed = int.from_bytes(digest[:8], 'little', signed=False)
    seed_rows.append({'replicate': rep, 'seed_label': label, 'sha256': hashlib.sha256(label.encode()).hexdigest(), 'pcg64_seed_uint64': seed})
    rng = np.random.Generator(np.random.PCG64(seed))
    bits = np.ones(NWIN, dtype=np.uint32)
    pos = np.zeros(NWIN, dtype=np.int16)
    active = np.ones(NWIN, bool)
    state = rng.random(NWIN) < pi_gap
    for _ in range(30):
        if not active.any():
            break
        k = np.ones(NWIN, dtype=np.int16)
        gm = active & state
        if gm.any():
            k[gm] = rng.choice(kgap, size=int(gm.sum()), p=pgap)
        newpos = pos + k
        valid = active & (newpos <= 30)
        vidx = np.flatnonzero(valid)
        if len(vidx):
            bits[vidx] |= np.uint32(1) << newpos[vidx].astype(np.uint32)
            pos[vidx] = newpos[vidx]
        active = valid & (pos < 30)
        if active.any():
            aidx = np.flatnonzero(active)
            probs = np.where(state[aidx], p11, p01)
            state[aidx] = rng.random(len(aidx)) < probs
    out[rep] = bits
    obs = (bits[:, None] >> slots[None, :] & 1).astype(np.uint8)
    retained = obs.sum(axis=1)
    last = (obs * slots[None, :]).max(axis=1)
    terminal = (30 - last) * 20
    cur = np.zeros(NWIN, np.int16)
    mx = np.zeros(NWIN, np.int16)
    for j in range(SLOTS):
        cur = np.where(obs[:, j] == 0, cur + 1, 0)
        mx = np.maximum(mx, cur)
    miss = SLOTS - retained
    sum_rows.append({'replicate': rep, 'n_masks': NWIN, 'mean_retained_points': retained.mean(), 'mean_missing_fraction': miss.mean() / SLOTS, 'median_missing_fraction': np.median(miss / SLOTS), 'p90_missing_fraction': np.quantile(miss / SLOTS, 0.9), 'mean_terminal_age_sec': terminal.mean(), 'p90_terminal_age_sec': np.quantile(terminal, 0.9), 'terminal_age_gt0_rate': (terminal > 0).mean(), 'terminal_age_ge60_rate': (terminal >= 60).mean(), 'mean_longest_missing_run_slots': mx.mean(), 'p90_longest_missing_run_slots': np.quantile(mx, 0.9), 'p99_longest_missing_run_slots': np.quantile(mx, 0.99)})
    print('rep', rep, 'mean miss', sum_rows[-1]['mean_missing_fraction'], 'term>0', sum_rows[-1]['terminal_age_gt0_rate'], flush=True)
out.flush()
pd.DataFrame(seed_rows).to_csv(D / 'empirical_markov_mask_seeds.csv', index=False)
pd.DataFrame(sum_rows).to_csv(D / 'empirical_markov_mask_summary_by_rep.csv', index=False)
params = {'version': VERSION, 'n_windows': NWIN, 'replicates': REPS, 'slots': SLOTS, 'nominal_step_sec': 20, 'history_sec': 600, 'pi_gap': pi_gap, 'p_gap_after_regular': p01, 'p_gap_after_gap': p11, 'gap_jump_k': kgap.tolist(), 'gap_jump_prob': pgap.tolist(), 'boundary_rule': 'raw inter-observation intervals >600 s excluded from calibration transitions and treated as active-segment boundaries', 'slot_mapping': 'k=max(1,min(30,floor(dt/20+0.5))) for dt<=600 s', 'cause_attribution': 'none; models observed spacing only'}
json.dump(params, open(D / 'empirical_markov_parameters.json', 'w'), indent=2)
print('overall')
print(pd.DataFrame(sum_rows).mean(numeric_only=True).to_dict())
