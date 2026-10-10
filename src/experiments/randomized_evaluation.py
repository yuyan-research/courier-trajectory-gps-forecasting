"""Deterministic missingness evaluation and courier-cluster uncertainty estimation.

Callable functions from the experiment implementation. Input caches and trained
checkpoints are supplied separately; no research data or weights are included.
The reproducibility identifiers embedded in the mask/bootstrap namespaces
are fixed components of the published randomization protocol.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch

N_BOOT = 2000
TRAINING_SEEDS = [20260811, 20260812, 20260813, 20260814, 20260815]

ALL_SCENARIOS = [
    "clean",
    "sample_40s",
    "sample_60s",
    "sample_120s",
    "sample_300s",
    "random_10",
    "random_30",
    "random_50",
    "burst_1m",
    "burst_3m",
    "burst_5m",
]

RANDOM_SCENARIOS = ["random_10", "random_30", "random_50"]

# Prespecified scenarios for the courier-level bootstrap.
BOOTSTRAP_KEY_SCENARIOS = [
    "clean",
    "sample_60s",
    "random_30",
    "burst_3m",
]

MODELS = [
    "LastPosition",
    "ConstantVelocity",
    "KalmanCV",
    "GRU_clean",
    "GRU_robust",
    "TCN_robust",
    "Transformer_robust",
]

LEARNED_MODELS = [
    "GRU_clean",
    "GRU_robust",
    "TCN_robust",
    "Transformer_robust",
]

CHECKPOINT_NAMES = {
    "GRU_clean": "primary_GRU_clean.pt",
    "GRU_robust": "primary_GRU_robust.pt",
    "TCN_robust": "primary_TCN_robust.pt",
    "Transformer_robust": "primary_Transformer_robust.pt",
}

HORIZONS = ["1m", "2m", "5m"]
HORIZON_SEC = np.array([60.0, 120.0, 300.0], dtype=np.float64)

MASK_NAMESPACE = "MEAS-MR-WP02|mask-v1"
BOOT_NAMESPACE = "MEAS-MR-WP02|courier-bootstrap-v1|2000"

def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def find_unique(root: Path, filename: str):
    hits = [p for p in root.rglob(filename) if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one {filename}; found {len(hits)}: {hits}"
        )
    return hits[0]


def import_canonical(source_path: Path):
    spec = importlib.util.spec_from_file_location(
        "trajectory_forecasting_core",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the model implementation.")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def compare_split_summaries(canonical_path: Path, derived_path: Path):
    a = pd.read_csv(canonical_path).fillna("")
    b = pd.read_csv(derived_path).fillna("")

    cols = sorted(set(a.columns) | set(b.columns))
    a = a.reindex(columns=cols).astype(str)
    b = b.reindex(columns=cols).astype(str)

    sort_cols = [c for c in ["protocol", "split"] if c in cols]
    if sort_cols:
        a = a.sort_values(sort_cols).reset_index(drop=True)
        b = b.sort_values(sort_cols).reset_index(drop=True)

    if a.shape != b.shape or not a.equals(b):
        raise RuntimeError(
            "Reconstructed split summary does not match the reference "
            "03_split_summary.csv."
        )


def root_seed_record(scenario, rep):
    txt = f"{MASK_NAMESPACE}|scenario={scenario}|rep={rep}"
    digest = hashlib.sha256(txt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex()


def window_seed(scenario, rep, canonical_window_index):
    txt = (
        f"{MASK_NAMESPACE}|scenario={scenario}|rep={rep}"
        f"|window_index={int(canonical_window_index)}"
    )
    digest = hashlib.sha256(txt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def make_seed_registry(test_n):
    rows = []
    idx = 0
    for sc in RANDOM_SCENARIOS:
        for rep in range(30):
            seed, digest = root_seed_record(sc, rep)
            rows.append(
                {
                    "index": idx,
                    "scenario": sc,
                    "replicate": rep,
                    "root_seed_u64": seed,
                    "root_sha256": digest,
                    "window_seed_rule":
                        'SHA256("MEAS-MR-WP02|mask-v1|scenario=<scenario>'
                        '|rep=<rep>|window_index=<canonical_window_index>") '
                        "first 8 bytes big-endian u64",
                    "n_test_windows": test_n,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def mask_matrix(test_idx, scenario, rep, n_points=31):
    n = len(test_idx)
    m = np.ones((n, n_points), dtype=bool)

    if scenario == "clean":
        return m

    if scenario.startswith("sample_"):
        sec = int(scenario.split("_")[1].replace("s", ""))
        k = max(1, int(round(sec / 20)))
        m[:] = False
        m[:, ::k] = True
        m[:, 0] = True
        return m

    if scenario.startswith("burst_"):
        mins = int(scenario.split("_")[1].replace("m", ""))
        n_drop = min(n_points - 1, int(round(mins * 60 / 20)))
        if n_drop:
            m[:, -n_drop:] = False
        m[:, 0] = True
        return m

    if scenario.startswith("random_"):
        rate = int(scenario.split("_")[1]) / 100.0
        for j, canonical_i in enumerate(test_idx):
            s = window_seed(scenario, rep, canonical_i)
            rng = np.random.Generator(np.random.PCG64(s))
            m[j] = rng.random(n_points) >= rate
            m[j, 0] = True
        return m

    raise ValueError(scenario)


def get_mask(mask_dir: Path, test_idx, scenario, rep):
    p = mask_dir / f"{scenario}__rep{rep:02d}.npz"

    if p.exists():
        z = np.load(p)
        packed = z["packed"]
        n_points = int(np.asarray(z["n_points"]).reshape(-1)[0])
        return np.unpackbits(
            packed,
            axis=1,
            count=n_points,
        ).astype(bool)

    m = mask_matrix(test_idx, scenario, rep)
    packed = np.packbits(m, axis=1)

    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        packed=packed,
        n_points=np.array([m.shape[1]], dtype=np.int16),
    )
    os.replace(tmp, p)
    return m


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device, weights_only=True)


def load_learned_models(mod, extract_root, device):
    models = {}
    checkpoint_hash = {}

    for name in LEARNED_MODELS:
        p = find_unique(extract_root, CHECKPOINT_NAMES[name])
        state = torch_load(p, device)

        model = mod.make_model(name).to(device)
        model.load_state_dict(state["state_dict"])
        model.eval()

        models[name] = model
        checkpoint_hash[name] = sha256_file(p)

    return models, checkpoint_hash


def learned_prediction(mod, model, data, test_idx, mask, device, batch_size):
    n = len(test_idx)
    pred_abs = np.empty((n, 3, 2), dtype=np.float32)

    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)

            idx = test_idx[s:e]
            x = data.x_hist[idx].astype(np.float64)
            y = data.y_hist[idx].astype(np.float64)
            t = data.t_hist[idx].astype(np.float64)
            mm = mask[s:e].copy()
            mm[:, 0] = True

            x0 = x[:, 0].copy()
            y0 = y[:, 0].copy()

            xf = np.empty_like(x)
            yf = np.empty_like(y)
            dt_since = np.zeros_like(t)

            last_x = x[:, 0].copy()
            last_y = y[:, 0].copy()
            last_t = t[:, 0].copy()

            for k in range(x.shape[1]):
                obs = mm[:, k]
                last_x[obs] = x[obs, k]
                last_y[obs] = y[obs, k]
                last_t[obs] = t[obs, k]

                xf[:, k] = last_x
                yf[:, k] = last_y
                dt_since[:, k] = np.maximum(0.0, t[:, k] - last_t)

            feats = np.stack(
                [
                    (xf - x0[:, None]) / 1000.0,
                    (yf - y0[:, None]) / 1000.0,
                    mm.astype(np.float64),
                    dt_since / 300.0,
                    t / 600.0,
                ],
                axis=2,
            ).astype(np.float32)

            xb = torch.from_numpy(feats).to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=use_cuda,
            ):
                p = model(xb)

            p = (
                p.detach()
                .cpu()
                .numpy()
                .reshape(-1, 3, 2)
                .astype(np.float32)
            )

            p[:, :, 0] = p[:, :, 0] * 1000.0 + x0[:, None]
            p[:, :, 1] = p[:, :, 1] * 1000.0 + y0[:, None]

            pred_abs[s:e] = p

    return pred_abs


def baseline_prediction(mod, model_name, data, test_idx, mask):
    n = len(test_idx)
    pred = np.empty((n, 3, 2), dtype=np.float32)

    if model_name == "LastPosition":
        ar = np.arange(mask.shape[1])[None, :]
        last = np.where(mask, ar, -1).max(axis=1)

        x = data.x_hist[test_idx]
        y = data.y_hist[test_idx]

        px = x[np.arange(n), last]
        py = y[np.arange(n), last]

        pred[:, :, 0] = px[:, None]
        pred[:, :, 1] = py[:, None]
        return pred

    for j, canonical_i in enumerate(test_idx):
        mm = mask[j]
        t = data.t_hist[canonical_i].astype(np.float64)[mm]
        x = data.x_hist[canonical_i].astype(np.float64)[mm]
        y = data.y_hist[canonical_i].astype(np.float64)[mm]

        if model_name == "ConstantVelocity":
            vx, vy, xlast, ylast, tlast = mod.robust_velocity_fit(
                t, x, y
            )
            for hix, h in enumerate(HORIZON_SEC):
                dt = float(h - tlast)
                pred[j, hix, 0] = xlast + vx * dt
                pred[j, hix, 1] = ylast + vy * dt

        elif model_name == "KalmanCV":
            pp = mod.kalman_predict_sequence(
                t,
                x,
                y,
                HORIZON_SEC,
            )
            pred[j] = pp.astype(np.float32)

        else:
            raise ValueError(model_name)

    return pred


def should_store_prediction(scenario, rep):
    # Full errors are retained for all conditions/replicates.
    # Predictions are persisted for the predeclared key conditions;
    # random_30 prediction is retained for replicate 0.
    if scenario in ("clean", "sample_60s", "burst_3m"):
        return True
    if scenario == "random_30" and rep == 0:
        return True
    return False


def result_paths(result_dir, model, scenario, rep):
    stem = f"{model}___{scenario}___rep{rep:02d}"
    return (
        result_dir / f"{stem}.npz",
        result_dir / f"{stem}.json",
    )


def valid_existing(npz_path, meta_path):
    if not npz_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return sha256_file(npz_path) == meta["sha256"]
    except Exception:
        return False


def save_result(
    path,
    meta_path,
    model,
    scenario,
    rep,
    pred,
    errors,
    checkpoint_sha,
):
    keep_pred = should_store_prediction(scenario, rep)

    tmp = path.with_suffix(".tmp.npz")

    if keep_pred:
        np.savez_compressed(
            tmp,
            error_m=errors.astype(np.float32),
            prediction_xy=pred.astype(np.float32),
        )
    else:
        np.savez_compressed(
            tmp,
            error_m=errors.astype(np.float32),
        )

    os.replace(tmp, path)

    meta = {
        "model": model,
        "scenario": scenario,
        "replicate": rep,
        "n_windows": int(errors.shape[0]),
        "horizons": HORIZONS,
        "contains_prediction": keep_pred,
        "status":
            "future reconstructed / derived output",
        "checkpoint_sha256": checkpoint_sha or "",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "created_utc": utcnow(),
    }
    atomic_json(meta, meta_path)


def reconstruct_all(
    mod,
    data,
    test_idx,
    mask_dir,
    result_dir,
    learned_models,
    checkpoint_hash,
    device,
    batch_size,
):
    total = len(MODELS) * (
        8 + 3 * 30
    )
    done = 0

    for model_name in MODELS:
        for scenario in ALL_SCENARIOS:
            reps = range(30) if scenario in RANDOM_SCENARIOS else range(1)

            for rep in reps:
                done += 1
                npz_path, meta_path = result_paths(
                    result_dir,
                    model_name,
                    scenario,
                    rep,
                )

                if valid_existing(npz_path, meta_path):
                    print(
                        f"[RESUME {done}/{total}] "
                        f"{model_name} {scenario} rep={rep}",
                        flush=True,
                    )
                    continue

                print(
                    f"[RUN {done}/{total}] "
                    f"{model_name} {scenario} rep={rep}",
                    flush=True,
                )

                mask = get_mask(
                    mask_dir,
                    test_idx,
                    scenario,
                    rep,
                )

                if model_name in LEARNED_MODELS:
                    pred = learned_prediction(
                        mod,
                        learned_models[model_name],
                        data,
                        test_idx,
                        mask,
                        device,
                        batch_size,
                    )
                    ck_sha = checkpoint_hash[model_name]
                else:
                    pred = baseline_prediction(
                        mod,
                        model_name,
                        data,
                        test_idx,
                        mask,
                    )
                    ck_sha = ""

                target = data.targets[test_idx].astype(np.float64)
                err = np.sqrt(
                    np.sum(
                        (
                            pred.astype(np.float64)
                            - target
                        ) ** 2,
                        axis=2,
                    )
                )

                save_result(
                    npz_path,
                    meta_path,
                    model_name,
                    scenario,
                    rep,
                    pred,
                    err,
                    ck_sha,
                )


def build_window_metadata(mod, data, test_idx, out):
    rows = []

    for rowpos, i in enumerate(test_idx):
        cc = int(data.courier_code[i])
        city_code = int(data.city_code[i])

        rows.append(
            {
                "result_row_position": rowpos,
                "canonical_window_index": int(i),
                "courier_code": cc,
                "courier_id": str(
                    data.courier_categories[cc]
                ),
                "city_code": city_code,
                "city": mod.CITIES[city_code],
                "date_key": int(data.date_key[i]),
                "cutoff_sec_of_day":
                    int(data.cutoff_sec_of_day[i]),
            }
        )

    atomic_csv(
        pd.DataFrame(rows),
        out / "TEST_WINDOW_METADATA.csv",
    )


def scan_result_manifest(result_dir: Path):
    rows = []

    for meta_path in sorted(result_dir.glob("*.json")):
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        npz_path = meta_path.with_suffix(".npz")

        if not npz_path.exists():
            raise RuntimeError(
                f"Missing result file for {meta_path}"
            )

        actual = sha256_file(npz_path)

        if actual != meta["sha256"]:
            raise RuntimeError(
                f"Result hash mismatch: {npz_path}"
            )

        rows.append(
            {
                **meta,
                "relative_result_file":
                    f"derived/window_outputs/{npz_path.name}",
                "row_alignment":
                    "TEST_WINDOW_METADATA.csv",
            }
        )

    return pd.DataFrame(rows)


def summarize_horizons(manifest, result_dir, out):
    rows = []

    for _, r in manifest.iterrows():
        z = np.load(result_dir / Path(
            r["relative_result_file"]
        ).name)

        e = z["error_m"].astype(np.float64)

        for hix, h in enumerate(HORIZONS):
            v = e[:, hix]
            rows.append(
                {
                    "model": r["model"],
                    "scenario": r["scenario"],
                    "replicate": int(r["replicate"]),
                    "horizon": h,
                    "n_windows": len(v),
                    "mean_error_m": float(np.mean(v)),
                    "median_error_m": float(np.median(v)),
                    "p90_error_m": float(
                        np.quantile(v, 0.90)
                    ),
                }
            )

    df = pd.DataFrame(rows)

    atomic_csv(
        df,
        out / "HORIZON_1_2_5_MIN_RESULTS.csv",
    )

    sto = df[df["scenario"].isin(RANDOM_SCENARIOS)].copy()
    atomic_csv(
        sto,
        out / "STOCHASTIC_REPLICATE_RESULTS.csv",
    )

    return df


def load_errors(result_dir, model, scenario, rep):
    p, _ = result_paths(
        result_dir,
        model,
        scenario,
        rep,
    )
    return np.load(p)["error_m"].astype(np.float64)


def bootstrap_condition_errors(result_dir):
    cond = {}

    for model in MODELS:
        for scenario in BOOTSTRAP_KEY_SCENARIOS:

            if scenario.startswith("random_"):
                arr = None
                for rep in range(30):
                    e = load_errors(
                        result_dir,
                        model,
                        scenario,
                        rep,
                    )
                    if arr is None:
                        arr = np.zeros_like(
                            e,
                            dtype=np.float64,
                        )
                    arr += e
                arr /= 30.0
            else:
                arr = load_errors(
                    result_dir,
                    model,
                    scenario,
                    0,
                )

            cond[(model, scenario)] = arr

    return cond


def full_stats(e):
    mean = np.mean(e, axis=0)
    q = np.quantile(e, [0.50, 0.90], axis=0)

    # horizon × [mean, median, p90]
    return np.stack(
        [
            mean,
            q[0],
            q[1],
        ],
        axis=1,
    )


def courier_cluster_bootstrap(
    data,
    test_idx,
    result_dir,
    out,
):
    cond = bootstrap_condition_errors(result_dir)
    keys = list(cond.keys())

    test_cc = data.courier_code[test_idx].astype(np.int64)
    unique_cc = np.unique(test_cc)

    seed_digest = hashlib.sha256(
        BOOT_NAMESPACE.encode("utf-8")
    ).digest()

    boot_seed = int.from_bytes(
        seed_digest[:8],
        "big",
    )

    rng = np.random.default_rng(boot_seed)

    draws = rng.integers(
        0,
        len(unique_cc),
        size=(N_BOOT, len(unique_cc)),
        dtype=np.int32,
    )

    np.savez_compressed(
        out / "BOOTSTRAP_DRAW_REGISTRY.npz",
        unique_test_courier_codes=unique_cc,
        sampled_courier_index=draws,
        seed_u64=np.array(
            [boot_seed],
            dtype=np.uint64,
        ),
    )

    atomic_json(
        {
            "namespace": BOOT_NAMESPACE,
            "seed_u64": boot_seed,
            "sha256": seed_digest.hex(),
            "draws": N_BOOT,
            "cluster": "courier",
            "sampling":
                "unique test couriers sampled with replacement; "
                "all windows of each sampled courier retained; "
                "multiplicity preserved",
            "same_draws_for_all_models_conditions_and_contrasts":
                True,
        },
        out / "BOOTSTRAP_DRAW_REGISTRY.json",
    )

    # Row positions in test-result arrays.
    positions = {
        int(c): np.where(test_cc == c)[0]
        for c in unique_cc
    }

    n_cond = len(keys)

    progress_path = out / "BOOTSTRAP_PROGRESS.npz"

    if progress_path.exists():
        z = np.load(progress_path, allow_pickle=True)
        old_keys = [
            tuple(x)
            for x in z["keys"].tolist()
        ]
        if old_keys != keys:
            raise RuntimeError(
                "Bootstrap progress key mismatch."
            )
        vals = z["vals"]
    else:
        vals = np.full(
            (
                n_cond,
                N_BOOT,
                3,  # horizons
                3,  # mean median p90
            ),
            np.nan,
            dtype=np.float64,
        )

    completed = np.where(
        np.all(
            np.isfinite(vals),
            axis=(0, 2, 3),
        )
    )[0]

    start_draw = (
        int(completed[-1]) + 1
        if len(completed)
        else 0
    )

    for d in range(start_draw, N_BOOT):
        sampled_codes = unique_cc[draws[d]]

        u, cnt = np.unique(
            sampled_codes,
            return_counts=True,
        )

        chunks = []
        for c, mult in zip(u, cnt):
            p = positions[int(c)]
            for _ in range(int(mult)):
                chunks.append(p)

        row_idx = np.concatenate(chunks)

        for ci, key in enumerate(keys):
            sample = cond[key][row_idx]
            vals[ci, d] = full_stats(sample)

        if (d + 1) % 25 == 0 or d + 1 == N_BOOT:
            tmp = progress_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                keys=np.array(
                    keys,
                    dtype=object,
                ),
                vals=vals,
            )
            os.replace(tmp, progress_path)

            print(
                f"[BOOTSTRAP] {d + 1}/{N_BOOT}",
                flush=True,
            )

    stat_names = ["mean", "median", "p90"]
    rows = []

    point = {
        k: full_stats(v)
        for k, v in cond.items()
    }

    for ci, (model, scenario) in enumerate(keys):
        for hix, horizon in enumerate(HORIZONS):
            for six, stat in enumerate(stat_names):
                dist = vals[ci, :, hix, six]

                lo, hi = np.quantile(
                    dist,
                    [0.025, 0.975],
                )

                rows.append(
                    {
                        "result_type": "level",
                        "model_or_contrast": model,
                        "scenario": scenario,
                        "horizon": horizon,
                        "statistic": stat,
                        "point_estimate_m":
                            float(point[(model, scenario)][hix, six]),
                        "ci_lower_m": float(lo),
                        "ci_upper_m": float(hi),
                        "bootstrap_draws": N_BOOT,
                        "cluster": "courier",
                        "ci_method":
                            "percentile 95%",
                        "paired_same_draws": True,
                        "stochastic_rep_handling":
                            (
                                "per-window mean across all 30 "
                                "locked SHA-256 replicates"
                                if scenario.startswith("random_")
                                else "not applicable"
                            ),
                    }
                )

    contrasts = [
        (
            "GRU_robust - GRU_clean",
            "GRU_robust",
            "GRU_clean",
        ),
        (
            "Transformer_robust - TCN_robust",
            "Transformer_robust",
            "TCN_robust",
        ),
    ]

    key_index = {
        k: i
        for i, k in enumerate(keys)
    }

    for label, a, b in contrasts:
        for scenario in BOOTSTRAP_KEY_SCENARIOS:
            ia = key_index[(a, scenario)]
            ib = key_index[(b, scenario)]

            diff_dist = vals[ia] - vals[ib]
            diff_point = (
                point[(a, scenario)]
                - point[(b, scenario)]
            )

            for hix, horizon in enumerate(HORIZONS):
                for six, stat in enumerate(stat_names):
                    dist = diff_dist[:, hix, six]
                    lo, hi = np.quantile(
                        dist,
                        [0.025, 0.975],
                    )

                    rows.append(
                        {
                            "result_type": "paired_contrast",
                            "model_or_contrast": label,
                            "scenario": scenario,
                            "horizon": horizon,
                            "statistic": stat,
                            "point_estimate_m":
                                float(diff_point[hix, six]),
                            "ci_lower_m": float(lo),
                            "ci_upper_m": float(hi),
                            "bootstrap_draws": N_BOOT,
                            "cluster": "courier",
                            "ci_method":
                                "percentile 95%",
                            "paired_same_draws": True,
                            "stochastic_rep_handling":
                                (
                                    "per-window mean across all 30 "
                                    "locked SHA-256 replicates"
                                    if scenario.startswith("random_")
                                    else "not applicable"
                                ),
                            "contrast_sign":
                                "negative means first model has lower error",
                        }
                    )

    atomic_csv(
        pd.DataFrame(rows),
        out / "BOOTSTRAP_CI_RESULTS.csv",
    )
