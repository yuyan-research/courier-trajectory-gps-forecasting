"""Fixed-seed training and paired architecture comparisons.

Five training seeds, matched-seed GRU comparisons and 30 stochastic-mask
replicates follow the reported experimental configuration. The functions
require separately obtained numerical windows and model checkpoints.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import randomized_evaluation as A

LOCKED_SEEDS = [
    20260811,
    20260812,
    20260813,
    20260814,
    20260815,
]

TRAIN_SPECS = [
    ("GRU_clean", "clean"),
    ("GRU_robust", "robust"),
    ("TCN_robust", "robust"),
    ("Transformer_robust", "robust"),
]

ALL_SCENARIOS = list(A.ALL_SCENARIOS)
RANDOM_SCENARIOS = list(A.RANDOM_SCENARIOS)
HORIZONS = list(A.HORIZONS)

METRICS = [
    "mean_error_m",
    "median_error_m",
    "p90_error_m",
]

def utcnow():
    return datetime.now(timezone.utc).isoformat()


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


def result_paths(result_dir, seed, model, scenario, rep):
    stem = (
        f"seed{seed}___{model}___"
        f"{scenario}___rep{rep:02d}"
    )
    return (
        result_dir / f"{stem}.npz",
        result_dir / f"{stem}.json",
    )


def valid_result(npz_path: Path, meta_path: Path):
    if not npz_path.exists() or not meta_path.exists():
        return False

    try:
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        return A.sha256_file(npz_path) == meta["sha256"]
    except Exception:
        return False


def load_checkpoint_model(mod, model_name, ckpt, device):
    state = A.torch_load(ckpt, device)

    model = mod.make_model(model_name).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    return model, state


def train_or_load(
    mod,
    model_name,
    training_mode,
    seed,
    data,
    train_idx,
    val_idx,
    device,
    out,
    hist_dir,
    epochs,
    batch_size,
    log,
):
    tag = f"seed_{seed}_{model_name}"

    ckpt = out / "checkpoints" / f"{tag}.pt"
    hist_path = hist_dir / f"{tag}_history.csv"
    complete_path = hist_dir / f"{tag}_COMPLETE.json"

    if (
        ckpt.exists()
        and hist_path.exists()
        and complete_path.exists()
    ):
        meta = json.loads(
            complete_path.read_text(encoding="utf-8")
        )

        if (
            meta.get("checkpoint_sha256")
            == A.sha256_file(ckpt)
        ):
            log.log(
                f"[RESUME TRAIN] seed={seed} "
                f"model={model_name}"
            )

            model, state = load_checkpoint_model(
                mod,
                model_name,
                ckpt,
                device,
            )

            hist = pd.read_csv(hist_path)

            return model, hist, ckpt, state

    log.log(
        f"[TRAIN] seed={seed} "
        f"model={model_name} mode={training_mode}"
    )

    # Critical pairing rule:
    # reset the SAME seed immediately before every model.
    # For GRU_clean and GRU_robust this gives identical
    # architecture initialization and initial shuffle RNG.
    mod.set_seed(seed)

    # Remove incomplete partial checkpoint if present.
    if ckpt.exists():
        ckpt.unlink()

    model, hist, produced_ckpt = mod.train_model(
        model_name=model_name,
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        device=device,
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        training_mode=training_mode,
        tag=tag,
        log=log,
    )

    if produced_ckpt != ckpt:
        raise RuntimeError(
            f"Unexpected checkpoint path: {produced_ckpt}"
        )

    hist = hist.copy()
    hist.insert(0, "training_seed", seed)

    atomic_csv(
        hist,
        hist_path,
    )

    state = A.torch_load(
        ckpt,
        device,
    )

    atomic_json(
        {
            "training_seed": seed,
            "model": model_name,
            "training_mode": training_mode,
            "epochs_cap": epochs,
            "batch_size": batch_size,
            "best_epoch": int(state["best_epoch"]),
            "best_val_loss":
                float(state["best_val_loss"]),
            "checkpoint_sha256":
                A.sha256_file(ckpt),
            "completed_utc": utcnow(),
        },
        complete_path,
    )

    return model, hist, ckpt, state


def save_eval_result(
    path,
    meta_path,
    seed,
    model_name,
    scenario,
    rep,
    error_m,
    checkpoint_sha,
):
    tmp = path.with_suffix(".tmp.npz")

    np.savez_compressed(
        tmp,
        error_m=error_m.astype(np.float32),
    )

    os.replace(tmp, path)

    meta = {
        "training_seed": seed,
        "model": model_name,
        "scenario": scenario,
        "replicate": rep,
        "n_windows": int(error_m.shape[0]),
        "horizons": HORIZONS,
        "checkpoint_sha256": checkpoint_sha,
        "mask_source":
            "SHA-256 deterministic evaluation-mask registry",
        "status":
            "fixed five-seed model comparison",
        "sha256": A.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "created_utc": utcnow(),
    }

    atomic_json(
        meta,
        meta_path,
    )


def evaluate_one_model(
    mod,
    model,
    model_name,
    seed,
    checkpoint_sha,
    data,
    test_idx,
    evaluation_mask_dir,
    device,
    batch_size,
    result_dir,
):
    target = data.targets[
        test_idx
    ].astype(np.float64)

    scenario_reps = (
        len(ALL_SCENARIOS)
        - len(RANDOM_SCENARIOS)
        + 30 * len(RANDOM_SCENARIOS)
    )

    counter = 0

    for scenario in ALL_SCENARIOS:

        reps = (
            range(30)
            if scenario in RANDOM_SCENARIOS
            else range(1)
        )

        for rep in reps:
            counter += 1

            npz_path, meta_path = result_paths(
                result_dir,
                seed,
                model_name,
                scenario,
                rep,
            )

            if valid_result(
                npz_path,
                meta_path,
            ):
                print(
                    f"[RESUME EVAL {counter}/{scenario_reps}] "
                    f"seed={seed} {model_name} "
                    f"{scenario} rep={rep}",
                    flush=True,
                )
                continue

            print(
                f"[EVAL {counter}/{scenario_reps}] "
                f"seed={seed} {model_name} "
                f"{scenario} rep={rep}",
                flush=True,
            )

            mask = A.get_mask(
                evaluation_mask_dir,
                test_idx,
                scenario,
                rep,
            )

            pred = A.learned_prediction(
                mod,
                model,
                data,
                test_idx,
                mask,
                device,
                batch_size,
            )

            err = np.sqrt(
                np.sum(
                    (
                        pred.astype(np.float64)
                        - target
                    ) ** 2,
                    axis=2,
                )
            )

            save_eval_result(
                npz_path,
                meta_path,
                seed,
                model_name,
                scenario,
                rep,
                err,
                checkpoint_sha,
            )


def load_error(
    result_dir,
    seed,
    model,
    scenario,
    rep,
):
    p, _ = result_paths(
        result_dir,
        seed,
        model,
        scenario,
        rep,
    )

    if not p.exists():
        raise RuntimeError(
            f"Missing evaluation result: {p}"
        )

    return np.load(
        p
    )["error_m"].astype(np.float64)


def aggregated_window_error(
    result_dir,
    seed,
    model,
    scenario,
):
    if scenario in RANDOM_SCENARIOS:
        acc = None

        for rep in range(30):
            e = load_error(
                result_dir,
                seed,
                model,
                scenario,
                rep,
            )

            if acc is None:
                acc = np.zeros_like(
                    e,
                    dtype=np.float64,
                )

            acc += e

        acc /= 30.0
        return acc

    return load_error(
        result_dir,
        seed,
        model,
        scenario,
        0,
    )


def summarize_horizon_results(result_dir, out):
    rows = []
    stochastic_rows = []

    for seed in LOCKED_SEEDS:
        for model, mode in TRAIN_SPECS:
            for scenario in ALL_SCENARIOS:

                e = aggregated_window_error(
                    result_dir,
                    seed,
                    model,
                    scenario,
                )

                for hix, horizon in enumerate(HORIZONS):
                    v = e[:, hix]

                    rows.append(
                        {
                            "training_seed": seed,
                            "model": model,
                            "training_mode": mode,
                            "scenario": scenario,
                            "horizon": horizon,
                            "n_windows": len(v),
                            "mean_error_m":
                                float(np.mean(v)),
                            "median_error_m":
                                float(np.median(v)),
                            "p90_error_m":
                                float(
                                    np.quantile(v, 0.90)
                                ),
                            "stochastic_rep_handling":
                                (
                                    "per-window mean across exactly "
                                    "30 locked SHA-256 replicates"
                                    if scenario in RANDOM_SCENARIOS
                                    else "not applicable"
                                ),
                        }
                    )

                if scenario in RANDOM_SCENARIOS:
                    for rep in range(30):
                        er = load_error(
                            result_dir,
                            seed,
                            model,
                            scenario,
                            rep,
                        )

                        for hix, horizon in enumerate(HORIZONS):
                            v = er[:, hix]

                            stochastic_rows.append(
                                {
                                    "training_seed": seed,
                                    "model": model,
                                    "training_mode": mode,
                                    "scenario": scenario,
                                    "replicate": rep,
                                    "horizon": horizon,
                                    "n_windows": len(v),
                                    "mean_error_m":
                                        float(np.mean(v)),
                                    "median_error_m":
                                        float(np.median(v)),
                                    "p90_error_m":
                                        float(
                                            np.quantile(v, 0.90)
                                        ),
                                }
                            )

    horizon_df = pd.DataFrame(rows)
    stochastic_df = pd.DataFrame(stochastic_rows)

    atomic_csv(
        horizon_df,
        out / "FIVE_SEED_HORIZON_RESULTS.csv",
    )

    atomic_csv(
        stochastic_df,
        out / "FIVE_SEED_STOCHASTIC_REPLICATE_RESULTS.csv",
    )

    return horizon_df, stochastic_df


def make_pair_results(
    horizon_df,
    model_first,
    model_second,
    label,
    out_path,
):
    rows = []

    for seed in LOCKED_SEEDS:
        for scenario in ALL_SCENARIOS:
            for horizon in HORIZONS:

                first = horizon_df[
                    (horizon_df["training_seed"] == seed)
                    & (horizon_df["model"] == model_first)
                    & (horizon_df["scenario"] == scenario)
                    & (horizon_df["horizon"] == horizon)
                ].iloc[0]

                second = horizon_df[
                    (horizon_df["training_seed"] == seed)
                    & (horizon_df["model"] == model_second)
                    & (horizon_df["scenario"] == scenario)
                    & (horizon_df["horizon"] == horizon)
                ].iloc[0]

                for metric in METRICS:
                    a = float(first[metric])
                    b = float(second[metric])
                    diff = a - b

                    rows.append(
                        {
                            "comparison": label,
                            "training_seed": seed,
                            "scenario": scenario,
                            "horizon": horizon,
                            "statistic":
                                metric.replace(
                                    "_error_m",
                                    "",
                                ),
                            "first_model":
                                model_first,
                            "second_model":
                                model_second,
                            "first_value_m": a,
                            "second_value_m": b,
                            "difference_first_minus_second_m":
                                diff,
                            "relative_difference_pct":
                                (
                                    100.0 * diff / b
                                    if b != 0
                                    else np.nan
                                ),
                            "first_model_lower_error":
                                bool(diff < 0),
                            "second_model_lower_error":
                                bool(diff > 0),
                        }
                    )

    df = pd.DataFrame(rows)

    atomic_csv(
        df,
        out_path,
    )

    return df


def summarize_pair(pair_df, out_path):
    rows = []

    group_cols = [
        "comparison",
        "scenario",
        "horizon",
        "statistic",
        "first_model",
        "second_model",
    ]

    for keys, g in pair_df.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        diff = (
            g["difference_first_minus_second_m"]
            .astype(float)
            .to_numpy()
        )

        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_locked_seeds": len(diff),
                "mean_difference_m":
                    float(np.mean(diff)),
                "sd_difference_m":
                    float(
                        np.std(
                            diff,
                            ddof=1,
                        )
                    ),
                "median_difference_m":
                    float(np.median(diff)),
                "min_difference_m":
                    float(np.min(diff)),
                "max_difference_m":
                    float(np.max(diff)),
                "seeds_first_model_lower_error":
                    int(np.sum(diff < 0)),
                "seeds_second_model_lower_error":
                    int(np.sum(diff > 0)),
                "seeds_exact_tie":
                    int(np.sum(diff == 0)),
                "all_five_seeds_same_direction":
                    bool(
                        np.all(diff < 0)
                        or np.all(diff > 0)
                    ),
            }
        )

    out = pd.DataFrame(rows)

    atomic_csv(
        out,
        out_path,
    )

    return out


def model_seed_summary(horizon_df, out_path):
    rows = []

    for (
        model,
        scenario,
        horizon,
    ), g in horizon_df.groupby(
        [
            "model",
            "scenario",
            "horizon",
        ],
        sort=False,
    ):
        for metric in METRICS:
            v = g[
                metric
            ].astype(float).to_numpy()

            rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "horizon": horizon,
                    "statistic":
                        metric.replace(
                            "_error_m",
                            "",
                        ),
                    "n_locked_seeds":
                        len(v),
                    "across_seed_mean_m":
                        float(np.mean(v)),
                    "across_seed_sd_m":
                        float(
                            np.std(
                                v,
                                ddof=1,
                            )
                        ),
                    "across_seed_median_m":
                        float(np.median(v)),
                    "across_seed_min_m":
                        float(np.min(v)),
                    "across_seed_max_m":
                        float(np.max(v)),
                }
            )

    out = pd.DataFrame(rows)

    atomic_csv(
        out,
        out_path,
    )

    return out


def collect_training_history(hist_dir, out):
    frames = []

    for p in sorted(
        hist_dir.glob(
            "seed_*_history.csv"
        )
    ):
        frames.append(
            pd.read_csv(p)
        )

    if not frames:
        raise RuntimeError(
            "No training history files found."
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    atomic_csv(
        df,
        out / "FIVE_SEED_TRAINING_HISTORY.csv",
    )

    return df


def checkpoint_manifest(out, hist_dir):
    rows = []

    for seed in LOCKED_SEEDS:
        for model, mode in TRAIN_SPECS:

            tag = f"seed_{seed}_{model}"
            ckpt = out / "checkpoints" / f"{tag}.pt"
            complete = (
                hist_dir
                / f"{tag}_COMPLETE.json"
            )

            if (
                not ckpt.exists()
                or not complete.exists()
            ):
                raise RuntimeError(
                    f"Incomplete training: {tag}"
                )

            meta = json.loads(
                complete.read_text(
                    encoding="utf-8"
                )
            )

            actual_hash = A.sha256_file(
                ckpt
            )

            if (
                actual_hash
                != meta["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    f"Checkpoint hash mismatch: {tag}"
                )

            rows.append(
                {
                    "training_seed": seed,
                    "model": model,
                    "training_mode": mode,
                    "checkpoint_file":
                        ckpt.name,
                    "checkpoint_sha256":
                        actual_hash,
                    "best_epoch":
                        meta["best_epoch"],
                    "best_val_clean_loss":
                        meta["best_val_loss"],
                }
            )

    df = pd.DataFrame(rows)

    atomic_csv(
        df,
        out / "FIVE_SEED_CHECKPOINT_MANIFEST.csv",
    )

    return df


def verify_eval_count(result_dir):
    json_files = list(
        result_dir.glob("*.json")
    )

    expected_per_model_seed = (
        len(ALL_SCENARIOS)
        - len(RANDOM_SCENARIOS)
        + 30 * len(RANDOM_SCENARIOS)
    )

    expected = (
        len(LOCKED_SEEDS)
        * len(TRAIN_SPECS)
        * expected_per_model_seed
    )

    if len(json_files) != expected:
        raise RuntimeError(
            f"Evaluation incomplete: "
            f"{len(json_files)}/{expected}"
        )

    for meta_path in json_files:
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        npz_path = meta_path.with_suffix(
            ".npz"
        )

        if not npz_path.exists():
            raise RuntimeError(
                f"Missing result: {npz_path}"
            )

        if (
            A.sha256_file(npz_path)
            != meta["sha256"]
        ):
            raise RuntimeError(
                f"Result hash mismatch: {npz_path}"
            )

    return expected
