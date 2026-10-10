#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Courier trajectory forecasting under sparse and intermittent GPS sensing
=======================================================================

A fixed experimental configuration for short-horizon courier forecasting.

Research question
-----------------
How much does short-horizon courier trajectory forecasting degrade as GPS
observations become sparse or intermittently unavailable, and which sequence
models remain most robust?

Ground truth
------------
The original LaDe ~20-second courier trajectory itself. No weak activity labels
are constructed.

Primary task
------------
Given the preceding ~10 minutes of courier GPS, forecast the courier position
at +1, +2 and +5 minutes.

Primary split
-------------
Courier-disjoint 70/15/15 within city.

Secondary validations
---------------------
1) Temporal holdout: early 60% dates train, next 10% validation, last 30% test.
2) Leave-one-city-out (LOCO): selected robust architecture trained on four
   cities, evaluated on the fifth city.

Sensor degradation scenarios
----------------------------
clean
sampling: 40s, 60s, 120s, 300s
random point loss: 10%, 30%, 50%
trailing burst outage before forecast cutoff: 1, 3, 5 minutes

Models
------
Classical:
  Last Position
  Constant Velocity
  Constant-Velocity Kalman Filter

Learned:
  GRU_clean       (trained only on clean histories)
  GRU_robust      (degradation-augmented training)
  TCN_robust
  Transformer_robust

Model selection
---------------
The best robust learned architecture is selected ONLY from the validation set
using mean ADE over four predeclared validation conditions:
  clean, sample_60s, random_30, burst_3m.
That architecture alone is retrained for temporal and LOCO validation.

Metrics
-------
Per horizon:
  mean position error (m)
  median position error (m)
  P90 position error (m)

Overall:
  ADE = mean error across 1/2/5 min horizons
  FDE = 5-min position error

Important leakage rules
-----------------------
- No courier identifier is a feature.
- Test couriers in the primary split are never used in training/validation.
- Sensor degradation is applied ONLY to the historical input.
- Future target points are always untouched.
- Coordinate features are relative within each window; raw city coordinate
  offsets are not used as shortcuts.
- No interpolation is used to create future ground truth.
- Windows with implausible jumps or poor original temporal coverage are
  excluded BEFORE any artificial degradation is applied.

Run
---
python src/trajectory_forecasting.py --root "/path/to/data_directory"

Useful options
--------------
--epochs 15
--max-windows 120000
--windows-per-courier-day 4
--run-loco 1
--rebuild-cache 0

Outputs
-------
<root>/results/trajectory_forecasting/
  00_protocol.txt
  01_data_quality.csv
  02_courier_split_manifest.csv
  03_split_summary.csv
  04_training_history.csv
  05_validation_model_selection.csv
  06_primary_courier_disjoint_metrics.csv
  07_temporal_holdout_metrics.csv
  08_loco_metrics.csv
  09_degradation_summary.csv
  10_run_summary.md
  windows_cache.npz
  checkpoints/*.pt
  trajectory_run.log
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError(
        "PyTorch is required for this experiment. Install a CUDA-enabled "
        "PyTorch build appropriate for your system, then rerun."
    ) from e


# ------------------------------ Constants -----------------------------------

SEED = 20260811
INPUT_POINTS = 31               # ~10 min at 20 s
TARGET_HORIZONS_SEC = np.array([60, 120, 300], dtype=np.int32)
TARGET_NAMES = ("1m", "2m", "5m")
MAX_TARGET_TIME_ERROR_SEC = 15
MIN_INPUT_DURATION_SEC = 540
MAX_INPUT_DURATION_SEC = 660
MAX_ORIGINAL_GAP_SEC = 60
MAX_STEP_SPEED_KMH = 150.0

CITIES = ("cq", "hz", "jl", "sh", "yt")
CITY_NAMES = {
    "cq": "Chongqing",
    "hz": "Hangzhou",
    "jl": "Jilin",
    "sh": "Shanghai",
    "yt": "Yantai",
}
CITY_MAP = {
    "上海市": "sh", "上海": "sh", "Shanghai": "sh",
    "杭州市": "hz", "杭州": "hz", "Hangzhou": "hz",
    "重庆市": "cq", "重庆": "cq", "Chongqing": "cq",
    "吉林市": "jl", "吉林": "jl", "Jilin": "jl",
    "烟台市": "yt", "烟台": "yt", "Yantai": "yt",
}
CITY_CODE = {c: i for i, c in enumerate(CITIES)}

SCENARIOS = (
    "clean",
    "sample_40s", "sample_60s", "sample_120s", "sample_300s",
    "random_10", "random_30", "random_50",
    "burst_1m", "burst_3m", "burst_5m",
)
MODEL_NAMES = (
    "LastPosition", "ConstantVelocity", "KalmanCV",
    "GRU_clean", "GRU_robust", "TCN_robust", "Transformer_robust",
)
ROBUST_SELECTION_SCENARIOS = ("clean", "sample_60s", "random_30", "burst_3m")

TRAIN_AUG_SCENARIOS = (
    "clean", "clean",
    "sample_40s", "sample_60s", "sample_120s",
    "random_10", "random_30", "random_50",
    "burst_1m", "burst_3m",
)


# ------------------------------- Utilities ----------------------------------

class Logger:
    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")
    def log(self, msg: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def ds_to_md(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("Int64")
    mm = (x // 100).astype("Int64")
    dd = (x % 100).astype("Int64")
    return mm.astype("string").str.zfill(2) + "-" + dd.astype("string").str.zfill(2)


def parse_lade_time(time_s: pd.Series, ds_s: pd.Series) -> pd.Series:
    """
    Explicit MM-DD in raw timestamp wins; ds is fallback.
    Synthetic year 2000 provides a shared chronological axis.
    """
    raw = time_s.astype("string").str.strip()
    fallback_md = ds_to_md(ds_s)

    md_parts = raw.str.extract(r"(?<!\d)(\d{1,2})-(\d{1,2})(?!\d)", expand=True)
    mm = pd.to_numeric(md_parts[0], errors="coerce").astype("Int64")
    dd = pd.to_numeric(md_parts[1], errors="coerce").astype("Int64")
    raw_md = mm.astype("string").str.zfill(2) + "-" + dd.astype("string").str.zfill(2)
    raw_md = raw_md.where(mm.between(1, 12) & dd.between(1, 31))
    md = raw_md.where(raw_md.notna(), fallback_md)

    clock = raw.str.extract(
        r"(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2})(?:\.\d+)?)?",
        expand=True,
    )
    hh = pd.to_numeric(clock["h"], errors="coerce").astype("Int64")
    mi = pd.to_numeric(clock["m"], errors="coerce").astype("Int64")
    ss = pd.to_numeric(clock["s"], errors="coerce").fillna(0).astype("Int64")

    valid = md.notna() & hh.between(0, 23) & mi.between(0, 59) & ss.between(0, 59)
    txt = (
        "2000-" + md
        + " " + hh.astype("string").str.zfill(2)
        + ":" + mi.astype("string").str.zfill(2)
        + ":" + ss.astype("string").str.zfill(2)
    )
    out = pd.Series(pd.NaT, index=time_s.index, dtype="datetime64[ns]")
    out.loc[valid] = pd.to_datetime(
        txt.loc[valid],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    return out


def planar_step_speed_kmh(x1, y1, x2, y2, dt_sec):
    dist_m = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (dist_m / dt_sec) * 3.6


def build_city_map(pickup: pd.DataFrame, delivery: pd.DataFrame):
    frames = []
    for df in (pickup, delivery):
        frames.append(pd.DataFrame({
            "courier_id": clean_id(df["delivery_user_id"]),
            "city": df["from_city_name"].astype("string").str.strip().map(CITY_MAP),
        }))
    e = pd.concat(frames, ignore_index=True).dropna()
    cc = e.groupby(["courier_id", "city"], observed=True).size().rename("n").reset_index()
    total = cc.groupby("courier_id")["n"].sum().rename("total")
    cc = cc.merge(total, on="courier_id")
    cc["share"] = cc["n"] / cc["total"]
    cc = cc.sort_values(["courier_id", "n"], ascending=[True, False])
    best = cc.drop_duplicates("courier_id", keep="first")
    best = best[best["share"] >= 0.95][["courier_id", "city"]].copy()
    return best


def nearest_target_index(times_sec: np.ndarray, base_idx: int, target_delta: int):
    target_t = times_sec[base_idx] + target_delta
    j = int(np.searchsorted(times_sec, target_t, side="left"))
    cand = []
    if j < len(times_sec):
        cand.append(j)
    if j - 1 > base_idx:
        cand.append(j - 1)
    cand = [k for k in cand if k > base_idx]
    if not cand:
        return None, None
    k = min(cand, key=lambda z: abs(times_sec[z] - target_t))
    err = abs(int(times_sec[k] - target_t))
    if err > MAX_TARGET_TIME_ERROR_SEC:
        return None, err
    return k, err


def make_windows_cache(
    root: Path,
    out: Path,
    max_windows: int,
    windows_per_cd: int,
    log: Logger,
):
    """
    Build a fixed, deterministic sample of high-quality trajectory windows.

    No artificial degradation is applied here. The cached history is the
    original observed trajectory.
    """
    raw_dir = root / "00_raw" / "LaDe" / "data_with_trajectory_20s"
    pu_path = raw_dir / "pickup_five_cities.pkl.xz"
    de_path = raw_dir / "delivery_five_cities.pkl.xz"
    tr_path = raw_dir / "courier_detailed_trajectory_20s.pkl.xz"

    log.log("Loading paired task files only to map anonymized courier IDs to cities...")
    pickup = pd.read_pickle(pu_path)
    delivery = pd.read_pickle(de_path)
    id_city = build_city_map(pickup, delivery)
    del pickup, delivery
    gc.collect()

    log.log(f"Courier-city map: {len(id_city):,} couriers.")

    log.log("Loading official 20-second trajectory pickle (~51M points)...")
    tr = pd.read_pickle(tr_path)
    log.log(f"Raw trajectory rows={len(tr):,}")

    traj = pd.DataFrame({
        "courier_id": clean_id(tr["postman_id"]),
        "traj_time": parse_lade_time(tr["gps_time"], tr["ds"]),
        "x": pd.to_numeric(tr["lng"], errors="coerce").astype("float64"),
        "y": pd.to_numeric(tr["lat"], errors="coerce").astype("float64"),
    })
    del tr
    gc.collect()

    traj = traj.merge(id_city, on="courier_id", how="inner")
    traj = traj.dropna(subset=["courier_id", "city", "traj_time", "x", "y"])
    finite = np.isfinite(traj["x"].to_numpy()) & np.isfinite(traj["y"].to_numpy())
    traj = traj.loc[finite].copy()

    traj["date_key"] = (
        traj["traj_time"].dt.month.astype(np.int16) * 100
        + traj["traj_time"].dt.day.astype(np.int16)
    )
    traj["sec_of_day"] = (
        traj["traj_time"].dt.hour.astype(np.int32) * 3600
        + traj["traj_time"].dt.minute.astype(np.int32) * 60
        + traj["traj_time"].dt.second.astype(np.int32)
    )

    # Encode IDs to reduce memory and accelerate groupby.
    traj["courier_id"] = traj["courier_id"].astype("category")
    courier_categories = traj["courier_id"].cat.categories.astype(str).tolist()
    traj["courier_code"] = traj["courier_id"].cat.codes.astype(np.int32)
    traj["city_code"] = traj["city"].map(CITY_CODE).astype(np.int8)

    log.log(
        f"Usable linked trajectory: {len(traj):,} points; "
        f"couriers={traj['courier_code'].nunique():,}; "
        f"courier-days={traj.groupby(['courier_code','date_key'], observed=True).ngroups:,}"
    )

    max_per_city = int(math.ceil(max_windows / len(CITIES)))
    x_hist = np.empty((max_windows, INPUT_POINTS), dtype=np.float32)
    y_hist = np.empty((max_windows, INPUT_POINTS), dtype=np.float32)
    t_hist = np.empty((max_windows, INPUT_POINTS), dtype=np.float32)
    targets = np.empty((max_windows, len(TARGET_HORIZONS_SEC), 2), dtype=np.float32)
    target_dt = np.empty((max_windows, len(TARGET_HORIZONS_SEC)), dtype=np.float32)
    courier_code_arr = np.empty(max_windows, dtype=np.int32)
    city_code_arr = np.empty(max_windows, dtype=np.int8)
    date_key_arr = np.empty(max_windows, dtype=np.int16)
    cutoff_sec_of_day_arr = np.empty(max_windows, dtype=np.int32)

    city_counts = {c: 0 for c in CITIES}
    nout = 0
    rejected_duration = 0
    rejected_target = 0
    rejected_gap_or_speed = 0

    log.log(
        f"Constructing up to {max_windows:,} windows "
        f"(cap ~{max_per_city:,}/city; <= {windows_per_cd} per courier-day)..."
    )

    group_cols = ["courier_code", "date_key"]
    for gi, ((cc, dk), g) in enumerate(
        traj.groupby(group_cols, sort=False, observed=True)
    ):
        city_code = int(g["city_code"].iloc[0])
        city = CITIES[city_code]
        if city_counts[city] >= max_per_city:
            continue

        if len(g) < INPUT_POINTS + 16:
            continue

        # Small per-group sort only; no 51M-row global sort.
        g = g.sort_values("sec_of_day")
        tt = g["sec_of_day"].to_numpy(dtype=np.int32)
        xx = g["x"].to_numpy(dtype=np.float64)
        yy = g["y"].to_numpy(dtype=np.float64)

        # Remove exact duplicate times for this courier-day.
        keep = np.r_[True, np.diff(tt) > 0]
        tt, xx, yy = tt[keep], xx[keep], yy[keep]
        if len(tt) < INPUT_POINTS + 16:
            continue

        dt = np.diff(tt).astype(np.float64)
        speed = planar_step_speed_kmh(xx[:-1], yy[:-1], xx[1:], yy[1:], dt)
        bad_interval = (
            (dt <= 0)
            | (dt > MAX_ORIGINAL_GAP_SEC)
            | ~np.isfinite(speed)
            | (speed > MAX_STEP_SPEED_KMH)
        )
        bad_cum = np.r_[0, np.cumsum(bad_interval.astype(np.int32))]

        candidates = np.arange(INPUT_POINTS - 1, len(tt) - 2, dtype=np.int32)
        # Require room for +5min.
        candidates = candidates[tt[candidates] + 300 <= tt[-1] + MAX_TARGET_TIME_ERROR_SEC]
        if len(candidates) == 0:
            continue

        # Sample candidate cutoffs across the courier-day rather than from
        # a contiguous early interval. Apply validity checks before selecting
        # up to `windows_per_cd` evenly spaced valid windows.
        if len(candidates) > windows_per_cd * 8:
            pick_idx = np.linspace(
                0, len(candidates) - 1,
                num=min(windows_per_cd * 8, len(candidates)),
                dtype=int,
            )
            candidates = candidates[pick_idx]

        valid_candidates = []
        for i in candidates:
            start = i - (INPUT_POINTS - 1)
            duration = int(tt[i] - tt[start])
            if duration < MIN_INPUT_DURATION_SEC or duration > MAX_INPUT_DURATION_SEC:
                rejected_duration += 1
                continue

            tjs = []
            terrs = []
            ok_targets = True
            for h in TARGET_HORIZONS_SEC:
                j, err = nearest_target_index(tt, int(i), int(h))
                if j is None:
                    ok_targets = False
                    break
                tjs.append(j)
                terrs.append(err)
            if not ok_targets:
                rejected_target += 1
                continue

            last_j = max(tjs)
            # bad_cum difference counts intervals from start through last_j-1.
            if int(bad_cum[last_j] - bad_cum[start]) > 0:
                rejected_gap_or_speed += 1
                continue

            valid_candidates.append((int(i), int(start), tjs, terrs))

        # Retain evenly spaced valid windows across the observed day.
        if len(valid_candidates) > windows_per_cd:
            sel = np.linspace(
                0, len(valid_candidates) - 1,
                num=windows_per_cd,
                dtype=int,
            )
            valid_candidates = [valid_candidates[k] for k in sel]

        for i, start, tjs, terrs in valid_candidates:
            if nout >= max_windows or city_counts[city] >= max_per_city:
                break

            base_t = float(tt[i])
            x_hist[nout] = xx[start:i+1].astype(np.float32)
            y_hist[nout] = yy[start:i+1].astype(np.float32)
            t_hist[nout] = (tt[start:i+1].astype(np.float32) - base_t)

            for hi, j in enumerate(tjs):
                targets[nout, hi, 0] = np.float32(xx[j])
                targets[nout, hi, 1] = np.float32(yy[j])
                target_dt[nout, hi] = np.float32(tt[j] - tt[i])

            courier_code_arr[nout] = int(cc)
            city_code_arr[nout] = city_code
            date_key_arr[nout] = int(dk)
            cutoff_sec_of_day_arr[nout] = int(tt[i])
            nout += 1
            city_counts[city] += 1

        if gi % 2000 == 0 and gi > 0:
            log.log(
                f"Processed courier-days={gi:,}; windows={nout:,}; "
                + ", ".join(f"{c}:{city_counts[c]:,}" for c in CITIES)
            )

        if nout >= max_windows:
            break
        if all(city_counts[c] >= max_per_city for c in CITIES):
            break

    log.log(f"Window construction finished: {nout:,} windows.")
    log.log(
        "City windows: " + ", ".join(f"{c}={city_counts[c]:,}" for c in CITIES)
    )

    # Trim arrays.
    x_hist = x_hist[:nout]
    y_hist = y_hist[:nout]
    t_hist = t_hist[:nout]
    targets = targets[:nout]
    target_dt = target_dt[:nout]
    courier_code_arr = courier_code_arr[:nout]
    city_code_arr = city_code_arr[:nout]
    date_key_arr = date_key_arr[:nout]
    cutoff_sec_of_day_arr = cutoff_sec_of_day_arr[:nout]

    cache = out / "windows_cache.npz"
    np.savez_compressed(
        cache,
        x_hist=x_hist,
        y_hist=y_hist,
        t_hist=t_hist,
        targets=targets,
        target_dt=target_dt,
        courier_code=courier_code_arr,
        city_code=city_code_arr,
        date_key=date_key_arr,
        cutoff_sec_of_day=cutoff_sec_of_day_arr,
        courier_categories=np.asarray(courier_categories, dtype=object),
    )

    quality_rows = [
        {"metric": "raw_linked_points", "value": len(traj)},
        {"metric": "windows", "value": nout},
        {"metric": "couriers_in_windows", "value": len(np.unique(courier_code_arr))},
        {"metric": "dates_in_windows", "value": len(np.unique(date_key_arr))},
        {"metric": "rejected_input_duration", "value": rejected_duration},
        {"metric": "rejected_target_alignment", "value": rejected_target},
        {"metric": "rejected_gap_or_speed", "value": rejected_gap_or_speed},
        {"metric": "target_1m_abs_time_error_sec_median", "value": float(np.median(np.abs(target_dt[:,0] - 60)))},
        {"metric": "target_2m_abs_time_error_sec_median", "value": float(np.median(np.abs(target_dt[:,1] - 120)))},
        {"metric": "target_5m_abs_time_error_sec_median", "value": float(np.median(np.abs(target_dt[:,2] - 300)))},
    ]
    for c in CITIES:
        quality_rows.append({
            "metric": f"windows_{c}",
            "value": int((city_code_arr == CITY_CODE[c]).sum()),
        })
    pd.DataFrame(quality_rows).to_csv(
        out / "01_data_quality.csv", index=False, encoding="utf-8-sig"
    )

    # Summarize hourly observation coverage.
    hour = (cutoff_sec_of_day_arr // 3600).astype(np.int16)
    hour_rows = []
    for city in CITIES:
        cm = city_code_arr == CITY_CODE[city]
        for h in range(24):
            hour_rows.append({
                "city": city,
                "hour_of_day": h,
                "windows": int(np.sum(cm & (hour == h))),
                "share_within_city": (
                    float(np.sum(cm & (hour == h)) / np.sum(cm))
                    if np.sum(cm) else np.nan
                ),
            })
    for h in range(24):
        hour_rows.append({
            "city": "POOLED",
            "hour_of_day": h,
            "windows": int(np.sum(hour == h)),
            "share_within_city": float(np.mean(hour == h)) if len(hour) else np.nan,
        })
    pd.DataFrame(hour_rows).to_csv(
        out / "01b_window_hour_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    del traj
    gc.collect()
    return cache


@dataclass
class WindowData:
    x_hist: np.ndarray
    y_hist: np.ndarray
    t_hist: np.ndarray
    targets: np.ndarray
    target_dt: np.ndarray
    courier_code: np.ndarray
    city_code: np.ndarray
    date_key: np.ndarray
    cutoff_sec_of_day: np.ndarray
    courier_categories: np.ndarray


def load_cache(cache_path: Path) -> WindowData:
    z = np.load(cache_path, allow_pickle=True)
    return WindowData(
        x_hist=z["x_hist"],
        y_hist=z["y_hist"],
        t_hist=z["t_hist"],
        targets=z["targets"],
        target_dt=z["target_dt"],
        courier_code=z["courier_code"],
        city_code=z["city_code"],
        date_key=z["date_key"],
        cutoff_sec_of_day=z["cutoff_sec_of_day"],
        courier_categories=z["courier_categories"],
    )


def make_courier_split(data: WindowData, out: Path):
    rng = np.random.default_rng(SEED)
    split_by_courier = {}
    rows = []

    for city in CITIES:
        code = CITY_CODE[city]
        couriers = np.unique(data.courier_code[data.city_code == code])
        rng.shuffle(couriers)
        n = len(couriers)
        n_train = max(1, int(n * 0.70))
        n_val = max(1, int(n * 0.15))
        tr = couriers[:n_train]
        va = couriers[n_train:n_train+n_val]
        te = couriers[n_train+n_val:]
        for c in tr:
            split_by_courier[int(c)] = "train"
        for c in va:
            split_by_courier[int(c)] = "val"
        for c in te:
            split_by_courier[int(c)] = "test"

        for c, sp in [(x, "train") for x in tr] + [(x, "val") for x in va] + [(x, "test") for x in te]:
            cid = str(data.courier_categories[int(c)])
            rows.append({
                "courier_code": int(c),
                "courier_id": cid,
                "city": city,
                "split": sp,
            })

    split = np.array([split_by_courier[int(c)] for c in data.courier_code], dtype="U5")
    pd.DataFrame(rows).to_csv(
        out / "02_courier_split_manifest.csv", index=False, encoding="utf-8-sig"
    )

    sum_rows = []
    for sp in ("train", "val", "test"):
        m = split == sp
        sum_rows.append({
            "protocol": "courier_disjoint",
            "split": sp,
            "windows": int(m.sum()),
            "couriers": int(len(np.unique(data.courier_code[m]))),
            "cities": int(len(np.unique(data.city_code[m]))),
        })

    dates = np.sort(np.unique(data.date_key))
    i1 = max(1, int(len(dates) * 0.60))
    i2 = max(i1 + 1, int(len(dates) * 0.70))
    train_dates = set(map(int, dates[:i1]))
    val_dates = set(map(int, dates[i1:i2]))
    test_dates = set(map(int, dates[i2:]))

    temporal_split = np.array([
        "train" if int(d) in train_dates else ("val" if int(d) in val_dates else "test")
        for d in data.date_key
    ], dtype="U5")

    for sp in ("train", "val", "test"):
        m = temporal_split == sp
        sum_rows.append({
            "protocol": "temporal_holdout",
            "split": sp,
            "windows": int(m.sum()),
            "couriers": int(len(np.unique(data.courier_code[m]))),
            "cities": int(len(np.unique(data.city_code[m]))),
            "dates": "|".join(map(str, sorted(np.unique(data.date_key[m]).tolist()))),
        })

    pd.DataFrame(sum_rows).to_csv(
        out / "03_split_summary.csv", index=False, encoding="utf-8-sig"
    )
    return split, temporal_split


# ---------------------------- Degradation -----------------------------------

def scenario_mask(n_points: int, scenario: str, sample_index: int, rep: int = 0):
    mask = np.ones(n_points, dtype=bool)
    if scenario == "clean":
        return mask

    if scenario.startswith("sample_"):
        sec = int(scenario.split("_")[1].replace("s", ""))
        k = max(1, int(round(sec / 20)))
        mask[:] = False
        mask[::k] = True
        mask[0] = True
        return mask

    if scenario.startswith("random_"):
        rate = int(scenario.split("_")[1]) / 100.0
        rng = np.random.default_rng(SEED + sample_index * 1009 + rep * 9176 + int(rate*100))
        mask = rng.random(n_points) >= rate
        mask[0] = True
        return mask

    if scenario.startswith("burst_"):
        mins = int(scenario.split("_")[1].replace("m", ""))
        n_drop = min(n_points - 1, int(round((mins * 60) / 20)))
        if n_drop > 0:
            mask[-n_drop:] = False
        mask[0] = True
        return mask

    raise ValueError(f"Unknown scenario: {scenario}")


def prepare_input_features(
    x: np.ndarray,
    y: np.ndarray,
    t_rel: np.ndarray,
    mask: np.ndarray,
):
    """
    Missing locations are forward-filled. Missingness is explicit via mask and
    time-since-last-observation, so the model never treats a forward-filled
    point as truly observed.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    t_rel = t_rel.astype(np.float64)
    mask = mask.astype(bool).copy()
    mask[0] = True

    x0 = float(x[0])
    y0 = float(y[0])

    xf = np.empty_like(x)
    yf = np.empty_like(y)
    dt_since = np.zeros(len(x), dtype=np.float64)

    last_x, last_y = x0, y0
    last_t = float(t_rel[0])
    for i in range(len(x)):
        if mask[i]:
            last_x, last_y = float(x[i]), float(y[i])
            last_t = float(t_rel[i])
        xf[i], yf[i] = last_x, last_y
        dt_since[i] = max(0.0, float(t_rel[i] - last_t))

    feats = np.column_stack([
        (xf - x0) / 1000.0,
        (yf - y0) / 1000.0,
        mask.astype(np.float32),
        dt_since / 300.0,
        t_rel / 600.0,
    ]).astype(np.float32)
    return feats, x0, y0


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        data: WindowData,
        indices: np.ndarray,
        training_mode: str,
        fixed_scenario: str = "clean",
        rep: int = 0,
    ):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training_mode = training_mode
        self.fixed_scenario = fixed_scenario
        self.rep = rep

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        if self.training_mode == "robust":
            scenario = TRAIN_AUG_SCENARIOS[np.random.randint(len(TRAIN_AUG_SCENARIOS))]
        elif self.training_mode == "clean":
            scenario = "clean"
        else:
            scenario = self.fixed_scenario

        mask = scenario_mask(INPUT_POINTS, scenario, i, self.rep)
        feats, x0, y0 = prepare_input_features(
            self.data.x_hist[i],
            self.data.y_hist[i],
            self.data.t_hist[i],
            mask,
        )
        target = self.data.targets[i].astype(np.float32).copy()
        target[:, 0] = (target[:, 0] - x0) / 1000.0
        target[:, 1] = (target[:, 1] - y0) / 1000.0
        target = target.reshape(-1).astype(np.float32)
        return torch.from_numpy(feats), torch.from_numpy(target), i


# ------------------------------- Models -------------------------------------

class GRURegressor(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden, num_layers=layers,
            batch_first=True, dropout=0.10 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class CausalBlock(nn.Module):
    def __init__(self, cin, cout, dilation):
        super().__init__()
        k = 3
        pad = dilation * (k - 1)
        self.pad = pad
        self.conv1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def crop(self, x):
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        residual = self.skip(x)
        y = self.crop(self.conv1(x))
        y = self.act(y)
        y = self.crop(self.conv2(y))
        if residual.shape[-1] != y.shape[-1]:
            residual = residual[:, :, -y.shape[-1]:]
        return self.act(y + residual)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.blocks = nn.Sequential(
            CausalBlock(input_dim, hidden, 1),
            CausalBlock(hidden, hidden, 2),
            CausalBlock(hidden, hidden, 4),
            CausalBlock(hidden, hidden, 8),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
    def forward(self, x):
        y = self.blocks(x.transpose(1, 2))
        return self.head(y[:, :, -1])


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, INPUT_POINTS, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.10, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.shape[1]]
        z = self.enc(z)
        return self.head(z[:, -1])


def make_model(name: str):
    if name.startswith("GRU"):
        return GRURegressor()
    if name.startswith("TCN"):
        return TCNRegressor()
    if name.startswith("Transformer"):
        return TransformerRegressor()
    raise ValueError(name)


# ---------------------------- Training / eval --------------------------------

def train_model(
    model_name: str,
    data: WindowData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device,
    out: Path,
    epochs: int,
    batch_size: int,
    training_mode: str,
    tag: str,
    log: Logger,
):
    model = make_model(model_name).to(device)
    train_ds = TrajectoryDataset(data, train_idx, training_mode=training_mode)
    val_ds = TrajectoryDataset(data, val_idx, training_mode="eval", fixed_scenario="clean")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = np.inf
    best_epoch = -1
    patience = 4
    stale = 0
    hist = []
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{tag}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss_sum += float(loss.detach().cpu()) * len(xb)
            train_n += len(xb)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=use_amp,
                ):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                val_loss_sum += float(loss.detach().cpu()) * len(xb)
                val_n += len(xb)

        tr_loss = train_loss_sum / max(train_n, 1)
        va_loss = val_loss_sum / max(val_n, 1)
        hist.append({
            "tag": tag,
            "model": model_name,
            "training_mode": training_mode,
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_clean_loss": va_loss,
        })
        log.log(
            f"[{tag}] epoch {epoch:02d}: train={tr_loss:.6f}, val_clean={va_loss:.6f}"
        )

        if va_loss < best - 1e-5:
            best = va_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best,
            }, ckpt)
        else:
            stale += 1
            if stale >= patience:
                log.log(f"[{tag}] early stopping at epoch {epoch}.")
                break

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    return model, pd.DataFrame(hist), ckpt


def predict_learned(
    model,
    data: WindowData,
    indices: np.ndarray,
    scenario: str,
    device,
    batch_size: int,
    rep: int = 0,
):
    ds = TrajectoryDataset(
        data, indices, training_mode="eval",
        fixed_scenario=scenario, rep=rep,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    pred_all = []
    idx_all = []
    model.eval()
    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for xb, _, ib in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                p = model(xb)
            pred_all.append(p.detach().cpu().numpy().reshape(-1, 3, 2))
            idx_all.append(np.asarray(ib))

    pred_km_rel = np.concatenate(pred_all, axis=0)
    idx = np.concatenate(idx_all).astype(np.int64)

    # Error does not require reconstructing the absolute origin because both
    # prediction and target are expressed relative to the same first point.
    target_rel = np.empty_like(pred_km_rel, dtype=np.float32)
    for j, i in enumerate(idx):
        x0 = data.x_hist[i, 0]
        y0 = data.y_hist[i, 0]
        target_rel[j, :, 0] = (data.targets[i, :, 0] - x0) / 1000.0
        target_rel[j, :, 1] = (data.targets[i, :, 1] - y0) / 1000.0

    err_m = np.sqrt(np.sum((pred_km_rel - target_rel)**2, axis=2)) * 1000.0
    return err_m, idx


def observed_arrays_for_scenario(data: WindowData, i: int, scenario: str, rep: int):
    mask = scenario_mask(INPUT_POINTS, scenario, i, rep)
    t = data.t_hist[i].astype(np.float64)
    x = data.x_hist[i].astype(np.float64)
    y = data.y_hist[i].astype(np.float64)
    return t[mask], x[mask], y[mask]


def baseline_last_position(data, indices, scenario, rep=0):
    errs = np.empty((len(indices), 3), dtype=np.float64)
    for j, i in enumerate(indices):
        t, x, y = observed_arrays_for_scenario(data, int(i), scenario, rep)
        px, py = x[-1], y[-1]
        target = data.targets[int(i)].astype(np.float64)
        errs[j] = np.sqrt((target[:,0]-px)**2 + (target[:,1]-py)**2)
    return errs


def robust_velocity_fit(t, x, y):
    """
    Linear fit over the most recent up to 6 observed points.
    Time is in seconds relative to cutoff; velocity output is coordinate units/s.
    """
    n = len(t)
    if n < 2:
        return 0.0, 0.0, float(x[-1]), float(y[-1]), float(t[-1])
    k = min(6, n)
    tt = t[-k:].astype(np.float64)
    xx = x[-k:].astype(np.float64)
    yy = y[-k:].astype(np.float64)
    tc = tt - tt.mean()
    denom = float(np.sum(tc**2))
    if denom <= 1e-9:
        vx = vy = 0.0
    else:
        vx = float(np.sum(tc * (xx - xx.mean())) / denom)
        vy = float(np.sum(tc * (yy - yy.mean())) / denom)
    return vx, vy, float(x[-1]), float(y[-1]), float(t[-1])


def baseline_constant_velocity(data, indices, scenario, rep=0):
    errs = np.empty((len(indices), 3), dtype=np.float64)
    for j, i in enumerate(indices):
        t, x, y = observed_arrays_for_scenario(data, int(i), scenario, rep)
        vx, vy, xlast, ylast, tlast = robust_velocity_fit(t, x, y)
        target = data.targets[int(i)].astype(np.float64)
        for hix, h in enumerate(TARGET_HORIZONS_SEC):
            future_dt = float(h - tlast)  # tlast <= 0
            px = xlast + vx * future_dt
            py = ylast + vy * future_dt
            errs[j, hix] = math.hypot(target[hix,0]-px, target[hix,1]-py)
    return errs


def kalman_predict_sequence(t, x, y, horizons):
    """
    Constant-velocity Kalman baseline in the assumed planar coordinate units.
    Fixed noise settings are intentionally not tuned on the test set.
    """
    if len(t) == 0:
        return np.zeros((len(horizons), 2))
    state = np.array([x[0], y[0], 0.0, 0.0], dtype=np.float64)
    P = np.diag([400.0, 400.0, 25.0, 25.0])
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
    R = np.eye(2) * (25.0**2)
    last_t = float(t[0])

    for k in range(1, len(t)):
        dt = max(1.0, float(t[k] - last_t))
        F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1],
        ], dtype=np.float64)
        q = 1.0
        G = np.array([[0.5*dt*dt,0],[0,0.5*dt*dt],[dt,0],[0,dt]], dtype=np.float64)
        Q = G @ (np.eye(2)*q) @ G.T
        state = F @ state
        P = F @ P @ F.T + Q

        z = np.array([x[k], y[k]], dtype=np.float64)
        innov = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ innov
        P = (np.eye(4) - K @ H) @ P
        last_t = float(t[k])

    preds = []
    for h in horizons:
        dt = float(h - last_t)
        preds.append([
            state[0] + state[2]*dt,
            state[1] + state[3]*dt,
        ])
    return np.asarray(preds)


def baseline_kalman(data, indices, scenario, rep=0):
    errs = np.empty((len(indices), 3), dtype=np.float64)
    for j, i in enumerate(indices):
        t, x, y = observed_arrays_for_scenario(data, int(i), scenario, rep)
        pred = kalman_predict_sequence(t, x, y, TARGET_HORIZONS_SEC)
        target = data.targets[int(i)].astype(np.float64)
        errs[j] = np.sqrt(np.sum((target - pred)**2, axis=1))
    return errs


def metric_rows_from_errors(
    errors_m: np.ndarray,
    model: str,
    scenario: str,
    protocol: str,
    scope: str,
    rep: int = 0,
):
    rows = []
    for hix, hname in enumerate(TARGET_NAMES):
        e = errors_m[:, hix]
        rows.append({
            "protocol": protocol,
            "scope": scope,
            "model": model,
            "scenario": scenario,
            "rep": rep,
            "horizon": hname,
            "n": len(e),
            "mean_error_m": float(np.mean(e)),
            "median_error_m": float(np.median(e)),
            "p90_error_m": float(np.quantile(e, 0.90)),
        })
    rows.append({
        "protocol": protocol,
        "scope": scope,
        "model": model,
        "scenario": scenario,
        "rep": rep,
        "horizon": "ADE",
        "n": len(errors_m),
        "mean_error_m": float(np.mean(errors_m)),
        "median_error_m": float(np.median(np.mean(errors_m, axis=1))),
        "p90_error_m": float(np.quantile(np.mean(errors_m, axis=1), 0.90)),
    })
    fde = errors_m[:, -1]
    rows.append({
        "protocol": protocol,
        "scope": scope,
        "model": model,
        "scenario": scenario,
        "rep": rep,
        "horizon": "FDE",
        "n": len(fde),
        "mean_error_m": float(np.mean(fde)),
        "median_error_m": float(np.median(fde)),
        "p90_error_m": float(np.quantile(fde, 0.90)),
    })
    return rows


def evaluate_baselines(
    data, indices, scenarios, protocol, scope, log
):
    rows = []
    for sc in scenarios:
        reps = range(3) if sc.startswith("random_") else range(1)
        for rep in reps:
            log.log(f"[{protocol}/{scope}] baselines scenario={sc} rep={rep}")
            for name, fn in [
                ("LastPosition", baseline_last_position),
                ("ConstantVelocity", baseline_constant_velocity),
                ("KalmanCV", baseline_kalman),
            ]:
                e = fn(data, indices, sc, rep)
                rows += metric_rows_from_errors(
                    e, name, sc, protocol, scope, rep
                )
    return rows


def evaluate_learned(
    model, model_name, data, indices, scenarios, protocol, scope,
    device, batch_size, log
):
    rows = []
    for sc in scenarios:
        reps = range(3) if sc.startswith("random_") else range(1)
        for rep in reps:
            log.log(f"[{protocol}/{scope}] {model_name} scenario={sc} rep={rep}")
            e, _ = predict_learned(
                model, data, indices, sc, device, batch_size, rep
            )
            rows += metric_rows_from_errors(
                e, model_name, sc, protocol, scope, rep
            )
    return rows


def validation_composite_ade(
    model, data, val_idx, device, batch_size
):
    vals = []
    for sc in ROBUST_SELECTION_SCENARIOS:
        e, _ = predict_learned(
            model, data, val_idx, sc, device, batch_size, rep=0
        )
        vals.append(float(np.mean(e)))
    return float(np.mean(vals)), vals


def degradation_summary(metrics: pd.DataFrame):
    """
    Relative ADE degradation compared with the same model/protocol/scope clean ADE.
    Random-replicate metrics are averaged first.
    """
    m = metrics[metrics["horizon"] == "ADE"].copy()
    agg = (
        m.groupby(["protocol","scope","model","scenario"], observed=True)
        .agg(
            ade_mean_m=("mean_error_m","mean"),
            ade_median_m=("median_error_m","mean"),
            ade_p90_m=("p90_error_m","mean"),
        )
        .reset_index()
    )
    clean = agg[agg["scenario"] == "clean"][
        ["protocol","scope","model","ade_mean_m"]
    ].rename(columns={"ade_mean_m":"clean_ade_mean_m"})
    out = agg.merge(clean, on=["protocol","scope","model"], how="left")
    out["relative_ADE_increase"] = (
        out["ade_mean_m"] - out["clean_ade_mean_m"]
    ) / out["clean_ade_mean_m"]
    return out


# ---------------------------- Main experiment --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-windows", type=int, default=120000)
    ap.add_argument("--windows-per-courier-day", type=int, default=4)
    ap.add_argument("--run-loco", type=int, default=1)
    ap.add_argument("--rebuild-cache", type=int, default=0)
    args = ap.parse_args()

    set_seed()

    root = Path(args.root).resolve()
    out = root / "results" / "trajectory_forecasting"
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "trajectory_run.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.log(f"Root: {root}")
    log.log(f"PyTorch={torch.__version__}; device={device}")
    if torch.cuda.is_available():
        log.log(f"GPU: {torch.cuda.get_device_name(0)}")

    protocol_text = f"""Formal fixed experiment protocol
Date seed: {SEED}
Input history: ~10 min, {INPUT_POINTS} original points
Targets: +1m, +2m, +5m
Window quality:
  original input duration {MIN_INPUT_DURATION_SEC}-{MAX_INPUT_DURATION_SEC}s
  original gap <= {MAX_ORIGINAL_GAP_SEC}s
  original step speed <= {MAX_STEP_SPEED_KMH} km/h
  target timing tolerance <= {MAX_TARGET_TIME_ERROR_SEC}s
Primary split: courier-disjoint 70/15/15 within city
Temporal split: first 60% dates train, next 10% val, last 30% test
LOCO: selected robust learned architecture only
Scenarios: {", ".join(SCENARIOS)}
Models: {", ".join(MODEL_NAMES)}
Robust architecture validation selection scenarios:
  {", ".join(ROBUST_SELECTION_SCENARIOS)}
Epoch cap: {args.epochs}
Batch size: {args.batch_size}
Maximum cached windows: {args.max_windows}
Maximum windows per courier-day: {args.windows_per_courier_day}
Window selection:
  valid candidate windows are sampled evenly across each courier-day's
  observed operating span; hour-of-day coverage is reported.
"""
    (out / "00_protocol.txt").write_text(protocol_text, encoding="utf-8")

    cache = out / "windows_cache.npz"
    if args.rebuild_cache or not cache.exists():
        cache = make_windows_cache(
            root, out, args.max_windows, args.windows_per_courier_day, log
        )
    else:
        log.log(f"Using existing cache: {cache}")

    data = load_cache(cache)
    log.log(
        f"Cached windows loaded: {len(data.x_hist):,}; "
        f"couriers={len(np.unique(data.courier_code)):,}"
    )

    split, temporal_split = make_courier_split(data, out)
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]

    histories = []
    learned_models = {}
    selection_rows = []

    train_specs = [
        ("GRU_clean", "clean"),
        ("GRU_robust", "robust"),
        ("TCN_robust", "robust"),
        ("Transformer_robust", "robust"),
    ]

    # Main training.
    for model_name, mode in train_specs:
        log.log(f"Training {model_name} on courier-disjoint training couriers...")
        model, hist, _ = train_model(
            model_name, data, train_idx, val_idx, device, out,
            args.epochs, args.batch_size, mode,
            tag=f"primary_{model_name}", log=log,
        )
        histories.append(hist)
        learned_models[model_name] = model

        comp, vals = validation_composite_ade(
            model, data, val_idx, device, args.batch_size
        )
        selection_rows.append({
            "model": model_name,
            "training_mode": mode,
            "validation_composite_ADE_m": comp,
            **{
                f"val_ADE_{sc}_m": vals[k]
                for k, sc in enumerate(ROBUST_SELECTION_SCENARIOS)
            },
        })
        log.log(f"{model_name} validation composite ADE={comp:.3f} m")

    pd.concat(histories, ignore_index=True).to_csv(
        out / "04_training_history.csv", index=False, encoding="utf-8-sig"
    )
    selection_df = pd.DataFrame(selection_rows).sort_values(
        "validation_composite_ADE_m"
    )
    selection_df.to_csv(
        out / "05_validation_model_selection.csv", index=False, encoding="utf-8-sig"
    )

    robust_candidates = selection_df[
        selection_df["training_mode"] == "robust"
    ]
    selected_arch = str(robust_candidates.iloc[0]["model"])
    log.log(f"Selected robust architecture from validation only: {selected_arch}")

    # Primary test.
    primary_rows = []
    primary_rows += evaluate_baselines(
        data, test_idx, SCENARIOS,
        protocol="courier_disjoint", scope="POOLED", log=log
    )
    for model_name, model in learned_models.items():
        primary_rows += evaluate_learned(
            model, model_name, data, test_idx, SCENARIOS,
            protocol="courier_disjoint", scope="POOLED",
            device=device, batch_size=args.batch_size, log=log
        )

    # City-specific primary metrics for learned models and CV baseline.
    for city in CITIES:
        city_test = test_idx[data.city_code[test_idx] == CITY_CODE[city]]
        if len(city_test) == 0:
            continue
        for sc in ("clean", "sample_60s", "random_30", "burst_3m", "burst_5m"):
            reps = range(3) if sc.startswith("random_") else range(1)
            for rep in reps:
                e = baseline_constant_velocity(data, city_test, sc, rep)
                primary_rows += metric_rows_from_errors(
                    e, "ConstantVelocity", sc, "courier_disjoint",
                    city, rep
                )
            for model_name, model in learned_models.items():
                primary_rows += evaluate_learned(
                    model, model_name, data, city_test, (sc,),
                    protocol="courier_disjoint", scope=city,
                    device=device, batch_size=args.batch_size, log=log
                )

    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(
        out / "06_primary_courier_disjoint_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    # Temporal holdout: retrain selected architecture only.
    temp_train = np.where(temporal_split == "train")[0]
    temp_val = np.where(temporal_split == "val")[0]
    temp_test = np.where(temporal_split == "test")[0]

    log.log(f"Retraining selected architecture {selected_arch} for temporal holdout...")
    temp_model, temp_hist, _ = train_model(
        selected_arch, data, temp_train, temp_val, device, out,
        args.epochs, args.batch_size, "robust",
        tag=f"temporal_{selected_arch}", log=log,
    )
    temp_hist.to_csv(
        out / "04b_temporal_training_history.csv", index=False, encoding="utf-8-sig"
    )

    temporal_rows = []
    temporal_rows += evaluate_baselines(
        data, temp_test, SCENARIOS,
        protocol="temporal_holdout", scope="POOLED", log=log
    )
    temporal_rows += evaluate_learned(
        temp_model, selected_arch, data, temp_test, SCENARIOS,
        protocol="temporal_holdout", scope="POOLED",
        device=device, batch_size=args.batch_size, log=log
    )
    temporal_df = pd.DataFrame(temporal_rows)
    temporal_df.to_csv(
        out / "07_temporal_holdout_metrics.csv",
        index=False, encoding="utf-8-sig"
    )

    # LOCO: selected architecture only.
    loco_rows = []
    if int(args.run_loco):
        for held_city in CITIES:
            held_code = CITY_CODE[held_city]
            test_loco = np.where(data.city_code == held_code)[0]
            trainval = np.where(data.city_code != held_code)[0]

            # Courier-disjoint validation from training cities.
            rng = np.random.default_rng(SEED + held_code)
            couriers = np.unique(data.courier_code[trainval])
            rng.shuffle(couriers)
            nval = max(1, int(len(couriers) * 0.10))
            val_couriers = set(map(int, couriers[:nval]))
            loco_val = trainval[
                np.array([int(c) in val_couriers for c in data.courier_code[trainval]])
            ]
            loco_train = trainval[
                np.array([int(c) not in val_couriers for c in data.courier_code[trainval]])
            ]

            log.log(
                f"LOCO heldout={held_city}: train={len(loco_train):,}, "
                f"val={len(loco_val):,}, test={len(test_loco):,}"
            )
            lm, lhist, _ = train_model(
                selected_arch, data, loco_train, loco_val, device, out,
                args.epochs, args.batch_size, "robust",
                tag=f"loco_{held_city}_{selected_arch}", log=log,
            )
            lhist.to_csv(
                out / f"04c_loco_{held_city}_training_history.csv",
                index=False, encoding="utf-8-sig"
            )

            for sc in ("clean", "sample_60s", "random_30", "burst_3m", "burst_5m"):
                reps = range(3) if sc.startswith("random_") else range(1)
                for rep in reps:
                    e = baseline_constant_velocity(data, test_loco, sc, rep)
                    loco_rows += metric_rows_from_errors(
                        e, "ConstantVelocity", sc,
                        "leave_one_city_out", held_city, rep
                    )
                loco_rows += evaluate_learned(
                    lm, selected_arch, data, test_loco, (sc,),
                    protocol="leave_one_city_out", scope=held_city,
                    device=device, batch_size=args.batch_size, log=log
                )

            del lm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    loco_df = pd.DataFrame(loco_rows)
    loco_df.to_csv(
        out / "08_loco_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # Combined degradation summary.
    combined = pd.concat(
        [primary_df, temporal_df, loco_df],
        ignore_index=True,
        sort=False,
    )
    deg = degradation_summary(combined)
    deg.to_csv(
        out / "09_degradation_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Human-readable summary focused on the predeclared primary test.
    p = primary_df[
        (primary_df["scope"] == "POOLED")
        & (primary_df["horizon"] == "ADE")
    ].copy()
    pagg = (
        p.groupby(["model","scenario"], observed=True)["mean_error_m"]
        .mean().reset_index()
    )

    lines = [
        "# Courier trajectory forecasting experiment summary",
        "",
        f"- Device: {device}",
        f"- Cached trajectory windows: {len(data.x_hist):,}",
        f"- Couriers represented: {len(np.unique(data.courier_code)):,}",
        f"- Selected robust architecture (validation only): {selected_arch}",
        "",
        "## Primary courier-disjoint pooled ADE",
    ]
    for model in MODEL_NAMES:
        g = pagg[pagg["model"] == model]
        if len(g) == 0:
            continue
        lines.append(f"### {model}")
        for _, r in g.iterrows():
            lines.append(
                f"- {r['scenario']}: ADE={r['mean_error_m']:.2f} m"
            )

    lines += [
        "",
        "## Interpretation rules fixed before reading the test results",
        "- The main scientific result is the degradation curve, not a claim that one neural architecture is universally best.",
        "- Strong evidence requires the selected learned model to remain competitive under courier-disjoint, temporal, and LOCO validation.",
        "- Random-loss and burst-outage results must be interpreted separately: a trailing burst removes the most recent location and is intentionally harder.",
        "- The 300s sampling and 5min burst conditions are stress tests, not expected operating points.",
        "- No additional window, city, horizon, speed threshold, or model will be chosen after seeing these results merely to improve performance.",
    ]
    (out / "10_run_summary.md").write_text("\n".join(lines), encoding="utf-8")

    log.log("FORMAL EXPERIMENT COMPLETE.")
    log.log(f"Selected architecture: {selected_arch}")
    log.log(f"Outputs: {out}")
    log.close()


if __name__ == "__main__":
    main()
