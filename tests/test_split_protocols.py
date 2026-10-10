"""Synthetic checks for courier-disjoint, temporal, LOCO and non-overlap protocols."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
from protocols_and_filters import selection_splits, nonoverlap


def test_strict_loco_and_temporal_separation_synthetic():
    # Artificial courier IDs: each one is in exactly one primary split and date.
    row = [(city, day, city * 10000 + day * 100 + j)
           for city in range(5) for day in range(1, 9) for j in range(2)]
    d = {
        "city_code": np.array([a for a, _, _ in row], dtype=np.int8),
        "date_key": np.array([b for _, b, _ in row], dtype=np.int32),
        "courier_code": np.array([c for _, _, c in row], dtype=np.int64),
    }
    primary = np.array(["train" if day <= 4 else ("val" if day == 5 else "test")
                        for _, day, _ in row])
    base, temporal, loco, dates = selection_splits(d, primary)
    assert len(loco) == 5 and len(dates) == 8
    for split in [base, temporal, *loco.values()]:
        ids = [set(d["courier_code"][split[k]]) for k in ("train", "val", "test")]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for held_city, split in enumerate(loco.values()):
        assert set(d["city_code"][split["test"]]) == {held_city}
        assert held_city not in set(d["city_code"][split["train"]])
        assert held_city not in set(d["city_code"][split["val"]])


def test_nonoverlap_uses_history_to_last_label_interval():
    # First two windows overlap; third has no shared history/forecast time.
    d = {
        "courier_code": np.array([1, 1, 1, 2]),
        "date_key": np.array([1, 1, 1, 1]),
        "cutoff_sec_of_day": np.array([1000, 1200, 2200, 1000]),
        "t_hist": np.tile(np.array([-600, -300, 0]), (4, 1)),
        "target_dt": np.tile(np.array([60, 120, 300]), (4, 1)),
    }
    kept = nonoverlap(d, np.arange(4))
    np.testing.assert_array_equal(kept, np.array([0, 2, 3]))


def test_architecture_choice_uses_validation_only_five_seeds():
    from validation_and_evaluation import select_strict_architecture
    selected, means = select_strict_architecture({
        "TCN_robust": [6.0, 5.0, 5.0, 5.0, 4.0],
        "Transformer_robust": [5.0, 4.0, 4.0, 4.0, 3.0],
    })
    assert selected == "Transformer_robust"
    assert means["Transformer_robust"] == 4.0


def test_kf_ukf_parameter_selection_on_synthetic_validation():
    from validation_and_evaluation import select_filter_parameters

    history_time = np.arange(-600, 1, 20, dtype=np.float64)
    horizons = np.array([60.0, 120.0, 300.0])
    x = np.stack([0.4 * history_time, 0.4 * history_time + 20])
    y = np.stack([0.05 * history_time, 0.05 * history_time + 5])
    future = np.stack([
        np.column_stack((0.4 * horizons + dx, 0.05 * horizons + dy))
        for dx, dy in ((0, 0), (20, 5))
    ])
    synthetic = {
        "x_hist": x,
        "y_hist": y,
        "t_hist": np.stack([history_time, history_time]),
        "targets": future,
    }
    rows, selected = select_filter_parameters(synthetic, np.array([0, 1]))
    assert len(rows) == 18  # Two filters, three process noises, three observation noises.
    assert set(selected) == {"KF_tuned", "UKF_CTRV"}
    assert all(np.isfinite(r["validation_ADE_mean"]) for r in rows)
    for setting in selected.values():
        assert setting["q"] in (0.1, 1.0, 10.0)
        assert setting["r"] in (10.0, 25.0, 50.0)
