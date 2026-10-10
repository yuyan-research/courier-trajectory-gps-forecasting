"""Synthetic tests for model outputs and deterministic observation masks."""

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/experiments"))
sys.path.insert(0, str(ROOT / "src"))

from randomized_evaluation import mask_matrix as evaluation_mask_matrix
from randomized_evaluation import window_seed as evaluation_window_seed
from sha256_missingness_masks import mask_matrix, root_seed, window_seed
from trajectory_forecasting import scenario_mask as training_mask


def test_sha256_mask_reference_vectors():
    with (ROOT / "tests/mask_anchor_vectors.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    for record in rows:
        scenario = record["scenario"]
        replicate = int(record["replicate"])
        index = int(record["window_index"])
        namespace = record["namespace"]

        seed, digest = window_seed(scenario, replicate, index, namespace)
        assert seed == int(record["window_seed_u64"])
        assert digest == record["window_sha256"]

        mask = mask_matrix([index], scenario, replicate, namespace=namespace)
        packed = bytes.fromhex(record["expected_mask_packed_hex"])
        np.testing.assert_array_equal(np.packbits(mask[0]).tobytes(), packed)
        if record["cohort"] == "primary_reference":
            np.testing.assert_array_equal(
                mask, evaluation_mask_matrix(np.array([index]), scenario, replicate)
            )
            assert seed == evaluation_window_seed(scenario, replicate, index)


def test_all_scenarios_synthetic_masks():
    indices = np.array([0, 1, 3, 11, 100])
    scenarios = [
        "clean", "sample_40s", "sample_60s", "sample_120s", "sample_300s",
        "random_10", "random_30", "random_50", "burst_1m", "burst_3m", "burst_5m",
    ]
    for scenario in scenarios:
        replicates = [0, 1, 7, 29] if scenario.startswith("random_") else [0]
        for replicate in replicates:
            mask = mask_matrix(indices, scenario, replicate)
            comparison = evaluation_mask_matrix(indices, scenario, replicate)
            assert mask.shape == (len(indices), 31)
            assert mask.dtype == bool
            assert mask[:, 0].all()
            np.testing.assert_array_equal(mask, comparison)


def test_training_and_evaluation_rng_are_distinct():
    evaluation = mask_matrix([0], "random_10", 0)[0]
    augmentation = training_mask(31, "random_10", 0, 0)
    assert not np.array_equal(evaluation, augmentation)


def test_model_forward_only_synthetic():
    import torch
    from trajectory_forecasting import INPUT_POINTS, make_model

    for name in ("GRU_clean", "GRU_robust", "TCN_robust", "Transformer_robust"):
        model = make_model(name)
        model.eval()
        with torch.no_grad():
            features = torch.zeros(2, INPUT_POINTS, 5, dtype=torch.float32)
            output = model(features)
        assert output.shape[0] == 2
        assert torch.isfinite(output).all()


def test_statistical_seed_registry():
    for scenario in ("random_10", "random_30", "random_50"):
        seeds = [root_seed(scenario, replicate)[0] for replicate in range(30)]
        assert len(set(seeds)) == 30


def test_in_memory_inference_binding_follows_fixed_mask_rule():
    import types
    from sha256_missingness_masks import patch_module_for_evaluation

    module = types.SimpleNamespace(scenario_mask=lambda *args: None)
    patched = patch_module_for_evaluation(module)
    observed = patched.scenario_mask(31, "random_30", 11, 5)
    expected = mask_matrix([11], "random_30", 5)[0]
    np.testing.assert_array_equal(observed, expected)
