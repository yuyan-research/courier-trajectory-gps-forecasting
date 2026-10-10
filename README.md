# Courier trajectory forecasting with sparse GPS observations

This repository contains the **core scientific implementation** for
*Performance evaluation of courier trajectory forecasting under sparse and
intermittent GPS sensing* (Measurement, manuscript MEAS-D-26-16741).

It provides the model definitions, sampling logic, observation-loss mechanisms,
training/evaluation functions, strictly held-out evaluation protocols, Kalman
filter implementations, and paired courier-cluster bootstrap. It is a
**methods-transparency release with synthetic regression checks**, not a
turnkey reproduction of every published result from raw LaDe data.

## Contents

- `src/trajectory_forecasting.py`: window definition, within-city courier-disjoint
  split, GRU/TCN/Transformer, training and prediction, simple motion baselines.
- `src/observation/`: empirical interval calibration, renewal/Markov masks,
  history and mobility characterization, raw-format-specific scanner.
- `src/experiments/randomized_evaluation.py`: 30-replicate SHA-256 evaluation
  masks, paired predictions and 2,000-draw courier-cluster uncertainty.
- `src/experiments/multiseed_comparison.py`: five predetermined training seeds,
  matched GRU and architecture evaluation.
- `src/experiments/protocols_and_filters.py`: strict LOCO and chronological
  splits, non-overlap, KF and coordinated-turn UKF, city/courier/date summaries.
- `src/experiments/strict_protocol_training.py`: strict-fold model fitting and
  reporting functions.
- `src/experiments/history_only.py`: history-only cohort construction with
  independently verified raw file SHA-256 digests.
- `src/experiments/sha256_missingness_masks.py`: fixed deterministic mask
  generator and in-memory binding for model inference.
- `src/experiments/validation_and_evaluation.py`: validation-only KF/UKF
  selection and five-seed architecture comparison.
- `src/experiments/courier_cluster_bootstrap.py`: fixed paired 2,000-draw
  courier bootstrap for primary model contrasts.

For a function-level guide, see [Methods-to-code map](METHODS_TO_CODE.md).

## Installation and checks

Use Python 3.10+ with a suitable CPU or GPU version of PyTorch. Dependencies are
listed by minimum supported package versions, *not* as an exact archived CUDA
or operating-system environment lockfile.

```bash
python -m pip install -r requirements.txt -r requirements-test.txt
python -m compileall -q src tests
python -m pytest -q tests
```

Tests operate on synthetic arrays and fixed deterministic reference vectors.
They check model forward shapes, random-mask consistency, fixed-seed generation,
courier-disjoint and non-overlapping split logic. They **do not** train models,
reconstruct published point estimates, or validate all dataset-dependent stages.

## Data and reproducibility boundary

The LaDe dataset can be obtained through the original provider:
<https://huggingface.co/datasets/Cainiao-AI/LaDe>. Please follow the provider's
current access requirements and license. This repository includes **no raw GPS
records, courier identifiers, window caches, split assignments, checkpoints,
model weights, original run logs or article figures**.

Selected source functions expect the original data schema, trusted preprocessed
window arrays and matching checkpoint structures. Exact split
assignments and the archived execution environment are not distributed. These components must be recreated or provided separately
for a numerical reproduction. Consequently, running `pytest` does not establish
that all numerical results in the article are independently reproducible.

The following interfaces are **not one-command public pipelines**:

- The empirical spacing scripts require source-format-specific generated shards
  in `COURIER_GPS_WORK_DIR/derived/`; the C scanner uses fixed source-format
  binary offsets and must never be run on other layouts. These must be validated for any independently downloaded file.
- The numerical evaluation functions accept arrays produced by
  the research pipeline, rather than downloading or regenerating every artifact.
- Protocol utilities expect a cache at `data/windows_cache.npz`, a corresponding
  `data/courier_split_manifest.csv`, and experiment result arrays where needed.
  The supplementary checkpoint-evaluation function accepts a `checkpoint_dir`
  argument (default: `checkpoints/` below the project root). These inputs are
  not provided.
- `history_only.make_history_only_cache` requires a caller-supplied input SHA-256
  manifest (`filename,size_bytes,sha256`) and trusted LaDe raw files. It verifies
  files before deserializing them.

The deterministic SHA-256 **namespace strings** and the empirical
Markov mask version label are intentionally preserved: changing these bytes
would change masks/seeds and therefore alter the verified scientific protocol.
In contrast, stochastic training augmentation uses its own random sequence.

Original coordinates, their projection and the conversion to physical metres
require verification with the dataset's original technical documentation.
This repository alone does not independently establish that a coordinate-unit
error corresponds exactly to one metre; interpret distance-unit labels with
that limitation. Observation interval statistics describe *observed spacing*,
not the physical causes of missing GPS records.

## Execution safety

Only process datasets and checkpoints from trusted sources. Pickle-derived
formats (`pd.read_pickle`, object-bearing NumPy archives) should not be loaded
from unknown parties. Some training and evaluation functions write or
replace files under the chosen output directory; use a dedicated, empty working
directory and keep a separate copy of valuable experimental artifacts. Do not
run the raw-object scanner on a file with an unverified binary layout.

## Rights and licensing

Copyright (c) 2026 YU YAN.

The source code and accompanying documentation in this repository are released under the MIT License. See [LICENSE](LICENSE) for the full license text.

The third-party LaDe dataset is not distributed with this repository and remains subject to its original provider's terms.
