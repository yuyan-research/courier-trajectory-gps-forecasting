# Methods-to-code map

| Scientific element | Implementation | Publicly tested |
|---|---|---|
| Courier-disjoint split, time-window formation, model definitions | `src/trajectory_forecasting.py` (`make_courier_split`, `make_windows_cache`, `make_model`) | Synthetic neural forward pass |
| Empirical GPS spacing and Markov observation loss | `src/observation/` (`calibrate_spacing.py`, `generate_empirical_masks.py`) | Source inspection only; raw-data shards absent |
| Fixed SHA-256 random masks and 30 repeats | `src/experiments/sha256_missingness_masks.py` and `randomized_evaluation.py` | Deterministic vectors and cross-implementation equality |
| Five-seed training, evaluation and matched pairs | `src/experiments/multiseed_comparison.py` | Seed registry; full training not rerun |
| Courier-disjoint, strictly chronological, strict LOCO and non-overlap evaluation | `src/experiments/protocols_and_filters.py` and `strict_protocol_training.py` | Synthetic split invariants |
| History-only sensitivity | `src/experiments/history_only.py` | Requires independently verified raw input; not run |
| Validation-only KF/UKF grid | `src/experiments/validation_and_evaluation.py` and `protocols_and_filters.py` | Source inspection; full empirical grid not rerun |
| Two thousand courier-level paired bootstrap resamples | `src/experiments/randomized_evaluation.py` (`courier_cluster_bootstrap`) and `courier_cluster_bootstrap.py` (`paired_bootstrap`) | Original algorithms retained; published intervals not rerun |
| Exact test-time mask binding for strict folds and history-only | `src/experiments/sha256_missingness_masks.py` (`patch_module_for_evaluation`) | Mask tests; end-to-end checkpoint inference not rerun |

**Reproduction scope.** This repository provides the model architectures,
observation-loss mechanisms, evaluation protocols and statistical estimators
for source inspection and focused synthetic checks. Evaluation with research
windows and trained weights requires external inputs described in the README.

**Numerical compatibility.** The SHA-256 mask namespaces and Markov observation
model version are fixed seed inputs. They are retained verbatim because even a
cosmetic change would produce different evaluation masks. Test vectors in
`tests/mask_anchor_vectors.csv` verify these deterministic specifications.
