# File inventory

## New V2 files

- `labels_v2.py`: convenient root-level CLI alias for V2 label audit.
- `xgb_v2.py`: corrected XGBoost baseline with target-specific sampling.
- `src/data/labels_v2.py`: label definitions and audit implementation.
- `src/data/export_labels_v2.py`: export all target/horizon labels and validity flags to Parquet.
- `src/data/dataset_v2.py`: graph-sequence dataset that consumes exported V2 label tables.
- `src/graph/physio_graph_v2.py`: 8 nodes, locked 10 directed edges, 2 node channels.
- `src/models/physiograph_model_v2.py`: two-channel GAT → TCN → MLP.
- `src/train/train_v2.py`: validation-selected training loop; does not load test data.
- `src/train/evaluate_v2.py`: frozen-checkpoint evaluation with required fixed threshold.
- `tests/test_v2_smoke.py`: synthetic label semantics test and optional PyG shape test.
- `requirements-v2.txt`: core package dependencies.
- `README.md`: setup, assumptions, commands, and cautions.
- `docs/IMPLEMENTATION_PLAN.md`: preflight and experiment checklist.

## Preserved baseline files

The remaining files under `src/` are copied from the supplied `fyp_icu2.zip` source tree. They are retained for reproducibility and V1 comparison; the V2 path uses the separate modules above. Existing checkpoints are not included in this pack.

## Deliberately not included

MIMIC-IV raw/processed patient data, credential files, local virtual environments, and old trained checkpoint weights are excluded. They should remain in the user's local secured project workspace.
- `src/models/xgboost_baseline.py`: module entry point for running the corrected baseline with `python -m src.models.xgboost_baseline`.
