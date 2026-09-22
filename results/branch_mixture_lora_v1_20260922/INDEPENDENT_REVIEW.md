# Independent read-only implementation review

Reviewer: data-engineer subagent, 2026-09-22, during main training and before TEST scoring. No GPU/optimizer execution or source edits by the reviewer.

No blocking correctness issue found in the reviewed scope:

- Raw ETTh2, prepared NPZ and both schedule SHA-256 values match DATA_AUDIT.json.
- PREFLIGHT and TRAINING_SEAL data/source/code hashes match current files.
- CPU tests: 11 passed. Both smoke arms match official native64/128 exactly, preserve frozen weights and produce finite gradients.
- Prediction flattening is origin then series, consistent with target arrays; branch/quantile/horizon reshaping matches declared distributions.
- Twice-pinball and sorted-atom empirical CRPS formulas match independent scalar/pairwise tests.
- Checkpoint selection reads VALIDATION only, affine fitting reads CALIBRATION only, selection hashes are sealed before TEST prediction.
- Target slices and observed context slices are disjoint; model-generated branch contexts have no target argument.
- Four fits / 2048 updates, identical within-seed initialization and schedule, fixed512 sensitivity and fixed reference forecasts match the registered scope.

Metadata clarification: `test_scored: false` in PREDICTIONS_MANIFEST.json records the state when all TEST predictions were saved before scoring. It is not a claim that finalization never scores TEST.

Exposure limitation: docs/ANCHOR_WINDOW_STUDY_PROTOCOL.md explicitly records ETTh2 and ETTm2 as resolutions of the same underlying transformer; prior project configurations use ETTm2. This is related-source pilot evidence, not an independent external source validation. This experiment's TEST remains excluded from its model/calibration selection.

Review is a read-only source/data audit, not a second independent training reproduction or a scientific-success endorsement.
