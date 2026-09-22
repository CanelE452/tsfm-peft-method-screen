# Pre-main read-only correctness review

Reviewed model.py, runner.py, preflight.py and attenuation.py against PROTOCOL.md before main training. A separate agent read the execution paths without changing code or running GPU work. No blocking issue was found. This is a code review, not an independent model implementation.

- CONTINUATION first call disables the adapter and gradients; the context manager restores the adapter for the second call.
- Native branching uses nine raw quantile-labelled first paths, then 81 atoms and the official quantile reduction. Composite parity is distinguished from native128 parity.
- Generated context is detached and never accepts future targets.
- Attenuation multiplies B only, reconstructs each candidate from original weights, and verifies lambda0/F0 and lambda1/archived VALIDATION equality.
- The four-fit ledger enforces paired initialization/schedules, two seeds and 2,048 main updates.
- Checkpoint selection uses VALIDATION, affine calibration uses CALIBRATION, and TEST checks the sealed selection/calibration hashes.

Fresh execution order: generate SOURCE_MANIFEST with `preflight.provenance()` before `attenuation.py`; then complete `preflight.py`, `runner.py all`, and `finalize.py`. Existing completed smoke receipts prevent silently repeating smoke optimizer steps. The current Stage D used that order and completed with exact endpoint parity.

The finalizer was subsequently reviewed at source commit 7091a6038ca1179dcb3077295e575ba8e6e4167e. No blocking issue was found in tensor axes, twice-pinball/81-atom CRPS, independent VAL selection/CAL grid recomputation, CAL/TEST first64 equality, practical decision direction, paired origin bootstrap or source/prediction/checkpoint hash checks. Syntax checks passed. No extra GPU/model/optimizer work was performed for this review.

The prediction manifest's `test_scored=false` records its creation before final scoring; VERIFICATION.json and decision.json describe the completed scoring stage. Model provenance is checked in preflight and again in the root post-run audit. Actual smoke receipts and CPU tests support the implementation review; they do not establish predictive benefit.
