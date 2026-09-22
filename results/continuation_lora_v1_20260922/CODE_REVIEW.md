# Pre-main read-only correctness review

Reviewed model.py, runner.py, preflight.py and attenuation.py against PROTOCOL.md before main training. A separate agent read the execution paths without changing code or running GPU work. No blocking issue was found. This is a code review, not an independent model implementation.

- CONTINUATION first call disables the adapter and gradients; the context manager restores the adapter for the second call.
- Native branching uses nine raw quantile-labelled first paths, then 81 atoms and the official quantile reduction. Composite parity is distinguished from native128 parity.
- Generated context is detached and never accepts future targets.
- Attenuation multiplies B only, reconstructs each candidate from original weights, and verifies lambda0/F0 and lambda1/archived VALIDATION equality.
- The four-fit ledger enforces paired initialization/schedules, two seeds and 2,048 main updates.
- Checkpoint selection uses VALIDATION, affine calibration uses CALIBRATION, and TEST checks the sealed selection/calibration hashes.

Fresh execution order: generate SOURCE_MANIFEST with `preflight.provenance()` before `attenuation.py`; then complete `preflight.py`, `runner.py all`, and `finalize.py`. Existing completed smoke receipts prevent silently repeating smoke optimizer steps. The current Stage D used that order and completed with exact endpoint parity.

Finalizer is reviewed separately after its implementation. Actual smoke receipts and CPU tests support the implementation review; they do not establish predictive benefit.
