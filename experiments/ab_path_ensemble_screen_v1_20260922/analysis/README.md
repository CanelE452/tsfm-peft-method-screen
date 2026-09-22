# Analysis and inference-only completion

The sole scientific contract remains `../contract/MASTER_CLI.txt`. Root training code and A/B data modules were sealed before main training. This subdirectory adds post-training inference diagnostics, fair timing, scoring and reporting without changing those sealed sources.

Executed in order:

1. `../runner.py`: fixed 24 fits, selection/CAL/baseline sealing, all selected TEST predictions, original runtime.
2. `diagnostic_predictions.py`: A checkpoint0 raw TEST predictions; no optimizer updates or alternative selection.
3. `verify_fair_runtime.py`: TRAIN batch1/8 equivalence for efficient GBQR and Chronos-2 wrappers (exact zero differences).
4. `fair_benchmark.py`: full opposing-order timing, 10 warmups and 30 measurements per arm/batch, preserving original runtime files. GBQR features once; Chronos-2 directly accepts CPU input. No model changes.
5. `finalize.py`: assert all predictions/runtime complete, verify seals, score raw/CAL, reproduce choices, compute paired 14-origin bootstrap and decisions.
6. `report.py`: figures and Korean reports from verified tables.
7. `final_audit.py`: arithmetic from tables, stored first64 preservation, identical GLOBAL3/CONTEXT3 forecasts, shared initial LoRA, weight normalization, cache manifest.

Execution used the exact Python3.11/CUDA environment recorded in each `ENVIRONMENT.json`. Training is not automatically replayable: existing fit keys and source seals reject duplicate execution. No follow-up experiment is scheduled.

For CPU reanalysis, the ignored local prediction/cache files identified in `LOCAL_CACHE_MANIFEST.json` are required. GitHub contains results and provenance, not raw data/model weights/prediction arrays. Reanalysis is not a new validation dataset or new model-selection opportunity.

Decision interpretation: the teacher-quality veto was conservatively implemented as requiring FULL9 not to be worse than calibrated F0 native/median. In this run FULL9 was better than both, so the interpretation does not affect any decision. A compression-only and B information-only recognize simple methods, not the proposed modules. All B neural arms are optimization-limited at the fixed 256-update endpoint.

Original CAL wall time was not separately instrumented. Reports identify its measured verification time explicitly rather than inventing an original timing. CPU-only baselines' GPU allocator peak is not their model memory requirement; CPU RAM was not separately profiled.
