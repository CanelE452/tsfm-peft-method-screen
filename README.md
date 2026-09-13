# Seven-candidate TSFM PEFT screen

Independent Chronos-2 development screen. Round 0 and Round 1 only; Round 2 requires subsequent user review. See [protocol](docs/MASTER_SCREEN_PROTOCOL.md) and [user specification](docs/USER_PROTOCOL.md).

No prior experiment outputs are used as new evidence. Raw data and cached model weights may be reused with provenance hashes. All source is local to this repository.

## Reproduce

```bash
uv venv --python 3.11
uv pip sync --python .venv/bin/python requirements-lock.txt
# Put the documented raw files under data/raw (see source manifests).
scripts/with_cuda.sh .venv/bin/python scripts/prepare_common_data.py
scripts/with_cuda.sh .venv/bin/python scripts/audit_environment.py
scripts/with_cuda.sh .venv/bin/python scripts/verify_all.py --preflight
scripts/with_cuda.sh .venv/bin/python scripts/screen_all.py
scripts/with_cuda.sh .venv/bin/python scripts/render_results.py
scripts/with_cuda.sh .venv/bin/python scripts/verify_all.py
```

The supervisor waits for 30 seconds without an external GPU compute process before starting. It never terminates another project's job. Each candidate has a 4-hour supervisor timeout; each fit/stream has a 2-hour guard. Terminal candidate directories and selection seals cannot be overwritten by the runner.

Raw inputs: Jena `mpi_roof_2021b.csv` from https://weather.bgc-jena.mpg.de/mpi_roof_2021b.zip; M5 `sales_train_evaluation.csv` from the user's existing competition download; ETTm2 from https://github.com/zhouhaoyi/ETDataset; Electricity from https://github.com/laiguokun/multivariate-time-series-data. Exact local SHA256 receipts are committed. Processed arrays are recreated independently. Jena is hourly top-of-hour; M5 daily; ETTm2 native15-minute; Electricity hourly.

Model: `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`. Cache this revision using Hugging Face before running; the experiment loader is offline-only. Model SHA256 is in the common integrity receipt.

On this machine only, kernel driver580.173.02 and globally installed userspace580.178.04 disagree. Matching official580.173.02 libraries were extracted (not installed) into ignored `.cache/nvidia-580.173.02`; the wrapper uses them only when the kernel version matches. Other machines can use their normal working CUDA installation. No system driver or unrelated repository is modified.

See [novelty boundaries](docs/NOVELTY_BOUNDARY.md). Candidate06 is stopped for direct overlap of the central neural-copula method; its raw data preparation is not a fitted result.

## Research reassessment after the latest STOP

A [Korean deep reassessment](research/peft_rethink_2026_09_13/REPORT.md) audits the failed recommendations and primary literature, then proposes a conditional next problem: representing temporally correlated uncertainty in future covariates with a single-pass PEFT model. It includes exact illustrative counterexamples and a staged comparison plan. Ordinary covariate adapters and ensemble distillation are established prior work; the new method, novelty and predictive benefit are **not yet validated**.

The recommendation is to verify that a legal-input multi-scenario reference actually improves prediction before implementing another candidate. This reassessment adds **zero fits and zero GPU jobs**. Existing STOP/FAIL outcomes remain unchanged. Supporting chart data and sources are provided as CSV files alongside the report.

## Block-conditioned distribution adaptation: completed 48-fit pilot

**48/48 fits completed; STOP** on the preregistered continuation gate. Two datasets × two seeds × six arms × two recipes, 5,760 proposed training iterations. On later chronological evaluation periods, the candidate's seed-mean primary loss is 0.407% worse on ETTm2 and 0.269% worse on Electricity than the predeclared conservative strongest-baseline comparator. All four selected candidate models have nonzero shape adaptation; safety and coverage conditions pass, but neither dataset meets the required 0.5% primary improvement.

The block rule selectively accepts shape updates and restores rejected Adam states, but shows no additional gain over pooled acceptance in this pilot. Conservative LoRA has lower seed-mean primary loss on both datasets. This is a completed learning comparison, not a GPU or implementation failure. GPU monitoring recorded 109 samples and at least 8,168MiB observed active free VRAM, with no external compute during active work. Startup waited for transient external work to leave.

All 39 tests pass; 224 prediction caches replay independently, all cached metrics replay exactly, and 215 historical result files are unchanged. The new heldout periods belong to existing source series, not external datasets. Architecture ingredients overlap prior work; no publication or novelty PASS is claimed. No further tuning followed this result.

See the [result and comparisons](results/block_shape_pilot/RESULT.md), [fixed protocol and literature boundaries](docs/BLOCK_SHAPE_PROTOCOL.md), and [verification](results/block_shape_pilot/verification.json).

## Frozen-past forecast-token adaptation: latest actual-learning pilot

**16/16 fits completed; FAIL on the fixed memory/quality gate.** Two datasets × four arms × two arm-appropriate recipes × first seed,3,840 actual optimizer updates (plus8 discarded smoke updates). The query candidate improves development primary loss by0.519% versus Standard on ETTm2, but is0.326% worse on Electricity. Its real optimizer-step peak is987.7MiB versus770.9MiB for Standard LoRA+BF16+checkpoint: **28.13% higher**, failing the required20% reduction. Selected query checkpoints are180/30, so this is actual adaptation evidence rather than an initial-state selection. It is one-seed development evidence, not an independent holdout or confirmed novel method.

GPU occupancy was monitored throughout:127 samples, minimum observed active free memory8,034MiB, maximum device use1,819MiB, no external compute PID and no contention pauses. A graphics-only startup wait was preserved and corrected before model execution. All arms have1,179,648 trainable parameters. Full cache/frozen forward/branch/clipping/optimizer-step costs count; the query branch was not checkpointed in this fixed implementation. See the [report](results/forecast_query_pilot/RESULT.md), [protocol](docs/FORECAST_QUERY_PROTOCOL.md), [GPU timeline](results/forecast_query_pilot/gpu_monitor.json), and [verification](results/forecast_query_pilot/verification.json).

```bash
scripts/with_cuda.sh .venv/bin/python scripts/finalize_forecast_query_pilot.py --verify-only
```

All106 saved V/E prediction caches replay, all8 validation selections were sealed before E arrays opened, and all prior results remain unchanged. Cumulative completed fits:62; stream attempts:9 (8 complete,1 historical abort). No second-seed or fixed-memory batch expansion followed this failed gate.

## Local residual-corrected LoRA: completed diagnostic

**STOP before learning.** Custom local backward preserves exact forward, B-gradient and input propagation; keeping every detail reproduces A-gradient to1.51e-7 relative error. At context4096 the sampled residual candidate reduces peak by9.02–9.06%, but A-gradient RMS error is12.94% on ETTm2 and36.33% on Electricity. Generic local FP16 reduces peak by10.98–11.02% with0.00618–0.01098% A-error. BF16 autocast plus checkpointing uses735–736MiB and takes0.879–0.888× FP32 standard forward/backward time, dominating the proposed candidate's memory/time in this setup.

The diagnosis completed32 cases,216 backwards and128 stochastic gradient-cache replays, with **zero optimizer updates and zero new fits**. Learning was authorized conditionally (maximum16 fits), but the fixed prerequisite failed. These are fixed-state gradient measurements, not forecasting-quality results. All arms retain FP32 AdamW moment buffers; no optimizer-step peak was measured. CARE/PRAC controls are explicitly scoped primitives, not official reproductions or matched-byte claims. See the [report](results/local_backward_feasibility/RESULT.md), [fixed protocol](docs/LOCAL_BACKWARD_PROTOCOL.md), and [verification](results/local_backward_feasibility/verification.json).

```bash
scripts/with_cuda.sh .venv/bin/python scripts/finalize_local_backward.py --verify-only
```

## Memory-efficient PEFT: completed global-compression feasibility study

**Bottleneck confirmed; temporal pair-mean primitive STOP.** At context4096, generic FP16 saved activations reduce measured forward/backward peak by32.57% with0.0895–0.1299% full-gradient relative error. Matched-byte INT8 reduces peak by42.41% with0.8968–1.0392% error; temporal pair means have66.24–77.10% error at essentially the same peak. Exact block checkpointing reduces peak by61.45% with exact gradients, at1.41–1.42× step time. These fixed-state gradient diagnostics are not forecasting-accuracy results, and Stage-B peaks omit optimizer states for all arms.

The two-dataset diagnostic completed8 warmup updates,8 context profiles and60 measured comparison backwards. No learning pilot or new V/E evaluation followed the failed temporal gate. One interrupted implementation attempt is preserved separately. Historical fits and results remain unchanged. See the [full report](results/memory_feasibility/RESULT.md), [fixed protocol](docs/MEMORY_FEASIBILITY_PROTOCOL.md), and [verification receipt](results/memory_feasibility/verification.json).

```bash
scripts/with_cuda.sh .venv/bin/python scripts/finalize_memory_feasibility.py --verify-only
```

## Freshness v2: completed additional experiment

The authorized asynchronous Freshness v2 experiment completed **12/12 fits** (3 arms × 2 recipes × 2 seeds), 4,320 updates. **FAIL**: Affine loses to Standard on seed30000 (−0.094734%F0) and Feature on seed30001 (−0.246892%F0); the two-seed mean gain is −0.170813%F0. Seed30001 clean degradation is 0.521507%F0, exceeding the 0.5% limit. The conditional scale/shift branches were active and trained. No further tuning or new-source experiment was launched.

See the [v2 report](results/candidate_01_v2/RESULT.md), [two-seed summary](results/screening_summary/freshness_v2_review.md), and [prespecified protocol](docs/CANDIDATE_01_V2.md). The v2 execution revision is `ebb1ee89ee64266fe1eebed41d0b6017f68250c3`. It reuses development E and is not independent holdout evidence. All original results remain unchanged. Cumulative work: 46 standard fits + 9 stream attempts (8 complete, 1 historical abort).

```bash
scripts/with_cuda.sh .venv/bin/python scripts/finalize_candidate_01_v2.py
scripts/with_cuda.sh .venv/bin/python scripts/verify_all.py
```

Verification requires the local ignored prediction/checkpoint caches. It checks the original screen, the Candidate05 recovery and the v2 experiment separately.

## Original seven-candidate outcome

The user-authorized Candidate05 recovery completed all five streams. Maturity-PEFT is **FAIL**: issued scaled 2-pinball 0.655929 versus TAFAS-like 0.630834, a gain of −4.2982% F0; adaptation overhead was +103.676% (limit 20%). F0 itself remains stronger at 0.583857. All seven candidates now have terminal decisions: six completed comparisons and Candidate06 stopped for novelty collision. No PASS and no Round2.

See the [latest seven-candidate review](results/screening_summary/latest_review.md) and [Candidate05 recovery report](results/candidate_05_repaired/RESULT.md). The recovery used the original fixed recipe and preserved all historical results. The three previously completed prediction arrays reproduce exactly. Cumulative work is 34 standard fits and 9 stream attempts (8 completed, 1 historical abort), including 5 newly authorized recovery streams.

Recovery verification and report regeneration (requires local prediction caches):

```bash
scripts/with_cuda.sh .venv/bin/python scripts/finalize_candidate_05_recovery.py
```

The same development evaluation origins were reused for implementation recovery; this is not independent holdout evidence. The recovery execution commit is recorded in its output directory.

## Historical screen outcome and code version

The completed screen produced 34 fits and 4 attempted streams (3 complete, 1 aborted). There were no PASS candidates; Round2 was not executed. See [ranking](results/screening_summary/ranking.md) and [all candidate metrics](results/screening_summary/all_candidates.csv).

Experiment outputs correspond to execution commit `4f0854b27db6950d8e9e7ecb70355c5388ca4fe7`. The original screen was followed by a [partial-label numerical repair](docs/POST_SCREEN_REPAIR.md), initially validated with regression tests and a train-only GPU backward smoke. Its historical Candidate05 record remains IMPLEMENTATION_BLOCKED. The subsequent authorized recovery above provides the repaired pilot result in a separate directory. Source verification checks the historical commit against each immutable selection contract.

To inspect the existing results, run `render_results.py` and `verify_all.py` through the wrapper. Final verification requires the local ignored prediction caches; these are not included in Git. The setup commands above describe a fresh environment, not authorization to overwrite terminal results or relabel them as runs of the repaired implementation.
