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

## Freshness v2: latest additional experiment

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
