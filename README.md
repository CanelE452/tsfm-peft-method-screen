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
