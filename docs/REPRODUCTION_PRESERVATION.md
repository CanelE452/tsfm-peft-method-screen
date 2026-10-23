# Reproduction preservation and relocation

The 2026-10-23 preservation work archives existing evidence. It does not run new
experiments, revise historical FAIL/STOP decisions, or establish that retraining on
a different GPU will be bitwise identical. See `RESULTS_INDEX.md` for the original
completion, failure, repair and verification boundaries.

## Material that must survive local cleanup

Keep the Git history and all tracked code, protocols, configurations, execution
receipts, results, research records and papers. Preserve the following additional
local files in checksum-verified release assets:

| Include | Reason |
| --- | --- |
| `.cache/` apart from the exclusions below | Original prediction and target arrays, processed inputs, augmentation schedules, selected/intermediate model and optimizer states, numerical diagnostics, download receipts and internal model snapshots. Historical finalizers use these files directly. |
| `data/` | Exact raw and processed input bytes, including M5, Jena, ETT and Electricity sources. Source URLs and checksums alone do not preserve a local input. |
| `runs/` | Owned-state experiment checkpoints and original local outputs. |
| `external/icdm-2026-reproduction/` without its environment and Git internals | The original source-paper simulator, its source and RAF data used for parity checks. |
| `.reproduction-dependencies/m5dataset/` | Five source/data/baseline dependencies otherwise lost when the separate M5 project is removed. |
| Other substantive ignored or untracked files | Preserve evidence outside the standard directories as well as any local notes. |

The cache includes Chronos-Bolt-small revision
`772f3d25d38aec6d914c8949dab4462e2d46f5d8`, Chronos-T5-small revision
`a971ba21945c4f1796b17a91fe69214b5f4ad472`, and Moirai source/model material.
Archive internal Hugging Face snapshots with their referenced `blobs/`; copying
only snapshot symlinks would create an incomplete backup. Internal data symlinks
also refer to other experiments in the same cache and must remain resolvable.

Two model snapshots are outside this project and must be archived separately or
retrieved at their exact revision and checked against the recorded file hashes:

| Model | Revision | Original location |
| --- | --- | --- |
| `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | `~/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/<revision>` |
| `AutonLab/MOMENT-1-small` | `411e288267f82cce86296dbe4d6c8bc533cc162f` | `~/.cache/huggingface/hub/models--AutonLab--MOMENT-1-small/snapshots/<revision>` |

Some historical MOMENT receipts contain an absolute snapshot path. Restoring only
the model bytes does not automatically rewrite those receipts. Keep them unchanged;
use an explicit relocated snapshot mapping when reproducing on another machine.

## Rebuildable material excluded from the binary evidence archive

These exact paths are Python installations rather than unique experiment outputs:

- `.venv/`
- `.venv-channel/`
- `.cache/tsfm_three_candidate_setup_20260925/envs/`
- `external/icdm-2026-reproduction/src/.venv/`

Capture each environment's interpreter version, package names and versions, and
available dependency lockfiles before removing it. `requirements-lock.txt` is the
main environment lock; auxiliary environments have distinct package sets and must
not be represented by that lock alone. Environment reconstruction requires access
to the recorded package versions. Package names and versions do not preserve
unpublished local package patches; preserve source checkouts and record such
differences separately where present.

Generated `__pycache__/`, `.pytest_cache/`, `.pyc` files, nested `.git/` internals,
and stale process state such as `.cache/gpu.lock` and `.cache/supervisor.pid` are not
scientific evidence. Preserve source checkout identities before excluding nested
Git internals. Keep original execution logs and terminal error records.

The optional `.cache/nvidia-580.173.02/` libraries are specific to the original
machine's kernel/userspace mismatch. The original integrity audit hashes their
bytes, so retaining this small compatibility payload preserves that receipt. A
new machine should normally use its own working driver; these files are not a
replacement for a GPU driver installation.

## Minimal M5 dependency

The M5 source was commit `03043927c1ea89e8569ab4e9d0bfe0b6d041a47f`.
`reproduction/M5_DEPENDENCY_MANIFEST.json` records the five original files and
their SHA-256 checksums, totaling **328,652,917 bytes**:

- `common/data_split.py` provides the exact 1,913-day/28-day split and ID ordering.
- `data/sales_train_evaluation.csv` contains the demand observations and truth.
- `data/calendar.csv` contains calendar/SNAP/event covariates.
- `data/sell_prices.csv` is used by the full LightGBM covariate controls.
- `models/naive/pred_snaive_week.npy` is read by the original precheck.

This is a dependency extract for TSFM experiments, not preservation of the separate
M5 project's research. The sales CSV duplicates the existing TSFM raw input; an
archive may deduplicate identical content if restoration recreates both paths.
M5 raw inputs originated from the user's competition download. Other original
screen source locations are listed in `README.md`; additional experiments record
their input receipts under the respective `results/` directories.

## Inspect a relocated M5-dependent script without running it

The original scripts contain fixed machine paths. They remain byte-for-byte
unchanged because their hashes are part of historical evidence. A small launcher
translates the two known project/M5 roots in memory, validates all five dependency
hashes, compiles the script without executing it, and prints the original and
translated source hashes plus the explicit path map:

```bash
python scripts/run_preserved_m5_experiment.py \
  --check-loader experiments/service_axis_v2_20260922/decide.py
```

`--check-loader` imports only the preserved split module's definitions and checks
the resolved data directory; it does not call `get_split`, load the CSV data,
construct models, infer, train or score an experiment. NumPy and pandas must be
available in the chosen interpreter. Without this flag, inspection uses only the
Python standard library. Set `TSFM_M5_ROOT` or `--m5-root` to override the extracted
dependency location; that location must contain `DEPENDENCY_MANIFEST.json`.
The launcher also maps the original absolute Hugging Face root to
`.reproduction-dependencies/huggingface/` and, in execution mode, sets `HF_HOME`
and `HF_HUB_CACHE` to that preserved cache. `TSFM_HF_ROOT` or `--hf-root` can
override this location. Absolute paths embedded in separately loaded historical
JSON receipts are not rewritten by this launcher; their relocation needs an
explicit receipt mapping when replaying experiments that consume such paths.

Actual execution requires the additional explicit `--execute` flag. It retains
the target's original side effects, which can include training, inference,
report rewriting and overwriting existing files. Execute only in an isolated
restored copy after reviewing the target. This preservation work does not invoke
that mode. The launcher does not repair unrelated absolute paths, missing package
versions, driver differences or pre-existing historical reproducibility gaps.

## Inspect MOMENT receipt relocation without running a channel experiment

`scripts/run_preserved_experiment.py` adds an explicit runtime adapter for paths
returned by `json.load` and `json.loads`, including absolute-path dictionary keys
used by historical file-hash receipts. It preserves the original JSON and Python
files. This handles the MOMENT loader's explicit `receipt['snapshot']`, for which
setting `HF_HOME` alone is insufficient. Inspect the representative channel
finalizer and load only the restored configuration with its recorded environment:

```bash
.venv-channel/bin/python scripts/run_preserved_experiment.py \
  --check-model-config scripts/finalize_channel_basis_pilot.py
```

This command compiles the target without executing it, maps the original MOMENT
receipt to the preserved snapshot, imports `momentfm`, and parses the embedded T5
configuration. It does not construct a model, load weights, infer, score, train or
rewrite results. After rebuilding the channel environment, substitute that
environment's interpreter for `.venv-channel/bin/python`.

Like the M5 launcher, `--execute` explicitly enables the target's original side
effects and should be used only in an isolated restored copy. Arguments after
`--` go to the target. During execution the adapter maps only three exact known
path prefixes: the project root, global Hugging Face root and M5 root. It prints
the root map, original/transformed source hashes, original MOMENT receipt hash,
and observed JSON path substitutions. The adapter does not rewrite arbitrary
text references or paths assembled independently through `Path.home()`, and it
does not transform imported Python modules. Historical checks that hash a JSON
object after parsing may need a distinction between original receipt bytes and
the mapped runtime object: serializing the mapped object changes path strings.
The launcher therefore establishes path handling for the documented operations,
not a blanket guarantee that every historical finalizer runs unchanged.

## Verification scope

For a read-only numerical check after restoring the assets, use Python with NumPy:

```bash
python scripts/verify_preserved_predictions.py --root /path/to/tsfm-restored
```

This independently accumulates per-channel MSE/MAE with `math.fsum` and checks
the original scalar, channel and raw-unit metric receipts for 36 evaluation
records: all eight phase-balance evaluation records and 28 resumed channel-basis
records. The latter include four explicitly labeled posthoc coefficient swaps.
It checks 6,480 metric scalar cells, 16 additional original scalar receipts, and
8,847,360 finite saved target values, plus prediction file SHA-256, evaluation
origins and TRAIN standard deviations. The original phase-balance tolerance is
`1e-12` absolute/relative; the channel-basis tolerance is `1e-10`.

The validator imports no experiment/model code and loads no checkpoint. It prints
its report to stdout; optional `--output /new/report.json` creates a new file and
refuses to overwrite an existing report. It accepts relative cache paths and
maps the original absolute project prefix to the supplied restored root. The
original-side check is recorded in `reproduction/PREDICTION_REPLAY_20261023.json`.

A small existing CPU test subset checks metric weighting/missing targets, channel
alignment and normalization, RNG preservation, and immutable selection contracts:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_protocol.py tests/test_priority12_channel_data.py \
  tests/test_common.py::test_metric_independent_missing_channel_scaling \
  tests/test_common.py::test_immutable_selection
```

These six tests passed during preservation. They use NumPy/PyTorch/pytest on CPU
and do not fit a forecasting model or open a GPU experiment. Reconstructing
predictions from model weights and checking every historical experiment remain
outside these selected checks.

Asset checksums, per-file manifests and restored-file verification establish that
the archived bytes survived upload and download. Loading a split module and
compiling a translated script establish only dependency and path portability for
those operations. Neither substitutes for replaying every historical finalizer
or retraining every run. Existing execution records remain the evidence for the
original numerical checks, including their documented limitations and errors.

## Build the archived AdaPTS source without its nested Git database

The original AdaPTS environment contained distribution `AdaPTS` version
`0.1.dev370+g8bf57c7ee`, installed in editable mode from the preserved source.
Its unchanged `pyproject.toml` obtains the version from `setuptools_scm` and has
no fallback. Removing the nested `.git/` therefore requires supplying the
recorded version explicitly when rebuilding the package:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ADAPTS='0.1.dev370+g8bf57c7ee' \
  uv pip install --python /path/to/rebuilt-adapts-env/bin/python \
  --no-deps --editable .cache/tsfm_three_candidate_setup_20260925/sources/AdaPTS
```

Reconstruct the scientific dependencies from the captured AdaPTS environment
separately; `--no-deps` deliberately avoids resolving or upgrading those
dependencies during this local-source installation. The normal isolated build
will need the source's small build-system requirements. For a separately
provisioned build environment, the tested tools were setuptools `84.0.0`, wheel
`0.48.0`, setuptools-scm `10.3.4`, vcs-versioning `2.6.0` and packaging `26.3`.

The package-specific variable and uppercase distribution-name normalization are
documented by [setuptools-scm](https://setuptools-scm.readthedocs.io/latest/config/#environment-variables).
This sets build metadata while preserving the archived source bytes; it does not
fabricate a Git checkout or infer a new version from the surrounding project.

`reproduction/ADAPTS_BUILD_CHECK_20261023.json` records a check on a temporary
source copy with both `.git/` and cached `*.egg-info` removed. Metadata generation
without the override failed as expected. With the exact recorded version, a
45,120-byte wheel built successfully; its metadata reported the original version
and it contained all 14 package Python source files. Only the five small
build-tool packages were installed in a temporary environment. Scientific/model
dependencies were not installed, no model was loaded, and no fit was run.
The archived source and binary archive were not modified. This verifies package
building and version metadata, not an AdaPTS experiment or a byte-identical wheel
across build timestamps and platforms.
