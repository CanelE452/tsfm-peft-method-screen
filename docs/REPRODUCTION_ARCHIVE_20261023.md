# Reproduction data and result archive — 2026-10-23

The user requested preservation of this project and Forecast Revision PEFT before
removing the local time-series workspaces. Existing scientific decisions and
sealed experiment source files are preserved. No new fits or research continuation
are part of this work.

The source, protocols, configurations, tracked results and lockfiles remain in
`CanelE452/tsfm-peft-method-screen`. The accompanying GitHub Release is
[`reproducibility-archive-20261023`](https://github.com/CanelE452/tsfm-peft-method-screen/releases/tag/reproducibility-archive-20261023).
Large input arrays, predictions, checkpoints and other originally ignored evidence
are split release assets rather than large Git blobs.

## Preserved files

`reproducibility/artifact-manifest.jsonl` records **28,613 artifacts** and
**73,310,977,615 logical file bytes** before compression. Each regular file has an
exact SHA-256 and byte count; symlinks retain their original target in the inventory.
The summary records the precise include and exclude rules.

The payload contains experiment caches, exact raw/processed data, run states,
the source-paper checkout and data, five required M5 dependency files, and the
Chronos-2 and MOMENT model snapshots formerly stored in the user's global cache.
Five substantive ignored result/log files outside the normal cache directories
are included explicitly. Internal Bolt/T5/Moirai sources and snapshots are retained
with their blob targets. Original NVIDIA compatibility libraries are preserved
for receipt checking. See [preservation details](REPRODUCTION_PRESERVATION.md).

Python environments, nested Git databases, compiled caches and small stale
lock/PID state are excluded. `reproducibility/environment-capture.json` and the
five environment freezes preserve the interpreter/package versions; main source
requirements and auxiliary environment installation records remain in Git.
`reproducibility/nested-source-provenance.json` records the six nested checkout
identities and local changes before their Git internals are removed.

## Restore

Use a machine with enough disk space for the downloaded compressed parts and the
73.31GB restored payload. Install Git, GitHub CLI, zstd, and Python 3.11 or newer.
Authenticate GitHub CLI when needed for repository access. The release tag fixes the
source version; later documentation-only verification commits may also be on main.

```bash
gh repo clone CanelE452/tsfm-peft-method-screen tsfm-restored
cd tsfm-restored
git checkout reproducibility-archive-20261023
gh release download reproducibility-archive-20261023 \
  --repo CanelE452/tsfm-peft-method-screen \
  --pattern 'tsfm-reproduction.tar.zst.part*' \
  --pattern artifact-manifest.jsonl --pattern SHA256SUMS \
  --dir ../tsfm-assets
(cd ../tsfm-assets && sha256sum -c SHA256SUMS)
cmp reproducibility/artifact-manifest.jsonl ../tsfm-assets/artifact-manifest.jsonl
python scripts/reproduction_archive.py verify \
  --manifest reproducibility/artifact-manifest.jsonl --parts ../tsfm-assets
python scripts/reproduction_archive.py restore \
  --manifest reproducibility/artifact-manifest.jsonl --parts ../tsfm-assets --root .
```

Restoration refuses to overwrite existing artifact paths. It checks every file,
preserves hardlinks, and relocates historical absolute symlinks into this checkout.
Keep the parts until restoration succeeds. The archive tool itself uses only the
standard library and zstd; it does not load pickle/model objects or run experiments.

Rebuild environments from the recorded Python versions and package sets. The
captured main environment has `transformers==5.17.0`, while the historical
`requirements-lock.txt` still records `4.57.6`; they are deliberately preserved as
different records. For the environment present at archival time, use the captured
portable requirements rather than silently substituting the older lock:

```bash
uv venv --python 3.11.15
uv pip sync --python .venv/bin/python reproducibility/requirements/main.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.reproduction-dependencies/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
```

Auxiliary environments have separate portable captures in
`reproducibility/requirements/`: `channel.txt` (Python 3.11.15), `adapts.txt`
(3.11.15), `msft.txt` (3.11.15), and `source-paper.txt` (3.12.13). Run their
installation from the project root so the preserved editable source checkouts
resolve. Do not replace them with the main requirements. Original, unmodified
freeze output remains under `reproducibility/environments/`.
Package retrieval, a compatible CUDA stack and suitable hardware
remain environment requirements. The recorded interpreter versions, original
source bytes and saved predictions support replay, but do not promise bitwise
identical retraining on a different driver or GPU.

Some sealed historical scripts and receipts contain original absolute paths.
Use the explicit portable launchers and root mappings documented in
`REPRODUCTION_PRESERVATION.md`; do not edit sealed source or receipt bytes merely
to satisfy an old checksum. The portable launchers translate the documented project and receipt paths without changing the preserved source files. The default M5 launcher verifies and inspects only:

```bash
.venv/bin/python scripts/run_preserved_m5_experiment.py --check-loader \
  experiments/service_axis_v2_20260922/decide.py
```

## Verification boundaries

The archive engine was checked with a multipart fixture containing regular files,
hardlinks, relative links, and absolute file/directory symlinks relocated to a new
root. Full payload verification checks all inventory members against the compressed
archive. Upload verification compares every release asset's server-computed
SHA-256 with its local checksum. The final upload/download/replay receipts are
attached to the release and retained under `reproducibility/` on main.

Byte preservation and selected replay checks do not mean every historical
finalizer or fit has been rerun. Preserve the documented FAIL/STOP decisions,
unexecuted stages and original verification limitations.
