# Candidate01 v2: asynchronous observation / clean-centered affine LoRA

## Authorization and scope

The user explicitly requested execution (“해줘”) after discussing the proposed maximum12-fit experiment in [their design review](USER_DESIGN_REVIEW_2026_09_13.md). This is one revised development candidate, not Round2 or a relabeling of the failed original gate. Original candidates, recovered Candidate05, data and selection artifacts stay unchanged. If v2 fails, stop this branch without extending LR, steps, thresholds, seeds or datasets.

## Method and controls

Pinned Chronos-2, 336 hourly context / 48 horizon, original 21 native quantiles, frozen backbone and head; all 96 attention q/k/v/o matrices receive rank8 alpha16 LoRA, zero B initialization. No new selector, backbone, output head or freezing policy.

- Standard: `W0 h + 2 B A h` with exactly the same corruption augmentation.
- Feature: existing additive `W0 h + 2 B (A h + C r + b)`; no alteration to its strong baseline implementation.
- Affine v2: `W0 h + 2 B [(1 + .5*(tanh(Wa r)-tanh(Wa r_clean))) * A h + Wb*(r-r_clean)]`.

`r=[observed_mask, age/336, trailing24 availability]`, patch-averaged for observed-context tokens. REG/future tokens use the last context state only. `r_clean=[1,0,1]`. The multiplicative factor lies in (0,2); additive shift is unrestricted and logged. At clean reference, the additional scale/shift is exactly zero. Shared A/B still adapt, so this guarantees the *standard LoRA form*, not preservation of F0's clean performance. The affine linear shift bias would cancel under centering and is omitted. Scale/shift weights start at zero. Extra parameters: Feature3072 (96×(3×8+8)), Affine4608 (96×3×16); shared LoRA1,179,648. The parameter difference is reported, not concealed as exact parameter matching.

All arms receive the same native relative-time encodings and actual observation masks through Chronos's context preparation. Feature/Affine receive identical r, which is derived solely from those causal masks. Standard has the same underlying information but no added state branch.

## Asynchronous observation process

The independent module `candidates/freshness_v2.py` is the source of the sealed rules. For each known origin and declared variant, SHA256 seeds a channel permutation and refresh offsets; neither target values nor experiment seed influences corruption. Refresh masks use **absolute hourly indices**, periods assigned separately to channels, and channel-specific offsets sampled in `[0,period)`. Block intervals end at different lags relative to context end and have different lengths. No forward fill. Existing natural missingness remains missing. No target is corrupted; no index at or after origin is accessed to create context/state.

| Variant | Four periods (hours) | Four block lengths | End lags from context end |
| --- | --- | --- | --- |
| refresh_train | 2,3,4,6 | — | — |
| block_train | — | 3,6,9,12 | 0,3,6,9 |
| combined_train | 2,3,4,6 | 3,6,9,12 | 0,3,6,9 |
| refresh_shift1 | 5,7,9,11 | — | — |
| refresh_shift2 | 7,9,11,13 | — | — |
| block_shift1 | — | 12,18,24,30 | 0,6,12,18 |
| block_shift2 | — | 18,24,30,36 | 0,9,18,27 |

Within each row the declared arrays share the deterministic channel permutation. “Unseen” denotes the held-out corruption regimes, not a claim that every individual length or channel assignment has never appeared. There is no claim that synthetic asynchrony exactly reproduces deployed sensor failures.

## Data, gate and learning

Reuse the sealed Jena hourly4-channel split and original train-only standard deviations. Gate17 origins, train28, V12, E30. Dataset raw/processed hashes are in the original manifest and copied into the v2 contract. The historical E has already been observed during development; this is explicitly **reused development E**, not fresh holdout evidence.

Before fitting, score frozen F0 on gate origins under clean + the three training corruptions. Continue only if their mean degradation is ≥2% and at least two corruptions degrade ≥1%. No evaluation-origin gate or data-dependent corruption choice.

Exactly `3 arms × 2 recipes × 2 seeds = 12 fits`, seeds30000/30001, 360 updates each, effective batch8 series (two four-channel origins), AdamW, weight_decay0, clipping1, dropout/TF32off. Same per-seed origin schedule in all six fits. Cycle clean/refresh_train/block_train/combined_train: 25% each. Both recipes keep a common LR for A/B; this is not LoRA+.

| Recipe | LoRA LR | Conditional LR (Feature and Affine equally) |
| --- | --- | --- |
| R0 | 3e-5 | 3e-4 |
| R1 | 1e-4 | 1e-3 |

Checkpoints0,4,8,15,30,60,120,180,240,360. V objective is the equal mean of clean and three training-corruption losses. Per arm/seed select lowest V loss, earliest step, recipe id. All **12 fits and all six selections** finish and are sealed before any v2 E access. Step0 remains eligible. The source revision, source hashes, config and schedules are recorded before training. No E-based checkpoint/recipe adjustment.

## Functionality and integrity

CPU tests check channel masks differ, observed values/times remain unchanged, causal state cannot change with later values, clean-centering exactness, scale bounds, both affine gradients and native identity. The first eight scheduled updates of every fit serve as train-only smoke, counted in 360: optimizer membership, finite/nonzero conditional gradients and update norms, and forecast change when replacing state by clean reference on the **same corrupted context**. Initial conditional gradient may be zero because B starts at zero; continued zero at update8 is an integrity failure. No separate exploratory GPU training runs.

Record branch gradient and update norms, conditional/task-driven LoRA gradient ratio, train state replacement forecast difference, affine multiplier range and shift range at small checkpoints. There is no regularizer here: do not interpret the branch-vs-LoRA gradient ratio as a regularizer/task gradient ratio. Verify frozen parameters, step0 F0 identity, saved-checkpoint replay, selection seals and independently recomputed saved-prediction metrics. Diagnostic forwards never update parameters or select checkpoints. An integrity failure stops the fixed candidate and preserves receipts.

## Decision before E

Primary per seed: equal mean E loss across four shifted corruption regimes. For each seed compare Affine against the better of Standard and Feature under this same primary metric. `gain =100*(baseline-proposed)/F0`; this is **%F0**, not percent relative to baseline. Require positive gain on both seeds, two-seed mean gain ≥1%F0, and clean degradation ≤0.5%F0 on **each seed** against the better clean baseline. Both comparisons must use the prespecified V-selected models. Record all per-regime metrics, including clean, even when they fail.

PASS requires all conditions; WEAK means positive gains on both seeds but a threshold miss; otherwise FAIL. Two-seed consistency is a stopping rule, not statistical significance. No weighted cross-candidate ranking or automatic scientific winner. Only a PASS warrants review for a subsequently frozen, new-source experiment; none is launched here.

## Prior knowledge boundary

[FiLM (AAAI2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11671) already introduces conditioning-based feature-wise affine transformations. This v2 does not claim that conditional affine modulation is new. Its narrow question is whether causal observation state and clean centering add practical value over the strong additive control in frozen-TSFM adaptation.

[LoRA+ (ICML2024)](https://proceedings.mlr.press/v235/hayou24a.html) studies unequal learning rates for A/B. That result does not establish the cause of any previous gate/GRU failure; our conditional-group recipes are a fixed engineering choice, not a LoRA+ reproduction or guaranteed cure. Both primary abstracts were checked before this execution. The earlier [bounded novelty review](NOVELTY_BOUNDARY.md) remains limited; a pilot PASS would not establish publishable novelty.

## Reproduction

```bash
scripts/with_cuda.sh .venv/bin/python -m pytest -q
scripts/with_cuda.sh .venv/bin/python scripts/run_candidate_01_v2.py
```

Start only once from a clean committed checkout. Dedicated paths `results/candidate_01_v2` and ignored `.cache/candidate_01_v2`. The supervisor locks the single GPU and waits for30 idle seconds; no external job is stopped. Each fit uses the existing2-hour guard; whole candidate4-hour supervisor. Large data/models/checkpoints/prediction arrays remain untracked.
