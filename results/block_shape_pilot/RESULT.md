# Block-conditioned shape adaptation: first pilot

**STOP** on the prospectively fixed research continuation gate. 48/48 fits, two datasets, two seeds, 5,760 proposed iterations. This is not a publication or novelty verdict.

Execution commit: f4fcf52c9428d2c3ae9f977abd810ce74231f3dc. All choices were sealed before heldout feature extraction and scoring. New chronological periods on existing source series; not external dataset replication.

| Dataset | Seed | Arm | Selected step | Scaled 2-pinball ↓ | 80% coverage | Width |
|---|---:|---|---:|---:|---:|---:|
| ettm2 | shared | F0 | — | 0.27264369 | 0.7101 | 5.2000 |
| ettm2 | shared | calibration | — | 0.27549542 | 0.7990 | 6.2400 |
| ettm2 | 30000 | lora | 0 | 0.27264369 | 0.7101 | 5.2000 |
| ettm2 | 30000 | split | 120 | 0.27378809 | 0.7083 | 5.1918 |
| ettm2 | 30000 | center | 60 | 0.27261799 | 0.7132 | 5.2000 |
| ettm2 | 30000 | anchor | 120 | 0.27381689 | 0.7093 | 5.1970 |
| ettm2 | 30000 | pooled | 120 | 0.27387630 | 0.7101 | 5.2202 |
| ettm2 | 30000 | block | 120 | 0.27388384 | 0.7090 | 5.2081 |
| ettm2 | 30000 | blend | 120 | 0.27378809 | 0.7083 | 5.1918 |
| ettm2 | 30001 | lora | 30 | 0.27115855 | 0.7135 | 5.1877 |
| ettm2 | 30001 | split | 30 | 0.27192670 | 0.7044 | 5.0709 |
| ettm2 | 30001 | center | 30 | 0.27209456 | 0.7139 | 5.2000 |
| ettm2 | 30001 | anchor | 30 | 0.27193410 | 0.7078 | 5.1079 |
| ettm2 | 30001 | pooled | 30 | 0.27202520 | 0.7137 | 5.2032 |
| ettm2 | 30001 | block | 30 | 0.27210740 | 0.7134 | 5.1972 |
| ettm2 | 30001 | blend | 30 | 0.27115855 | 0.7135 | 5.1877 |
| electricity | shared | F0 | — | 0.17712342 | 0.7996 | 31.6218 |
| electricity | shared | calibration | — | 0.17712342 | 0.7996 | 31.6218 |
| electricity | 30000 | lora | 120 | 0.17702609 | 0.8063 | 31.7417 |
| electricity | 30000 | split | 30 | 0.17705776 | 0.8018 | 31.6400 |
| electricity | 30000 | center | 30 | 0.17706885 | 0.8014 | 31.6218 |
| electricity | 30000 | anchor | 30 | 0.17705792 | 0.8018 | 31.6397 |
| electricity | 30000 | pooled | 30 | 0.17705861 | 0.8021 | 31.6510 |
| electricity | 30000 | block | 30 | 0.17706314 | 0.8022 | 31.6487 |
| electricity | 30000 | blend | 120 | 0.17702609 | 0.8063 | 31.7417 |
| electricity | 30001 | lora | 120 | 0.17612318 | 0.8066 | 32.8644 |
| electricity | 30001 | split | 30 | 0.17703702 | 0.8031 | 31.6495 |
| electricity | 30001 | center | 30 | 0.17704871 | 0.8027 | 31.6218 |
| electricity | 30001 | anchor | 30 | 0.17703725 | 0.8031 | 31.6488 |
| electricity | 30001 | pooled | 30 | 0.17703591 | 0.8032 | 31.6654 |
| electricity | 30001 | block | 30 | 0.17703701 | 0.8029 | 31.6775 |
| electricity | 30001 | blend | 120 | 0.17612318 | 0.8066 | 32.8644 |

## What this pilot established

The update mechanism was active: every selected candidate had a nonzero shape change. It nevertheless failed the primary criterion on both datasets. This is a learning-quality failure under the fixed protocol, not an implementation block or GPU capacity failure.
Conservative LoRA has lower seed-mean primary loss than the block candidate on both datasets. Relative to pooled acceptance, the block variability penalty changes heldout loss very little and provides no gain here. Preventing some training shape steps was therefore insufficient to improve generalization.
The ETTm2 scalar calibration baseline illustrates a tradeoff: 80% coverage approaches its nominal target, but its primary loss worsens. Coverage alone would give a misleading success signal. On Electricity, F0 already has coverage near 80%.
These findings reject this particular first rule as a current lead; they do not establish that every form of uncertainty-aware PEFT must fail.

## Fixed decision

- ettm2: block / strongest baseline seed-mean primary ratio **1.004073** (required <=0.995), coverage-error difference +0.002197. Checks: {'primary': False, 'f0_safety': True, 'coverage': True, 'active_shape': True}.
  Seed 30000: strongest comparator center; selected block step 120, mean absolute log-gap displacement 0.0126571.
  Seed 30001: strongest comparator lora; selected block step 30, mean absolute log-gap displacement 0.00240333.
- electricity: block / strongest baseline seed-mean primary ratio **1.002693** (required <=0.995), coverage-error difference -0.003906. Checks: {'primary': False, 'f0_safety': True, 'coverage': True, 'active_shape': True}.
  Seed 30000: strongest comparator lora; selected block step 30, mean absolute log-gap displacement 0.00429391.
  Seed 30001: strongest comparator lora; selected block step 30, mean absolute log-gap displacement 0.00277716.

The strongest comparator is the predeclared conservative E-oracle across separately V-selected baselines. No candidate checkpoint or recipe was selected by E.

## Mechanism and resources

| Dataset | Seed | Rule | Recipe | Accepted shape proposals /120 | Best step | Peak allocated MiB | Median step ms |
|---|---:|---|---:|---:|---:|---:|---:|
| ettm2 | 30000 | pooled | 0 | 113 | 120 | 90.3 | 6.31 |
| ettm2 | 30000 | pooled | 1 | 99 | 60 | 90.3 | 6.11 |
| ettm2 | 30000 | block | 0 | 82 | 120 | 90.3 | 6.18 |
| ettm2 | 30000 | block | 1 | 46 | 60 | 90.3 | 6.06 |
| ettm2 | 30001 | pooled | 0 | 120 | 30 | 90.3 | 7.34 |
| ettm2 | 30001 | pooled | 1 | 84 | 30 | 90.3 | 7.15 |
| ettm2 | 30001 | block | 0 | 66 | 30 | 90.3 | 7.03 |
| ettm2 | 30001 | block | 1 | 29 | 30 | 90.3 | 7.53 |
| electricity | 30000 | pooled | 0 | 119 | 30 | 90.3 | 6.13 |
| electricity | 30000 | pooled | 1 | 115 | 0 | 90.3 | 6.49 |
| electricity | 30000 | block | 0 | 115 | 30 | 90.3 | 6.08 |
| electricity | 30000 | block | 1 | 92 | 0 | 90.3 | 6.46 |
| electricity | 30001 | pooled | 0 | 119 | 30 | 90.3 | 6.01 |
| electricity | 30001 | pooled | 1 | 106 | 0 | 90.3 | 5.76 |
| electricity | 30001 | block | 0 | 113 | 30 | 90.3 | 5.80 |
| electricity | 30001 | block | 1 | 76 | 0 | 90.3 | 5.96 |

GPU monitor: 109 samples; minimum observed active free VRAM 8168 MiB; samples with external compute 0. Model work pauses at step boundaries when other compute appears. Worker wall time: 294.2s.

Frozen feature extraction total: 7.1s. Cached adapter steps exclude repeated backbone computation. Full training-step resources remain in fits.json; this is not an equal-end-to-end-cost comparison or memory-efficiency claim.

## Interpretation limits

The architecture overlaps established monotonic quantile calibration and output-adapter methods. The tested incremental hypothesis is shape-only finite-step acceptance penalized by temporal-block variability. This is not a confidence interval or a distribution-free calibration guarantee.
Only two existing source datasets and one later split are evaluated. Two seeds address optimizer variation, not independent sampling. Four original channels, fixed learning-rate grids and one anchor coefficient limit the comparison. No official AdaPTS or delta-Adapter reproduction is claimed. A positive continuation decision would still require stronger baselines, additional sources and a full novelty assessment.

## Verification

224 saved prediction caches replay the primary loss with maximum independent error 8.33e-17; every reported cached metric replays exactly. Selection seals, checkpoint hashes, blend/calibration choices, and 215 historical files verified. Gate decisions replay from recorded block changes. No additional tuning or experiment expansion.

Protocol and primary literature links: [fixed protocol](../../docs/BLOCK_SHAPE_PROTOCOL.md).
