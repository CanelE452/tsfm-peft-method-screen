# Block-conditioned shape updates: fixed first learning pilot

User authorization: September 13, 2026, “그러면 너가 말한거 해볼수있어?” after the discussion of developing a PEFT method that separates predictive center and spread. This authorizes this new bounded project stage. Historical FAIL/STOP results and their gates remain unchanged.

## Question and status

Does a temporal-block variability penalty on proposed distribution-shape updates improve probabilistic adaptation beyond a small unconstrained adapter, fixed spread, output anchoring, pooled training-loss acceptance, and conservative LoRA?

This is a concrete **candidate learning rule**, not established novelty, a calibrated statistical test, or a paper-ready method. A positive pilot only authorizes considering further research; it cannot establish publication success.

## Model and proposed update

Pinned Chronos-2, context 4096, horizon 48, first four existing channels. Its encoder and original decoder are frozen. We cache its final three hidden tokens and its sorted raw quantiles in context-normalized units. Features are standardized with nonlearned layer normalization. Labels never enter the features. Cache creation cost is measured separately.

Two rank-8 SiLU bottlenecks produce a center displacement and 20 adjacent quantile-gap multipliers per horizon. Their output matrices start at zero. Positive multipliers exp(0.5 tanh(a/0.5)) preserve ordering, and cumulative gaps extend in both directions from the median. This preserves the *raw-scale* F0 gap structure at initialization, up to rounding. Center and shape have 6,272 and 8,704 parameters respectively (14,976 total). This is feature-adapter PEFT, not internal attention LoRA; the architecture itself is not claimed novel.

Every training iteration:
1. Compute native asinh-space 2-pinball gradients on the same two sampled origins for every arm.
2. Clip the joint parameter gradient to norm 1, then apply the center AdamW step.
3. Propose the shape AdamW step. Compare shape-before and shape-after losses on all 120 training origins, **with the updated center held constant**.
4. Aggregate paired changes inside eight fixed contiguous temporal blocks. Accept only if mean(block changes) + std(block changes)/sqrt(8) <= 0. Otherwise restore both shape parameters **and shape AdamW moments/step counters**.

This is a training-data acceptance heuristic inspired by stochastic line search. The block term does not create a confidence interval: contexts overlap, time blocks can remain dependent, and labels are reused adaptively. It gives no population-risk or coverage guarantee. Repeated overlapping windows inside a block do not increase the count of blocks. Gate work uses the same training labels available to every arm and never uses V/E labels.

## Fixed comparisons and budget

Two datasets (ETTm2, Electricity), two seeds (30000/30001), six arms, two learning rates each: **48 fits**, 120 proposed iterations each, checkpoints 0/30/60/120; 5,760 proposed iterations. Rejected shape steps are separately counted; no claim of 5,760 accepted shape updates. No smoke optimizer updates.

- lora: original 1,179,648-parameter attention LoRA, BF16 and exact block checkpointing; learning rates 1e-5/3e-5.
- split: unconstrained center/shape adapter, both branches trained.
- center: same center branch, frozen original shape (6,272 active parameters).
- anchor: same split architecture, native loss + 1.0 mean squared log-gap multiplier. This is a simple output-anchor baseline, not an official L2-SP reproduction.
- pooled: identical trial-step machinery, accepts mean training loss change <= 0, with zero block-variability penalty.
- block: proposed rule with penalty coefficient 1.

All adapter learning rates are 3e-4/1e-3, AdamW weight decay zero. Adapter arithmetic is FP32 on frozen BF16 encoder features. No full-finetuning, rank search, penalty search, or official AdaPTS/delta-Adapter reproduction is included.

Additional conservative baselines use **no extra learned fits**:
- Prediction blending: F0 + alpha(adapted-F0), alpha in {0.25,0.5,1}, across every checkpoint and all noncandidate arms. It intentionally gets a larger V-selection budget than the candidate.
- Scalar calibration: common normalized center shift {-0.2,0,0.2} and width multiplier {0.8,1,1.2}, nine fixed choices selected on V. This is a coarse baseline, not conformal calibration.

A winning candidate would still need stronger tuned regularization, richer calibration and relevant published baselines before a methods claim.

## Chronology and data exposure

Training origins: eight blocks beginning at 10240 + 512b, b=0..7; offsets 0..448 step32 within each block (120 origins). Training target intervals overlap within blocks, and do not cross between blocks. Histories overlap between blocks; statistical independence is not claimed. Training spans 4,096 timestamps, with 480 covered target timestamps per block. V origins: 16384..17824 step96 (16). E origins: 20480..23456 step96 (32). Horizon 48, so V/E target windows do not overlap.

These are later than the earlier repository evaluation periods (ETTm2 original evaluation ended before 9,600). They are new heldout periods for this pilot on already-used sources, not new independent datasets. Foundation-model pretraining overlap is unknown.

Mechanical staging opens the existing compressed data container and writes predetermined development and heldout arrays **without heldout statistics or inspection**. The worker loads only the development slice for training and selection. Scaling is per-channel training-period standard deviation on [10240,14336), with floor 1e-6. After every arm/seed checkpoint and blending/calibration choice is sealed and hashed, the heldout loader and heldout feature extraction are used. The data-container opening during staging must not be described as never reading any heldout bytes.

## Metrics, selection, and continuation rule

Primary: raw-scale equal-channel scaled 2-pinball, existing 21 quantiles. Also median MAE, quantile-mean MSE, 80% interval coverage and width, train loss, acceptance fraction, selected shape displacement, parameter counts and resources.

Each arm/seed chooses minimum V primary, then earliest step, then recipe index. All choices across both datasets are sealed before E scoring.

For **each dataset**:
- seed-mean block primary <= 0.995 times seed-mean strongest baseline;
- every seed block/F0 <= 1.01;
- seed-mean absolute 80% coverage error <= corresponding strongest baseline error + 0.01;
- every selected block checkpoint has step > 0 and nonzero shape displacement.

The strongest comparator is selected separately per seed by minimum **E** primary among F0, LoRA, split, center, anchor, pooled, selected blend, and selected scalar calibration. This E-oracle comparator is a deliberately conservative screen; E never selects candidate weights or recipes. Ties follow the report's fixed arm ordering.

Failure of any condition means STOP for this fixed pilot. A pass is called PASS_CONTINUATION, not a publication PASS. There is no inherited memory-reduction requirement and no relabeling of older memory failures. No tuning after E in this run.

## Resources and reproducibility

GPU lock; 30 seconds with no external compute PID, >=4GiB free and utilization <90% before model loading. Bounded startup wait 12 hours. Monitor at least every five seconds at step boundaries; pause for external compute or free VRAM <1GiB. Never terminate external work or change drivers. Existing system and per-fit guards remain.

Measure optimizer-step peak allocated/reserved and wall time, including gate forwards and state snapshots. Feature caching is legal because the backbone is fixed; its one-time cost is reported. These are not equal-end-to-end-cost or equal-parameter comparisons with LoRA. No speed or memory publication claim follows from cached adapter training.

Commit all code/configuration before execution. Preserve prediction caches, train schedules, checkpoint hashes, choices, all fit outcomes, and historical result hashes. Replay saved metrics independently. No rewriting failed attempts.

## Prior-work boundary checked before execution

- [AdaPTS, ICML 2025](https://arxiv.org/abs/2502.10235): probabilistic adaptation of foundation models already exists.
- [The Forecast After the Forecast, 2026](https://arxiv.org/abs/2601.20280): bounded output adapters and monotonic quantile calibration directly overlap the architecture ingredients here. No novelty claim for those ingredients.
- [Beyond Accuracy](https://arxiv.org/abs/2510.16060): TSFM calibration has already been studied; “analyze calibration” alone is not a novel method.
- [Domain Generalization via Gradient Surgery](https://arxiv.org/abs/2108.01621) and [PCGrad](https://arxiv.org/abs/2001.06782): gradient agreement/conflict mitigation is established. This pilot evaluates finite proposed steps, not a claim to invent gradient agreement.
- [Painless Stochastic Gradient](https://arxiv.org/abs/1905.09997): training-loss-based step acceptance/line search is established.
- [L2-SP](https://arxiv.org/abs/1802.01483) and [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html): anchoring and pretrained/finetuned interpolation are established.

The unvalidated remaining question is the *added value of block-conditioned shape-only acceptance* over these simpler explanations. This limited search does not establish novelty by absence.
