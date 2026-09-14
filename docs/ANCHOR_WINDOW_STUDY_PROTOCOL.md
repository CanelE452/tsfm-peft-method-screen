# Anchoring under a controlled optimization-window budget

Frozen before any new heldout forecast. Run: `anchor_window_study_20260914`.

The completed reassessment found that native + uniform anchor improved ETTh1 validation by 0.60% but worsened Traffic by 0.81%. DualClock remained essentially tied after a fourfold training extension, and event perturbations had very small validation effects. These observations motivate a bounded anchoring-condition experiment, not a claim that anchoring is novel or will pass publication review.

## Intervention and evaluation

Three sources, two optimization-window budgets (32 and 233), two seeds (34000, 34001), two arms (native, native + uniform anchor 0.1), two learning rates (3e-5, 1e-4): **48 fits / 43,200 updates**. Each fit receives 900 updates with validation at 0, 150, 450 and 900. Native asinh loss is divided by 21, as in the previous factorial diagnostic. Rank-8 LoRA trains 1,179,648 parameters. BF16 autocast, FP32 parameters and AdamW, gradient clipping 1, zero weight decay, context 1024, horizon 48, batch two origins × four channels. Minibatches are identical across arms and learning rates within each source, window budget and seed.

The 32 windows are deterministic evenly spaced members of the 233-window grid, including both endpoints. Their calendar span and channel scales match. Scale is the float64 standard deviation over the common training interval, ignoring missing values. Contexts expose past observations and dense training targets overlap. Consequently this is an **optimization-window intervention**, not strict label scarcity, nor a comparison at equal unique target count.

All 48 fits finish before any E forecasts. Each arm selects its LR/checkpoint using V only, including step 0. A second explicit seal freezes all 24 choices and 12 optional V policies before the heldout loader opens. The policy selects plain or anchored using V, with exact ties favoring plain; it has more model-selection budget than either individual arm and is not a proposed PEFT method. Frozen-model E predictions and both selected arms are evaluated for every case. No E result changes a hyperparameter or determines which case is reported.

| Source | Channels | Train origins (inclusive) | V origins | E origins |
|---|---|---|---|---|
| Beijing PM2.5 | pm2.5, DEWP, TEMP, PRES | 2048:32:9472 | 10752:96:12192 | 13312:96:16288 |
| ETTm2 later period | HUFL, HULL, MUFL, MULL | 28672:32:36096 | 37376:96:38816 | 39936:96:42912 |
| Electricity unused meters | zero-based source columns 4–7 | 2048:32:9472 | 10752:96:12192 | 13312:96:16288 |

Beijing is new within this project. ETTm2 uses a later chronology after the previous maximum scored prefix 26064. Electricity uses unused channels of an existing source. These are not three wholly independent new corpora, and foundation-model pretraining overlap is unknown. ETTm2 is 15-minute resolution (12-hour forecast); the other two are hourly (48-hour forecast). Missing Beijing targets are masked without imputation. Mechanical schema/timestamp/missingness checks precede training but do not use evaluation performance. Raw hashes and staging hashes are frozen in the config and execution contract.

Sources: [UCI Beijing PM2.5, CC BY 4.0](https://archive.ics.uci.edu/dataset/381/beijing%2Bpm2%2B5%2Bdata), [official ETT dataset](https://github.com/zhouhaoyi/ETDataset), [electricity distribution](https://github.com/laiguokun/multivariate-time-series-data). ETTm2 and ETTh2 are resolutions of the same underlying transformer, so ETTh2 was not counted as an independent new source.

## Interpretation fixed before E

Primary metric is equal-channel scaled 2-pinball. Report both seeds, every source × budget, the balanced macro relative gain over native LoRA, and sparse-minus-dense gain in percentage points. Also report both arms against F0 and the V-only policy. A four-block chronological bootstrap (eight E origins per block, 2,000 paired resamples, shared across arms/seeds/channels) supplies descriptive intervals only. Two budgets reuse the same E targets; six cells are not six independent datasets. Seeds are held fixed in the bootstrap.

A **conditional follow-up signal**, not statistical significance or a paper PASS, means sparse macro gain > 0, mean sparse-minus-dense gain > 0, and positive sparse-minus-dense interaction in at least two of three sources. All other outcomes are recorded as the window-budget hypothesis not supported by this study. No requirement that every individual dataset improve; no retrospective threshold changes. A signal still needs a distinct mechanism, novelty review and broader independent evaluation. A negative result calls for train/V mechanism diagnosis before another intervention, rather than continued tuning on these E targets.

## Execution and provenance

`scripts/run_anchor_window_study.py stage`, `smoke`, then commit the complete protocol and smoke receipt, then `start`. The detached controller runs training → seal → heldout evaluation → independent verification and report → tests and scoped commit/push. Live status: `.cache/anchor_window_study_20260914/queue_status.json`; training: `results/anchor_window_study_20260914/anchor/status.json`. A 12-update real-data smoke covers each source and arm. Earlier executed sources and result files remain unchanged.

GPU startup requires 30 seconds without an external compute process and at least 4 GiB free; training pauses at step boundaries for external compute or free memory below 1 GiB. RAM/GPU guards remain enabled. Whole controller cap is eight hours, each fit cap 30 minutes, no implicit retries. An execution error is inconclusive, not scientific failure. Each phase logs its exit and preserves partial artifacts. The controller automatically completes the specified study and archival; it does not autonomously invent a subsequent scientific hypothesis or send a Discord notification.
