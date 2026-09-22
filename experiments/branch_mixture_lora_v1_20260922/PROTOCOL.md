# Native branch-mixture LoRA: bounded pilot

Status at registration: design before new training or target-score inspection. The user authorized designing and executing the first proposed candidate on 2026-09-22. Existing HIER/MAG/FR/rollout/three-candidate runs are preserved. This file is this experiment's execution contract; older contracts are provenance, not inherited instructions.

## Purpose and scope

Determine whether scoring the mixture of native quantile branches is preferable to averaging component scores when adapting Chronos-Bolt-small beyond its native 64-step horizon. A limited positive pilot is useful; universal SOTA, an arbitrary percentage threshold, and a paper PASS are not objectives. Audience: user deciding whether this small PEFT training method merits further work. No automatic follow-up runs.

Alternatives considered: (a) native branch mixture training (chosen: actual prior training/inference mismatch), (b) native quantile-spacing residual head (lower compute, closer to existing monotone heads), (c) latent temporal-convolution adapter (useful transfer but close speech/vision precedent). LoRA and CRPS are existing methods. Ensemble rollout CRPS also exists in weather foundation models; this is a proposed narrow adaptation, not a novelty claim.

## Fixed design

- Dataset: official ETTh2, all seven channels independently, no performance-based channel selection. Audit past repository exposure and publish the result; new run does not imply globally unseen benchmark/pretraining data.
- Context 512 hourly slots; horizon 128 (native block 1: 1-64, continuation block 2: 65-128). One continuation controls branching cost.
- Chronological row splits: TRAIN 0-60%, CALIBRATION 60-70%, VALIDATION 70-80%, TEST 80-100%. Daily origins aligned to row mod 24=0; targets entirely within role, context strictly before origin. Context may include already observed rows of earlier roles. TRAIN-only per-series population standard deviation; no outcome-based selection, interpolation or leakage.
- Backbone: amazon/chronos-bolt-small, immutable pretrained revision from previous source manifest; verify weight SHA and current official code provenance. Same isolated Python 3.11 environment may be reused read-only. No previously trained weights.
- q/v LoRA rank 8, alpha16, dropout0, bias none; FP32, deterministic operations, model eval mode to disable pretrained dropout while retaining gradients. Same pretrained model and same LoRA initialization within paired seed.
- Arms COMPONENT and MIXTURE. First block identical mean twice-pinball over nine native quantiles. Second block: COMPONENT averages CRPS of nine equally weighted 9-atom component distributions; MIXTURE uses CRPS of their equally weighted 81-atom mixture. Total loss = .5 first + .5 second, each divided by TRAIN sigma then averaged. No other loss terms.
- Native branches use raw quantile-labelled paths (not pointwise sorting before feedback), exactly as official code. First-block branch inputs detached for both arms. Gradients update LoRA through current predictions; no end-to-end gradient through generated context. No target values in branch construction.
- Empirical CRPS: mean(abs(z-y)) - sum((2*i-M-1)*sort(z)_i)/M^2, i=1..M. This is the exact score of the declared discrete distribution, not exact continuous-model CRPS. No IID/fair-ensemble M(M-1) correction. COMPONENT uses the same distribution representation as MIXTURE.
- Seeds 92231/92232, no selection seed or LR search. Fixed AdamW lr1e-4, betas(.9,.999), eps1e-8, wd0, gradient norm clip1, batch4, 512 updates per fit. Two arms x two seeds = 4 fits / 2048 main updates maximum; smoke 2 updates per arm at seed92230 = 4 updates maximum. No performance-triggered extra LR/rank/loss/seed/dataset runs.
- Both arms execute one first-block B forward and one B*9 continuation forward and corresponding backwards per update: 2 conditional batch calls, 40 series-context evaluations per update. Same batch schedule within seed. Full continuation batch retained for mixture gradients. Report actual wall time and peak allocated/reserved memory.
- Checkpoint candidates INIT,128,256,512, selected by VALIDATION block2 scaled twice-pinball after native 81-to-9 compression and common output sort. Tie earlier step. No TEST or CAL selection of checkpoint. Report fixed512 as an unselected sensitivity alongside selected models (no extra fits).
- Simple output calibration for every selected model and zero-shot reference: CAL-only per-block alpha in {.5,.75,1,1.25,1.5,2,3}, beta in {-.5,-.25,0,.25,.5}; q'=median+beta*sigma+alpha*(q-median). Select minimum scaled twice-pinball; tie identity distance,alpha,abs(beta),beta. Not fed back into context.
- Fixed zero-shot references: F0_NATIVE official branching and CHRONOS2_DIRECT at horizon128/context512, independent univariate groups. These are practical references, not matched-compute causal controls. No added ordinary/median LoRA fit in this 4-fit screen; therefore no claim of superiority to all standard LoRA recipes or isolation of general rollout exposure.

## Preflight and evaluation

Require real TRAIN-input native parity at 64 and128, both base and initial LoRA, smoke-trained native parity, same-seed initialization, frozen parameter preservation, finite gradients, component/mixture identity and scalar-loss/gradient checks, context target independence, raw-data SHA/schema/time/leakage audit, GPU and package check. Stop before main training if these fail.

Primary: paired MIXTURE vs COMPONENT on TEST block2 TRAIN-scaled mean twice-pinball of common sorted official nine-quantile output, after each arm's VAL checkpoint selection. Report seed-specific and mean-of-seed scores, not ensemble predictions. Secondary: identical CAL opportunity, full128/block1 metrics, raw MAE, scaled RMSE, coverage80, width, quantile crossing, exact empirical mixture CRPS and component-minus-mixture disagreement, fixed512 results. Report channel/origin/lead tables. Saved TEST predictions and selection/calibration hashes sealed before TEST scoring.

Uncertainty: paired noncircular moving-block bootstrap of seven daily origins, 2000 draws, RNG92239; all channels/horizons/seeds kept paired. This reflects origin variation conditional on two seeds and this series set, not population-of-datasets or reliable between-seed uncertainty. Overlapping horizons are not independent samples.

Interpretation predeclared: both seed primary effects positive -> repeat-direction positive pilot signal; mixed directions -> inconclusive; neither positive -> no positive signal. Report CI without turning arbitrary 1%/p-value into universal gate. CAL reversal or worse than simple F0 calibration weakens practical value and must be prominent. No novelty/paper PASS. Never call execution/data/resource failure a scientific negative. Stop after reporting regardless of outcome.

## Provenance and outputs

Source manifest, requirements lock, data and GPU audit, smoke receipts, capped optimizer ledger, paired initial/schedule hashes, checkpoint/selection seal, prediction manifests, independent scalar checks, seed/origin/channel/lead/resource tables, PNG figures embedded in REPORT_KO.md, FINAL_DECISION.md. Scoped commits/push per repository authorization. Raw data/model weights/checkpoints/prediction arrays stay ignored locally. Protocol/source hash sealed before training; preserve and disclose implementation repairs.

## Primary references

- Current official branching: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py
- TSFM calibration and branch/trajectory comparisons: https://arxiv.org/html/2510.16060v2 (its historical Bolt description is not current native-code behavior).
- AIFS-CRPS: https://www.nature.com/articles/s44387-026-00073-7
- Aurora 1.5: https://www.microsoft.com/en-us/research/publication/aurora-1-5-fine-tuning-a-foundation-model-for-medium-range-ensemble-weather-prediction/
- Prior local rollout training: ../rollout_uncertainty_peft_v1_20260921/PROTOCOL.md
