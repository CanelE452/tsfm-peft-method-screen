# Bounded continuation-only LoRA follow-up

Registered before new Solar forecast scores. User approval: “그렇게 해줘” after the two-stage proposal. This is one bounded refinement of the completed branch-mixture pilot, not an automatic search or a restart of previous studies. Prior outcomes and raw caches remain unchanged.

## Purpose and rationale

Test whether restricting LoRA to continuation preserves useful pretrained forecasts while adding value over both shared-block LoRA and frozen native branching with the same simple calibration. Previous ETTh2 MIXTURE improved on VAL but lost to calibrated F0 on TEST; this is not proof of overtraining or that first-block damage caused the loss. Goal is a modest practical PEFT result, not paper PASS or universal superiority.

Alternatives: (1) weight-change attenuation (cheap diagnostic), (2) continuation-only gating (chosen structural refinement), (3) quantile-spacing head (deferred; no automatic execution). Shared parameter interference and moving generated inputs motivate the gating hypothesis; neither mechanism is already proven causal. Aurora already supports `from_second` LoRA, so gating is known-method transfer rather than new on its own.

## Stage D: no-fit attenuation diagnostic, old ETTh2 VALIDATION only

- Read only the two previous selected MIXTURE checkpoints (seeds92231/92232) and audited ETTh2 prepared packet.
- W=W0+lambda*DeltaW, lambda in {0,.25,.5,1}. Multiply only learned LoRA B factors; keep A unchanged. No lambda-squared mistake; not output interpolation.
- Evaluate old VALIDATION tail65–128 scaled twice-pinball, common final sort. Save all seed/lambda scores and inference receipts. Choose global lambda by mean seed VAL score, tie smaller lambda. Validate lambda0 matches F0 and lambda1 reproduces archived VAL predictions.
- No old TEST inference, scoring or lambda selection. Zero optimizer updates. This is an exposed-development diagnostic, not independent confirmation.
- Diagnostic cannot modify the next stage's fixed LR/rank/loss/seeds/steps. The new structural comparison uses full lambda1 for both arms, as proposed; no automatic attenuation search on Solar.

## Stage S: new prespecified Solar source, four fits

Data: official LSTNet multivariate-time-series-data Solar-energy `solar_AL.txt.gz`. Verify README time resolution; average disjoint groups of six10-minute rows into hourly slots. No invented calendar/timezone. First8 columns in SHA256('continuation-v1|solar|'+zero-based-column) order among TRAIN-only finite positive-variance columns. No performance/zero-rate filtering. Audit raw SHA/schema/aggregation/missingness and repository exposure before prediction. Public pretraining overlap remains unknown.

Chronological hourly TRAIN60%, CALIBRATION10%, VALIDATION10%, TEST20%; context512, horizon128, stride24, targets wholly within role. TRAIN-only per-series population std. Context may use already observed earlier-role rows. No targets in generated branch inputs. One source, all8 selected series, no outcome-based source replacement.

Backbone: same immutable pretrained Chronos-Bolt-small revision, FP32, eval-mode/dropout disabled, q/v LoRA rank8/alpha16/dropout0/biasnone, all other parameters frozen. New seeds92241/92242. Equal seed initialization and batch schedule. AdamW1e-4, betas(.9,.999), eps1e-8, wd0, clip1, batch4,512updates. 2 arms x2seeds=4fits/2048mainupdates; separate2updates per arm smoke=4updates cap. No LR/rank/loss/graph/seed expansion.

- SHARED: exact previous MIXTURE algorithm. LoRA active for first64 and continuation64. Loss=.5 first64 native twice-pinball + .5 next64 equal-weight81-atom mixture CRPS, divided by TRAINsigma. Generated first predictions detached.
- CONTINUATION: disable adapter for first64, use these fixed F0 nine raw quantile-labelled paths as continuation inputs, enable LoRA in the second call only. Loss=.5 next64 mixture CRPS. The .5 coefficient keeps tail gradient scaling fixed. First64 has no backward and is exactly F0. The adapter acts on all tokens in the second call, not exclusively generated tokens. This tests gating + fixed first paths + removing first loss as a bundle; does not individually identify their effects. For horizon128 its generated inputs come from the composite model's frozen first stage.
- Both use native branching reshape9x9 and native empirical nine-quantile reduction, no pointwise sorting before feedback. Exact empirical CRPS denominator M^2, not fair-IID correction. Both perform2conditional forward calls per update; SHARED2backwards, CONTINUATION1backward. Report actual compute/memory difference, do not call this equal-FLOP training.

Checkpoint candidates INIT/128/256/512; select each arm/seed solely by VAL tail sorted native9 quantile scaled twice-pinball; tie earlier step. Report fixed512 sensitivity without extra fits. Same CAL-only per64block affine grid: alpha(.5,.75,1,1.25,1.5,2,3), beta(-.5,-.25,0,.25,.5), tie score then squared distance to identity,alpha,abs(beta),beta. CONTINUATION first block must yield the identical F0 calibration. No feedback of calibration into rollout.

Fixed references: F0 native branching, Chronos-2 direct128/context512 independent univariate groups; no new fits. This study does not isolate benefits relative to ordinary one-block/median LoRA or demonstrate a unique new principle.

## Gates, scores, decision, stop

Before main: official source/weight hashes, Python/GPU/package check, data audit, CPU loss and gradient tests, zero-LoRA native parity, and smoke. SHARED must match official native64/128; CONTINUATION must match adapter-off official first64 + adapter-on official continuation calls. After smoke and full fits, raw first64 for CONTINUATION must remain identical to F0; CAL first-block coefficients must also match. Stop and classify implementation/data/resource failure separately.

Primary practical comparison: selected CONTINUATION vs equally calibrated F0 on TEST65–128 scaled twice-pinball. Structural comparison: CONTINUATION vs SHARED with and without identical CAL opportunity. Also report first64,full128,seed/channel/origin/lead tables, raw MAE, coverage,width,RMSE,empirical mixture CRPS, fixed512 and resources. Do not hide negative channels or use them for new selection. All checkpoint/calibration hashes sealed and TEST predictions saved before score inspection.

Seven daily-origin moving-block bootstrap,2000draws,RNG92249; all series/seeds/horizons kept paired. CI is conditional time-origin uncertainty, not reliable two-seed or dataset-population confidence. Mean is seed-score mean, not forecast ensemble.

If both seeds improve over calibrated F0, label practical-direction pilot signal and report CI/size and SHARED comparison separately. Otherwise no repeatable practical advantage; stop this current branching refinement after this single attempt. Do not automatically train the deferred head or change evaluation criteria. If SHARED also improves, attribute generic adaptation vs gating separately. No arbitrary1% gate; no novelty/paperPASS or global PEFT rejection.

Outputs: Korean report with figures, final decision, all tables, source/data/environment/optimizer/prediction/artifact manifests, checks, scoped commits/push. Raw/weights/checkpoints/arrays remain ignored locally. Training code and protocol hashes sealed; repairs must be logged without silently spending extra updates.

Sources: https://arxiv.org/abs/2109.01903 (WiSE-FT); https://microsoft.github.io/aurora/api.html (from_second); https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py ; https://github.com/laiguokun/multivariate-time-series-data . Prior source code reused with provenance from `branch_mixture_lora_v1_20260922`; original artifacts are preserved.
