# Frozen development protocol

The complete user contract is USER_PROTOCOL.md. Numerical settings are in configs. All seven recipes, selection rules and thresholds are fixed before any Round 1 E score. Round 0 uses a separate construct partition preceding training; it is not the final E partition. Same context information, sampling and optimization budgets apply within each candidate. Daily M5 and 15-minute ETTm2 use 336/48 native steps; hourly Jena uses hours.

Default effective batch is eight series; multivariate groups never cross origins. All methods have the same batches within their candidate. Native Chronos normalized 21-quantile loss is used for optimization, train-standard-deviation scaled raw 2-pinball for selection/evaluation. Validation ties select the earliest checkpoint then lowest LR. Step 0 is eligible. No E-based adjustment or retry of a completed valid fit.

Resource guards abort unsafe/nonfinite jobs and retain failure receipts. Completed selections are immutable. All fit counts include unsuccessful attempted fits and are capped. Round 2 is never launched by this repository.

Earlier FR-LoRA pilot E results are known; Jena uses a fresh August–November 2021 interval in this screen. Historical benchmark datasets may overlap foundation-model pretraining: results are development evidence, not proof of unseen-pretraining generalization.

Memory guard uses strict CommitLimit only in Linux overcommit mode 2. This host uses heuristic mode 0, with baseline Committed_AS already above CommitLimit; here an 8 GiB growth limit, 6 GiB process RSS cap and 2 GiB available-RAM floor apply. No system setting is modified.
