# Post-screen numerical repair

Update 2026-09-13: the user subsequently authorized completion (“나머지해줘”). All five repaired streams have now completed; the result is FAIL. See [recovery report](../results/candidate_05_repaired/RESULT.md) and [recovery contract](CANDIDATE_05_RECOVERY.md). The account below describes the historical state before this separately recorded recovery.

All Round 0 / Round 1 results were executed from commit `4f0854b27db6950d8e9e7ecb70355c5388ca4fe7`. Every sealed contract's source hash matches that historical commit; `results/screening_summary/execution_source.json` records the proof.

Candidate05 stopped during the first partial-label TAFAS-like update. The native task formula normalized NaN targets before masking. Although the forward loss was finite, the gradient into trainable input-normalization location/scale contained `0 * NaN`. Standard attention LoRA keeps those input statistics outside its gradient path; completed offline fits are not affected by this specific issue.

After the entire screen stopped, main was repaired to replace unavailable labels before nonlinear normalization. Regression tests verify finite location/scale gradients and exact preservation of the prior frozen-input loss and prediction gradients. A train-only GPU GCM backward probe also passed, with zero optimizer updates and no E predictions.

This is an implementation repair, not a rerun or a positive Maturity-PEFT result. The original Candidate05 verdict stays IMPLEMENTATION_BLOCKED. Three streams completed, one TAFAS-like stream aborted, and the proposed Maturity stream was not started. Stored issued forecasts and the nonzero-exit trace are preserved. Partial metrics were recovered solely by replaying saved predictions; missing per-stream timing/peak-memory and drift measurements are explicitly unavailable.

A new, separately reviewed Round1 stream would be needed to evaluate the repaired implementation. Round2 was not executed. Historical results must be audited against their recorded execution revision, not silently relabeled as outputs of the repaired code.
