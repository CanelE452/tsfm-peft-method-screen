# Candidate 03

[확인] ETTm2 native15-minute steps, all7 channels. Phase changes add only masked padding, preserving normalization inputs and original absolute-time features. Canonical0 has no padding, phase4/8/12 has left phase and right16-phase padding; future embedding/timing unchanged. Exact roundtrip and canonical native-patching parity required. R0 uses128 distinct fixed origins before training. Train/V phases0,8; E phases0,4,12. Ordinary/conditioned rank8 final-hidden adapters supplement identical attention LoRA. Variance uses scale-normalized quantile forecasts across0,4,12. Phase gate is continuous sin/cos, not a learned unseen-phase embedding.

See USER_PROTOCOL.md for fixed PASS thresholds. Non-PASS with a positive gain over all simple baselines is WEAK; otherwise FAIL. NO_PROBLEM, INVALID_CONSTRUCT and NOVELTY_COLLISION halt before fitting. These are development decisions, not paper evidence.
