# Candidate 01

[확인] Per-step observation state has exactly three components: observed mask, age/336, and trailing24 availability. Patch means condition rank8 attention updates, with last context state for REG/future tokens. FEATURE adds a learned rank-space feature term; proposed multiplies rank activations. Both expose identical state and train only LoRA/conditioning parameters. Train block6/refresh2 alternate; validation uses both seen corruptions. E primary averages block12/block24/refresh4/refresh8; clean is separate. Blocks remove the most recent n context values for every channel. Stale R0 carries the last value over the final24h. No labels are corrupted.

See USER_PROTOCOL.md for fixed PASS thresholds. Non-PASS with a positive gain over all simple baselines is WEAK; otherwise FAIL. NO_PROBLEM, INVALID_CONSTRUCT and NOVELTY_COLLISION halt before fitting. These are development decisions, not paper evidence.
