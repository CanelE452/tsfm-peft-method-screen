# Candidate 04

[확인] Paired origin o/o+24 each has336 context and48 horizon, one pair of4 channels gives8 series per batch. Train/V/E are new Jena intervals. Regularizer uses only exact overlap early24:48 and late0:24, all21 raw quantiles divided by train scale. Anchor averages all correction cells. Single common lambda=clip(0.01*first train task/first train raw-stability,0.001,1). This rule handles zero initial FR/anchor by using the nonzero raw-stability reference; no extra warmup fit. Standard ignores lambda. All comparisons include step0.

See USER_PROTOCOL.md for fixed PASS thresholds. Non-PASS with a positive gain over all simple baselines is WEAK; otherwise FAIL. NO_PROBLEM, INVALID_CONSTRUCT and NOVELTY_COLLISION halt before fitting. These are development decisions, not paper evidence.
