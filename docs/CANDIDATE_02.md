# Candidate 02

[확인] Train-only M5 selection: zero fraction >=0.5, at least40 positive days, nonzero scale on days337..1400; SHA256 item-id ordering chooses256. Event GRU hidden32 uses only events in the336-step window, positions/gaps/log1p magnitudes/age. Summary baseline has last gap/mean gap/last magnitude/positive rate and hidden32 projection. Both supplement identical attention LoRA with rank8 native-hidden fusion. V uses5 origins and all256 items. E uses6 origins/allitems. Robustness requires median per-item effect>0, positive mean after removing top10% effects and positive upper-quartile zero-fraction subgroup effect.

See USER_PROTOCOL.md for fixed PASS thresholds. Non-PASS with a positive gain over all simple baselines is WEAK; otherwise FAIL. NO_PROBLEM, INVALID_CONSTRUCT and NOVELTY_COLLISION halt before fitting. These are development decisions, not paper evidence.
