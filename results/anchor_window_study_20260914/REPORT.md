# Anchoring × optimization window budget

Decision: **WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED**. This is not a paper PASS.

48 fits, 43,200 updates, two seeds; all validation choices sealed before 30 heldout forecasts.
Positive gain means lower scaled 2-pinball than plain native LoRA. No all-dataset improvement gate.

| Source | Windows | Plain | Anchor | Gain % | Seed gains % | Descriptive block interval % |
|---|---:|---:|---:|---:|---|---|
| beijing | 32 | 0.210505 | 0.210123 | +0.181 | +99.737, +99.900 | -0.177, +0.817 |
| beijing | 233 | 0.220126 | 0.210941 | +4.172 | +97.150, +94.549 | +1.463, +7.779 |
| ettm2_later | 32 | 0.129203 | 0.129203 | +0.000 | +100.000, +100.000 | +0.000, +0.000 |
| ettm2_later | 233 | 0.129203 | 0.129203 | +0.000 | +100.000, +100.000 | +0.000, +0.000 |
| electricity_new | 32 | 0.219443 | 0.219929 | -0.221 | +100.281, +100.162 | -0.943, +0.358 |
| electricity_new | 233 | 0.220893 | 0.223473 | -1.168 | +100.975, +101.362 | -4.494, +0.916 |

Macro gain: +0.494%. Sparse-minus-dense interaction: -1.015 percentage points.

The two budgets reuse the same evaluation period; they are not independent datasets. Four chronological blocks with two fixed seeds give descriptive uncertainty only. Beijing is a new project source; ETTm2 is a later period of a previously used source, and electricity uses previously unused source-order channels 4–7. Foundation-model pretraining overlap is not excluded.
Training scales use the common training interval. This changes optimization windows, not strict label availability. Dense targets overlap; sparse targets do not. All arms use identical steps and within-budget sampling. The optional validation policy compares two arms and therefore has more selection budget; it is not a new PEFT method.
ETTm2 forecasts 12 hours; hourly sources forecast 48 hours. Missing Beijing targets are masked, not imputed. No evaluation score sets hyperparameters.

Next action: Do not launch more uniform-anchor variants on these same E targets. Inspect train/V forgetting and gradient interference to identify a mechanism before preregistering a different intervention; retain these negative and mixed results.

Integrity: 282 prediction caches independently replayed; maximum metric discrepancy 1.11e-16; 859 historical files unchanged.
