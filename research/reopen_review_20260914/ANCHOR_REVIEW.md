# Anchoring retrospective classification

ANCHOR_CONDITIONAL. Original WINDOW_BUDGET_HYPOTHESIS_NOT_SUPPORTED remains unchanged. No repeated fit. Positive macro but Beijing positive, ETTm2 zero, Electricity negative; sparse32 does not show the proposed advantage.

Source-balanced macro +0.494126%; sparse32 -0.013348%; dense233 +1.001601%; sparse-minus-dense -1.014949 percentage points.

| source/budget | native loss | anchor loss | anchor/native gain | native/F0 gain | anchor/F0 gain |
|---|---:|---:|---:|---:|---:|
| beijing_32 | 0.210504945 | 0.210123045 | +0.181421% | +0.391963% | +0.572673% |
| beijing_233 | 0.220125890 | 0.210941239 | +4.172454% | -4.160535% | +0.185515% |
| ettm2_later_32 | 0.129203173 | 0.129203173 | +0.000000% | +0.000000% | +0.000000% |
| ettm2_later_233 | 0.129203173 | 0.129203173 | +0.000000% | +0.000000% | +0.000000% |
| electricity_new_32 | 0.219442918 | 0.219928910 | -0.221466% | +1.046145% | +0.826996% |
| electricity_new_233 | 0.220893477 | 0.223472745 | -1.167652% | +0.392042% | -0.771033% |

All seed-level losses, F0 references, corrected gains and selections: ANCHOR_EFFECTS.csv. Historical seed_gains_percent was a ratio×100, not a gain; the prior ERRATUM is preserved. 30 original E caches independently replayed here; no new E forecasts.
