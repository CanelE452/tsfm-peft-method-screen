# Censor tail-gradient diagnostic

New fits0, model backward0. Existing CDF only, raw-output autograd on selected checkpoints. All360 original train sampling batches replayed for each of two states. Censoring is the original synthetic sale cap, not real stockout annotations. This is a snapshot diagnostic, not historical gradient telemetry.

| state | censored occurrences | saturated | zero gradient | saturated zero | censor-loss share |
|---|---:|---:|---:|---:|---:|
| CENSORED_LOSS_LORA | 7957 | 2.727158% | 2.727158% | 2.727158% | 18.247093% |
| CENSOR_PRESERVE_LORA | 7957 | 2.789996% | 2.789996% | 2.789996% | 18.549105% |

Materiality is interpreted from the continuous fractions and loss share, not a fabricated performance cutoff. Support-gap quantiles are preserved in summary.json; distances use raw sales units. A large zero-gradient share motivates a corrected tail-loss design but no corrected loss or Censor v2 is trained here.

Final classification: CENSOR_IMPLEMENTATION_LIMIT_MATERIAL. The 2.73–2.79% saturated-zero-gradient occurrences carry 18.25–18.55% of the censor loss component; this is a material implementation limitation in that term, not proof that it caused all past failure. See FINAL_ASSESSMENT.md.
