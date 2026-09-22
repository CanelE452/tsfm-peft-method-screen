# Final decision

EXECUTION: COMPLETE
SCIENTIFIC_SIGNAL: POSITIVE_PILOT_SIGNAL
PRACTICAL_ADVANTAGE: NOT_ESTABLISHED_VS_CALIBRATED_F0
PRIMARY_RELATIVE_GAIN_PERCENT: +2.4742921
CALIBRATED_RELATIVE_GAIN_PERCENT: +2.7368611
FITS: 4 / 4
MAIN_UPDATES: 2048 / 2048
SMOKE_UPDATES: 4 / 4
NOVELTY: NOT_ESTABLISHED
PAPER_PASS: NOT_CLAIMED
AUTOMATIC_FOLLOWUP: NONE

Positive pilot signal means the preregistered direction criterion (MIXTURE better than COMPONENT in both seeds), not established practical value. Primary effect CI includes zero. MIXTURE is 1.8523% worse than equally calibrated frozen F0 on the primary horizon; paired conditional effect CI [-0.013746, -0.000188]. The loss change reduces harm relative to component training but has not established additional forecasting value over the simple reference.

The result concerns ETTh2 512→128, frozen Chronos-Bolt-small q/v rank8 LoRA, component versus mixture empirical CRPS on native branches. It does not isolate generic rollout benefit or compare all standard LoRA recipes. Both arm selection and output calibration exclude TEST. See REPORT_KO.md for uncertainty, practical references and data exposure.
