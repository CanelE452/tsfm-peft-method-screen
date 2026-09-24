# Bounded information-contract recheck v2

COMPLETED_BOUNDED_RECHECK. Descriptive retrospective comparison, not a PEFT success/failure claim.

Future plans are identical across arms. CURRENT intentionally removes past-command information; this is an information ablation, not an equal-information PEFT comparison.
History summaries and raw lags use the same bounded raw history with different representations.
All five tasks and fixed-alpha controls are retained. No DEV-based task/alpha selection.
Command-change panels are operational time windows, not measured physical tau or validated complexity.
DEV episodes are 5/6. Episode 7 is not opened or hashed. Two DEV episodes do not justify population uncertainty estimates.

| task | comparison | panel | DEV episodes supported | positive episodes | mean MAE reduction K |
|---|---|---|---:|---:|---:|
| SUPPLY_15M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.0155599886 |
| SUPPLY_15M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.0175549178 |
| SUPPLY_15M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 1 | 0.00797981975 |
| SUPPLY_15M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 1 | 0.000124282608 |
| SUPPLY_30M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 1 | -0.0194557019 |
| SUPPLY_30M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 1 | -0.0198296362 |
| SUPPLY_30M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 1 | 0.00808402082 |
| SUPPLY_30M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 1 | 0.00175965098 |
| ZONE_120M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.126831847 |
| ZONE_120M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.0460581605 |
| ZONE_120M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.126831847 |
| ZONE_120M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.0460581605 |
| ZONE_360M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.410372992 |
| ZONE_360M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.205997875 |
| ZONE_360M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.410372992 |
| ZONE_360M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.205997875 |
| ZONE_60M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.0291633603 |
| ZONE_60M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | ALL | 2 | 0 | -0.017145333 |
| ZONE_60M | HISTORY_LAGS_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.0278377942 |
| ZONE_60M | HISTORY_SUMMARY_PLAN_TUNED__vs__CURRENT_PLAN_TUNED | RECENT_OR_PLANNED_CHANGE | 2 | 0 | -0.0168229817 |

A positive number favors the history model in the specified comparison only.
See predictions/, origins/, SCORES.csv, PAIRED_EFFECTS.csv, EFFECT_SUMMARY.json and PRE_DEV_SEAL.json for exact support.
No absolute physical-error sufficiency threshold, significance claim, new complexity index or automatic continuation decision is defined.
