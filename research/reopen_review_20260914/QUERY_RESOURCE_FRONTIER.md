# Query resource frontier

QUERY_RESOURCE_FRONTIER. New forecasting fits0, V/E array reads0, old equal-time quality reused. Shared-state equivalence includes output/loss/raw and clipped gradient/update/Adam.

| dataset | arm | CP blocks | old mean loss | peak MiB | median ms |
|---|---|---:|---:|---:|---:|
| ettm2 | standard | 0 | 0.261981203 | 1828.17 | 116.952 |
| ettm2 | standard | 3 | 0.261981203 | 1631.61 | 128.440 |
| ettm2 | standard | 6 | 0.261981203 | 1331.67 | 143.247 |
| ettm2 | standard | 9 | 0.261981203 | 1032.53 | 162.055 |
| ettm2 | standard | 12 | 0.261981203 | 732.78 | 168.257 |
| ettm2 | query | 0 | 0.262948363 | 948.10 | 126.688 |
| ettm2 | query | 12 | 0.262948363 | 683.88 | 195.227 |
| ettm2 | side | 0 | 0.264141433 | 759.86 | 78.310 |
| ettm2 | head | 0 | 0.264900897 | 610.90 | 43.879 |
| electricity | standard | 0 | 0.083995113 | 1828.17 | 119.023 |
| electricity | standard | 3 | 0.083995113 | 1631.61 | 129.246 |
| electricity | standard | 6 | 0.083995113 | 1331.67 | 145.220 |
| electricity | standard | 9 | 0.083995113 | 1032.53 | 160.384 |
| electricity | standard | 12 | 0.083995113 | 732.78 | 169.614 |
| electricity | query | 0 | 0.084965421 | 948.10 | 130.686 |
| electricity | query | 12 | 0.084965421 | 683.88 | 195.396 |
| electricity | side | 0 | 0.085775292 | 759.86 | 75.896 |
| electricity | head | 0 | 0.087089462 | 610.90 | 40.079 |

ettm2 Query CP0: QUERY_RESOURCE_FRONTIER; dominators: none.

ettm2 Query CP12: QUERY_RESOURCE_FRONTIER; dominators: none.

electricity Query CP0: QUERY_RESOURCE_FRONTIER; dominators: none.

electricity Query CP12: QUERY_RESOURCE_FRONTIER; dominators: none.

Old quality scores come from selected two-seed models at the historical nominal equal-time budget. These new measurements are three fixed-train-batch steps at seed30000, not new quality experiments or proof that storage changes quality. Near timing ties are descriptive, not statistically established superiority. Peak reserved and start allocated plus each measured step are in measurements.json. Side may be faster/smaller yet worse quality; such a tradeoff is not domination.

[Quality-memory-time plot](../../results/reopen_query_resource_20260914/quality_memory_time.png)
