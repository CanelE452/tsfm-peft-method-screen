# PatchPhase v2 support-complete

PATCH_V2_MIXED. 12/12 fits, 8640 updates; smoke6 updates separate. Historical Candidate03 verdict unchanged. ETTm2 reused development follow-up, not fresh test.

| seed | conditioned loss | augmented loss | conditioned gain | variance reduction | canonical loss C/A |
|---|---:|---:|---:|---:|---:|
| 30000 | 0.322540991 | 0.318144009 | -1.382073% | +0.433635% | 0.309720098/0.313502232 |
| 30001 | 0.317554773 | 0.317557554 | +0.000876% | +0.060328% | 0.308970833/0.308975431 |

Canonical loss is separate from unseen primary. Both seeds must favor primary and phase variance; a positive result with ambiguous clean retention is conservatively MIXED. No arbitrary 30% robustness gate. Full per-phase/V/checkpoint details reside in results/patchphase_v2_support_complete/evaluation.json and trajectory.json.
The final-hidden branch is closed if NOT_SUPPORTED; MIXED is insufficient for automatic expansion. No patch-level v3 constructed.
