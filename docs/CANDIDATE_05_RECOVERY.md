# Candidate05 implementation recovery

Authorized by the user on 2026-09-13: “나머지해줘”, following the report that Candidate05 was incomplete. Execute exactly five repaired Round1 streams (F0, IMMEDIATE_LORA, WAIT_FULL, TAFAS_LIKE, MATURITY_PEFT), using the original data, origins, seed30000, LR1e-4, preservation lambda1, eight updates per eligible origin, model, metrics and PASS thresholds. No hyperparameter search or Round2 execution.

The original screen remains under `results/candidate_05` and its prediction cache stays unchanged. The recovery uses `results/candidate_05_repaired` and `.cache/candidate_05_repaired`. This adds at most five streaming runs to the historical four attempts; do not claim the cumulative work fits the original five-stream budget. No standard fits are added.

The execution commit is recorded before opening E. The numerical repair sanitizes missing targets before nonlinear normalization. Recovery plumbing only adds explicit output/cache paths, overwrite protection and incremental completed-arm receipts. Method equations and the fixed recipe do not change. All five arms are rerun together for timing comparability and to verify the three formerly completed arms reproduce.

The same development E origins were already partly evaluated. This is a bug-recovery comparison, not an untouched holdout or independent replication. Prior observations are not used to alter the recipe or select methods. The novelty-stopped Candidate06 remains stopped.

Run once from a clean committed checkout:

```bash
scripts/with_cuda.sh .venv/bin/python scripts/resume_candidate_05.py
```

The supervisor obtains the common GPU lock, waits for 30 idle seconds, never kills external processes, and enforces the original timeouts. Existing recovery artifacts prevent an automatic rerun.
