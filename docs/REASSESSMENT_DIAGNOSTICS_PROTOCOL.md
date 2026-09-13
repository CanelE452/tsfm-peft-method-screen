# Anchoring factorial / DualClock budget diagnostics

User authorized both bounded diagnostics after the full reassessment. This is an additional development experiment, not a replacement for any historical FAIL or STOP.

**Questions and fixed budget**

- Anchoring: objective (native asinh loss /21 versus raw sorted scaled 2-pinball) × uniform F0 preservation (absent versus coefficient0.1). Four arms × ETTh1/Traffic × seeds33000/33001 × learning rates3e-5/1e-4 =32 fits, each900 updates. Rank8 LoRA,1,179,648 parameters, BF16 autocast,FP32 AdamW,clip1,weight decay0. Context1024, horizon48, batch2 origins×4channels. All arms share each seed's minibatch stream. The coefficient is not tuned here. A teacher pass caches F0 train predictions; its time is reported separately. Native+anchor is the previously missing factorial cell.
- DualClock: original Standard LoRA, EventSummary and DualClock architectures; seeds30000/30001; both original learning rates.12 fits, each1440 updates, four times the original360. FP32 and original unscaled native loss preserve the original optimization recipe. Same256 M5 items and context336/horizon48; batch8 independent item/windows. Each arm receives the same seed-specific stream, including the historical seed30000 prefix. Extra parameters: summary12704, DualClock16192; do not claim equal parameter counts or wall time.
- Maximum44 fit attempts,46,080 training updates,4h queue wall and30min per fit. No retry of a failed attempt; failures remain execution-inconclusive. Train-only GPU smoke is14 additional updates (seven arms×2). Historical results are immutable. No further topic generation, experiment recursion or messaging automation.

**Data access and selection**

Use only the already hashed ETTh1/Traffic development arrays ending at V target end12240 and M5 fit.npz ending1650. No raw-source download, E loader or E score exists in this runner. Train origins/scales and V origins match the prior experiments. This is adaptive train/V reuse, not independent holdout confirmation. Old results informed these questions; all new recipes and analyses are fixed before the new runs.

Anchor checkpoints0/150/450/900. DualClock checkpoints0/4/8/15/30/60/120/180/240/360/720/1080/1440. V scaled2-pinball selects the lowest loss, breaking ties by earlier step then lower LR. Step0 remains eligible. Save every checkpoint prediction and parameter hash; independently replay metrics and reload each fit's winner.

**Analysis fixed before execution**

Report all dataset/seed/LR results, not only wins. Anchoring: pairwise anchor-minus-plain effects within each objective at every fixed checkpoint/LR, at the final budget, and after V selection. Report the difference between raw/native anchoring effects descriptively; it is not proof of a causal dataset mechanism. The same raw-space penalty is an operational factorial intervention, not an assertion that its relative gradient strength is equal under both losses. Record actual task/regularizer magnitudes.

DualClock: compare all methods at both360 and1440 budgets and at fixed360/1440 endpoints for each LR. The best over a larger V checkpoint set cannot worsen by construction, so selected-V improvement alone cannot demonstrate convergence or generalization. Record whether selection still hits the budget endpoint and the old/new seed30000 trajectory discrepancy; do not silently assume historical parity.

At the V-selected Summary and DualClock checkpoints, retain the original context/backbone and evaluate: zero event embedding, rotation of event representations across32-item evaluation chunks, and complete output-adapter removal. These are diagnostic interventions, potentially outside the training distribution; they do not establish unique causal event utility or replace a retrained matched-capacity control. Do not choose a new model from these ablations. CPU tests check normal-mode equivalence to the original EventAdapter, and the exact original360-step sampling prefix. Smoke checks F0 identity, finite real updates, active event encoder updates and frozen weights.

No all-dataset improvement gate and no new scientific PASS/FAIL label. Results are recorded as completed diagnostics with effect directions and unresolved questions. A later heldout study must predeclare operating conditions, minimum practically useful effects, aggregation/uncertainty, controls and fresh evaluation before using its outcomes. No new E is opened automatically after these diagnostics.

**GPU and execution**

One GPU, one serial queue, project lock. Wait30s with no external compute, at least4GiB free and utilization below90%. Check about every5s and pause at training/evaluation boundaries on another compute process or less than1GiB free. Never stop another job. Existing RAM/commit,8GiB allocated and512MiB physical-headroom guards apply; queue timeout also bounds GPU waiting. All task/loss/gradient, timing and peak ledgers are retained.

Implementation: `src/tsfm_peft_screen/reassessment.py`, `scripts/run_reassessment_diagnostics.py`, `configs/reassessment_diagnostics_20260914.json`.

Commands:

```
scripts/with_cuda.sh .venv/bin/python scripts/run_reassessment_diagnostics.py smoke
# Commit tested code/config/smoke before execution.
scripts/with_cuda.sh .venv/bin/python scripts/run_reassessment_diagnostics.py start
scripts/with_cuda.sh .venv/bin/python scripts/run_reassessment_diagnostics.py status
CUDA_VISIBLE_DEVICES='' scripts/with_cuda.sh .venv/bin/python scripts/run_reassessment_diagnostics.py finalize --verify-only
```

The process survives terminal disconnection and finalizes metrics after the two jobs. This is local job execution, not a Codex CLI conversation-resume mechanism. Reports distinguish implementation failures from scientific evidence. Raw models, checkpoints, arrays and logs stay ignored; compact provenance, comparisons and results are committed on main.
