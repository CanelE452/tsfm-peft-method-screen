# Repository collaboration instructions

The user requests that completed work, including earlier work, be reviewable on GitHub.

- After completing an authorized unit of work, run the appropriate checks, commit its code, protocols, results and verification records, and push to `origin/main`. Existing user authorization covers these routine commits and pushes; do not ask again. Follow any later user instruction that changes this scope.
- Use ordinary commits on main, without force-pushing or rewriting history. Inspect the diff before staging; do not include unrelated changes or secrets.
- Verify that the pushed commit is on the remote and report the commit or result link. If a push fails, state that clearly; a local commit is not a completed push.
- Keep `docs/RESULTS_INDEX.md` current with links to completed work and its verification. Distinguish completed runs from proposals, preflight checks and ongoing work.
- Preserve archived experiment outputs and original FAIL/STOP decisions. Add corrections or reinterpretations separately with provenance.
- Keep ignored raw data, model weights and prediction caches local; publish their manifests and verification scope. Do not imply that GitHub alone contains everything needed for numerical replay when local caches are required.
- This publication workflow does not authorize new experiments, automatic research continuation, or external notifications.
