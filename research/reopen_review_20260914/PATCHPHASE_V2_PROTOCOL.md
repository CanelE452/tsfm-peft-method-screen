# PatchPhase v2 support-complete protocol

User-authorized 12 fits: 3 arms × seeds 30000/30001 × LR 3e-5/1e-4. 720 updates each; checkpoints 0,60,120,240,360,540,720. Original ETTm2 Panel: 7 original channels, 336 context,48 horizon, original train/V/E origins and train scales. E is previously used development data, not fresh test.

Standard attention rank8 LoRA; augmented adds ordinary rank8 final-hidden adapter; conditioned adds the same adapter with 2-dimensional sin/cos gate. Both adapters share initial weights per seed. FP32 as original Candidate03; native loss unchanged, AdamW weight_decay0, clipping1. All parameters use the same candidate LR. One origin/all7 series per update, identical per-seed origin schedule across methods/LRs; phases cycle [0,2,4,6,8,10,12,14]. V unseen [1,5,9,13], E unseen [3,7,11,15]. Phase0 clean retention reported separately. Selection minimum mean V per-phase score, earlier step then lower LR; all six choices sealed before E.

Every phase preserves observations, missingness, absolute timestamps, origin, horizon; only masked patch padding changes. Smoke: each arm 16 phase forwards and 2 actual train steps, 54 forward/6 optimizer updates separate from 8640 main updates. No retries or budget extension. No original files/results modified.

Report unseen mean scaled pinball, per-phase losses, train-scale-normalized prediction variance across E phases, canonical score, per-seed conditioned-vs-augmented and augmented-vs-standard differences. No 30% variance gate. Support requires same favorable primary/variance directions in both seeds and no material canonical loss, reported quantitatively; canonical materiality has no invented numerical threshold, so ambiguous retention yields MIXED. No source-independent conclusion from two seeds. No v3.

Existing .cache/gpu.lock, RAM/GPU guard, 2h cap retained. RustDesk exact display process <=512MiB allowed under earlier user sharing authorization; other compute jobs cause resource stop. Code/smoke/tests committed and pushed before main run. Results committed/pushed after independent verification.
