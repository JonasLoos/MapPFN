# Bug audit & clean-base reset (2026-06-07)

Context: the full-scale headline runs used gradient accumulation (bs=32 + accum=2), which
introduced several footguns. Decision: **switch back to no accumulation** (accum=1, bs=32) with a
slightly lower Muon LR (1e-2 → **7e-3**), establish a clean run base, and audit the codebase for
bugs introduced since the original paper baseline.

## Method
Diffed every core file against the paper baseline `8235431` ("Update code", the last upstream state
before our edits start at `160ceaa`) and audited three areas in parallel (data pipeline, training/
optimizer, eval/model/checkpoint). Each finding was verified in code; the optimizer fix was checked
numerically on CPU against `optax.adamw`.

## Findings

| # | Area | Severity | Origin | Status |
|---|------|----------|--------|--------|
| 1 | Muon path applied **zero weight decay** | MED | NEW (`35192f7`, muon added) | **FIXED** |
| 2 | `worker_init_fn` on val/test/predict broke eval reproducibility | HIGH* | NEW (v1noise fixes) | **FIXED** |
| 3 | WSD `total_steps` sized in micro-steps → cooldown skipped at accum>1 | HIGH* | PRE-EXISTING (paper) | Documented; N/A at accum=1 |
| 4 | Logged `train/lr` compressed ×accum (display only) | LOW | NEW (accum) | **FIXED** |
| — | EMA gating under accum, attention XLA fallback, checkpointer teardown, `.ys[0]` shape fix, `assign_split` vectorization | — | NEW | verified **correct** (no bug) |

\* HIGH only manifests at the non-clean settings (num_workers>0 / accum>1); the clean base
(num_workers=0, accum=1) was already immune. They are fixed/documented so the trap is gone.

### 1. Muon weight decay (FIXED) — `map_pfn/train/muon.py`, `jax_lightning.py:231`
`muon_adamw` never applied weight decay (docstring claimed "handled in the outer chain", but the
chain was only `clip → optimizer`), while the AdamW path used `weight_decay=1e-5`. So every Muon
run trained with **WD=0** — a (small, 1e-5) but real confound in all Muon-vs-AdamW comparisons.
Fix: `muon_adamw` now takes `weight_decay` and applies optax-AdamW-style **decoupled** WD to all
params (`update += -lr·wd·param`), threaded via the params optax passes into `update`. Verified on
CPU: wd=0 reproduces the old updates exactly; wd>0 matches `optax.adamw` bit-for-bit on Adam-routed
params and shifts Muon-routed params by exactly `-lr·wd·W`.

### 2. worker_init_fn on eval loaders (FIXED) — `map_pfn/data/data_module.py`
`_seed_worker` (a correct fix for the **train** loader, where forked workers shared one RNG) was
also attached to the **val/test/predict** loaders. There it overwrote the rng that
`set_predict_seed` installs for the two-round baseline resampling in `eval/evaluate.py`, and since
no fixed `Generator` is set, `info.seed` is uncontrolled → eval became non-reproducible at
num_workers>0. Fix: `_seed_worker` is now **train-loader only** (docstring updated).

### 3. WSD total_steps at accum>1 (documented; not hit by clean base) — `base_config.py:131`
`total_steps="${globals: num_steps}"` is in micro-steps, but under `MultiSteps(every_k=accum)` the
schedule advances once per `accum` micro-steps, so at accum>1 it reaches only `num_steps/accum`
updates and the WSD decay never fires. This is **pre-existing in the paper baseline** (default
accum=8). Our headline runs avoided it because `bench/run_full.sh`/`train_eval_variant.sh` pass an
explicit `total_steps = num_steps/accum` (`SCHED_TOTAL`). At the clean base (accum=1) the default is
correct. Added a comment documenting the caveat rather than restructuring the config.

### 4. Logged LR ×accum artifact (FIXED) — `jax_lightning.py:122`
Logged `lr_schedule(global_step)` used micro-steps against an update-sized schedule, so the *logged*
LR looked like it hit 0 at the midpoint (the *applied* LR was correct). Now logs
`lr_schedule(global_step // accum)`. No-op at accum=1.

## Clean base for further runs
- **Recipe:** Muon@7e-3, bs=32, **accum=1**, ns=200, WSD (warmup 0.02 / decay 0.3), euler eval,
  num_workers=0. Effective batch = 32.
- **Driver:** `bench/run_full.sh <variant> [seed] [steps] [load_ckpt]` (train + zero-shot eval on
  Frangieh+Papalexi). `total_steps` defaults to `STEPS` (correct at accum=1); windowed resume pins
  it via `SCHED_TOTAL`. The 11M arch is selected via `EMBED/NBLK/NHEAD/NREG` env.
- Removed the accum-specific `bench/phase2_window.sh`; `bench/eval_hl.sh` now defaults to the clean
  small/accum=1 settings.
- The accum>1 code path still exists (guarded, no-op at accum=1) but is no longer used or trusted.

## Bottom line
Two real new bugs (Muon WD, eval-loader RNG) introduced after the paper baseline — both fixed. One
pre-existing latent bug (WSD@accum>1) documented and side-stepped by the accum=1 base. All other
post-baseline changes (attention fallback, checkpointer, EMA gating, shape fix) verified correct.
