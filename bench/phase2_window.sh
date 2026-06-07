#!/usr/bin/env bash
# One training WINDOW of the 11M headline run (Phase 2), resumable across SLURM
# allocations. Train-only (SKIP_EVAL=1); real-data eval is run separately by
# bench/eval_hl.sh once the full run (and its WSD cooldown) has completed.
#
# The 11M @ 20k micro-steps (~7h at ~0.77 it/s) exceeds the 5h gpu partition cap,
# so it is split into windows of <=~13k micro-steps. Resume preserves the LR
# schedule because the schedule count lives in opt_state (serialized) and advances
# once per ACCUM micro-batches; we therefore PIN total_steps to SCHED_FULL (the
# full optimizer-update budget for the WHOLE run = TOTAL_MICRO/ACCUM) in every
# window, and only vary num_steps (this window's micro-batch budget).
#
# Usage:
#   bash bench/phase2_window.sh <window_micro_steps> [load_ckpt]
# Examples:
#   bash bench/phase2_window.sh 10000                  # window 1 (fresh)
#   bash bench/phase2_window.sh 10000 outputs/prior_v1noise_full_s42_hl11m/model.ckpt  # window 2 (resume)
set -euo pipefail
cd /workdir

STEPS="${1:?window micro-steps}"
LOAD="${2:-}"
SCHED_FULL="${SCHED_FULL:-10000}"   # full optimizer-update budget (20000 micro / accum 2)
PRIOR="${PRIOR:-datasets/synthetic/sergio_v1noise_full.h5ad}"
RUNDIR="${RUNDIR:-outputs/prior_v1noise_full_s42_hl11m}"

export PYTHONPATH=/workdir PYTHONUNBUFFERED=1 SKIP_EVAL=1
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export JAX_COMPILATION_CACHE_DIR=/workdir/.jax_cache

echo "=== PHASE2 11M window: steps=$STEPS(micro) sched_full=$SCHED_FULL load='${LOAD:-none}' rundir=$RUNDIR prior=$PRIOR ==="
ARGS=(
  cfg=map_pfn_rna cfg/datamodule=frangieh
  cfg.datamodule.prior_dataset_path="$PRIOR"
  cfg.datamodule.dataset.num_samples=200 cfg.datamodule.batch_size=32
  cfg.datamodule.num_workers=0 cfg.datamodule.persistent_workers=false
  cfg.module.model.decoder.embed_dim=192 cfg.module.model.decoder.cond_dim=192
  cfg.module.model.decoder.num_heads=6 cfg.module.model.decoder.num_blocks=6
  cfg.module.model.decoder.num_reg_tokens=6 cfg.module.model.cond_dim=192
  cfg.module.gradient_accumulation_steps=2 cfg.module.optimizer_name=muon
  cfg.module.lr_schedule.peak_value=1e-2 cfg.module.lr_schedule.decay_frac=0.3
  cfg.module.lr_schedule.warmup_frac=0.02 cfg.module.lr_schedule.total_steps="$SCHED_FULL"
  cfg.module.eval_solver=euler cfg.module.step_size=0.1
  cfg.globals.num_steps="$STEPS" cfg.seed=42
  cfg.trainer.val_check_interval=2000 cfg.trainer.limit_val_batches=2
  +cfg.trainer.num_sanity_val_steps=0
  hydra.run.dir="$RUNDIR"
)
[ -n "$LOAD" ] && ARGS+=(cfg.load_checkpoint="$LOAD")
.venv/bin/python map_pfn/scripts/train.py "${ARGS[@]}"
echo "=== window done: $(ls -la "$RUNDIR/model.ckpt" 2>/dev/null) ==="
