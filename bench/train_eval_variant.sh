#!/usr/bin/env bash
# Train the small 3.36M model on a SERGIO prior VARIANT, then eval zero-shot on
# Frangieh + Papalexi. Controlled prior comparison: all variants share this recipe
# (AdamW LR=3e-3, ns=100, bs=64, WSD cooldown 0.3). v0base = matched baseline.
#
# Usage: bash bench/train_eval_variant.sh <variant> [seed] [steps]
set -euo pipefail
cd /workdir

V="${1:?variant}"           # v0base | v1noise | v2dag | v3combo
SEED="${2:-42}"
STEPS="${3:-6000}"
PRIOR="datasets/synthetic/sergio_${V}.h5ad"
RUNDIR="outputs/prior_${V}_s${SEED}"
TAG="${V}_s${SEED}"

export PYTHONPATH=/workdir PYTHONUNBUFFERED=1
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export JAX_COMPILATION_CACHE_DIR=/workdir/.jax_cache

echo "=== TRAIN $V seed=$SEED steps=$STEPS prior=$PRIOR ==="
.venv/bin/python map_pfn/scripts/train.py \
  cfg=map_pfn_rna cfg/datamodule=frangieh \
  cfg.datamodule.prior_dataset_path="$PRIOR" \
  cfg.datamodule.dataset.num_samples=100 \
  cfg.datamodule.batch_size=64 \
  cfg.datamodule.num_workers=0 cfg.datamodule.persistent_workers=false \
  cfg.module.model.decoder.embed_dim=128 cfg.module.model.decoder.cond_dim=128 \
  cfg.module.model.decoder.num_heads=4 cfg.module.model.decoder.num_blocks=4 \
  cfg.module.model.decoder.num_reg_tokens=4 cfg.module.model.cond_dim=128 \
  cfg.module.gradient_accumulation_steps=1 cfg.module.optimizer_name=adamw \
  cfg.module.lr_schedule.peak_value=3e-3 cfg.module.lr_schedule.decay_frac=0.3 \
  cfg.module.lr_schedule.warmup_frac=0.02 \
  cfg.module.eval_solver=euler cfg.module.step_size=0.1 \
  cfg.globals.num_steps="$STEPS" cfg.seed="$SEED" \
  cfg.trainer.val_check_interval=2000 cfg.trainer.limit_val_batches=2 \
  +cfg.trainer.num_sanity_val_steps=0 \
  hydra.run.dir="$RUNDIR"

CKPT="$RUNDIR/model.ckpt"
echo "=== ckpt: $CKPT ==="
ls -la "$CKPT"

echo "=== EVAL Frangieh ==="
NS=200 OPT=adamw .venv/bin/python eval_downstream.py small datasets/single_cell/frangieh.h5ad "$CKPT" "fr_${TAG}"
echo "=== EVAL Papalexi ==="
NS=200 OPT=adamw .venv/bin/python eval_downstream.py small datasets/single_cell/papalexi.h5ad "$CKPT" "pa_${TAG}"
echo "=== VARIANT $V DONE ==="
