#!/usr/bin/env bash
# Canonical full-scale training+eval driver — the CLEAN base for further runs.
#
# Deliberately NO gradient accumulation (accum=1) and num_workers=0: those two were
# the source of recent bugs (EMA-vs-accum, lr-schedule total_steps under MultiSteps,
# forked-worker shared RNG). Effective batch = BS. Recipe:
#   optimizer Muon@7e-3, bs=32, ns=200, WSD (warmup 0.02 / decay 0.3), euler eval.
# Arch defaults to the small 3.36M model; override via env for the 11M (see below).
#
# Usage:
#   bash bench/run_full.sh <variant> [seed] [steps] [load_ckpt]
# Examples:
#   bash bench/run_full.sh v1noise_full 42 10000
#   EMBED=192 NBLK=6 NHEAD=6 NREG=6 ARCH=med bash bench/run_full.sh v1noise_full 42 20000
#   # resume a windowed run (5h cap): pass the previous ckpt as the 4th arg, same STEPS-remaining
#   bash bench/run_full.sh v1noise_full 42 8000 outputs/prior_v1noise_full_s42/model.ckpt
set -euo pipefail
cd /workdir

V="${1:?variant (e.g. v1noise_full)}"
SEED="${2:-42}"
STEPS="${3:-10000}"          # this run's micro-step budget (Stopper). accum=1 => 1 update/step.
LOAD="${4:-}"               # optional checkpoint to resume from (windowed runs)
# Schedule length in optimizer updates. accum=1 so this == micro-steps. For a SINGLE-shot run
# leave it = STEPS (default). For a WINDOWED/resumed run pin it to the FULL budget across all
# windows (the schedule count lives in the serialized opt_state and continues on resume), and
# bump STEPS to the cumulative target for this window.
SCHED_TOTAL="${SCHED_TOTAL:-$STEPS}"

# --- recipe (the clean base) ---
OPT="${OPT:-muon}"; LR="${LR:-7e-3}"     # Muon@7e-3 (was 1e-2; lowered for the smaller eff-batch)
BS="${BS:-32}"; NS="${NS:-200}"
EMBED="${EMBED:-128}"; NBLK="${NBLK:-4}"; NHEAD="${NHEAD:-4}"; NREG="${NREG:-4}"
ARCH="${ARCH:-}"; VAL_INT="${VAL_INT:-2000}"
SUF="${ARCH:+_$ARCH}"
PRIOR="${PRIOR:-datasets/synthetic/sergio_${V}.h5ad}"
RUNDIR="outputs/prior_${V}_s${SEED}${SUF}"
TAG="${V}_s${SEED}${SUF}"

export PYTHONPATH=/workdir PYTHONUNBUFFERED=1
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export JAX_COMPILATION_CACHE_DIR=/workdir/.jax_cache

echo "=== TRAIN $V seed=$SEED steps=$STEPS accum=1 ns=$NS bs=$BS opt=$OPT lr=$LR arch=${EMBED}/${NBLK}blk/${NHEAD}h/${NREG}reg prior=$PRIOR load='${LOAD:-none}' ==="
ARGS=(
  cfg=map_pfn_rna cfg/datamodule=frangieh
  cfg.datamodule.prior_dataset_path="$PRIOR"
  cfg.datamodule.dataset.num_samples="$NS"
  cfg.datamodule.batch_size="$BS"
  cfg.datamodule.num_workers=0 cfg.datamodule.persistent_workers=false
  cfg.module.model.decoder.embed_dim="$EMBED" cfg.module.model.decoder.cond_dim="$EMBED"
  cfg.module.model.decoder.num_heads="$NHEAD" cfg.module.model.decoder.num_blocks="$NBLK"
  cfg.module.model.decoder.num_reg_tokens="$NREG" cfg.module.model.cond_dim="$EMBED"
  cfg.module.gradient_accumulation_steps=1 cfg.module.optimizer_name="$OPT"
  cfg.module.lr_schedule.peak_value="$LR" cfg.module.lr_schedule.decay_frac=0.3
  cfg.module.lr_schedule.warmup_frac=0.02 cfg.module.lr_schedule.total_steps="$SCHED_TOTAL"
  cfg.module.eval_solver=euler cfg.module.step_size=0.1
  cfg.globals.num_steps="$STEPS" cfg.seed="$SEED"
  cfg.trainer.val_check_interval="$VAL_INT" cfg.trainer.limit_val_batches=2
  +cfg.trainer.num_sanity_val_steps=0
  hydra.run.dir="$RUNDIR"
)
[ -n "$LOAD" ] && ARGS+=(cfg.load_checkpoint="$LOAD")
# SKIP_EVAL: train only; the slow post-fit synthetic test + classical baselines are skipped.
# Real-data zero-shot eval is run separately below (and again standalone via bench/eval_hl.sh).
SKIP_EVAL=1 .venv/bin/python map_pfn/scripts/train.py "${ARGS[@]}"

CKPT="$RUNDIR/model.ckpt"
echo "=== ckpt: $CKPT ==="; ls -la "$CKPT"

echo "=== EVAL (zero-shot, ns=200, accum=1) ==="
ACCUM=1 OPT="$OPT" EMBED="$EMBED" NBLK="$NBLK" NHEAD="$NHEAD" NREG="$NREG" \
  bash bench/eval_hl.sh "$CKPT" "$TAG"
echo "=== $V ${ARCH:-small} DONE ==="
