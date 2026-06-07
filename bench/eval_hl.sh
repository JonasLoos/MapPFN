#!/usr/bin/env bash
# Eval a checkpoint zero-shot on Frangieh + Papalexi.
# Defaults = the clean base: small 3.36M arch (128/4/4/4), Muon, accum=1, NS=200.
# Override via env for the 11M model (EMBED=192 NBLK=6 NHEAD=6 NREG=6) or other configs.
# ACCUM must match the checkpoint's training accum (opt_state structure); the clean
# base trains at accum=1, so leave it at 1 unless evaluating an old accum>1 ckpt.
#
# Usage: bash bench/eval_hl.sh <ckpt> <tag>
#   writes fr_<tag>.json and pa_<tag>.json
set -euo pipefail
cd /workdir
CKPT="${1:?ckpt path}"; TAG="${2:?tag}"
EMBED="${EMBED:-128}"; NBLK="${NBLK:-4}"; NHEAD="${NHEAD:-4}"; NREG="${NREG:-4}"
OPT="${OPT:-muon}"; ACCUM="${ACCUM:-1}"; NS="${NS:-200}"
export PYTHONPATH=/workdir PYTHONUNBUFFERED=1
export XLA_FLAGS=--xla_cpu_use_thunk_runtime=false
export JAX_COMPILATION_CACHE_DIR=/workdir/.jax_cache
for DS in frangieh papalexi; do
  pfx=$([ "$DS" = frangieh ] && echo fr || echo pa)
  echo "=== EVAL $DS tag=$TAG ckpt=$CKPT ==="
  NS="$NS" OPT="$OPT" ACCUM="$ACCUM" EMBED="$EMBED" NBLK="$NBLK" NHEAD="$NHEAD" NREG="$NREG" \
    .venv/bin/python eval_downstream.py small "datasets/single_cell/${DS}.h5ad" "$CKPT" "${pfx}_${TAG}"
done
echo "=== eval done: $TAG ==="
