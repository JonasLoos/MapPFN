#!/usr/bin/env bash
# Eval a headline-run checkpoint zero-shot on Frangieh + Papalexi.
# Defaults = the 11M arch (192/6/6/6), Muon, accum=2, NS=200. Override via env for
# the 3.36M model (EMBED=128 NBLK=4 NHEAD=4 NREG=4) or other configs.
#
# Usage: bash bench/eval_hl.sh <ckpt> <tag>
#   writes fr_<tag>.json and pa_<tag>.json
set -euo pipefail
cd /workdir
CKPT="${1:?ckpt path}"; TAG="${2:?tag}"
EMBED="${EMBED:-192}"; NBLK="${NBLK:-6}"; NHEAD="${NHEAD:-6}"; NREG="${NREG:-6}"
OPT="${OPT:-muon}"; ACCUM="${ACCUM:-2}"; NS="${NS:-200}"
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
