"""Create a smaller SERGIO prior for fast iteration.

Keeps a random subset of biological `context` values (all their cells, all
splits, all treatments + control), preserving the per-context in-context
structure needed for num_shots demonstrations. Reads the source in backed mode
so we don't materialize 27 GB just to subset it.

Usage:
  PYTHONPATH=/workdir .venv/bin/python bench/make_sergio_subset.py <n_contexts> [out_path]
"""

from __future__ import annotations

import sys
import time

import anndata as ad
import numpy as np

SRC = "datasets/synthetic/sergio.h5ad"
N_CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = sys.argv[2] if len(sys.argv) > 2 else f"datasets/synthetic/sergio_sub{N_CTX}.h5ad"
SEED = 0


def t() -> float:
    return time.perf_counter()


def main() -> None:
    t0 = t()
    A = ad.read_h5ad(SRC, backed="r")
    print(f"[read backed] {t() - t0:.1f}s  shape={A.shape}", flush=True)
    obs = A.obs
    print(f"columns: {list(obs.columns)}", flush=True)
    n_ctx = obs["context"].nunique()
    n_trt = obs["treatment"].nunique()
    print(f"n_contexts={n_ctx}  n_treatments={n_trt}", flush=True)
    print(f"split counts:\n{obs['split'].value_counts()}", flush=True)
    # treatments per context (non-control)
    nctrl = obs[obs["treatment"] != "control"].groupby("context", observed=True)["treatment"].nunique()
    print(f"non-control treatments/context: min={nctrl.min()} median={int(nctrl.median())} max={nctrl.max()}", flush=True)

    rng = np.random.default_rng(SEED)
    all_ctx = obs["context"].unique()
    keep = set(rng.choice(all_ctx, size=min(N_CTX, len(all_ctx)), replace=False).tolist())
    mask = obs["context"].isin(keep).to_numpy()
    print(f"keeping {len(keep)} contexts -> {mask.sum()} cells", flush=True)

    t0 = t()
    sub = A[mask].to_memory()
    print(f"[to_memory] {t() - t0:.1f}s  sub.shape={sub.shape}", flush=True)
    t0 = t()
    sub.write_h5ad(OUT)
    print(f"[write] {t() - t0:.1f}s -> {OUT}", flush=True)

    # report resulting train-pair count proxy
    s = sub.obs
    train_pairs = s[s["split"].isin(["train", "control"])].groupby(
        ["context", "treatment"], observed=True).ngroups
    print(f"resulting (context,treatment) groups in train+control: {train_pairs}", flush=True)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
