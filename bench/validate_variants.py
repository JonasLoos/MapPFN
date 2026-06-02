"""Validate prior variants: do their marginals + DE structure move toward real?
Prints one summary row per dataset. Run AFTER gen_prior_variant.py.

Usage: PYTHONPATH=/workdir .venv/bin/python bench/validate_variants.py
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
MAXCELLS = 30_000

FILES = {
    "v0base": "datasets/synthetic/sergio_v0base.h5ad",
    "v1noise": "datasets/synthetic/sergio_v1noise.h5ad",
    "v2dag": "datasets/synthetic/sergio_v2dag.h5ad",
    "v3combo": "datasets/synthetic/sergio_v3combo.h5ad",
    "--Frangieh": "datasets/single_cell/frangieh.h5ad",
    "--Papalexi": "datasets/single_cell/papalexi.h5ad",
}


def dense(x):
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def de_stats(A, max_perts=40):
    # Deterministic: first `max_perts` perturbations by sorted name, all their cells.
    obs = A.obs
    trt = obs["treatment"].astype(str).to_numpy()
    ctx = obs["context"].astype(str).to_numpy() if "context" in obs else np.zeros(len(obs), str)
    ctrl = trt == "control"
    perts = sorted(t for t in np.unique(trt) if t != "control")
    ndegs, lfcs = [], []
    for p in perts[:max_perts]:
        pmask = trt == p
        if pmask.sum() < 20:
            continue
        pctx = ctx[pmask]
        dom = max(set(pctx), key=list(pctx).count)
        cm = ctrl & (ctx == dom)
        if cm.sum() < 20:
            cm = ctrl
        Xp = dense(A.X[np.where(pmask)[0]])
        Xc = dense(A.X[np.where(cm)[0]])
        lfc = np.log2((Xp.mean(0) + 1e-6) / (Xc.mean(0) + 1e-6))
        nd = 0
        for g in range(Xp.shape[1]):
            if abs(lfc[g]) <= 0.2:
                continue
            try:
                _, pv = mannwhitneyu(Xp[:, g], Xc[:, g])
            except ValueError:
                continue
            if pv < 0.01:
                nd += 1
                lfcs.append(abs(lfc[g]))
        ndegs.append(nd)
    return np.mean(ndegs) if ndegs else 0, (np.median(lfcs) if lfcs else 0)


def main():
    print(f"{'dataset':<12}{'fracZero':>9}{'medLib':>8}{'medGmean':>9}{'medGstd':>9}"
          f"{'DEGs/prt':>9}{'medLFC':>8}")
    print("-" * 64)
    for name, path in FILES.items():
        A = ad.read_h5ad(path)
        obs = A.obs
        trt = obs["treatment"].astype(str).to_numpy()
        ctrl = trt == "control"
        idx = np.where(ctrl)[0] if ctrl.sum() else np.arange(len(obs))
        if len(idx) > MAXCELLS:
            idx = RNG.choice(idx, MAXCELLS, replace=False)
        X = dense(A.X[idx])
        fz = (X == 0).mean()
        lib = np.median(X.sum(1))
        gmean = np.median(X.mean(0))
        gstd = np.median(X.std(0))
        ndeg, mlfc = de_stats(A)
        print(f"{name:<12}{fz:>9.3f}{lib:>8.1f}{gmean:>9.3f}{gstd:>9.3f}{ndeg:>9.1f}{mlfc:>8.3f}")


if __name__ == "__main__":
    main()
