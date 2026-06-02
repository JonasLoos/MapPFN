"""Model-free diagnostic: how does the SERGIO prior's marginal/effect distribution
differ from real Frangieh + Papalexi? Identifies which prior levers (DAG structure
vs technical noise) to change to improve zero-shot transfer.

Compares, on CONTROL cells (unperturbed baseline) unless noted:
  - sparsity (fraction zeros) per cell & per gene
  - per-gene mean / std distributions
  - library size (row sum) distribution  [post normalize+log1p these are ~comparable]
  - global value histogram
  - DE structure: # DEGs/perturbation, effect-size (|logFC|) distribution

Usage: PYTHONPATH=/workdir .venv/bin/python bench/prior_vs_real_stats.py
"""
from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")

DSETS = {
    "SERGIO": "datasets/synthetic/sergio_sub400.h5ad",
    "Frangieh": "datasets/single_cell/frangieh.h5ad",
    "Papalexi": "datasets/single_cell/papalexi.h5ad",
}
RNG = np.random.default_rng(0)
MAXCELLS = 40_000  # subsample for speed


def dense(x):
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def load(path):
    A = ad.read_h5ad(path)
    return A


def pct(a, qs=(5, 25, 50, 75, 95)):
    return "  ".join(f"p{q}={np.percentile(a, q):.3f}" for q in qs)


def marginal_report(name, A):
    obs = A.obs
    is_ctrl = (obs["treatment"].astype(str) == "control").to_numpy()
    if is_ctrl.sum() == 0:  # SERGIO uses "control" too; fall back to all
        is_ctrl = np.ones(len(obs), bool)
    idx = np.where(is_ctrl)[0]
    if len(idx) > MAXCELLS:
        idx = RNG.choice(idx, MAXCELLS, replace=False)
    X = dense(A.X[idx])
    print(f"\n===== {name}  (control cells: {len(idx)}, genes={X.shape[1]}) =====")
    # sparsity
    fz_cell = (X == 0).mean(1)
    fz_gene = (X == 0).mean(0)
    print(f"  frac_zero overall = {(X == 0).mean():.3f}")
    print(f"  frac_zero per-cell : {pct(fz_cell)}")
    print(f"  frac_zero per-gene : {pct(fz_gene)}")
    # library size (row sum)
    lib = X.sum(1)
    print(f"  library (row sum)  : mean={lib.mean():.2f}  {pct(lib)}")
    # per-gene mean/std
    gmean = X.mean(0)
    gstd = X.std(0)
    print(f"  per-gene mean      : {pct(gmean)}")
    print(f"  per-gene std       : {pct(gstd)}")
    # nonzero value distribution
    nz = X[X > 0]
    print(f"  nonzero values     : mean={nz.mean():.3f}  {pct(nz)}")
    return dict(fz=(X == 0).mean(), gmean=gmean, gstd=gstd, lib=lib, nz=nz)


def de_report(name, A, max_perts=40):
    """Wilcoxon DE test treatment vs control, matching eval thresholds p<0.01 & |lfc|>0.2."""
    obs = A.obs
    trt = obs["treatment"].astype(str).to_numpy()
    ctx = obs["context"].astype(str).to_numpy() if "context" in obs else np.zeros(len(obs), str)
    ctrl_mask = trt == "control"
    if ctrl_mask.sum() == 0:
        print(f"  [{name}] no control label; skip DE")
        return
    perts = [t for t in np.unique(trt) if t != "control"]
    RNG.shuffle(perts)
    perts = perts[:max_perts]
    ndegs = []
    lfcs = []
    for p in perts:
        # match context: use cells of the perturbation's dominant context
        pmask = trt == p
        if pmask.sum() < 20:
            continue
        pctx = ctx[pmask]
        dom = max(set(pctx), key=list(pctx).count)
        cm = ctrl_mask & (ctx == dom)
        if cm.sum() < 20:
            cm = ctrl_mask
        Xp = dense(A.X[np.where(pmask)[0]])
        Xc = dense(A.X[np.where(cm)[0]])
        mp, mc = Xp.mean(0), Xc.mean(0)
        lfc = np.log2((mp + 1e-6) / (mc + 1e-6))
        ndeg = 0
        for g in range(Xp.shape[1]):
            if abs(lfc[g]) <= 0.2:
                continue
            try:
                _, pv = mannwhitneyu(Xp[:, g], Xc[:, g], alternative="two-sided")
            except ValueError:
                continue
            if pv < 0.01:
                ndeg += 1
                lfcs.append(abs(lfc[g]))
        ndegs.append(ndeg)
    ndegs = np.array(ndegs)
    print(f"\n  [{name}] DE over {len(ndegs)} perts: "
          f"DEGs/pert mean={ndegs.mean():.2f} median={np.median(ndegs):.0f} "
          f"max={ndegs.max() if len(ndegs) else 0}")
    if lfcs:
        print(f"           |logFC| of DEGs: {pct(np.array(lfcs))}")


def main():
    stats = {}
    cache = {}
    for name, path in DSETS.items():
        A = load(path)
        cache[name] = A
        stats[name] = marginal_report(name, A)
    print("\n\n########## DE STRUCTURE ##########")
    for name in DSETS:
        de_report(name, cache[name])

    print("\n\n########## SUMMARY: prior vs real mismatch ##########")
    print(f"{'metric':<22}{'SERGIO':>10}{'Frangieh':>10}{'Papalexi':>10}")
    print(f"{'frac_zero':<22}" + "".join(f"{stats[n]['fz']:>10.3f}" for n in DSETS))
    print(f"{'median gene-mean':<22}" + "".join(f"{np.median(stats[n]['gmean']):>10.3f}" for n in DSETS))
    print(f"{'median gene-std':<22}" + "".join(f"{np.median(stats[n]['gstd']):>10.3f}" for n in DSETS))
    print(f"{'median library':<22}" + "".join(f"{np.median(stats[n]['lib']):>10.2f}" for n in DSETS))
    print(f"{'mean nonzero val':<22}" + "".join(f"{stats[n]['nz'].mean():>10.3f}" for n in DSETS))


if __name__ == "__main__":
    main()
