"""Why does a SERGIO KO hit ~38/50 genes regardless of GRN structure?
Separate three candidate causes:
  (1) graph reachability  -> avg #descendants of the KO'd node in the DAG
  (2) normalize_total compositional coupling (50-gene library) -> raw vs normalized DEGs
  (3) raw causal ripple   -> DEGs on raw counts

For two structure settings (v1noise-default vs v4xsparse), sample GRNs, and for each
(context, KO) measure descendants + DEGs on RAW counts and on normalize_total'd counts.

Usage: PYTHONPATH=/workdir .venv/bin/python bench/deg_breadth_diag.py
"""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
from scipy.stats import mannwhitneyu

from map_pfn.data.sergio_dataset import SergioDataset

warnings.filterwarnings("ignore")

V1MARG = dict(dropout_q_range=(10.0, 45.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.0, 6.5))
CONFIGS = {
    "v1noise": V1MARG,
    "v4sparse": dict(**{**V1MARG, "num_groups_range": (6, 10), "modularity_range": (1000.0, 4000.0),
                        "regulators_per_gene_range": (1.0, 1.3), "delta_out_range": (15.0, 40.0)}),
    "v4xsparse": dict(**{**V1MARG, "num_groups_range": (10, 16), "modularity_range": (3000.0, 8000.0),
                         "regulators_per_gene_range": (1.0, 1.1), "delta_out_range": (40.0, 100.0),
                         "delta_in_range": (80.0, 250.0)}),
    # v5real: sparse-modular GRN (breadth) + stronger dropout cut & higher baseline to
    # counter the sparsity that a sparse GRN introduces -> aim DEGs~12-16 AND fracZero~0.25.
    "v5real": dict(dropout_q_range=(5.0, 28.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.5, 7.0),
                   mr_high_range=(3.5, 6.0), num_groups_range=(8, 14), modularity_range=(2000.0, 6000.0),
                   regulators_per_gene_range=(1.0, 1.4), delta_out_range=(30.0, 80.0),
                   delta_in_range=(50.0, 200.0)),
}
N_CTX = 6
N_KO = 14  # KOs per context to test


def normalize_total_log1p(X):
    # mimic sc.pp.normalize_total (median target) + log1p, on these 50 genes
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1
    target = np.median(X.sum(1)[X.sum(1) > 0]) if np.any(X.sum(1) > 0) else 1.0
    return np.log1p(X / tot * target)


def degs(Xp, Xc, thr=0.2):
    mp, mc = Xp.mean(0), Xc.mean(0)
    lfc = np.log2((mp + 1e-6) / (mc + 1e-6))
    n = 0
    for g in range(Xp.shape[1]):
        if abs(lfc[g]) <= thr:
            continue
        try:
            _, pv = mannwhitneyu(Xp[:, g], Xc[:, g])
        except ValueError:
            continue
        if pv < 0.01:
            n += 1
    return n


def descendants_of(beta_dag, node):
    G = nx.DiGraph()
    n = beta_dag.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if beta_dag[i, j] != 0:
                G.add_edge(i, j)
    return len(nx.descendants(G, node)) if node in G else 0


def main():
    for name, ov in CONFIGS.items():
        ds = SergioDataset(num_genes=50, num_samples=100, num_contexts=N_CTX, seed=7, **ov)
        desc, deg_norm, n_edges, fz = [], [], [], []
        for cseed in ds.context_seeds:
            G = ds.sample_grn(int(cseed))
            beta_dag = ds._remove_cycles(G)
            beta_dag = ds._ensure_mrs(beta_dag)
            n_edges.append(int((beta_dag != 0).sum()))
            ctrl = ds.sample_condition(int(cseed), None)
            if ctrl is None:
                continue
            Xc = normalize_total_log1p(ctrl["data"])
            fz.append((Xc == 0).mean())
            for k in range(50)[:N_KO]:
                r = ds.sample_condition(int(cseed), k)
                if r is None:
                    continue
                desc.append(descendants_of(beta_dag, k))
                deg_norm.append(degs(normalize_total_log1p(r["data"]), Xc))
        print(f"\n=== {name} ===  edges={np.mean(n_edges):.0f}  fracZero={np.mean(fz):.3f}")
        print(f"  #descendants of KO: mean={np.mean(desc):.1f} median={np.median(desc):.0f} max={np.max(desc)}")
        print(f"  DEGs/KO (per-context, normalized): {np.mean(deg_norm):.1f}   [real: Frangieh~16, Papalexi~5]")


if __name__ == "__main__":
    main()
