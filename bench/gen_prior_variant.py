"""Generate SERGIO prior VARIANTS to improve real-data transfer.

Motivated by bench/prior_vs_real_stats.py, which showed the default SERGIO prior
mismatches real Frangieh/Papalexi on three axes:
  - too sparse           (frac_zero 0.65 vs real 0.18-0.24)
  - too small a library  (median 32 vs real 77-91)
  - effects too strong/global (36/50 DEGs/pert, |logFC|~1.1 vs real 5-16 DEGs, |logFC|~0.3)

Strategy: BROADEN the prior's support to cover the real (subtle/sparse) regime,
rather than narrowing it. Each variant widens or shifts a few generating ranges.

Usage:
  PYTHONPATH=/workdir .venv/bin/python bench/gen_prior_variant.py <variant> [n_contexts] [num_samples]
Writes datasets/synthetic/sergio_<variant>.h5ad
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from torch.utils.data import DataLoader

from map_pfn.data.sergio_dataset import SergioDataset
from map_pfn.data.utils import BatchKeys, ColumnNames, assign_split, collate_fn
from map_pfn.utils.helpers import cpu_count

# Each entry overrides SergioDataset kwargs vs the default config.
# Defaults (for reference):
#   regulators_per_gene_range=(1.5,3.0)  num_groups_range=(1,3)  modularity_range=(1,900)
#   interaction_k_range=(1,5)  noise_s_range=(0.5,1.5)  decay_range=(0.5,1.0)
#   dropout_q_range=(45,82)  dropout_k_range=(8,8)  library_mu_range=(4.5,6.0)
VARIANTS: dict[str, dict] = {
    # control: current defaults, regenerated at this scale for a fair comparison
    "v0base": {},
    # V1 - realistic technical noise: less dropout (match real sparsity), a touch
    # more library, slightly less simulation noise. DAG/effects unchanged.
    "v1noise": dict(
        dropout_q_range=(10.0, 45.0),
        noise_s_range=(0.3, 1.0),
        library_mu_range=(5.0, 6.5),
    ),
    # V2 - WEAKER effects: genuinely reduce interaction strength (lower the UPPER
    # bound of interaction_k) and soften the Hill response (lower hill_n) so a KO
    # produces smaller fold-changes; sparser DAGs (r>=1.0) for fewer downstream DEGs.
    # (The earlier v2 mistakenly widened hill_n UPWARD -> sharper switches -> larger
    # LFC; fixed here.)  NB regulators_per_gene>=1.0 (smallworld beta=1-1/r >= 0).
    "v2dag": dict(
        regulators_per_gene_range=(1.0, 2.5),
        interaction_k_range=(0.3, 2.5),
        num_groups_range=(1, 6),
        hill_n_range=(1.0, 1.8),
    ),
    # V3 - combined: realistic noise + genuinely weaker effects.
    "v3combo": dict(
        dropout_q_range=(10.0, 45.0),
        noise_s_range=(0.3, 1.0),
        library_mu_range=(5.0, 6.5),
        regulators_per_gene_range=(1.0, 2.5),
        interaction_k_range=(0.3, 2.5),
        num_groups_range=(1, 6),
        hill_n_range=(1.0, 1.8),
    ),
    # --- DEG-BREADTH sweep: keep v1noise marginals, make the GRN sparse+modular so a
    # KO stays inside its module (real perts hit ~5-16 genes, SERGIO default ~38). ---
    # v4mod: many modules + high modularity (within-group edges dominate -> contained).
    "v4mod": dict(
        dropout_q_range=(10.0, 45.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.0, 6.5),
        num_groups_range=(6, 10), modularity_range=(1000.0, 4000.0),
    ),
    # v4sparse: + sparser DAG (r->1) and fewer hubs (high delta_out) so fewer descendants.
    "v4sparse": dict(
        dropout_q_range=(10.0, 45.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.0, 6.5),
        num_groups_range=(6, 10), modularity_range=(1000.0, 4000.0),
        regulators_per_gene_range=(1.0, 1.3), delta_out_range=(15.0, 40.0),
    ),
    # v4xsparse: aggressive - many tiny modules, very high modularity, near-tree, no hubs.
    "v4xsparse": dict(
        dropout_q_range=(10.0, 45.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.0, 6.5),
        num_groups_range=(10, 16), modularity_range=(3000.0, 8000.0),
        regulators_per_gene_range=(1.0, 1.1), delta_out_range=(40.0, 100.0),
        delta_in_range=(80.0, 250.0),
    ),
    # v5real: THE breadth fix. Sparse-modular GRN (DEGs/KO ~17.5, real range) + stronger
    # dropout cut & higher baseline to keep marginals realistic (fracZero ~0.26 like v1noise).
    # Targets BOTH real marginals AND real DEG breadth. (deg_breadth_diag.py.)
    "v5real": dict(
        dropout_q_range=(5.0, 28.0), noise_s_range=(0.3, 1.0), library_mu_range=(5.5, 7.0),
        mr_high_range=(3.5, 6.0), num_groups_range=(8, 14), modularity_range=(2000.0, 6000.0),
        regulators_per_gene_range=(1.0, 1.4), delta_out_range=(30.0, 80.0),
        delta_in_range=(50.0, 200.0),
    ),
}


def main() -> None:
    variant = sys.argv[1]
    n_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    # optional 4th arg = output basename (no extension); defaults to sergio_<variant>.
    # lets a re-generation at a different scale (e.g. full 6000-ctx/ns200) avoid
    # clobbering the small factorial file sergio_<variant>.h5ad.
    out_name = sys.argv[4] if len(sys.argv) > 4 else f"sergio_{variant}"
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {list(VARIANTS)}")

    out = Path("datasets/synthetic") / f"{out_name}.h5ad"
    out.parent.mkdir(parents=True, exist_ok=True)

    overrides = VARIANTS[variant]
    print(f"[{variant}] n_ctx={n_ctx} num_samples={num_samples} overrides={overrides}", flush=True)

    ds = SergioDataset(num_genes=50, num_samples=num_samples, num_contexts=n_ctx, seed=42, **overrides)
    nw = max(1, cpu_count() - 1)
    loader = DataLoader(ds, batch_size=10, num_workers=nw, persistent_workers=False,
                        collate_fn=collate_fn, drop_last=False)

    t0 = time.perf_counter()
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i % 20 == 0:
            done = sum(len(x[BatchKeys.TREATMENT_ID]) for x in batches)
            print(f"  ...{done} conditions  ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[gen] {time.perf_counter() - t0:.0f}s", flush=True)

    treatment_ids = np.concatenate([b[BatchKeys.TREATMENT_ID] for b in batches])
    context_ids = np.concatenate([b[BatchKeys.CONTEXT_ID] for b in batches])
    treatments = np.concatenate([b[BatchKeys.TREATMENT] for b in batches])
    data = np.concatenate([b[BatchKeys.DATA] for b in batches])
    del batches  # free ~12GB of per-batch buffers before building/normalizing the 61M-cell AnnData

    _, ns, ng = data.shape
    X = data.reshape(-1, ng)
    treatment_ids = np.repeat(treatment_ids, ns)
    context_ids = np.repeat(context_ids, ns)
    treatments = np.repeat(treatments, ns, axis=0)

    adata = ad.AnnData(
        X=X,
        obs={ColumnNames.CONTEXT: context_ids, ColumnNames.TREATMENT: treatment_ids},
        obsm={ColumnNames.TREATMENT: treatments},
        uns={"commit_hash": f"prior_variant_{variant}"},  # DataModule reads uns/commit_hash
    )
    # same post-processing as generate_data.py for sergio_grn
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata = assign_split(adata, val_share=0.1, test_share=0.5, seed=42)

    adata.write_h5ad(out)
    print(f"[done] wrote {out}  shape={adata.shape}  frac_zero={(adata.X == 0).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
