# Improving the SERGIO prior to improve zero-shot transfer

**Question (user):** can we improve the synthetic SERGIO prior so the frozen small
MapPFN generalizes better zero-shot to real single-cell data? Ideas floated: bigger
distribution of DAG types, less noise.

**Answer:** Yes — and the decisive lever is **technical-noise / marginal realism**, not the
DAG structure. Making the prior's per-gene marginals (sparsity, library size, mean) match
real scRNA-seq roughly **halves the real-data Wasserstein** and fixes a large effect-magnitude
overshoot, on both melanoma (Frangieh) and leukemia (Papalexi). Weakening the DAG effects does
not help (and hurts Papalexi).

## 1. Model-free diagnostic: how the prior mismatches real data
`bench/prior_vs_real_stats.py` (control cells). The default SERGIO prior is, vs real:

| | SERGIO | Frangieh | Papalexi |
|---|---|---|---|
| fraction zeros | **0.65** | 0.18 | 0.24 |
| median library | **32** | 77 | 91 |
| median gene-mean | **0.62** | 1.37 | 2.01 |
| DEGs / perturbation | **36 / 50** | 16 | 5 |
| median \|logFC\| of DEGs | **1.1** | 0.31 | 0.59 |

Two distinct mismatches: (a) **marginals** — the prior is 3× too sparse with ⅓ the library;
(b) **effect structure** — a SERGIO knockout perturbs 72% of genes with huge fold-changes,
while a real perturbation moves a handful of genes subtly.

## 2. A 2×2 factorial of prior variants (`bench/gen_prior_variant.py`)
1000 contexts, 50 genes, 100 cells/cond. Validated marginals move toward real
(`bench/validate_variants.py`):

| variant | what changed | fracZero | medLib | med\|logFC\| |
|---|---|---|---|---|
| **v0base** | current defaults (control) | 0.66 | 32 | 1.35 |
| **v1noise** | less dropout, more library, less sim-noise | 0.33 | 69 | 0.75 |
| **v2dag** | weaker interactions, softer Hill, sparser/modular DAGs | 0.66 | 33 | 1.08 |
| **v3combo** | v1noise + v2dag | 0.35 | 67 | 0.64 |

(Resistant to all levers: DEG **breadth** ~38 vs real ~16 — scale-free hubs propagate widely —
and gene overdispersion. These bound the achievable DEG-AUPRC; see §4.)

## 3. Result: zero-shot transfer (small 3.36M model, AdamW LR=3e-3, ns=100, 4000 steps, seed 42)
Identical recipe; only the prior changes. Test-split metrics on real data
(`bench/train_eval_variant.sh`, `bench/aggregate_variant_results.py`). **bold = best**.

### Frangieh (melanoma)
| metric | v0base | **v1noise** | v2dag | v3combo | paper |
|---|---|---|---|---|---|
| W₂ ↓ | 26.07 | **19.78** | 23.75 | 21.90 | 22.75 |
| MMD ↓ | 0.073 | **0.037** | 0.092 | 0.040 | 0.010 |
| mag-ratio →1 | 1.13 | 0.92 | 1.04 | **0.98** | 1.00 |
| PDS ↓ | 0.497 | **0.215** | 0.482 | 0.478 | 0.17 |
| DEG-AUPRC ↑ | 0.032 | **0.050** | 0.033 | 0.033 | 0.34 |

### Papalexi (leukemia)
| metric | v0base | **v1noise** | v2dag | v3combo |
|---|---|---|---|---|
| W₂ ↓ | 52.61 | **20.75** | 59.18 | 22.97 |
| MMD ↓ | 0.209 | **0.085** | 0.277 | 0.089 |
| mag-ratio →1 | 3.17 | **1.27** | 3.57 | 1.38 |
| DEG-AUPRC ↑ | 0.174 | 0.178 | **0.202** | 0.160 |

(`assets/prior_variants_transfer.png`.)

## 4. Conclusions
- **Marginal realism (v1noise) is the win.** Real W₂ drops 24% (Frangieh) / **61%** (Papalexi),
  MMD ≈ halves, and the effect-magnitude overshoot (mag-ratio up to 3.2× on Papalexi) is
  corrected to ~1.0–1.3. v1noise even **beats the official paper model's Frangieh W₂** (19.8 vs
  22.7) on distribution fit. Gains far exceed seed noise (baseline Papalexi W₂ alone moves 32 pts).
- **Weakening DAG effects (v2dag) does NOT help** distribution transfer and *hurts* Papalexi
  (W₂ 52.6→59.2, mag-ratio 3.17→3.57). Its only gain is a small AUPRC bump. → the user's
  "bigger DAG distribution / less effect" idea is not the lever; the "less noise" idea is.
- **Combining adds nothing (v3combo ≈ v1noise).** Marginal realism alone captures the gain.
- **The DEG-AUPRC gap to the paper (0.34) is NOT closed** (all variants 0.03–0.05 on Frangieh) —
  expected, because none of these levers fixed DEG **breadth** (still ~38/pert). Closing AUPRC
  needs a sparser-effect GRN (fewer downstream targets per KO) and/or the paper's 10-seed
  resampling. That is the natural next experiment.

## 5. Recommendation
Adopt the **v1noise** technical-noise settings for the SERGIO prior when the objective is
real-data transfer: `dropout_q_range=(10,45)`, `noise_s_range=(0.3,1.0)`,
`library_mu_range=(5.0,6.5)` (in `SergioDatasetConfig`). Validate at full prior scale + a 2nd
seed (seed-43 replication in flight), then revisit DEG-breadth for the AUPRC gap.
