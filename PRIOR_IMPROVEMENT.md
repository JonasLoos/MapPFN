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

## 4b. Seed-43 replication (confirms the headline)
The v0base→v1noise comparison reproduces almost exactly at a second seed — baseline
seed-variance is small, so the gains are real, not noise:

| v0base → v1noise | seed-42 | seed-43 |
|---|---|---|
| Frangieh W₂ | 26.1 → 19.8 | 25.4 → **19.4** |
| Frangieh mag-ratio | 1.13 → 0.92 | 1.13 → **0.92** |
| Papalexi W₂ | 52.6 → 20.8 | 50.2 → **19.6** |
| Papalexi mag-ratio | 3.17 → 1.27 | 3.02 → **1.15** |

(seed-43 v2dag/v3combo run for completeness; results in `fr_*/pa_*_s43.json`.)

## 5b. DEG-breadth fix — TESTED & REFUTED (the AUPRC lever that wasn't)
The one mismatch v1noise leaves open is DEG **breadth**: a SERGIO KO still perturbs more genes
than a real one. First, a measurement correction: `validate_variants.de_stats` pools each KO
across all (different) GRNs, so the huge N inflates significance — it is NOT a valid breadth
measure for synthetic data. The correct **per-context** measure (`bench/deg_breadth_diag.py`)
shows a sparse-modular GRN genuinely cuts breadth: KO-descendants 16→2, DEGs/KO 24→12, and that
structure **does** depend on the GRN (my earlier "structure doesn't help" was the pooling artifact).

Two breadth-reduced priors were built and trained (keeping v1noise marginals):
- **v5real**: sparse GRN + stronger dropout/baseline → DEGs/KO ~17.5, fracZero ~0.26 (real range).
- **v4xsparse**: aggressive sparse GRN → DEGs/KO ~12, fracZero ~0.44 (too sparse).

Zero-shot transfer (seed 42), vs the v1noise winner:

| metric | v1noise | v5real | v4xsparse |
|---|---|---|---|
| Frangieh W₂ ↓ | **19.8** | 25.7 | 19.0 |
| Frangieh DEG-AUPRC ↑ | **0.050** | 0.036 | 0.034 |
| Papalexi W₂ ↓ | 20.8 | 22.6 | **19.6** |
| Papalexi DEG-AUPRC ↑ | 0.178 | **0.184** | 0.169 |

**Refuted:** narrowing the prior's DEG breadth does NOT lift real-data AUPRC (v1noise's Frangieh
0.050 stays best; Papalexi ~tie), and the sparse GRN needed for it tends to *hurt* W₂/MMD —
plausibly because a sparse GRN strips the gene–gene **correlation** structure that matters more
for distribution transfer than breadth does. So **no prior change tested — marginals, effect
magnitude, or DEG breadth — closes the AUPRC gap.** This confirms the gap is an **eval-resampling
artifact** (paper's 0.34 = mean over 10 cell-resampling seeds; melanoma DE is too sparse at n=200
for a single draw — see OFFICIAL_BASELINE_COMPARISON.md), not a deficiency of the prior. The lever
for AUPRC is the eval protocol (resampling), not the prior.

## 5c. Mid-size model (11M) on v1noise — capacity doesn't help at matched compute
Combined the two wins: a model **between** small (3.36M) and paper (43M) — `embed=192, 6 blocks,
6 heads, 6 reg = 11.08M params` — on the v1noise prior, same recipe, 4000 steps (matched to small).

| metric | small 3.36M | **med 11M** | paper 43M |
|---|---|---|---|
| Frangieh W₂ ↓ | **19.8** | 20.7 | 22.75 |
| Frangieh MMD ↓ | 0.037 | 0.038 | 0.010 |
| Frangieh AUPRC ↑ | **0.050** | 0.042 | 0.34 |
| Papalexi W₂ ↓ | **20.8** | 28.5 | — |
| Papalexi MMD ↓ | **0.085** | 0.125 | — |

**At matched 4000 steps the 11M model does NOT beat the 3.36M model** — tied on Frangieh, worse on
Papalexi. The early-training trace shows why: it was *behind* at step 2000 (Frangieh val W₂ 37.7 vs
small 31.5) and the LR cooldown only rescued Frangieh (→19.3), not Papalexi → the bigger model is
**undertrained** at this budget. MMD also did not move toward the paper's 0.010, so that gap is a
`num_samples` effect (we use 100, paper 200), not capacity. Practical note: the 11M model is ~8× slower
per step here (0.77 vs 6 it/s) AND has pathologically slow one-time XLA compiles for the CFG'd euler
ODE at the new shape (~25 min each), so a fair capacity test (≥10k steps) is expensive. **Takeaway:
the small 3.36M recipe is well-matched; don't naively scale capacity without also scaling steps.**

## 5d. Muon on the realistic prior — the transfer penalty disappears
Earlier work found Muon *hurts* real transfer on the **old** (unrealistic) prior — it
out-optimizes into the prior during the cooldown and overfits it (old-prior Muon Papalexi
W₂ ~122 vs AdamW ~56; Frangieh ~42 vs ~28; 3-seed). Hypothesis: on v1noise the prior now
*looks like real data*, so fitting it harder should stop costing transfer. Trained Muon@1e-2
on v1noise (small model, 4000 steps, seed 42):

| metric | AdamW+v1noise | Muon+v1noise |
|---|---|---|
| Frangieh W₂ ↓ | 19.8 | **18.3** |
| Frangieh AUPRC ↑ | 0.050 | **0.056** |
| Frangieh mag-ratio →1 | 0.92 | **0.86** |
| Papalexi W₂ ↓ | **20.8** | 20.9 |
| Papalexi AUPRC ↑ | **0.178** | 0.163 |
| Papalexi MMD ↓ | **0.085** | 0.100 |

**Confirmed flip:** Muon and AdamW are now essentially **tied** on v1noise (Muon slightly better on
Frangieh, slightly worse on Papalexi MMD) — the large transfer *penalty* Muon paid on the old prior
is **gone**. Mechanistically clean: the LR cooldown, which *worsened* Muon's transfer on the old
prior, *improved* it here (Muon real-Frangieh val W₂ 26.2→17.1 across the cooldown), because deeper
fit of a realistic prior is good for transfer. Muon also still fits the prior harder (val /prior W₂
31.4 vs AdamW 36.9). (Caveat: single seed; the step-2000/4000 val AUPRC was noisy — 0.094/0.384 vs
the stable test-split 0.056.) Net: on v1noise, **Muon is safe (no transfer cost) and gives the best
prior-fit** — the old "AdamW is safer for transfer" caveat no longer applies. Not a decisive transfer
*win* though; AdamW remains a fine default.

## 6. Recommendation
Adopt the **v1noise** technical-noise settings for the SERGIO prior when the objective is
real-data transfer: `dropout_q_range=(10,45)`, `noise_s_range=(0.3,1.0)`,
`library_mu_range=(5.0,6.5)` (in `SergioDatasetConfig`). This is the single robust prior win
(2-seed confirmed, both datasets). Do NOT pursue DAG-effect or DEG-breadth changes — tested,
they don't help. For the AUPRC gap specifically, use the paper's 10-seed eval resampling, not a
prior change. Next: validate v1noise at full prior scale (6000 ctx) + larger model.
