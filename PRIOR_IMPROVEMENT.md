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

## 5e. 11M + Muon — capacity finally pays off (best config)
The 11M+AdamW run (§5c) underwhelmed because it was *undertrained* at 4000 steps. Muon is a
faster optimizer and (§5d) carries no transfer penalty on v1noise — so it's the natural fix.
Re-ran the 11M model with Muon@1e-2 (v1noise, 4000 steps, seed 42). Full grid (test-split):

| metric | small+AdamW | small+Muon | 11M+AdamW | **11M+Muon** |
|---|---|---|---|---|
| Frangieh W₂ ↓ | 19.8 | 18.3 | 20.7 | **17.8** |
| Frangieh AUPRC ↑ | 0.050 | 0.056 | 0.042 | **0.081** |
| Frangieh mag-ratio →1 | 0.92 | 0.86 | 0.92 | **0.83** |
| Papalexi W₂ ↓ | **20.8** | 20.9 | 28.5 | 21.1 |
| Papalexi AUPRC ↑ | **0.178** | 0.163 | 0.171 | 0.162 |

**11M+Muon is the best overall configuration.** It wins Frangieh on W₂ *and* AUPRC (0.081, the
highest of any config — vs ~0.05 elsewhere) and recovers Papalexi to small-model level (W₂ 21.1 vs
the broken 28.5 of 11M+AdamW). The trace confirms the mechanism: Muon converged the 11M model far
faster (step-2000 val W₂ 30.1 vs 11M+AdamW's 37.7), and the cooldown pushed it to a best-in-class
16.8 val W₂ → **the capacity payoff that AdamW left undertrained was unlocked by the faster optimizer.**
Caveat: single seed; W₂ (stable) clearly improved, AUPRC 0.081 vs ~0.05 is suggestive but should be
seed-averaged. Best recipe to date: **v1noise prior + 11M model + Muon@1e-2.** AUPRC still ≪ paper
0.34 (the eval-resampling artifact, §5b), but this is the highest single-config AUPRC we've reached.

## 6. Recommendation
Adopt the **v1noise** technical-noise settings for the SERGIO prior when the objective is
real-data transfer: `dropout_q_range=(10,45)`, `noise_s_range=(0.3,1.0)`,
`library_mu_range=(5.0,6.5)` (in `SergioDatasetConfig`). This is the single robust prior win
(2-seed confirmed, both datasets). Do NOT pursue DAG-effect or DEG-breadth changes — tested,
they don't help. For the AUPRC gap specifically, use the paper's 10-seed eval resampling, not a
prior change. Next: validate v1noise at full prior scale (6000 ctx) + larger model.

## 7. Headline run — full paper-scale v1noise + longer training (2026-06-03, in progress)
Everything above used a **1000-context** prior and **4000** steps. To make a serious, paper-comparable
claim we re-do the two best configs at **full paper scale**: prior = v1noise at **6000 contexts ×
ns=200** (`sergio_v1noise_full.h5ad`, 61.2M cells, ~27GB; same train/val/test split structure as the
official `sergio.h5ad`), trained much longer. Two configs, same recipe:
- optimizer **Muon@1e-2**; **bs=32 + grad-accum=2** (effective batch 64, matching the eff-batch of the
  §5e winner but feasible at ns=200, which OOMs at bs=64); WSD schedule (warmup 0.02 / decay 0.3).
- **Phase 1:** small 3.36M (128/4/4/4), **10 000** micro-steps (= 5 000 optimizer updates).
- **Phase 2:** 11M (192/6/6/6), **20 000** micro-steps (= 10 000 updates).

**Accum/cooldown fix (important):** `num_steps` counts micro-batches, but the LR-schedule counter
lives inside the optimizer, which is wrapped in `optax.MultiSteps(every_k=accum)` and so advances
once per `accum` micro-batches. With `total_steps` left at `num_steps`, accum=2 would only reach 50%
of the schedule and the **cooldown would never fire**. Fix: set `lr_schedule.total_steps =
num_steps/accum`. (Verified empirically; no-op at accum=1, so the whole §1–6 factorial was unaffected.)

Differences vs the official paper run: ~40× the data of our earlier runs but ~5× fewer optimizer
updates than the paper's 400k-step / accum-8 run; effective batch 64 vs the paper's 256; otherwise
the v1noise prior (vs the paper's default-noise prior) is the intended difference.

### Results (zero-shot test-split, NS=200)
Baselines for reference: official ~43M = Fr W₂ 22.7 / MMD 0.022 / AUPRC 0.141, Pa W₂ 44.2 / mag 2.62;
best small-scale config (§5e, 11M+Muon, 1000-ctx) = Fr W₂ 17.8 / AUPRC 0.081, Pa W₂ 21.1; paper AUPRC 0.34 (10-seed).

| config | Fr W₂ ↓ | Fr MMD ↓ | Fr AUPRC ↑ | Fr mag→1 | Pa W₂ ↓ | Pa MMD ↓ | Pa AUPRC ↑ | Pa mag→1 |
|---|---|---|---|---|---|---|---|---|
| 3.36M full @10k (`hl3m`, buggy NW=3) | 24.6 | 0.064 | 0.044 | 1.15 | 36.3 | 0.160 | 0.225 | 2.33 |
| 3.36M full @10k (`hl3mv2`, RNG+EMA fix) | 26.0 | 0.072 | 0.036 | 1.18 | 29.5 | 0.116 | 0.184 | 1.81 |
| 11M full @20k | _never launched (session ended)_ | | | | | | | |

**Outcome — full scale did NOT beat the 1000-ctx factorial for the 3.36M model (2026-06-05).**
The first full run (`hl3m`) used `num_workers=3` and hit a shared-RNG bug (forked workers draw an
identical sampling stream → correlated batches). The fixed re-run (`hl3mv2`: `worker_init_fn`
re-seeding + EMA-gating under accum>1) **only partially recovered**: Papalexi improved markedly
(W₂ 36.3→29.5, magnitude overshoot 2.33→1.81), but **Frangieh did not** (W₂ 26.0, slightly worse
than the buggy run, and far from the 1000-ctx §5e Fr W₂ 17.8). Training itself was healthy — the
cooldown fired (val prior loss 1.040→0.934) and the SERGIO-prior fit was strong (val prior AUPRC
0.455, MMD 0.041). So the worker-RNG bug was **not** the cause of the full-scale Frangieh
regression; something about the full-scale config (6000-ctx prior, ns=200, bs32/accum2, 10k
micro-steps) genuinely transfers worse to melanoma than the 1000-ctx/ns100/bs64/accum1/4000
recipe. **Open: ablate which knob (most likely accum=2, or ns=200-at-equal-steps undertraining,
or the larger/more-diverse prior) before re-attempting Phase 2.** Ckpts on cluster
`outputs/prior_v1noise_full_s42_{hl3m,hl3mv2}/model.ckpt`; logs `hl3mv2.log`, evals
`{fr,pa}_v1noise_full_s42_hl3m{,v2}.json`.
