# Official MapPFN checkpoint vs our small model — held-out SERGIO, 2026-06-01

Reference-point eval of the **official paper checkpoint** (`marvinsxtr/MapPFN`,
`model.ckpt`, 519 MB, JAX/Equinox) against our small 3.36M model, to anchor the
training-efficiency study against the published model.

## Protocol (identical for both models — the whole point)
- Script: `eval_official_sergio.py` (official) / `eval_small_sergio.py` (ours).
- `trainer.test` on `datasets/synthetic/sergio.h5ad` held-out **test** split,
  `num_samples=200`, `num_nodes=50`, `seed=42`, `guidance=2.0`, euler ODE.
- `num_workers=0` (forking a dataloader worker while JAX is multithreaded
  deadlocks the test loop at 0/1 — must disable).
- Ran on sess_010c (A100-40GB), a **separate GPU** from the live training run, so
  no contention. JAX preallocation off (`XLA_PYTHON_CLIENT_PREALLOCATE=false`).
- Official model = config defaults (embed 256 / 8 blocks / 8 reg, accum=8).
  Small model = embed 128 / 4 blocks / 4 heads / 4 reg, **accum=1** (must set, or
  the optax.MultiSteps `inner_opt_state` leaf mismatches the checkpoint pytree).
- Small ckpt = seed-42 AdamW@3e-3 final 15k (`outputs/2026-06-01/13-28-45/model.ckpt`),
  the run whose `/prior` (val split) AUPRC = 0.341.

## Results — held-out SERGIO test split (1 batch = 32 contexts)

| metric        | Official (~43M, 519MB) | Small (3.36M) | better |
|---------------|------------------------|---------------|--------|
| deg_auprc     | 0.175                  | 0.179         | tie    |
| wasserstein   | **24.74**              | 25.51         | ~tie   |
| mmd           | 0.0224                 | **0.0130**    | small  |
| mean_rmse     | 0.289                  | **0.207**     | small  |
| var_rmse      | 0.227                  | **0.193**     | small  |
| mean_r2       | 0.926                  | **0.944**     | small  |
| var_r2        | 0.882                  | **0.895**     | small  |
| pds (↓)       | **0.285**              | 0.467         | official |
| w_mag_ratio   | 1.099                  | 1.035         | small  |

Raw: `official_eval_sergio.json`, `small_eval_sergio.json`.

## Findings

1. **The earlier AUPRC "inversion" was a split artifact, not a real gap.**
   Our small model reports `/prior` AUPRC ≈ 0.34 during training — but that is the
   **val** split. On the **test** split (what the official eval uses) the same
   checkpoint scores **0.179**, essentially identical to the official **0.175**.
   The val and test SERGIO splits differ enough that AUPRC is not comparable across
   them; only same-split numbers are.

2. **On the SERGIO prior, our 3.36M specialized model is competitive with — and on
   most distributional metrics slightly better than — the ~13×-larger official
   generalist** (lower MMD/RMSE, higher R²; AUPRC and W₂ tied; but official wins
   PDS 0.285 vs 0.467 — it discriminates perturbations better even on the prior).
   Plausible: the
   small model is trained only on SERGIO, while the official model spends capacity
   generalizing across many priors and to real single-cell data.

## Caveats (do NOT over-claim)
- **Single split, single seed, only 32 contexts** → noisy, especially deg_auprc.
  The clean "small ≥ official" pattern needs more contexts / seeds to confirm.
- This tests **only the SERGIO-prior regime**. The official model's headline claim
  is *transfer to real single-cell data* (frangieh/papalexi) — NOT tested here, and
  a SERGIO-only small model would not be expected to match that.
- Inference hyperparams (guidance=2.0, ns=200) are the `load_model` defaults, used
  identically for both; not necessarily optimal for the official model.

## Downstream REAL data — zero-shot (2026-06-01, the paper's headline track)
Same harness (`eval_downstream.py`), same protocol for both models: `trainer.test`,
ns=200, seed=42, guidance=2.0, euler, `num_workers=0`. Official ckpt vs our small
seed-42 AdamW ckpt. Each real dataset's test loader = **1 batch (32 contexts)**.
Raw: `frangieh_{official,small}.json`, `papalexi_{official,small}.json`.

### Frangieh (melanoma, 248 perts / 3 ctx) — OFFICIAL WINS DECISIVELY
| metric      | official | small  | paper (official) |
|-------------|----------|--------|------------------|
| deg_auprc   | **0.141**| 0.039  | 0.34             |
| wasserstein | 22.69    | 23.32  | 22.75            |
| mmd         | **0.0225**| 0.100 | ~0.010           |
| mean_rmse   | **0.220**| 0.395  | 0.13             |
| mean_r2     | **0.976**| 0.902  | -                |
| mag_ratio   | 1.034    | 1.065  | 1.00             |
| pds (↓)     | **0.182**| 0.463  | -                |
Official beats small on AUPRC (3.6×), MMD (4.4×), RMSE, R², PDS; W₂ and MR ~tie.
**Our SERGIO-specialized small model does NOT transfer to real melanoma; the official
generalist does.** This is the expected, important result and the inverse of the
SERGIO-prior comparison above.

### Papalexi (leukemia, 26 perts / 1 ctx) — BOTH POOR, small slightly ahead
| metric      | official | small  | paper (official) |
|-------------|----------|--------|------------------|
| deg_auprc   | 0.157    | **0.192**| 0.16           |
| wasserstein | 44.19    | **37.94**| -              |
| mmd         | 0.195    | **0.157**| -              |
| mean_rmse   | 0.780    | **0.702**| -              |
| mag_ratio   | 2.625    | 2.275  | -                |
| pds         | 0.485    | 0.477  | -                |
Both models **overshoot effect magnitude ~2.3–2.6×** (mag_ratio≫1) → both genuinely
bad on Papalexi, the hard 26-pert/1-ctx case the paper flags as degrading. Small edges
official on every metric here, but off a low base.

### FULL-COVERAGE eval (2026-06-01) — resolves the coverage-vs-protocol question
The tables above use the fixed 32-pert holdout test loader. To check whether the gap to
paper (0.34) was a coverage artifact, re-ran with **every perturbation as a query**
(Frangieh=150 perts / 5 batches, Papalexi=23 perts). `eval_downstream.py ... full`.
NB bug fixed first: `trainer.test(datamodule=...)` re-runs `datamodule.setup('test')`
and reverts the swapped dataset → the full loader must be passed DIRECTLY to `trainer.test`.
Raw: `*_full.json`.

| Frangieh        | official 32 | **official 150 (full)** | small 150 (full) | paper |
|-----------------|-------------|-------------------------|------------------|-------|
| deg_auprc       | 0.141       | **0.117**               | 0.038            | 0.34  |
| wasserstein     | 22.69       | 19.98                   | 24.67            | 22.75 |
| mmd             | 0.0225      | 0.0272                  | 0.110            | ~0.010|
| mean_rmse       | 0.220       | 0.214                   | 0.447            | 0.13  |
| mag_ratio       | 1.034       | **1.000**               | 1.249            | 1.00  |
| pds (↓)         | 0.182       | **0.057**               | 0.159            | -     |

Papalexi full (23 perts): official auprc 0.144 / W₂ 43.7; small auprc **0.197** / W₂ 39.0
— same picture as the 1-batch version (small ≥ official, both poor, MR ~2.4–2.6).

**CONCLUSION — the paper-gap is a PROTOCOL/inference difference, NOT coverage (earlier
"coverage artifact" claim was WRONG).** Full coverage did not lift official-Frangieh AUPRC
toward 0.34 — it went slightly *down* (0.141→0.117). Meanwhile W₂ improved (19.98) and
mag_ratio became a perfect 1.000, so the *distribution fit* is sound; it is specifically the
**DE-gene AUPRC** that sits far below the paper. Likely causes (unresolved): inference config
not matching the paper (guidance, num_samples=200, num_shots=8), the DE-test thresholds
(p<0.01, |lfc|>0.2), or the released `model.ckpt` differing from the Table-3 model. Resolving
it needs the paper's exact real-data eval config.
- **The official≫small comparison is unaffected** (both evaluated identically): full-coverage
  Frangieh official 0.117 vs small 0.038 AUPRC — official still clearly transfers, small doesn't.

### Net takeaway
Official model wins where it matters (Frangieh, the richer dataset); on the degenerate
Papalexi case neither model works and ours is no worse. Combined with the SERGIO result:
**our small model is a strong SERGIO-prior fitter but a weak real-data generalizer — it
trades transfer for cheap in-distribution quality.**

## DEEP DIVE: why our official-Frangieh AUPRC (0.14) ≠ paper (0.34) — 2026-06-01
Read the paper (arXiv 2601.21092) Table 3 + appendix, audited our eval code, confirmed
our repo IS the paper's eval code (map_pfn/eval + scripts/train.py test()).

**Paper Table 3 (pre-trained), mean±std over 10 resampling seeds:**
- Melanoma(Frangieh): W₂22.75 MMD0.0101 RMSE0.13 PDS0.17 MR1.00 **AUPRC0.34**
- Leukemia(Papalexi): W₂44.42 MMD0.192  RMSE0.78 PDS0.49 MR2.56 **AUPRC0.16**

**Paper protocol (appendix A.4/C.1/D.1):** Dopri5 solver; classifier-free guidance ω=2.0
by default; n=200 cells/condition; holdout context = the **IFN-γ** cell line; test = **half
the treatments of the holdout context** (the other half → train); 50 genes.

**Our eval matches ALL of that:** guidance=2.0, dopri5 (load_model defaults — we never set
euler here), n=200, num_shots=8, and our baked frangieh split holds out exactly **IFNγ**
(verified: test/test_other = IFNγ; train/val = Control+Co-culture). So my earlier
full-coverage (150 perts, all 3 contexts) was the WRONG protocol — paper uses IFNγ-only.

**The validation that proves our harness is correct:** **Papalexi reproduces the paper on
EVERY metric** (W₂ 44.19/44.42, MMD 0.195/0.192, RMSE 0.78/0.78, PDS 0.485/0.49, MR
2.63/2.56, AUPRC 0.157/0.16). So checkpoint, guidance, solver, n=200, num_shots, the
Wilcoxon DE test, and the AUPRC computation are all correct.

**The residual gap is Frangieh-specific:** on the correct IFNγ holdout we match the paper on
W₂ (22.69), PDS (0.182), MR (1.03) — but AUPRC 0.14 vs 0.34, MMD 0.0225 vs 0.0101, RMSE
0.220 vs 0.13 are ~2× off. W₂/MR/PDS matching means the predicted *distribution scale* is
right; the off-metrics (DE-AUPRC, MMD, RMSE) are the ones sensitive to fine-grained
per-gene structure / sample noise. Ruled OUT: guidance, solver, n_samples, holdout context,
coverage, checkpoint identity (all proven correct by the Papalexi match + protocol audit).

**ROOT CAUSE (model-free evidence) — melanoma DE ground-truth is too sparse at n=200.**
Ran the exact Wilcoxon DE test (p<0.01, |lfc|>0.2) on the REAL data only (true int vs
control), counting ground-truth DEGs/perturbation:
| ground-truth DEGs/pert | n=200 | n=1000 |
|------------------------|-------|--------|
| Melanoma (IFNγ)        | **0.6** (baseline 0.013) | 4.0 |
| Leukemia               | **4.9** (baseline 0.098) | 7.0 |
At the eval's n=200, leukemia has ~8× richer DE signal than melanoma. So leukemia's AUPRC is
well-defined & stable → reproduces the paper exactly. Melanoma at n=200 has almost no
detectable true DEGs (0.6/pert ≈ ~15 positives pooled over ~25 test perts) → the single-draw
AUPRC is noise-dominated and unreliable. **Our one-seed 0.14 is a single noisy draw of a
high-variance quantity; the paper's 0.34 is the "mean over 10 resampling seeds" — and that
averaging is *essential precisely because* melanoma DE is this sparse, while leukemia (rich
DE) is stable at one draw.** This cleanly explains why leukemia matches and melanoma doesn't,
and why W₂/MR/PDS (not DE-thresholded) match for melanoma while DE-AUPRC doesn't.

**Single-seed-variance hypothesis — TESTED and REFUTED.** Ran the IFNγ melanoma model eval
over 10 resampling seeds (0–9): AUPRC = **0.122 ± 0.022** (range 0.095–0.177; W₂ stably ~22.7,
matching paper; MMD stably ~0.0215). So averaging seeds does NOT lift melanoma AUPRC toward the
paper's 0.34 — the single-seed 0.141 was representative, and the gap is **real, systematic, and
low-variance, not resampling noise.** My variance explanation was wrong (and so was the earlier
"coverage" one). The DE-sparsity fact (0.6 DEGs/pert at n=200) is real but makes melanoma AUPRC
stably ~0.12, not high-variance.

**HONEST STATUS — the melanoma AUPRC/MMD/RMSE gap vs paper is UNRESOLVED.** What is firmly
established: (a) our harness is correct — Papalexi reproduces the paper exactly on all 6 metrics;
(b) we match the paper's stated protocol (guidance 2.0, Dopri5, n=200, IFNγ holdout 50%); (c) for
melanoma, W₂/MR/PDS match the paper but AUPRC (0.12), MMD (~0.0215), RMSE (~0.22) are stably 2–3×
off; (d) it's not coverage, not seed variance, not the holdout context. Per-seed raw: `fr_off_s0..s9.json`.

**Cell-count lever — also TESTED and REFUTED.** Swept cells/population at fixed seed:
| NS (cells/pop) | AUPRC | W₂ | MMD |
|----------------|-------|------|--------|
| 200 (paper)    | 0.122 | 23.0 | 0.0228 |
| 400            | 0.102 | 18.5 | 0.0368 |
| 600 / 1000     | OOM (dense attention is O(seq²), seq = cells × (shots+1)) | | |
More cells made melanoma AUPRC slightly *worse* (0.102), not better, and W₂ *diverged* from the
paper (18.5 vs paper 22.75 — which n=200 matches best). So despite the ground-truth DEGs/pert
rising 0.6→4.0 from n=200→1000, the model's AUPRC does not climb toward 0.34. **The
DE-power/cell-count explanation is refuted too.**

**THREE hypotheses now refuted by experiment: (1) coverage, (2) single-seed variance, (3) cell
count.** The melanoma AUPRC gap (our 0.12 vs paper 0.34) is robust to every protocol lever we
can vary, while Papalexi reproduces the paper exactly. **Status: genuinely unresolved — and not
a bug on our side.** The one lever we could NOT test (dense attention OOMs as it grows seq) is
**num_shots / amount of in-context conditioning** — the paper's Fig 5a explicitly states
performance improves monotonically with more interventional experiments in context, so a
larger context at eval is the most likely remaining explanation, but it requires chunked/flash
attention (not wired up here) to test at this gene/cell scale. Other residual possibilities: the
released ckpt differing from the exact Table-3 melanoma model (it matches Papalexi, but melanoma's
DE metrics may be more sensitive), or a subtle preprocessing/gene-selection difference in our
downloaded frangieh.h5ad that affects DE but not n=200 W₂. Raw: `fr_off_ns{200,400}.json`.

## Muon-vs-AdamW DOWNSTREAM TRANSFER (2026-06-02) — the SERGIO-prior win does NOT transfer
Evaluated all 6 SERGIO-pretrained small checkpoints (Muon@1e-2 and AdamW@3e-3, seeds 42/43/44)
zero-shot on real Frangieh + Papalexi (eval-seed 42; raw: `fr_*.json`, `pa_*.json`,
`{frangieh,papalexi}_small.json`). Seed-means:

| dataset  | metric    | Muon@1e-2 mean | AdamW@3e-3 mean | better |
|----------|-----------|----------------|-----------------|--------|
| Frangieh | AUPRC     | 0.122*         | 0.040           | noise* |
| Frangieh | W₂ ↓      | 42.2           | **28.1**        | AdamW  |
| Frangieh | MMD ↓     | 0.128          | **0.087**       | AdamW  |
| Papalexi | AUPRC     | 0.191          | 0.181           | ~tie   |
| Papalexi | W₂ ↓      | 122.4          | **56.3**        | AdamW  |
| Papalexi | MMD ↓     | 0.238          | **0.154**       | AdamW  |

\* Muon's Frangieh-AUPRC mean is driven by ONE outlier seed (s43=0.265; the other two are
0.055/0.045 ≈ AdamW). Real-data AUPRC is noise-dominated (the small model sits near the floor),
so AUPRC differences here are NOT meaningful.

**Finding: AdamW transfers BETTER to real data on every distribution metric (W₂ and MMD, both
datasets — 4/4 mean-comparisons), while real-data AUPRC is a noisy tie.** This is the inverse of
the SERGIO-`/prior` result (where Muon won AUPRC/W₂/MMD at all 3 seeds). So **Muon mildly OVERFITS
the SERGIO prior**: it fits the in-distribution prior better but generalizes WORSE to real
single-cell data. Muon: lower prior-W₂ / higher real-W₂; AdamW: the opposite — a textbook
fit-vs-transfer tradeoff.

**Implication for the Muon-adoption verdict (TRAINING_EFFICIENCY.md):** the "adopt Muon@1e-2"
call was made purely on the SERGIO-`/prior` metric. If the actual goal is **real-data transfer**
(the paper's application), this result says Muon does NOT help and AdamW is the safer choice on
distribution fit. Adopt Muon only if optimizing the SERGIO prior itself is the objective.
**Caveat:** single eval-seed per model → real metrics are noisy; the W₂/MMD pattern is consistent
across 4 comparisons × 3 seeds (convincing as a direction) but absolute values swing a lot.

## WHY Muon transfers worse — it's the COOLDOWN, i.e. "victim of better optimization" (2026-06-02)
Extracted prior-fit AND real-Frangieh-val transfer at all 6 val checkpoints for all runs (pp.py
on the training logs). The real-transfer gap is NOT present mid-training — it opens entirely in the
WSD LR-cooldown (last 30%, 10.5k→15k). Seed-means, `W2real` = real-Frangieh val W₂ (↓ = better transfer):

| optim | prior-AUPRC 10k→15k | real-W₂ 10k→15k |
|-------|--------------------|------------------|
| Muon  | 0.341 → **0.365** (+0.024) | 26.8 → **54.5** (+28, collapses) |
| AdamW | 0.331 → 0.334 (+0.003)     | 24.6 → 32.9 (+8, mild)          |
(per-seed real-W₂ @15k: Muon 63.6/47.9/51.9, AdamW 27.1/38.6.)

**Diagnostic:** at MATCHED prior-fit (≤10k, pre-cooldown, both at prior-AUPRC ~0.33–0.35) the two
optimizers transfer the SAME (real-W₂ ~24–30 for both). The divergence appears only in the deep
cooldown, exactly where Muon drives prior-fit to 0.37 (a level AdamW never reaches). AdamW barely
moves in the final cooldown; Muon converts the low-LR phase into a big prior gain that is
**prior-overfitting** — bought prior-AUPRC, paid back in transfer.

**Conclusion:** Muon's worse transfer is NOT an optimizer-specific generalization defect — it's a
consequence of Muon being the *more effective* optimizer. It exploits the cooldown to descend
deeper into the SERGIO-prior-specific minimum (which doesn't transfer); AdamW stays effectively
under-converged on the prior and keeps better transfer. At equal prior-fit they're equivalent.
Practical implication: prior-AUPRC is a partly-misleading objective (its cooldown gains are
overfitting); for real-data transfer, early-stop before the deep cooldown or use a shallower
cooldown — and don't pick the optimizer on prior-AUPRC alone. Caveat: real-W₂ is val-split /
noisy (limit_val_batches=2); pattern is consistent across 3 Muon + 2 AdamW seeds and agrees with
the clean test-split finals (Muon real-W₂ 42 vs AdamW 28). Clean confirmation would need
intermediate checkpoints eval'd downstream at matched prior-fit.

## Cheap next steps if we want to firm this up
- **DONE: full Frangieh coverage** (150 perts) — showed the paper-gap is protocol, not
  coverage (see above). The remaining unknown is the paper's exact real-data inference config.
- Re-run both at seeds 43/44 (changes the 32-context draw) for a band on each metric.
- Eval the small **muon1e2** seed-42 ckpt the same way (set optimizer_name=muon for
  the opt_state template) — does its val-split AUPRC edge persist on the test split?
- Optional: also eval official on frangieh/papalexi to reproduce its real-data
  numbers as a sanity check that our harness matches the paper.
