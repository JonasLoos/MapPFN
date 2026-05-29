# Downstream evaluation: SERGIO → real single-cell (finding)

**Date:** 2026-05-29
**Context:** Follow-up to the wall-clock optimization on the LinearSCM (d=20) benchmark.
This investigates the paper's *other* evaluation tracks that our optimization never touched.

## The paper has three eval tracks; our optimization only covered one

| Track | Setup | Headline metric | Our status before this |
|---|---|---|---|
| 1. **LinearSCM** (d=20) | controlled synthetic SCM (paper Table 2) | W₂/MMD/RMSE/PDS/MR | ✅ matched/beat paper (see memory) |
| 2. **SERGIO → zero-shot real** (d=50) | pretrain on SERGIO synthetic prior, eval zero-shot on Frangieh/Papalexi (paper Table 3) | **DEG AUPRC** | ❌ never run |
| 3. **Fine-tuned on real** (d=50) | fine-tune pretrained model on real data (paper Table 3) | DEG AUPRC | ❌ never run |

Our optimized LinearSCM checkpoint **cannot** be reused for tracks 2/3:
`treatment_in`/`cond_in` are `Linear(in_dim, cond_dim)` and `decoder.noise_dim = in_dim`, so
`in_dim` is baked into the weights (20 vs the 50 genes of the real data), and it was trained on the
wrong prior (LinearSCM, not SERGIO).

## What was run

Track 2, zero-shot Frangieh, with our optimized recipe applied to the d=50 setting:

```
cfg=map_pfn_rna  cfg/datamodule=frangieh        # fit on SERGIO prior, test zero-shot on Frangieh
gradient_accumulation_steps=1  LR peak=1e-3  WSD warmup=0.02 decay=0.3
batch_size=64  step_size=0.05  num_steps=1500   # COMPRESSED probe (paper uses 400k)
model: embed=128, 4 blocks, 4 heads, 4 reg tokens  -> 3.36M params (in_dim=50)
```

`DataModule` with `prior_dataset_path=sergio.h5ad` trains on the SERGIO prior and validates/tests on
the real Frangieh split in one run.

## Result: the LinearSCM compression does NOT transfer to the downstream task

Zero-shot Frangieh metrics over training (val split, same Dopri5 ODE eval used for test):

| step | Frangieh AUPRC ↑ | Frangieh W₂ ↓ | Frangieh MR (→1) | SERGIO-prior AUPRC |
|---|---|---|---|---|
| 500  | 0.116 | 205  | 10.2 | 0.366 |
| 1000 | 0.087 | 154  | 7.6  | 0.408 |
| 1500 | **0.076** | 95.4 | 4.64 | 0.383 |
| **paper (400k)** | **0.34** | ~22 | ~1.0 | — |

- The model **is** learning the SERGIO prior task (prior AUPRC ~0.37–0.41, distribution metrics improving).
- On the real data, distribution fit steadily improves (W₂ 205→95, magnitude-ratio 10×→4.6× overshoot),
  **but DEG AUPRC declines** (0.116 → 0.076). The model specializes to SERGIO statistics that do not
  zero-shot-transfer to real-melanoma differential-expression ranking at this step count.

### Important caveat: our model is also ~8× smaller than the paper's

The gap above reflects **two** differences from the paper at once, not just step count:

| | embed_dim | cond_dim | blocks | heads | reg tokens | params (d=50) |
|---|---|---|---|---|---|---|
| **Ours** | 128 | 128 | 4 | 4 | 4 | 3.36M |
| **Paper default (`map_pfn_rna`)** | 256 | 256 | 8 | 4 | 8 | ~25–30M (est., not measured) |

The repo's `MMDiTConfig` defaults are `embed=256, cond=256, 8 blocks, 8 reg tokens`, and
`MapPFNRNATrainingRunConfig` inherits them unchanged (it only overrides `num_steps=400000`,
`num_shots=8`). We deliberately fixed the small `128/4` arch during the LinearSCM optimization and
carried it into the downstream probe. So the zero-shot gap is **~250× fewer steps AND ~8× less model
capacity** combined — reduced capacity likely contributes, and a fair comparison would match the paper's
`256/8` arch. (Our d=50 model is otherwise identical to our d=20 model — 3.36M vs 3.35M params, the
~15.5k difference being only the `in_dim`-dependent I/O projections.)

**Conclusion:** the wall-clock compression that reached paper quality on LinearSCM in ~3000 steps does
**not** shortcut the SERGIO→real downstream task. The bottleneck there is not optimizer-update
efficiency but genuinely learning a transferable perturbation map from a much richer/more diverse prior.
Reproducing the paper's zero-shot AUPRC (0.34) needs far more training — the paper's 400k steps ≈ **~44h
of pure compute** at the warm cluster throughput (~2.5 it/s), across many 2h windows with checkpointing.
That full reproduction was deliberately **not** pursued.

## Practical notes for anyone resuming track 2/3

- **Datasets** (HF `marvinsxtr/MapPFN`, dataset repo): `sergio.h5ad` (27 GB, 50-dim prior),
  `frangieh.h5ad` (50 MB), `papalexi.h5ad` (38 MB). Downloaded to the cluster under
  `datasets/synthetic/` and `datasets/single_cell/`.
- **Throughput (d=50, A100):** cold first run 0.36 it/s (NFS-cold SERGIO + cold XLA); warm 2.5 it/s.
  SERGIO loads to ~32 GB RSS (~4 min cold, fast once page-cached; node has 755 GB RAM).
- **Cold XLA compile of the d=50 Dopri5 val/test ODE is ~18 min** (vs ~12–60 s at d=20) — one-time,
  persists in `$JAX_COMPILATION_CACHE_DIR=/workdir/.jax_cache`.
- **Two pipeline issues to fix before the fine-tune track:**
  1. `Checkpointer.teardown()` (`map_pfn/utils/lightning.py`) calls `remove_checkpoint(...)`, deleting
     `model.ckpt` at the end of every stage — so no checkpoint persists after a clean run. Must persist
     it (copy before teardown / override) to feed `cfg.load_checkpoint` for fine-tuning.
  2. After `Stopper` triggers, `train.py`'s `trainer.test()` + `evaluate_baselines()` produced no output
     and the run exited 0. The final validation runs the same ODE eval, so it is the usable zero-shot
     signal, but the formal test/baseline path should be debugged.
- **Cluster run invocation:** after a compute-session reactivation, training must be launched with
  `PYTHONPATH=/workdir` (the editable install finder doesn't auto-load) and `PYTHONUNBUFFERED=1`;
  venv is python3.12 at `.venv/bin/python`.
