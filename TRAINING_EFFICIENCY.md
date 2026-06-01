# Training-efficiency study (small 3.36M model, d=50 downstream)

Goal: approach the paper's downstream performance with **much less compute**, keeping
our small `embed=128, 4-block, 4-head, 4-reg-token` arch (≈8× smaller than the paper's
`256/8` default). Work done autonomously on 2026-05-29.

## Phase A — fast iteration loop (done)

Built the infrastructure to iterate quickly on the otherwise very expensive downstream loop:

- **Profiling** (`bench/profile_baseline.py`) overturned the initial assumption: the
  dataloader is **52 ms/batch (not the bottleneck)**; the earlier "~50% GPU util" is the
  small model underutilizing the A100, not data starvation. So `num_workers`/spawn work was
  dropped as low-value.
- **SERGIO subset** (`bench/make_sergio_subset.py`): full SERGIO is 61.2M cells / 6000
  contexts → 27 GB, ~300 s to load+build per run. A 400-context subset (`sergio_sub400.h5ad`,
  ~7%) loads+builds in ~11 s → ~30× faster startup. Used for **relative** method comparison;
  finalists to be validated on full SERGIO.
- **Robust attention** (`map_pfn/models/utils.py`): the cuDNN flash-attention path threw
  `No valid engine configs` on fresh compiles on this cluster. Default is now a plain dense
  XLA attention (compiles in seconds; cuDNN kept opt-in via `MAPPFN_ATTN_IMPL=cudnn`). Dense
  attention is O(seq²); seq ≈ num_contexts×num_samples×2, so we run at **num_samples=100**
  (num_samples=200 OOMs at ~30 GB). ~1.5 it/s warm.
- **Euler eval** (`solve_ode` + `cfg.module.eval_solver=euler`): avoids the ~18 min cold
  Dopri5 compile for validation. (Euler val is still ~19 s/batch — set
  `val_check_interval≥400`, `limit_val_batches=2` to keep overhead down.)
- **Checkpoint persistence**: `Checkpointer.teardown` no longer deletes `model.ckpt`
  (previously removed it every stage, leaving nothing for fine-tuning/resume).
- **Protocol**: `bash run_exp.sh <name> cfg.globals.num_steps=N cfg.trainer.callbacks.2.max_steps=N <overrides>`
  → full log `exp_<name>.log`; `python bench/extract_val.py exp_<name>.log` prints the val curve.
  make_step **persists in the JIT cache**, so same-code reruns skip the ~4 min compile.

**Metric for efficiency comparisons:** `val/loss` and `val/wasserstein/prior` (clean,
monotonic). Real-data zero-shot `deg_auprc` is too noisy at these step counts (~0.03–0.07)
for fine comparisons, but is reported as the ultimate target.

## Phase B — sample efficiency (in progress)

All on the subset proxy, num_samples=100, WSD schedule (warmup 0.02 / decay 0.3), 800 steps.

### Learning rate (clear win)
`val/loss` (lower=better) and `prior_W2` (lower=better):

| LR | loss@400 | loss@600 | prior_auprc@800 | prior_W2@800 | real test AUPRC |
|---|---|---|---|---|---|
| 1e-3 (old default) | 1.32 | 1.22 | 0.283 (@1200) | 58.7 (@1200) | ~0.037 |
| **3e-3** | **1.26** | **1.10** | **0.293** | 69.4 | **0.075** |
| 5e-3 | 1.30 | — | 0.290 | 66.2 | — |

**LR=3e-3 is the sweet spot**: reaches at ~800 steps what 1e-3 reaches at ~1200 (≈33% fewer
steps) and ~2× better real-data zero-shot AUPRC. 5e-3 is noisier/no better; 1e-3 is slow.
(Contradicts the earlier LinearSCM "1e-3≈3e-4" note — different scale/task.) **Adopt 3e-3.**

### Muon (implemented, ADOPTED for SERGIO — verdict reversed 2026-06-01)
`optax.contrib.muon` is incompatible with equinox's `None`-filtered param tree, so
`map_pfn/train/muon.py` implements `muon_adamw`: Newton-Schulz orthogonalized momentum for
2-D weight matrices, AdamW for everything else (norms/biases/embeddings), routed per-leaf via
`jax.tree.map`. Enable with `cfg.module.optimizer_name=muon` (`peak_value=1e-2`).

**DECISIVE 3-seed result (full SERGIO, ns=200, bs=32, 15k steps, WSD cooldown 0.3), held-out
SERGIO `/prior` (the clean deciding metric, NOT the noisy real-Frangieh AUPRC):**

| optim | AUPRC s42/s43/s44 | **mean AUPRC** | mean W₂ | mean MMD |
|---|---|---|---|---|
| AdamW@3e-3 | 0.341 / 0.332 / 0.336 | 0.336 | 29.97 | 0.0289 |
| **Muon@1e-2** | 0.369 / 0.349 / 0.378 | **0.365** | **28.87** | **0.0242** |

**Muon@1e-2 beats AdamW@3e-3 at ALL THREE seeds on AUPRC (+8.6% mean), W₂, and MMD, at equal
compute.** The gap widens through the WSD cooldown. → **adopt Muon@1e-2 for SERGIO pretraining.**

This REVERSES the earlier "not adopted" call below, which was based on the noisy real-Frangieh
subset AUPRC at ns=100/800 steps (0.022 vs 0.051) — that signal is near its noise floor and not
a reliable optimizer ranking (see caveat). On the stable SERGIO-prior metric at ns=200 with full
runs + seed-averaging, Muon is the clear winner.

<details><summary>Superseded subset evidence (ns=100, 800 steps) — kept for history</summary>

| run (800 steps) | loss@400 | prior_W2@800 | real test AUPRC (noisy) | test W2 |
|---|---|---|---|---|
| AdamW@3e-3 | 1.37 | 70.0 | 0.051 | 147 |
| Muon@3e-3 | 1.36 | 71.8 | 0.022 | 150 |
| Muon@1e-2 | **1.24** | **65.0** | 0.022 | **131** |

Old read: Muon better on synthetic-prior W₂ but worse on real-data AUPRC → "not adopted, LR was
the lever." Overturned by the seeded ns=200 result above; the real-AUPRC signal was noise.
</details>

### Measurement caveat (important)
At ≤800 steps on the subset, real-data Frangieh test AUPRC sits near its noise floor
(~0.02–0.05; the Frangieh test split is tiny). Method ranking by AUPRC there is unreliable, and
cross-session numbers differ (env-version sensitivity). Distributional metrics (W₂/MMD/AUPRC)
are also not comparable across different `num_samples`, so we hold `num_samples=100` fixed and
treat synthetic-prior `val/loss` + `W₂` as the fast proxy — but the **metric that decides** is
real-data AUPRC from a longer, full-prior run.

### Meaningful validation — RESULT (headline)
`exp_fullsergio_lr3e3`: full SERGIO prior (6000 contexts) + LR=3e-3 + small 3.36M model,
num_samples=100, 10k steps (~40× less compute than the paper's 400k, ~8× smaller model).
Held-out **real Frangieh test** metrics vs the paper's zero-shot Table 3:

| metric | ours (10k, small) | paper (400k) |
|---|---|---|
| DEG AUPRC ↑ | **0.21** | 0.34 |
| W₂ ↓ | **21.4** | 22.75 |
| MR (→1) | **0.99** | 1.00 |
| PDS ↓ | **0.12** | 0.17 |
| MMD ↓ | 0.061 | 0.010 |

**⚠️ The single-seed numbers above are NOT reliable** — see the seeded result below. The real
Frangieh test set is small, so the test metrics have high run-to-run variance.

### Seeded result (n=3, the honest figure)
Same recipe, 3 seeds (42/43/44), 10k steps each:

| metric | ours (mean ± std) | paper (400k) |
|---|---|---|
| DEG AUPRC ↑ | **0.17 ± 0.04** | 0.34 |
| W₂ ↓ | **31 ± 11** | 22.75 |
| PDS ↓ | 0.30 ± 0.17 | 0.17 |
| MR (→1) | **0.97** ± 0.02 | 1.00 |
| MMD ↓ | 0.094 ± 0.04 | 0.010 |

(per-seed AUPRC: 0.21 / 0.16 / 0.14; per-seed W₂: 21 / 43 / 29.)

**Honest takeaway:** the small model **approaches but does not match** the paper — roughly **half
the DEG-AUPRC**, W₂ within ~1.4×, and **MR matches (~1.0)** — at **~40× less compute and 8× fewer
parameters**. The earlier "matches/beats paper on W₂/PDS" was an optimistic single-seed (42) draw;
across seeds W₂ is 31±11. Only MR is low-variance.

### What the noise means for optimization
- Real-data Frangieh test metrics are **eval-noise-limited** at this dataset scale (val AUPRC
  bounced 0.03→0.22→0.03 across checkpoints of one run; W₂ swings 21–43 across seeds). You **cannot
  finely rank training tweaks by single-run real AUPRC** — must seed-average.
- More steps did **not** reliably help (25k seed-42 test AUPRC 0.05, W₂ 39 — within the noise band,
  with mag_ratio drifting >1, hinting at mild SERGIO-prior overfitting). ~10k is a fine budget.
- `num_samples` is genuinely metric-confounding: it lifts the *synthetic-prior* AUPRC (0.21→0.33
  from 100→200 cells) but on the tiny *real* test the effect is buried in noise (num_samples=200,
  matched sample-budget, 20k steps: AUPRC 0.069, W₂ 30 — indistinguishable from num_samples=100).
- Robust wins that survive seeding: **LR=3e-3** (vs 1e-3), the overall efficient recipe, and —
  on the stable SERGIO-prior metric at ns=200, 3 seeds — **Muon@1e-2** (see the reversed Muon
  verdict above; +8.6% prior AUPRC at all 3 seeds). **More steps** did not survive as a clear win.
- The deciding signal must be the clean held-out **SERGIO `/prior`** metric (ns=200), NOT real-Frangieh
  AUPRC. Aside (2026-06-01): the official-checkpoint baseline study found our real-Frangieh AUPRC
  is dominated by sparse DE signal at n=200 (0.6 ground-truth DEGs/pert vs leukemia 4.9) — the paper's
  0.34 is a 10-resample mean; a single draw is unreliable. See OFFICIAL_BASELINE_COMPARISON.md.

## Remaining directions (not yet done)
- Muon via custom integration (potentially stacks with LR=3e-3).
- Loss/path: minibatch OT-CFM coupling, time-sampling shift (current: logit-normal `sigmoid(N(0,1))`).
- `num_samples` efficiency: does 64 cells/population train as well as 100/200 at lower per-step cost?
- bf16 model weights for throughput.
- Validate the best combined recipe on **full SERGIO** + longer run; compare to paper Table 3.
- (Deferred) custom kernels — only once compute-bound; currently small model underutilizes the GPU.

## Cluster friction notes
- Partition `gpu-2h`; sessions idle-deactivate after ~30 min and polling does **not** reliably
  reset the timer → keep individual runs ≤ ~800 steps (~15–20 min) and reactivate between.
- Always launch training with `PYTHONPATH=/workdir XLA_FLAGS=--xla_cpu_use_thunk_runtime=false`
  and `.venv/bin/python` (python3.12).
