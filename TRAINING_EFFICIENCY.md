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

### Muon (blocked)
`optax.contrib.muon` is incompatible with equinox's `None`-filtered param tree
(`ValueError: Expected None, got 4` in its internal ndim-routing mask) and with
`optax.MultiSteps` (fixed by skipping MultiSteps at accum=1). Using Muon needs a custom
Newton-Schulz / `multi_transform` integration that partitions matrix vs non-matrix params
with a label tree matching the filtered structure. **TODO.**

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
