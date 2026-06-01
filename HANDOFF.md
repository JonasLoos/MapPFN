# Handoff — Muon vs AdamW on SERGIO (ns=200), 2026-06-01

Continuation of the training-efficiency study (small 3.36M model, `embed=128/4-block/4-head/4-reg`).
Goal this round: **test Muon further with longer runs at num_samples=200, training + evaluating mainly
on SERGIO**, deciding on the *clean held-out SERGIO `/prior` metrics* (loss, W₂, DEG-AUPRC, MMD) —
NOT the noisy real-Frangieh AUPRC that confounded the earlier "Muon not adopted" call.

## What was run
All full-SERGIO, ns=200, bs=32 (ns=200 OOMs at bs=64 dense attn), 15k steps, WSD cooldown 0.3,
peak LR: AdamW=3e-3, Muon=1e-2. Muon = custom `map_pfn/train/muon.py` (Newton-Schulz on 2-D weight
matrices, AdamW on norms/biases/embeddings). Enable via `cfg.module.optimizer_name=muon`.

1. **Subset LR probe** (sergio_sub400, 1.5k steps): picked Muon@1e-2 (Muon needs higher LR than Adam).
2. **Seed-42 head-to-head (COMPLETE):** `exp_full_{adamw3,muon1e2}_ns200.log`.
3. **Seed-43 replication (INTERRUPTED):** `exp_full_{muon1e2_s43,adamw3_s43}.log`. muon-s43 finished;
   adamw-s43 only reached ~step 10.3k before the user interrupted for cluster setup updates.

## Results — SERGIO held-out `/prior` (the deciding metric)

Seed-42 final @15k:  AdamW loss0.718 / W₂29.5 / **AUPRC0.341** / MMD0.0284
                     Muon  loss0.723 / W₂29.0 / **AUPRC0.369** / MMD0.0238
  → Muon better on AUPRC (+8%), MMD (−16%), W₂; AdamW a hair better on raw FM loss. Muon led at
    **all 6 checkpoints** (2.5k→15k); the gap *widened through the cooldown*.

Seed-43:  muon-s43 final @15k = loss0.726 / W₂28.6 / **AUPRC0.349** / MMD0.0223
          adamw-s43 @10k (last captured) = loss0.793 / W₂30.4 / **AUPRC0.336** / MMD0.0252  ← no 15k final

## Honest takeaway (do NOT over-claim)
- Seed-42 = a **clear Muon win**. Seed-43 = the two were **tied (adamw even slightly ahead) pre-cooldown**
  (10k: muon 0.329 vs adamw 0.336); muon-s43 cooled to 0.349, and adamw-s43's cooldown-final was *not
  captured* but would likely land ~0.34–0.35 → **seed-43 looks like a near-tie.**
- Net: **Muon@1e-2 is never worse than AdamW@3e-3 and sometimes clearly better, at equal compute.**
  Worth adopting, but it is *not* the decisive uniform win that seed-42 alone implies. Needs seed-averaging.

## Other findings this round
- **ns=200 lifts the stable prior AUPRC to ~0.34–0.37** (vs ~0.25 at ns=100) → use ns=200 as the yardstick.
- The small model's SERGIO-prior fit **largely plateaus by ~2.5k steps** (AUPRC ~0.33, W₂ ~29); the
  **cooldown (last 30%) gives the real final boost** (→0.34–0.37). So the model is *capacity-limited*, not
  training-limited, on prior AUPRC; remaining headroom is in MMD/W₂/real-transfer.
- **Throughput:** ns=200/bs=32 warm ~3–5 it/s, but full-SERGIO degrades over a long run (down toward
  ~1 it/s late — likely NFS random-read on the 27 GB h5ad). 15k ≈ 2–2.5 h/run. First-val ODE compile is
  slow (~10–18 min, two val loaders); subsequent vals fast.
- **End-of-run real-Frangieh `test()` logs no metrics (known silent-skip bug)** → rely on `/prior` val.

## To continue (cheap → decisive)
1. **Finish seed-43 + add seed-44:** rerun `adamw3_s43` to 15k (it died at ~10.3k), then run
   `{muon1e2,adamw3}_s44`. Decide Muon adoption on the 2–3-seed average of prior AUPRC/MMD/W₂.
2. If Muon confirmed ≥ AdamW: update `TRAINING_EFFICIENCY.md` (reverse the old "Muon not adopted"),
   keep `optimizer_name=muon, peak_value=1e-2` as default-ish for SERGIO.
3. **Next lever (untouched): minibatch OT-CFM coupling** in `map_pfn/loss/loss_fn.py` (`fm_loss` is plain
   rectified-flow: `x_t=(1-t)x0+t·x1`, MSE on `v=x1-x0`, logit-normal `t`). Couple noise↔cells *within each
   population* (Sinkhorn, vmapped) to straighten paths — targets MMD/W₂ which don't plateau. Test on the
   subset first; watch the per-step cost (200×200 OT per population × bs).
4. Cheaper levers also open: bf16 weights for throughput; AdamW b2=0.95; longer-wall partition (gpu-2d)
   to escape the 5h cap for these ~2h runs.

## Cluster how-to (recipe)
- Sessions: project `jonasloos-mappfn`, gpu-5h, idle 300 min. Both deactivated. Reactivate, then datasets
  live at `datasets/synthetic/sergio.h5ad` (27 GB) + `sergio_sub400.h5ad` (subset, session-6bb4 only) +
  `datasets/single_cell/{frangieh,papalexi}.h5ad`. Env: `.venv` (py3.12) on NFS; if fresh,
  `uv sync --all-extras && uv pip install torchdyn torch_geometric`; verify `from map_pfn.configs.train import config_stores`.
- Launch full run: `bash run_exp_full.sh <name> cfg.seed=<S> cfg.globals.num_steps=15000 cfg.trainer.callbacks.2.max_steps=15000 cfg.module.optimizer_name={adamw|muon} cfg.module.lr_schedule.peak_value={3e-3|1e-2}`
  (run_exp_full.sh exists on both sessions; run_exp.sh = subset variant). Always `PYTHONPATH=/workdir`.
- Harvest `/prior` curve: `tail -c 3000000 exp_<name>.log | tr '\r' '\n' > /tmp/x; .venv/bin/python /workdir/pp.py /tmp/x`
  (`pp.py` = per-step dedup parser, lives in /workdir; tail first — logs get huge). NOTE log timestamps are
  +0200 local; the SLURM 5h wall cap runs on UTC.
