"""Baseline performance profiling for the d=50 SERGIO->Frangieh downstream loop.

Measures, with wall-clock timers:
  1. DataModule construction (h5ad load) and setup() (dataset build / groupby).
  2. Per-batch dataloader cost (num_workers=0) over several iterations.
  3. Train make_step: first-step compile vs steady-state.
  4. ODE eval: Dopri5 (compile + solve) vs a fixed-step Euler variant
     (compile + solve), with a metric comparison so we know Euler is faithful.

Run:  PYTHONPATH=/workdir .venv/bin/python bench/profile_baseline.py [dataset_path]
"""

from __future__ import annotations

import sys
import time
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from hydra_zen import builds, instantiate

from map_pfn.configs.train.base_config import (
    DataModuleConfig,
    JaxLightningModuleConfig,
    LRScheduleConfig,
    MapPFNConfig,
    MMDiTConfig,
    PerturbationDatasetConfig,
)
from map_pfn.eval.metrics import compute_distribution_metrics

DATA = sys.argv[1] if len(sys.argv) > 1 else "datasets/single_cell/frangieh.h5ad"
PRIOR = sys.argv[2] if len(sys.argv) > 2 else "datasets/synthetic/sergio.h5ad"
SEED = 42
IN_DIM = 50


def t() -> float:
    return time.perf_counter()


def main() -> None:
    print(f"=== profiling: data={DATA} prior={PRIOR} ===", flush=True)

    # ---- 1. DataModule load + setup ----
    t0 = t()
    dataset_cfg = PerturbationDatasetConfig(seed=SEED, num_samples=200)
    datamodule = instantiate(
        DataModuleConfig(
            dataset=dataset_cfg,
            dataset_path=DATA,
            prior_dataset_path=PRIOR,
            ood=False,
            num_shots=8,
            batch_size=64,
            num_workers=0,
            persistent_workers=False,
        )
    )
    t_load = t() - t0
    print(f"[load] DataModule construct (read_h5ad x2): {t_load:.1f}s", flush=True)

    t0 = t()
    datamodule.setup("fit")
    t_build = t() - t0
    n_prior = len(datamodule.prior_train_dataset)
    print(f"[build] setup('fit'): {t_build:.1f}s  | prior_train samples={n_prior}", flush=True)

    # ---- 2. dataloader per-batch cost (num_workers=0) ----
    loader = datamodule.train_dataloader()
    it = iter(loader)
    batch_times = []
    for i in range(6):
        t0 = t()
        batch = next(it)
        batch_times.append(t() - t0)
    # first iter includes worker/startup; report both
    print(f"[dataloader] per-batch (nw=0) first={batch_times[0]:.3f}s "
          f"steady={sum(batch_times[1:]) / len(batch_times[1:]):.3f}s  "
          f"shapes obs={tuple(batch['obs_data'].shape)} int={tuple(batch['int_data'].shape)}", flush=True)

    # ---- 3. model + make_step compile vs steady ----
    mmdit = MMDiTConfig(noise_dim=IN_DIM, embed_dim=128, cond_dim=128,
                        num_heads=4, num_blocks=4, num_reg_tokens=4, key=builds(jr.key, SEED))
    model_cfg = MapPFNConfig(decoder=mmdit, in_dim=IN_DIM, cond_dim=128, key=builds(jr.key, SEED))
    module = instantiate(JaxLightningModuleConfig(
        model=model_cfg, lr_schedule=LRScheduleConfig(total_steps=3000, peak_value=1e-3),
        guidance=2.0, key=builds(jr.key, SEED), gradient_accumulation_steps=1, step_size=0.05))
    module.configure_optimizers()

    from map_pfn.data.utils import BatchKeys, filter_batch
    try:
        step_times = []
        for i in range(4):
            b = next(it)
            b = filter_batch(b, keys=[BatchKeys.OBS_DATA, BatchKeys.INT_DATA, BatchKeys.TREATMENT])
            b = eqx.filter_shard(b, module.data_sharding)
            module.train_key, tk = jr.split(module.train_key)
            t0 = t()
            loss, aux, module.model, module.opt_state, module.ema_state = type(module).make_step(
                module.model, b, module.loss_fn, module.opt_state, module.optimizer,
                module.ema_state, module.ema, tk, module.model_sharding)
            loss.block_until_ready()
            step_times.append(t() - t0)
        print(f"[make_step] compile(step0)={step_times[0]:.1f}s steady={sum(step_times[1:]) / 3:.3f}s", flush=True)
        module.ema_model = eqx.combine(module.ema_state.ema, eqx.filter(module.model, eqx.is_array, inverse=True))
    except Exception as e:  # noqa: BLE001
        print(f"[make_step] SKIPPED (standalone harness compile error): {type(e).__name__}: {str(e)[:120]}", flush=True)
        module.ema_model = module.model  # forward-only ODE comparison still valid

    # ---- 4. ODE eval: Dopri5 vs Euler ----
    val_loader = datamodule.val_dataloader()[0]  # frangieh val
    vb = next(iter(val_loader))
    vb = filter_batch(vb, keys=[BatchKeys.OBS_DATA, BatchKeys.INT_DATA, BatchKeys.TREATMENT])
    obs = jnp.asarray(vb["obs_data"]); intd = jnp.asarray(vb["int_data"]); trt = jnp.asarray(vb["treatment"])
    bsz, nc, s, d = intd.shape
    obs_cond, int_cond = obs[:, :-1] if False else obs, intd[:, :-1]
    # mirror module.step's conditioning split
    from map_pfn.data.utils import unpack_batch
    obs_full, int_full, obs_data_cond, int_data_cond, treatment = unpack_batch(vb)
    b2, _, s2, d2 = int_full.shape

    def make_solver(solver, step_size):
        @eqx.filter_jit
        def run(model, key):
            x_init = jr.normal(key, (b2, s2, d2))
            in_axes = (0, 0, None if int_data_cond is None else 0, 0)

            def vf(tt, x, args):
                tt = jnp.array(tt)
                mc = jax.vmap(partial(model, t=tt, drop_cond=False), in_axes=in_axes)
                mu = jax.vmap(partial(model, t=tt, drop_cond=True), in_axes=in_axes)
                vc = mc(x, obs_data_cond, int_data_cond, treatment)
                vu = mu(x, obs_data_cond, int_data_cond, treatment)
                return 2.0 * vc + (1 - 2.0) * vu

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(vf), solver=solver, t0=0.0, t1=1.0, dt0=step_size,
                y0=x_init, saveat=diffrax.SaveAt(t1=True),
                max_steps=4096,
            )
            return sol.ys[0]
        return run

    int_eval = int_full[:, -1]
    obs_eval = obs_full[:, -1]
    results = {}
    for name, solver, ss in [("dopri5@0.05", diffrax.Dopri5(), 0.05),
                             ("euler@0.05(20)", diffrax.Euler(), 0.05),
                             ("euler@0.1(10)", diffrax.Euler(), 0.1),
                             ("euler@0.2(5)", diffrax.Euler(), 0.2)]:
        run = make_solver(solver, ss)
        k = jr.PRNGKey(0)
        t0 = t(); out = run(module.ema_model, k); out.block_until_ready(); t_comp = t() - t0
        t0 = t(); out = run(module.ema_model, jr.PRNGKey(1)); out.block_until_ready(); t_solve = t() - t0
        mk = jr.PRNGKey(7)
        m = compute_distribution_metrics(obs_eval, int_eval, out, key=mk)
        results[name] = (t_comp, t_solve, m.get("wasserstein"), m.get("deg_auprc"), m.get("mmd"))
        print(f"[ode] {name:16s} compile={t_comp:6.1f}s solve={t_solve:6.2f}s "
              f"W2={m.get('wasserstein'):.2f} auprc={m.get('deg_auprc'):.3f} mmd={m.get('mmd'):.4f}", flush=True)

    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
