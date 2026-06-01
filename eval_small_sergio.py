"""Evaluate a SMALL (embed=128/4-block/4-head/4-reg) MapPFN checkpoint on the
SAME held-out SERGIO test split + protocol as eval_official_sergio.py, so the
small model and the official 519MB checkpoint are directly comparable.

Usage: .venv/bin/python eval_small_sergio.py <checkpoint_path> [limit] [seed]
"""
import json
import sys

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
from map_pfn.utils.lightning import JaxTrainer, TestMetrics

ckpt = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
num_nodes = 50

# Small architecture matching run_exp_full.sh (embed/cond=128, 4 blocks/heads/reg).
mmdit = MMDiTConfig(
    embed_dim=128,
    cond_dim=128,
    noise_dim=num_nodes,
    num_heads=4,
    num_blocks=4,
    num_reg_tokens=4,
    key=builds(jr.key, seed),
)
model_cfg = MapPFNConfig(decoder=mmdit, in_dim=num_nodes, cond_dim=128, key=builds(jr.key, seed))
module = instantiate(
    JaxLightningModuleConfig(
        model=model_cfg,
        lr_schedule=LRScheduleConfig(total_steps=50_000),
        guidance=2.0,
        # Match the training run: accum=1 means no optax.MultiSteps wrapper, so the
        # opt_state pytree matches the checkpoint (default accum=8 adds inner_opt_state).
        gradient_accumulation_steps=1,
        key=builds(jr.key, seed),
    )
)
datamodule = instantiate(
    DataModuleConfig(
        dataset=PerturbationDatasetConfig(seed=seed, num_samples=200),
        dataset_path="datasets/synthetic/sergio.h5ad",
        ood=False,
    )
)
module.load_checkpoint(ckpt)

datamodule.num_workers = 0
if hasattr(datamodule, "persistent_workers"):
    datamodule.persistent_workers = False

trainer = JaxTrainer(
    callbacks=[TestMetrics(seed=seed)],
    enable_model_summary=False,
    limit_test_batches=limit,
    logger=False,
)
trainer.test(module, datamodule=datamodule)

metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
metrics["_ckpt"] = ckpt
metrics["_limit_test_batches"] = limit
metrics["_seed"] = seed
print("METRICS_JSON=" + json.dumps(metrics), flush=True)
with open("small_eval_sergio.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[done] wrote small_eval_sergio.json", flush=True)
