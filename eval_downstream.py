"""Zero-shot eval of a MapPFN checkpoint on a real single-cell dataset
(Frangieh / Papalexi) — the paper's headline track.

Handles both the official full model and our small model with one code path:
  which = "official" -> default arch (embed256/8-block/8-reg, accum=8)
  which = "small"    -> embed128/4-block/4-head/4-reg, accum=1

Usage:
  python eval_downstream.py <official|small> <dataset_path> <ckpt> <tag> [limit] [seed]
Writes <tag>.json.
"""
import json
import os
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

which = sys.argv[1]
dataset_path = sys.argv[2]
ckpt = sys.argv[3]
tag = sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 999
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
full = "full" in sys.argv[7:]  # full-coverage: every perturbation a query (not just the 32-pert holdout)
num_nodes = 50
num_samples = int(os.environ.get("NS", "200"))  # cells per population (paper=200)

if which == "small":
    mmdit = MMDiTConfig(
        embed_dim=128, cond_dim=128, noise_dim=num_nodes,
        num_heads=4, num_blocks=4, num_reg_tokens=4, key=builds(jr.key, seed),
    )
    model_cfg = MapPFNConfig(decoder=mmdit, in_dim=num_nodes, cond_dim=128, key=builds(jr.key, seed))
    accum = 1
elif which == "official":
    mmdit = MMDiTConfig(noise_dim=num_nodes, key=builds(jr.key, seed))  # defaults = paper arch
    model_cfg = MapPFNConfig(decoder=mmdit, in_dim=num_nodes, key=builds(jr.key, seed))
    accum = 8
else:
    raise SystemExit(f"which must be official|small, got {which}")

module = instantiate(
    JaxLightningModuleConfig(
        model=model_cfg,
        lr_schedule=LRScheduleConfig(total_steps=50_000),
        guidance=2.0,
        gradient_accumulation_steps=accum,
        key=builds(jr.key, seed),
    )
)
datamodule = instantiate(
    DataModuleConfig(
        dataset=PerturbationDatasetConfig(seed=seed, num_samples=num_samples),
        dataset_path=dataset_path,
        ood=False,
    )
)
module.load_checkpoint(ckpt)

datamodule.num_workers = 0
if hasattr(datamodule, "persistent_workers"):
    datamodule.persistent_workers = False

datamodule.setup("test")

trainer = JaxTrainer(
    callbacks=[TestMetrics(seed=seed)],
    enable_model_summary=False,
    limit_test_batches=limit,
    logger=False,
)

if full:
    # Full-coverage: every non-control condition becomes a query, with num_shots
    # demos drawn from the full pool of the same context (self-excluded). Covers
    # all Frangieh perts, not just the fixed 32-pert holdout.
    # NB: pass the loader DIRECTLY to trainer.test — calling trainer.test(datamodule=...)
    # re-runs datamodule.setup('test') and would silently revert to the 32-pert holdout.
    from torch.utils.data import DataLoader

    full_ds = datamodule.dataset(
        query_adata=datamodule.adata, context_adata=datamodule.adata, num_shots=datamodule.num_shots
    )
    print(f"[info] FULL coverage: {len(full_ds)} query perturbations", flush=True)
    loader = DataLoader(
        full_ds, batch_size=datamodule.batch_size, num_workers=0,
        collate_fn=datamodule._collate_fn, drop_last=False, shuffle=False,
    )
    print(f"[info] {which} {dataset_path}: test batches={len(loader)} bs={datamodule.batch_size} full=True", flush=True)
    trainer.test(module, dataloaders=loader)
else:
    tl = datamodule.test_dataloader()
    if isinstance(tl, list):
        tl = tl[0]
    print(f"[info] {which} {dataset_path}: test batches={len(tl)} bs={getattr(tl,'batch_size','?')} limit={limit}", flush=True)
    trainer.test(module, datamodule=datamodule)

metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
metrics.update({"_which": which, "_dataset": dataset_path, "_ckpt": ckpt, "_seed": seed, "_limit": limit})
print("METRICS_JSON=" + json.dumps(metrics), flush=True)
with open(f"{tag}.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[done] wrote {tag}.json", flush=True)
