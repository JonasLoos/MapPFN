"""Evaluate the official marvinsxtr/MapPFN checkpoint on held-out SERGIO.

Produces the same distribution metrics (deg_auprc, wasserstein, mmd, rmse, r2)
that our small-model runs report as `/prior`, so the numbers are directly
comparable. Writes results to official_eval_sergio.json.

Usage: .venv/bin/python eval_official_sergio.py [limit_test_batches] [seed]
"""
import json
import sys

from map_pfn.eval.evaluate import load_model
from map_pfn.utils.lightning import JaxTrainer, TestMetrics

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 8
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

trainer, module, datamodule = load_model(
    method="map_pfn",
    checkpoint_path="checkpoints/model.ckpt",
    dataset_path="datasets/synthetic/sergio.h5ad",
    num_samples=200,
    num_nodes=50,
    seed=seed,
)

# Force single-process data loading: forking dataloader workers while JAX is
# multithreaded deadlocks (the test loop hangs at 0/1). Training used num_workers=0.
datamodule.num_workers = 0
if hasattr(datamodule, "persistent_workers"):
    datamodule.persistent_workers = False

datamodule.setup("test")
tl = datamodule.test_dataloader()
if isinstance(tl, list):
    tl = tl[0]
print(f"[info] full test batches={len(tl)} batch_size={getattr(tl, 'batch_size', '?')} limit={limit}", flush=True)

# Rebuild trainer so we can bound the number of test batches and disable loggers.
trainer = JaxTrainer(
    callbacks=[TestMetrics(seed=seed)],
    enable_model_summary=False,
    limit_test_batches=limit,
    logger=False,
)
trainer.test(module, datamodule=datamodule)

metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
metrics["_limit_test_batches"] = limit
metrics["_num_samples"] = 200
metrics["_seed"] = seed
print("METRICS_JSON=" + json.dumps(metrics), flush=True)
with open("official_eval_sergio.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[done] wrote official_eval_sergio.json", flush=True)
