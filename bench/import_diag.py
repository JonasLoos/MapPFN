"""Step-by-step import timer to find which import in the train.py chain hangs
on a fresh gpu-5h node. Prints a timestamped line after each import (flushed), so
the last line before a stall/timeout identifies the culprit (the NEXT import).
Run: timeout 400 .venv/bin/python bench/import_diag.py > import_diag.log 2>&1
"""
import time

T0 = time.time()


def t(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


t("start")
import os  # noqa: E402

t(f"os (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}, WANDB_MODE={os.environ.get('WANDB_MODE')})")
import numpy  # noqa: E402, F401

t("numpy")
import torch  # noqa: E402, F401

t("torch")
import jax  # noqa: E402

t("jax (module)")
print("  jax.devices:", jax.devices(), flush=True)
t("jax.devices()")
import equinox  # noqa: E402, F401

t("equinox")
import anndata  # noqa: E402, F401

t("anndata")
import scanpy  # noqa: E402, F401

t("scanpy")
import lightning  # noqa: E402, F401

t("lightning")
from lightning.pytorch.loggers import WandbLogger  # noqa: E402, F401

t("WandbLogger")
import hydra_zen  # noqa: E402, F401

t("hydra_zen")
import wandb  # noqa: E402, F401

t("wandb")
from map_pfn.data.sergio_dataset import SergioDataset  # noqa: E402, F401

t("sergio_dataset (Rust ext)")
from map_pfn.utils.config import run  # noqa: E402, F401

t("map_pfn.utils.config")
from map_pfn.configs.train.base_config import TrainingRun  # noqa: E402, F401

t("base_config")
from map_pfn.eval.evaluate import evaluate_baselines  # noqa: E402, F401

t("eval.evaluate")
print("ALL_IMPORTS_OK", flush=True)
