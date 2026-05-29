"""Extract the validation-metric curve from a captured training log.

Lightning logs val metrics only into the tqdm postfix (no CSV when wandb is
off), so we parse those lines. Prints one row per validation (distinct
global_step that carries a val/ metric).

Usage: python bench/extract_val.py <logfile> [metric1 metric2 ...]
Default metrics: val/loss val/deg_auprc val/deg_auprc/prior val/wasserstein val/wasserstein/prior
"""

from __future__ import annotations

import re
import sys

LOG = sys.argv[1]
METRICS = sys.argv[2:] or [
    "val/loss",
    "val/deg_auprc",
    "val/wasserstein",
    "val/deg_auprc/prior",
    "val/wasserstein/prior",
]

kv = re.compile(r"([A-Za-z0-9_/]+)=([-+0-9.einf]+)")

with open(LOG, encoding="utf-8", errors="ignore") as f:
    text = f.read().replace("\r", "\n")

rows: dict[int, dict[str, str]] = {}
for line in text.split("\n"):
    if "val/deg_auprc=" not in line:
        continue
    d = dict(kv.findall(line))
    step_s = d.get("train/global_step") or d.get("global_step")
    if step_s is None:
        continue
    try:
        step = int(float(step_s))
    except ValueError:
        continue
    rows[step] = d  # last write for a given step wins

steps = sorted(rows)
hdr = ["step", *METRICS]
print("\t".join(hdr))
prev = None
for s in steps:
    d = rows[s]
    vals = [d.get(m, "-") for m in METRICS]
    if vals == prev:  # collapse the per-step repeats of an unchanged val result
        continue
    prev = vals
    print("\t".join([str(s), *vals]))
