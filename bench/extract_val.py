"""Extract the validation-metric curve from a captured training log.

Lightning logs val metrics only into the tqdm postfix (no CSV when wandb is
off), so we parse those lines. Robust to scientific-notation steps (e.g.
``train/global_step=1e+4``) and to the real vs /prior dataloader metrics.

Usage: python bench/extract_val.py <logfile> [metric1 metric2 ...]
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

kv = re.compile(r"([A-Za-z0-9_/]+)=([-+0-9][-+0-9.eE]*)")

with open(LOG, encoding="utf-8", errors="ignore") as f:
    text = f.read().replace("\r", "\n")


def to_step(s: str) -> int | None:
    try:
        return int(round(float(s)))
    except (TypeError, ValueError):
        return None


rows: dict[int, dict[str, str]] = {}
for line in text.split("\n"):
    if "val/deg_auprc=" not in line:
        continue
    d = dict(kv.findall(line))
    step = to_step(d.get("train/global_step") or d.get("global_step") or "")
    if step is None:
        continue
    rows[step] = d  # last write for a given step wins

print("\t".join(["step", *METRICS]))
prev = None
for s in sorted(rows):
    vals = [rows[s].get(m, "-") for m in METRICS]
    if vals == prev:  # collapse per-step repeats of an unchanged val result
        continue
    prev = vals
    print("\t".join([str(s), *vals]))
