"""Aggregate + plot zero-shot transfer of the prior-variant models.
Reads fr_<tag>.json / pa_<tag>.json (from eval_downstream.py) for each variant and
prints a comparison table; optionally writes a bar-chart figure.

Usage: python bench/aggregate_variant_results.py [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VARIANTS = ["v0base", "v1noise", "v2dag", "v3combo"]
LABELS = {
    "v0base": "v0 baseline",
    "v1noise": "v1 realistic-noise",
    "v2dag": "v2 weak-effects",
    "v3combo": "v3 combined",
}
# lower-is-better for all except auprc/r2/mag_ratio(→1)
METRICS = ["deg_auprc", "wasserstein", "mmd", "mean_rmse", "mean_r2", "pds", "wasserstein_mag_ratio"]
SEED = sys.argv[1] if len(sys.argv) > 1 else "42"


def load(tag):
    p = Path(f"{tag}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def find_metric(d, name):
    # TestMetrics keys may be suffixed (e.g. "test/deg_auprc" or "deg_auprc/...")
    for k, v in d.items():
        if k == name or k.endswith("/" + name) or k.startswith(name):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def table(ds):
    print(f"\n===== {ds.upper()} (zero-shot, seed {SEED}) =====")
    rows = {}
    for v in VARIANTS:
        d = load(f"{ds}_{v}_s{SEED}")
        rows[v] = d
    hdr = f"{'variant':<20}" + "".join(f"{m.replace('wasserstein','W2').replace('_','.')[:9]:>10}" for m in METRICS)
    print(hdr)
    for v in VARIANTS:
        d = rows[v]
        if d is None:
            print(f"{LABELS[v]:<20}{'(missing)':>10}")
            continue
        cells = []
        for m in METRICS:
            val = find_metric(d, m)
            cells.append(f"{val:>10.4f}" if val is not None else f"{'-':>10}")
        print(f"{LABELS[v]:<20}" + "".join(cells))
    return rows


def main():
    fr = table("fr")
    pa = table("pa")
    print("\nNote: deg_auprc/mean_r2 higher=better; W2/mmd/rmse/pds lower=better; "
          "wasserstein_mag_ratio →1 ideal. Real metrics are noisy at single seed.")


if __name__ == "__main__":
    main()
