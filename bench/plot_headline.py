"""Plots for the full-scale v1noise headline run (3.36M, hl3m/hl3mv2) + the
1000-ctx factorial comparison. Produces assets/headline_*.png.

Two figures:
  headline_training.png  - how the hl3mv2 run trained (train loss, val loss,
                           SERGIO-prior W2/AUPRC, real-Frangieh-val W2/AUPRC vs step)
  headline_transfer.png  - zero-shot test-split transfer (W2, AUPRC, mag-ratio)
                           across configs: factorial v0base/v1noise/+Muon/11M+Muon,
                           full-scale hl3m/hl3mv2, official 43M baseline.

Run: .venv/bin/python bench/plot_headline.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PLOT = ROOT / "_plotdata"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Parse the hl3mv2 training log into per-step train loss + val checkpoints.
# ----------------------------------------------------------------------------
def parse_train_log(path: Path):
    # The lightning bar's `train/global_step` is rounded to 1 sig-fig (5e+3) and the
    # epoch progress counter `N/EPOCH_LEN` wraps each epoch, so the true micro-step is
    # epoch*EPOCH_LEN + N. EPOCH_LEN is read from the bar ("N/<EPOCH_LEN>").
    text = path.read_text(errors="replace").replace("\r", "\n")
    num = r"([-\d.eE+]+)"
    loss_by_step = {}  # true_step -> last loss
    lr_by_step = {}     # true_step -> MAX lr (the bar logs lr=0.000 on validation
                        # repaints and sub-resolution steps; max recovers the real value)
    val_rows = {}    # true_step -> dict (first time each val-tuple appears)
    seen_val = set()
    epoch = 0
    epoch_len = None
    for line in text.split("\n"):
        em = re.search(r"Epoch (\d+):", line)
        if em:
            epoch = int(em.group(1))
        pm = re.search(r"\|\s*(\d+)/(\d+) ", line)
        if not pm or "train/loss=" not in line:
            continue
        n = int(pm.group(1))
        epoch_len = int(pm.group(2))
        step = epoch * epoch_len + n
        tl = re.search(r"train/loss=" + num, line)
        lr = re.search(r"train/lr=" + num, line)
        try:
            loss_by_step[step] = float(tl.group(1))
            if lr:
                lr_by_step[step] = max(lr_by_step.get(step, -1.0), float(lr.group(1)))
        except (ValueError, AttributeError):
            pass
        if "val/wasserstein/prior=" in line:
            keys = ["val/loss", "val/wasserstein", "val/deg_auprc",
                    "val/loss/prior", "val/wasserstein/prior", "val/deg_auprc/prior",
                    "val/mmd/prior", "val/wasserstein_mag_ratio/prior"]
            m = {k: re.search(rf"{re.escape(k)}=" + num, line) for k in keys}
            if all(v is not None for v in m.values()):
                key = tuple(round(float(v.group(1)), 5) for v in m.values())
                if key not in seen_val:
                    seen_val.add(key)
                    val_rows[step] = {k: float(v.group(1)) for k, v in m.items()}
    steps = np.array(sorted(loss_by_step))
    tloss = np.array([loss_by_step[s] for s in steps])
    tlr = np.array([lr_by_step.get(s, np.nan) for s in steps])
    vsteps = np.array(sorted(val_rows))
    vals = {k: np.array([val_rows[s][k] for s in vsteps])
            for k in next(iter(val_rows.values()))}
    return steps, tloss, tlr, vsteps, vals


steps, tloss, tlr, vsteps, vals = parse_train_log(PLOT / "hl3mv2.log")
print(f"train points: {len(steps)} (max step {steps.max():.0f}); val checkpoints at {vsteps}")

# ----------------------------------------------------------------------------
# FIGURE 1: training dynamics
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(
    "Full-scale v1noise headline run — 3.36M, Muon@1e-2, bs32+accum2, ns200, 10k micro-steps (hl3mv2)",
    fontsize=12, fontweight="bold",
)

# IMPORTANT: the logged `train/lr` (line 122 in jax_lightning.py) evaluates the schedule
# at `global_step` = MICRO-steps, but the schedule's total_steps was sized in OPTIMIZER
# UPDATES (=num_steps/accum). Under accum>1 the optimizer is wrapped in MultiSteps, so the
# APPLIED lr advances once per accum micro-steps. => the logged lr is horizontally
# compressed by `accum` (appears to hit 0 at step 5000), but the lr actually applied to
# the weights decays over the full run. Reconstruct the true applied lr = logged(step/accum).
ACCUM = 2
true_lr = np.interp(steps / ACCUM, steps, np.nan_to_num(tlr, nan=0.0))
PEAK = np.nanmax(tlr) if np.isfinite(tlr).any() else 0.01
_decaying = steps[(true_lr < 0.99 * PEAK) & (steps > 500)]
COOLDOWN_START = int(_decaying.min()) if len(_decaying) else 7000   # ~7000 (last 30%)


def labelpts(ax, xs, ys, fmt="{:.3f}", dy=0.0, color="k"):
    for x_, y_ in zip(xs, ys):
        ax.annotate(fmt.format(y_), (x_, y_), textcoords="offset points",
                    xytext=(0, 6 + dy), ha="center", fontsize=7, color=color)


def cooldown_span(ax):
    ax.axvspan(COOLDOWN_START, 10000, color="orange", alpha=0.10)   # true WSD cooldown


# (a) train loss + LR
a = ax[0, 0]
def smooth(y, w=101):  # centered moving average, edge-correct via convolution with counts
    if len(y) < w:
        return y
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")
a.plot(steps, tloss, color="0.8", lw=0.5, label="train/loss (raw)")
a.plot(steps, smooth(tloss), color="C0", lw=1.8, label="train/loss (smoothed)")
a.set_xlabel("micro-step")
a.set_ylabel("flow-matching loss", color="C0")
a.set_ylim(0.6, 1.8)
a.set_title("(a) training loss + LR schedule")
a2 = a.twinx()
a2.plot(steps, true_lr, color="C3", lw=1.6, ls="--", label="train/lr (applied)")
a2.plot(steps, tlr, color="C3", lw=0.8, ls=":", alpha=0.45,
        label="train/lr (logged — accum artifact)")
a2.set_ylabel("learning rate", color="C3")
a2.set_ylim(0, PEAK * 1.15)
cooldown_span(a)
a.text((COOLDOWN_START + 10000) / 2, 1.68, "WSD\ncooldown", ha="center",
       color="darkorange", fontsize=8)
a.legend(loc="upper left", fontsize=7)

# (b) val loss: prior vs real-Frangieh
b = ax[0, 1]
b.plot(vsteps, vals["val/loss/prior"], "o-", color="C2", label="SERGIO prior (val)")
b.plot(vsteps, vals["val/loss"], "s--", color="C1", label="real Frangieh (val)")
labelpts(b, vsteps, vals["val/loss/prior"], color="C2")
labelpts(b, vsteps, vals["val/loss"], dy=-16, color="C1")
cooldown_span(b)
b.set_xlabel("micro-step"); b.set_ylabel("flow-matching loss")
b.set_title("(b) validation loss"); b.legend(fontsize=8); b.grid(alpha=0.3)

# (c) Wasserstein: prior vs real
c = ax[1, 0]
c.plot(vsteps, vals["val/wasserstein/prior"], "o-", color="C2", label="SERGIO prior (val)")
c.plot(vsteps, vals["val/wasserstein"], "s--", color="C1", label="real Frangieh (val)")
labelpts(c, vsteps, vals["val/wasserstein/prior"], "{:.1f}", color="C2")
labelpts(c, vsteps, vals["val/wasserstein"], "{:.1f}", dy=-16, color="C1")
cooldown_span(c)
c.set_xlabel("micro-step"); c.set_ylabel("Wasserstein-2  ↓")
c.set_title("(c) distribution fit (W₂)"); c.legend(fontsize=8); c.grid(alpha=0.3)

# (d) DEG AUPRC: prior vs real
d = ax[1, 1]
d.plot(vsteps, vals["val/deg_auprc/prior"], "o-", color="C2", label="SERGIO prior (val)")
d.plot(vsteps, vals["val/deg_auprc"], "s--", color="C1", label="real Frangieh (val)")
labelpts(d, vsteps, vals["val/deg_auprc/prior"], color="C2")
labelpts(d, vsteps, vals["val/deg_auprc"], dy=-16, color="C1")
cooldown_span(d)
d.set_xlabel("micro-step"); d.set_ylabel("DEG AUPRC  ↑")
d.set_title("(d) DE-gene recovery (AUPRC)"); d.legend(fontsize=8); d.grid(alpha=0.3)
d.text(0.5, -0.32,
       "The WSD cooldown (orange, true applied LR over steps 7k-10k) deepens the SERGIO-prior "
       "fit (prior AUPRC ↑, loss ↓)\nwhile real-Frangieh W₂/AUPRC do NOT improve — the "
       "prior-vs-transfer tradeoff. (The logged train/lr is compressed ×accum, a display-only\n"
       "artifact of jax_lightning.py:122; the applied LR is correct. Only mid + final "
       "validations were logged.)",
       transform=d.transAxes, ha="center", va="top", fontsize=8, color="0.3")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(ASSETS / "headline_training.png", dpi=130)
print("wrote", ASSETS / "headline_training.png")


# ----------------------------------------------------------------------------
# 2. Transfer comparison across configs.
# ----------------------------------------------------------------------------
def load(p):
    return json.loads(Path(p).read_text())


def from_block(txtfile, ckpt_substr, ds):
    """Pull one JSON object out of a concatenated variant_results_*.txt."""
    blocks = re.findall(r"\{.*?\}", (ROOT / txtfile).read_text(), re.S)
    for blk in blocks:
        try:
            o = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if ckpt_substr in o.get("_ckpt", "") and ds in o.get("_dataset", ""):
            return o
    raise KeyError(f"{ckpt_substr}/{ds} not in {txtfile}")


# config -> (frangieh dict, papalexi dict)
configs = [
    ("v0base\n(orig prior)", "0.6",
     from_block("variant_results_s42.txt", "prior_v0base_s42/", "frangieh"),
     from_block("variant_results_s42.txt", "prior_v0base_s42/", "papalexi")),
    ("v1noise\n(realistic)", "C0",
     from_block("variant_results_s42.txt", "prior_v1noise_s42/", "frangieh"),
     from_block("variant_results_s42.txt", "prior_v1noise_s42/", "papalexi")),
    ("v1noise\n+Muon", "C9",
     from_block("variant_results_muon_s42.txt", "prior_v1noise_s42_muon/", "frangieh"),
     from_block("variant_results_muon_s42.txt", "prior_v1noise_s42_muon/", "papalexi")),
    ("11M\n+Muon", "C4",
     from_block("variant_results_medmuon_s42.txt", "prior_v1noise_s42_medmuon/", "frangieh"),
     from_block("variant_results_medmuon_s42.txt", "prior_v1noise_s42_medmuon/", "papalexi")),
    ("full hl3m\n(buggy NW)", "C5",
     load(PLOT / "fr_v1noise_full_s42_hl3m.json"),
     load(PLOT / "pa_v1noise_full_s42_hl3m.json")),
    ("full hl3mv2\n(RNG fix)", "C3",
     load(PLOT / "fr_v1noise_full_s42_hl3mv2.json"),
     load(PLOT / "pa_v1noise_full_s42_hl3mv2.json")),
    ("official\n43M", "0.3",
     load(ROOT / "frangieh_official.json"),
     load(ROOT / "papalexi_official.json")),
]

labels = [c[0] for c in configs]
colors = [c[1] for c in configs]
x = np.arange(len(configs))

metrics = [
    ("test/wasserstein", "Wasserstein-2  ↓", None),
    ("test/deg_auprc", "DEG AUPRC  ↑", 0.34),  # paper line
    ("test/wasserstein_mag_ratio", "magnitude ratio  →1", 1.0),
]

fig2, axes = plt.subplots(3, 2, figsize=(13, 11))
fig2.suptitle(
    "Zero-shot transfer (test split, ns=200, seed 42) — prior-variant factorial → full-scale → official",
    fontsize=12, fontweight="bold",
)
for col, (ds_name, idx) in enumerate([("Frangieh (melanoma)", 2), ("Papalexi (leukemia)", 3)]):
    for row, (key, ylab, ref) in enumerate(metrics):
        axx = axes[row, col]
        vals_bar = [configs[i][idx][key] for i in range(len(configs))]
        bars = axx.bar(x, vals_bar, color=colors, edgecolor="k", lw=0.5)
        for xi, v in zip(x, vals_bar):
            axx.text(xi, v, f"{v:.3f}" if v < 2 else f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7)
        if ref is not None:
            axx.axhline(ref, color="red", ls=":", lw=1.2,
                        label=("paper 0.34 (10-seed)" if "auprc" in key else
                               ("ideal=1" if "mag" in key else None)))
            if axx.get_legend_handles_labels()[1]:
                axx.legend(fontsize=7, loc="upper right")
        axx.set_xticks(x)
        axx.set_xticklabels(labels, fontsize=7)
        axx.set_ylabel(ylab)
        if row == 0:
            axx.set_title(ds_name, fontsize=11, fontweight="bold")
        axx.grid(axis="y", alpha=0.3)
        # headroom for labels
        top = max(vals_bar) * 1.18
        axx.set_ylim(0, max(top, (ref or 0) * 1.1))

fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig(ASSETS / "headline_transfer.png", dpi=130)
print("wrote", ASSETS / "headline_transfer.png")
