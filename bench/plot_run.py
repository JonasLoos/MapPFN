"""Plot training dynamics for one run_*.log (a bench/run_full.sh training log).

Parses the lightning progress bar (train loss + applied lr, with epoch-wrap handling and
prior-vs-real validation curves) and writes a 4-panel figure: (a) train loss + LR,
(b) val loss, (c) val Wasserstein-2, (d) val DEG-AUPRC, each with the WSD cooldown shaded.
Also prints a spikiness summary and the per-validation prior/real table.

Usage: .venv/bin/python bench/plot_run.py <log_path> <out_png> [title]
  e.g. .venv/bin/python bench/plot_run.py _plotdata/run_clean_11m.log assets/clean_11m_training.png "11M clean base"
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

log_path = Path(sys.argv[1])
out_png = Path(sys.argv[2])
title = sys.argv[3] if len(sys.argv) > 3 else log_path.stem

txt = log_path.read_text(errors="replace").replace("\r", "\n")
NUM = r"([-\d.eE+]+)"
VK = ["val/loss", "val/wasserstein", "val/deg_auprc", "val/wasserstein_mag_ratio",
      "val/loss/prior", "val/wasserstein/prior", "val/deg_auprc/prior", "val/mmd/prior",
      "val/wasserstein_mag_ratio/prior"]

loss_by, lr_by, valrows, seen = {}, {}, {}, set()
epoch, elen = 0, 8436
for line in txt.split("\n"):
    em = re.search(r"Epoch (\d+):", line)
    epoch = int(em.group(1)) if em else epoch
    pm = re.search(r"\|\s*(\d+)/(\d+) ", line)
    if not pm or "train/loss=" not in line:
        continue
    n = int(pm.group(1)); elen = int(pm.group(2)); step = epoch * elen + n
    tl = re.search(r"train/loss=" + NUM, line); lr = re.search(r"train/lr=" + NUM, line)
    if tl:
        loss_by[step] = float(tl.group(1))
    if lr:
        lr_by[step] = max(lr_by.get(step, -1.0), float(lr.group(1)))  # max defeats accum/val 0.000 repaints
    if "val/wasserstein/prior=" in line:
        m = {k: re.search(re.escape(k) + "=" + NUM, line) for k in VK}
        if all(m.values()):
            key = tuple(round(float(v.group(1)), 5) for v in m.values())
            if key not in seen:
                seen.add(key); valrows[step] = {k: float(v.group(1)) for k, v in m.items()}

steps = np.array(sorted(loss_by)); tl = np.array([loss_by[s] for s in steps])
lr = np.array([lr_by.get(s, np.nan) for s in steps])
vs = np.array(sorted(valrows)); V = {k: np.array([valrows[s][k] for s in vs]) for k in VK}
total = int(steps.max()) if len(steps) else 0
cd0 = int(total * 0.7)  # WSD decay_frac=0.3


def movmed(y, w=151):
    return np.array([np.median(y[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(y))])


warm = steps > 500
res = (tl[warm] - movmed(tl)[warm]) if warm.any() else np.array([0.0])
sd = res.std()
print(f"train steps={len(steps)} (max {total})")
print(f"loss: start~{tl[steps<200].mean():.2f}  end~{tl[steps>0.98*total].mean():.3f}")
print(f"spikiness: resid std={sd:.4f}  max|resid|={np.abs(res).max():.3f}  #(>5sd)={(np.abs(res)>5*sd).sum()}")
for s in vs:
    r = valrows[s]
    print(f"  step {s:6d} | PRIOR loss {r['val/loss/prior']:.3f} W2 {r['val/wasserstein/prior']:5.1f} "
          f"AUPRC {r['val/deg_auprc/prior']:.3f} | REAL-Fr W2 {r['val/wasserstein']:5.1f} "
          f"AUPRC {r['val/deg_auprc']:.3f} magR {r['val/wasserstein_mag_ratio']:.2f}")


def smooth(y, w=101):
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


fig, ax = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(title, fontweight="bold")
a = ax[0, 0]
a.plot(steps, tl, color="0.8", lw=.5, label="train/loss raw")
a.plot(steps, smooth(tl), color="C0", lw=1.8, label="smoothed")
a.axvspan(cd0, total, color="orange", alpha=.10); a.set_ylim(0.6, 1.8)
a.set_title("(a) train loss + LR"); a.set_xlabel("step"); a.set_ylabel("loss", color="C0")
a2 = a.twinx(); a2.plot(steps, lr, color="C3", ls="--", lw=1.3); a2.set_ylabel("lr", color="C3")
a.legend(loc="upper right", fontsize=7)
for axx, key_p, key_r, ttl in [
    (ax[0, 1], "val/loss/prior", "val/loss", "(b) val loss"),
    (ax[1, 0], "val/wasserstein/prior", "val/wasserstein", "(c) val Wasserstein-2"),
    (ax[1, 1], "val/deg_auprc/prior", "val/deg_auprc", "(d) val DEG-AUPRC"),
]:
    axx.plot(vs, V[key_p], "o-", color="C2", label="prior")
    axx.plot(vs, V[key_r], "s--", color="C1", label="real Frangieh")
    axx.axvspan(cd0, total, color="orange", alpha=.10)
    axx.set_title(ttl); axx.set_xlabel("step"); axx.legend(fontsize=8); axx.grid(alpha=.3)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=130)
print("wrote", out_png)
