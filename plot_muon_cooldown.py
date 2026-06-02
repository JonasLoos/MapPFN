"""Muon vs AdamW: the SERGIO-prior win is a cooldown-overfit that hurts real transfer.

Data = prior-fit (held-out SERGIO /prior DEG-AUPRC) and real transfer (Frangieh val
W2, lower=better) at all 6 val checkpoints of each 15k run (ns=200, WSD cooldown 0.3).
"""
import matplotlib.pyplot as plt
import numpy as np

# step, prior_AUPRC, real_W2 (Frangieh val)  -- per run
RUNS = {
    "muon": {
        42: [(3500, 0.342, 34.10), (5000, 0.335, 21.10), (7500, 0.335, 26.10),
             (10000, 0.342, 29.50), (12500, 0.344, 30.60), (15000, 0.369, 63.60)],
        43: [(3500, 0.332, 30.80), (5000, 0.318, 30.40), (7500, 0.326, 26.90),
             (10000, 0.329, 27.40), (12500, 0.354, 28.20), (15000, 0.349, 47.90)],
        44: [(3500, 0.340, 31.20), (5000, 0.343, 22.10), (7500, 0.353, 21.10),
             (10000, 0.351, 23.60), (12500, 0.352, 29.30), (15000, 0.378, 51.90)],
    },
    "adamw": {
        43: [(3500, 0.341, 51.30), (5000, 0.333, 26.50), (7500, 0.331, 39.50),
             (10000, 0.331, 25.30), (12500, 0.333, 33.50), (15000, 0.332, 27.10)],
        44: [(3500, 0.339, 46.60), (5000, 0.319, 21.50), (7500, 0.327, 19.60),
             (10000, 0.330, 23.90), (12500, 0.321, 25.30), (15000, 0.336, 38.60)],
    },
}
COL = {"muon": "#c0392b", "adamw": "#2471a3"}
LAB = {"muon": "Muon@1e-2", "adamw": "AdamW@3e-3"}
STEPS = [3500, 5000, 7500, 10000, 12500, 15000]
COOLDOWN_START = 10500  # WSD decay_frac=0.3 of 15k

def mean_curve(opt, idx):
    arr = np.array([[row[idx] for row in seed] for seed in RUNS[opt].values()])
    return arr.mean(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# --- Panel A: prior-fit vs step ---
ax = axes[0]
for opt in RUNS:
    for seed, rows in RUNS[opt].items():
        s, p = [r[0] for r in rows], [r[1] for r in rows]
        ax.plot(s, p, color=COL[opt], alpha=0.25, lw=1)
    ax.plot(STEPS, mean_curve(opt, 1), color=COL[opt], lw=2.5, marker="o", label=LAB[opt])
ax.axvspan(COOLDOWN_START, 15000, color="gray", alpha=0.12)
ax.text(12750, ax.get_ylim()[0] if False else 0.317, "LR cooldown", ha="center", color="gray", fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("SERGIO /prior DEG-AUPRC  (↑ better)")
ax.set_title("A. Prior fit — Muon pulls ahead in the cooldown"); ax.legend(loc="upper left")

# --- Panel B: real transfer vs step ---
ax = axes[1]
for opt in RUNS:
    for seed, rows in RUNS[opt].items():
        s, w = [r[0] for r in rows], [r[2] for r in rows]
        ax.plot(s, w, color=COL[opt], alpha=0.25, lw=1)
    ax.plot(STEPS, mean_curve(opt, 2), color=COL[opt], lw=2.5, marker="o", label=LAB[opt])
ax.axvspan(COOLDOWN_START, 15000, color="gray", alpha=0.12)
ax.text(12750, 60, "LR cooldown", ha="center", color="gray", fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("real Frangieh W$_2$  (↓ better transfer)")
ax.set_title("B. Real transfer — Muon collapses in the cooldown"); ax.legend(loc="upper left")

# --- Panel C: the frontier (real transfer vs prior fit) ---
ax = axes[2]
for opt in RUNS:
    for seed, rows in RUNS[opt].items():
        for step, p, w in rows:
            final = step == 15000
            ax.scatter(p, w, color=COL[opt], s=120 if final else 45,
                       edgecolor="black" if final else "none", lw=1.2 if final else 0,
                       alpha=0.9 if final else 0.5, zorder=3 if final else 2)
# mean trajectory arrows pre-cooldown -> final
for opt in RUNS:
    pm, wm = mean_curve(opt, 1), mean_curve(opt, 2)
    ax.plot(pm, wm, color=COL[opt], lw=1.5, alpha=0.6)
    ax.annotate("", xy=(pm[-1], wm[-1]), xytext=(pm[-2], wm[-2]),
                arrowprops=dict(arrowstyle="-|>", color=COL[opt], lw=2))
ax.scatter([], [], color="gray", s=45, alpha=0.5, label="≤12.5k (pre-cooldown)")
ax.scatter([], [], color="gray", s=120, edgecolor="black", label="15k (post-cooldown)")
ax.set_xlabel("SERGIO /prior DEG-AUPRC  (↑ better fit)")
ax.set_ylabel("real Frangieh W$_2$  (↓ better transfer)")
ax.set_title("C. Frontier: same curve until Muon over-fits the prior")
ax.legend(loc="upper left", fontsize=8)

fig.suptitle("Muon's prior-AUPRC win is a cooldown-overfit: at matched prior-fit both transfer "
             "equally;\nthe gap opens only where Muon out-optimizes AdamW into the prior "
             "(real W$_2$ = Frangieh val, noisy)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("assets/muon_cooldown_overfit.png", dpi=140)
print("saved assets/muon_cooldown_overfit.png")
