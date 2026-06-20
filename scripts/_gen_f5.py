"""Generate F5: per-battery PSI-max vs RMSE scatter, colored by temperature group."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data = json.loads(Path("data/processed/modeling/drift_coverage_linkage.json").read_text())
records = data["records"]

# Known temperature groups (from model_metrics.json split_metadata)
TEMP_MAP = {
    "B0030": "hot", "B0033": "room", "B0036": "room",
    "B0041": "cold", "B0044": "cold", "B0052": "cold",
}
COLOR_MAP = {"hot": "#e6900a", "room": "#2196F3", "cold": "#9c27b0"}

valid = [r for r in records
         if r.get("psi_max") is not None and r.get("rmse") is not None]

fig, ax = plt.subplots(figsize=(7, 5))

plotted = {"hot": False, "room": False, "cold": False}
for r in valid:
    bat = r["battery_id"]
    grp = TEMP_MAP.get(bat, "room")
    psi = r["psi_max"]
    rmse = r["rmse"]
    col = COLOR_MAP[grp]
    label = grp if not plotted[grp] else None
    ax.scatter(psi, rmse, color=col, s=120, zorder=3, label=label)
    ax.annotate(bat, (psi, rmse), textcoords="offset points",
                xytext=(6, 3), fontsize=8)
    plotted[grp] = True

ax.axhline(y=44.74, color="grey", linestyle="--", alpha=0.6, linewidth=1,
           label="Multi-seed mean RMSE (44.74)")

ax.set_xlabel("Per-battery PSI_max (distribution shift from training)", fontsize=11)
ax.set_ylabel("XGBoost RMSE (cycles)", fontsize=11)
ax.set_title("Feature Distribution Shift vs Prediction Difficulty\n(Per Test Battery, Seed 42)", fontsize=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)

# Add annotation box
ax.text(0.97, 0.97,
    "Cold batteries: highest RMSE\n"
    "Conformal cov (cold) = 74.5%\nvs 80% target",
    transform=ax.transAxes, fontsize=8,
    va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = Path("paperprep/fig_drift_coverage.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()
