# Databricks notebook source
# MAGIC %pip install matplotlib==3.9.0 pyarrow==16.1.0 scipy==1.13.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy import stats
from pathlib import Path

FEATURES_PATH  = "/Volumes/critical_minerals/geochem/data/05_features.parquet"
OUTPUTS_PATH   = "/Volumes/critical_minerals/geochem/data/outputs"
Path(OUTPUTS_PATH).mkdir(exist_ok=True)

df = pd.read_parquet(FEATURES_PATH)
print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f5f5f3",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
print("Ready.")

# COMMAND ----------

# Shows log-normal behaviour — proves you understand geochemical data

elements = [
    ("copper_ppm",     "copper_log",     "Copper",     "#e74c3c"),
    ("nickel_ppm",     "nickel_log",     "Nickel",     "#3498db"),
    ("cobalt_ppm",     "cobalt_log",     "Cobalt",     "#9b59b6"),
    ("gold_ppm",       "gold_log",       "Gold",       "#f1c40f"),
    ("molybdenum_ppm", "molybdenum_log", "Molybdenum", "#1abc9c"),
    ("zinc_ppm",       "zinc_log",       "Zinc",       "#e67e22"),
]
elements = [(a,b,n,c) for a,b,n,c in elements if a in df.columns]

fig, axes = plt.subplots(2, len(elements), figsize=(20, 8))
fig.suptitle(
    "USGS Geochemical Survey — Element Distributions\n"
    "Raw concentrations (top) vs log-transformed (bottom)",
    fontsize=14, fontweight="bold"
)

for i, (raw_col, log_col, name, color) in enumerate(elements):
    raw_vals = df[raw_col].dropna()
    log_vals = df[log_col].dropna()

    # Raw
    axes[0, i].hist(raw_vals.clip(upper=raw_vals.quantile(0.99)),
                    bins=50, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
    axes[0, i].set_title(f"{name}\n(raw ppm)", fontsize=10)
    axes[0, i].set_xlabel("ppm")
    if i == 0: axes[0, i].set_ylabel("Count")

    # Log transformed + normal curve overlay
    axes[1, i].hist(log_vals, bins=50, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.3, density=True)
    mu, sigma = log_vals.mean(), log_vals.std()
    x = np.linspace(log_vals.min(), log_vals.max(), 200)
    axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma),
                    color="black", linewidth=1.5, linestyle="--", label="Normal fit")
    axes[1, i].set_title(f"log({name}+1)", fontsize=10)
    axes[1, i].set_xlabel("log(ppm+1)")
    if i == 0: axes[1, i].set_ylabel("Density")
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUTS_PATH}/01_element_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_element_distributions.png")

# COMMAND ----------

# The centrepiece visualisation — what Earth AI produces daily

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Copper Geochemical Anomaly — United States\n"
    "74,349 real USGS National Geochemical Survey samples",
    fontsize=14, fontweight="bold"
)

# ── Left: raw concentration ──────────────────────────────────────────
ax1 = axes[0]
clip_val = df["copper_ppm"].quantile(0.98)
sc1 = ax1.scatter(
    df["longitude"], df["latitude"],
    c=df["copper_ppm"].clip(upper=clip_val),
    cmap="YlOrRd",
    norm=mcolors.LogNorm(
        vmin=max(df["copper_ppm"].quantile(0.02), 0.1),
        vmax=clip_val
    ),
    s=1.5, alpha=0.6, linewidths=0
)
ax1.set_xlim(-128, -65)
ax1.set_ylim(24, 50)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Raw Copper Concentration (ppm)", fontsize=11)
ax1.set_facecolor("#e8f4f8")
plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Cu (ppm)")

# ── Right: z-score anomaly ───────────────────────────────────────────
ax2 = axes[1]
zvals = df["copper_zscore"].clip(-3, 5)
sc2 = ax2.scatter(
    df["longitude"], df["latitude"],
    c=zvals,
    cmap="RdYlBu_r",
    norm=mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=5),
    s=1.5, alpha=0.6, linewidths=0
)
ax2.set_xlim(-128, -65)
ax2.set_ylim(24, 50)
ax2.set_xlabel("Longitude")
ax2.set_title("Copper Anomaly Z-score\n(red = exploration target)", fontsize=11)
ax2.set_facecolor("#e8f4f8")
plt.colorbar(sc2, ax=ax2, shrink=0.7).set_label("Z-score")

# Mark top 1% anomalies
top1 = df[df["copper_zscore"] >= df["copper_zscore"].quantile(0.99)]
ax2.scatter(top1["longitude"], top1["latitude"],
            s=12, facecolors="none", edgecolors="black",
            linewidths=0.6, zorder=5, label=f"Top 1% (n={len(top1):,})")
ax2.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_PATH}/02_copper_anomaly_map.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_copper_anomaly_map.png")

# COMMAND ----------

# The final targeting output — what goes to the drill team

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "Composite Mineralisation Score — Exploration Targeting Output\n"
    "Weighted anomaly index: Cu(35%) + Ni(25%) + Co(15%) + Mo(10%) + Au(15%)",
    fontsize=13, fontweight="bold"
)

# ── Left: full dataset score map ─────────────────────────────────────
ax1 = axes[0]
score = df["mineralisation_score"]
sc1 = ax1.scatter(
    df["longitude"], df["latitude"],
    c=score,
    cmap="inferno",
    norm=mcolors.TwoSlopeNorm(
        vmin=score.quantile(0.02),
        vcenter=score.median(),
        vmax=score.quantile(0.98)
    ),
    s=1.5, alpha=0.7, linewidths=0
)
ax1.set_xlim(-128, -65)
ax1.set_ylim(24, 50)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title("Mineralisation Score (all samples)", fontsize=11)
ax1.set_facecolor("#1a1a2e")
plt.colorbar(sc1, ax=ax1, shrink=0.7).set_label("Score")

# ── Right: drill targets only ────────────────────────────────────────
ax2 = axes[1]
background = df[~df["is_drill_target"]]
targets    = df[df["is_drill_target"]]

ax2.scatter(background["longitude"], background["latitude"],
            c="#2c3e50", s=1, alpha=0.3, linewidths=0, label="Background")
ax2.scatter(targets["longitude"], targets["latitude"],
            c=targets["mineralisation_score"],
            cmap="YlOrRd", s=12, alpha=0.9,
            linewidths=0, zorder=5,
            label=f"Drill targets (n={len(targets):,})")

ax2.set_xlim(-128, -65)
ax2.set_ylim(24, 50)
ax2.set_xlabel("Longitude")
ax2.set_title("Priority Drill Targets (top 2%)\nColour = mineralisation score", fontsize=11)
ax2.set_facecolor("#1a1a2e")
ax2.legend(fontsize=9, loc="lower right",
           facecolor="#2c3e50", labelcolor="white")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_PATH}/03_mineralisation_score.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_mineralisation_score.png")

# COMMAND ----------

# Shows deposit-type signatures — proves domain knowledge

log_cols = [c for c in df.columns if c.endswith("_log")]
corr     = df[log_cols].dropna().corr()
labels   = [c.replace("_log","").replace("_"," ").title() for c in corr.columns]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Element Correlation Analysis — Deposit Type Signatures",
             fontsize=13, fontweight="bold")

# Heatmap
im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(labels)))
axes[0].set_yticks(range(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].set_title("Pearson Correlation\n(log-transformed concentrations)", fontsize=11)
plt.colorbar(im, ax=axes[0], shrink=0.8).set_label("Correlation")

for i in range(len(corr)):
    for j in range(len(corr)):
        val   = corr.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        axes[0].text(j, i, f"{val:.2f}", ha="center",
                     va="center", fontsize=7, color=color)

# Cu vs Mo scatter — porphyry copper pathfinder
if "copper_log" in df.columns and "molybdenum_log" in df.columns:
    data  = df[["copper_log", "molybdenum_log",
                "mineralisation_score"]].dropna()
    sc    = axes[1].scatter(
        data["copper_log"], data["molybdenum_log"],
        c=data["mineralisation_score"],
        cmap="plasma", alpha=0.4, s=3, linewidths=0
    )
    m, b, r, p, _ = stats.linregress(
        data["copper_log"], data["molybdenum_log"]
    )
    x_line = np.linspace(data["copper_log"].min(),
                         data["copper_log"].max(), 100)
    axes[1].plot(x_line, m*x_line+b, color="black",
                 linewidth=1.5, linestyle="--", label=f"r = {r:.2f}")
    axes[1].set_xlabel("log(Cu + 1)", fontsize=11)
    axes[1].set_ylabel("log(Mo + 1)", fontsize=11)
    axes[1].set_title(
        "Cu vs Mo — Porphyry Copper Pathfinder\n"
        "Colour = mineralisation score",
        fontsize=11
    )
    axes[1].legend(fontsize=10)
    plt.colorbar(sc, ax=axes[1], shrink=0.8).set_label("Mineralisation score")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_PATH}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 04_correlation_heatmap.png")

# COMMAND ----------

print("=" * 60)
print("CRITICAL MINERALS PIPELINE — COMPLETE")
print("=" * 60)
print(f"Data source:      USGS National Geochemical Survey")
print(f"Raw samples in:   74,408")
print(f"Clean samples:    {len(df):,}")

ppm_cols    = [c for c in df.columns if c.endswith("_ppm")]
log_cols    = [c for c in df.columns if c.endswith("_log")]
zscore_cols = [c for c in df.columns if c.endswith("_zscore")
               and "local" not in c]
local_cols  = [c for c in df.columns if c.endswith("_zscore_local")]
ratio_cols  = [c for c in df.columns if c.startswith("ratio_")]
cell_cols   = [c for c in df.columns if "_cell_" in c]

print(f"\nFeatures produced:")
print(f"  Concentrations (ppm):      {len(ppm_cols)}")
print(f"  Log transforms:            {len(log_cols)}")
print(f"  Global z-scores:           {len(zscore_cols)}")
print(f"  Local z-scores (geology):  {len(local_cols)}")
print(f"  Pathfinder ratios:         {len(ratio_cols)}")
print(f"  Grid aggregates:           {len(cell_cols)}")
print(f"  Total feature columns:     {len(df.columns)}")

print(f"\nExploration outputs:")
print(f"  Priority drill targets:    {df['is_drill_target'].sum():,} samples")
print(f"  Grid cells surveyed:       {df['grid_id'].nunique():,}")
print(f"  Top target location:")
top = df.nsmallest(1, "targeting_rank").iloc[0]
print(f"    Sample: {top['sample_id']}")
print(f"    Lat/Lon: {top['latitude']:.3f}, {top['longitude']:.3f}")
print(f"    Geology: {top['geology_unit']}")
print(f"    Cu: {top['copper_ppm']:.1f} ppm")
print(f"    Score: {top['mineralisation_score']:.3f}")

print(f"\nOutput files saved to:")
for f in Path(OUTPUTS_PATH).glob("*.png"):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:.0f} KB")
print("=" * 60)

# COMMAND ----------

# Display clickable download links for all output files
from IPython.display import HTML

files = list(Path(OUTPUTS_PATH).glob("*.png"))
links = ""
for f in sorted(files):
    size_kb = f.stat().st_size / 1024
    links += f'<p><a href="/files/Volumes/critical_minerals/geochem/data/outputs/{f.name}" download>{f.name}</a> — {size_kb:.0f} KB</p>'

display(HTML(f"<h3>Download your output maps:</h3>{links}"))

# COMMAND ----------

