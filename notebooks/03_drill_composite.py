# Databricks notebook source
# MAGIC %pip install numpy pandas pyarrow==16.1.0 matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np

DRILL_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/"

desurveyed_df = pd.read_parquet(DRILL_PATH + "03_desurveyed.parquet")

print(f"Loaded: {len(desurveyed_df):,} intervals")
print(f"Columns: {list(desurveyed_df.columns)}")
display(desurveyed_df.head(3))

# COMMAND ----------

# Same BDL substitution as Project 1
# Must do this BEFORE compositing
# If BDL values are negative during compositing the weighted average is wrong

element_cols = ["Cu_ppm", "Ni_ppm", "Co_ppm", "Au_ppb", "Mo_ppm"]

print("BDL substitution (negative = below detection limit):")
print("-" * 50)

for col in element_cols:
    bdl_count = (desurveyed_df[col] < 0).sum()
    desurveyed_df[col] = desurveyed_df[col].apply(
        lambda x: abs(x) / 2.0 if pd.notna(x) and x < 0 else x
    )
    print(f"  {col:<12} {bdl_count:>4} BDL values substituted with abs(val)/2")

print(f"\nNo negative values remain:")
for col in element_cols:
    neg = (desurveyed_df[col] < 0).sum()
    print(f"  {col:<12} {neg} negatives remaining")

# COMMAND ----------

COMPOSITE_LENGTH = 2.0  # metres — industry standard

def composite_hole(hole_df, composite_length=2.0):
    """
    Aggregate 1m assay intervals into composite_length intervals
    using length-weighted mean.

    WHY LENGTH-WEIGHTED MEAN?
    A 0.8m interval at 500 ppm contributes LESS total metal
    than a 1.2m interval at 500 ppm.
    Simple mean ignores this. Length-weighted mean corrects for it.

    Formula:
    composite_grade = sum(grade * length) / sum(length)
    """
    element_cols = ["Cu_ppm", "Ni_ppm", "Co_ppm", "Au_ppb", "Mo_ppm"]
    hole_df = hole_df.sort_values("From").reset_index(drop=True)

    composites = []

    # Find start and end of this hole's assay record
    min_depth = hole_df["From"].min()
    max_depth = hole_df["To"].max()

    # Generate composite boundaries
    comp_starts = np.arange(min_depth, max_depth, composite_length)

    for comp_from in comp_starts:
        comp_to = comp_from + composite_length

        # Find all 1m intervals that overlap this composite window
        overlap = hole_df[
            (hole_df["From"] < comp_to) &
            (hole_df["To"]   > comp_from)
        ].copy()

        if len(overlap) == 0:
            continue

        # Calculate actual overlap length for each interval
        # An interval might only partially overlap the composite window
        overlap["overlap_from"] = overlap["From"].clip(lower=comp_from)
        overlap["overlap_to"]   = overlap["To"].clip(upper=comp_to)
        overlap["weight"]       = overlap["overlap_to"] - overlap["overlap_from"]

        # Remove zero-weight rows
        overlap = overlap[overlap["weight"] > 0]
        if len(overlap) == 0:
            continue

        total_weight = overlap["weight"].sum()

        # Skip if core recovery was very poor
        # WHY? If only 20% of core was recovered, the composite
        # grade is not representative of the full interval
        avg_recovery = (
            overlap["Recovery_pct"] * overlap["weight"]
        ).sum() / total_weight if "Recovery_pct" in overlap.columns else 100

        # Build composite row
        comp = {
            "HoleID":       hole_df["HoleID"].iloc[0],
            "From":         comp_from,
            "To":           comp_to,
            "MidDepth":     (comp_from + comp_to) / 2,
            "Length":       total_weight,
            "Recovery_pct": round(avg_recovery, 1),
            "n_samples":    len(overlap),
        }

        # Length-weighted mean for each element
        for col in element_cols:
            valid = overlap[overlap[col].notna()]
            if len(valid) == 0:
                comp[col] = np.nan
            else:
                w = valid["weight"].values
                v = valid[col].values
                comp[col] = (v * w).sum() / w.sum()

        # Length-weighted mean XYZ (midpoint coordinates)
        for coord in ["X", "Y", "Z"]:
            if coord in overlap.columns:
                valid = overlap[overlap[coord].notna()]
                if len(valid) > 0:
                    w = valid["weight"].values
                    v = valid[coord].values
                    comp[coord] = (v * w).sum() / w.sum()
                else:
                    comp[coord] = np.nan

        composites.append(comp)

    return pd.DataFrame(composites)


# Test on one hole
test_hole = desurveyed_df[desurveyed_df["HoleID"] == "DH001"]
test_comp = composite_hole(test_hole, COMPOSITE_LENGTH)

print(f"DH001: {len(test_hole)} x 1m intervals → {len(test_comp)} x 2m composites")
print(f"\nRaw 1m intervals (first 6):")
display(test_hole[["From","To","Cu_ppm","Au_ppb","X","Y","Z"]].head(6))
print(f"\nComposited 2m intervals (first 3):")
display(test_comp[["From","To","Length","Cu_ppm","Au_ppb","X","Y","Z"]].head(3))

# COMMAND ----------

print("Compositing all 50 drill holes...")
print(f"Composite length: {COMPOSITE_LENGTH}m")
print("-" * 40)

all_composites = []

for hole_id in desurveyed_df["HoleID"].unique():
    hole_df = desurveyed_df[desurveyed_df["HoleID"] == hole_id]
    comp_df = composite_hole(hole_df, COMPOSITE_LENGTH)
    all_composites.append(comp_df)

composites_df = pd.concat(all_composites, ignore_index=True)

print(f"Input:   {len(desurveyed_df):,} x 1m intervals")
print(f"Output:  {len(composites_df):,} x 2m composites")
print(f"Ratio:   {len(desurveyed_df)/len(composites_df):.1f}x reduction")
display(composites_df.head(5))

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Grade Distribution — 1m Raw vs 2m Composites\n"
    "Compositing smooths noise while preserving the mineralised zone signal",
    fontsize=13, fontweight="bold"
)

elements = [
    ("Cu_ppm",  "Copper (ppm)",     "#e74c3c"),
    ("Ni_ppm",  "Nickel (ppm)",     "#3498db"),
    ("Co_ppm",  "Cobalt (ppm)",     "#9b59b6"),
    ("Au_ppb",  "Gold (ppb)",       "#f1c40f"),
    ("Mo_ppm",  "Molybdenum (ppm)", "#1abc9c"),
]

for i, (col, label, color) in enumerate(elements):
    ax = axes[i // 3, i % 3]

    raw_vals  = desurveyed_df[col].dropna()
    comp_vals = composites_df[col].dropna()

    # Clip to 99th percentile for display
    clip_val = raw_vals.quantile(0.99)
    ax.hist(raw_vals.clip(upper=clip_val),  bins=50, alpha=0.5,
            color=color,   label="1m raw",        density=True)
    ax.hist(comp_vals.clip(upper=clip_val), bins=50, alpha=0.5,
            color="black", label="2m composite",  density=True)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Concentration")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

# Hide unused subplot
axes[1, 2].set_visible(False)

plt.tight_layout()
OUTPUT_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/outputs/"
plt.savefig(OUTPUT_PATH + "02_grade_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_grade_distributions.png")

# COMMAND ----------

# Show how copper grade changes with depth
# This should show the mineralised zone at 80-150m clearly

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle(
    "Copper Grade vs Depth\nMineralised zone visible at 80-150m",
    fontsize=13, fontweight="bold"
)

# ── Left: all holes overlaid ──
ax1 = axes[0]
for hole_id in composites_df["HoleID"].unique()[:20]:  # first 20 holes
    hole = composites_df[composites_df["HoleID"] == hole_id]
    hole = hole.dropna(subset=["Cu_ppm"])
    if len(hole) == 0:
        continue
    ax1.plot(
        hole["Cu_ppm"].clip(upper=500),
        hole["MidDepth"],
        alpha=0.3, linewidth=0.8, color="#e74c3c"
    )

ax1.axhspan(80, 150, alpha=0.15, color="gold", label="Mineralised zone (80-150m)")
ax1.set_xlabel("Copper (ppm)")
ax1.set_ylabel("Depth (m)")
ax1.invert_yaxis()
ax1.set_title("All holes (first 20)", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 500)

# ── Right: average grade per depth interval ──
ax2 = axes[1]
depth_avg = composites_df.groupby("MidDepth")["Cu_ppm"].mean()
ax2.barh(depth_avg.index, depth_avg.values,
         height=1.8, color="#e74c3c", alpha=0.7)
ax2.axhspan(80, 150, alpha=0.15, color="gold", label="Mineralised zone")
ax2.set_xlabel("Average Copper (ppm)")
ax2.set_ylabel("Depth (m)")
ax2.invert_yaxis()
ax2.set_title("Average grade at each depth\n(all 50 holes)", fontsize=11)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "03_grade_vs_depth.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_grade_vs_depth.png")

# COMMAND ----------

composites_df.to_parquet(DRILL_PATH + "04_composites.parquet", index=False)

print("=" * 55)
print("STAGE 4 — COMPOSITING COMPLETE")
print("=" * 55)
print(f"Raw 1m intervals:    {len(desurveyed_df):,}")
print(f"2m composites:       {len(composites_df):,}")

ppm_cols = ["Cu_ppm", "Ni_ppm", "Co_ppm", "Mo_ppm"]
ppb_cols = ["Au_ppb"]

print(f"\nGrade summary (composites):")
print(f"{'Element':<12} {'Min':>8} {'Median':>8} {'Mean':>8} {'Max':>8}")
print("-" * 50)
for col in ppm_cols + ppb_cols:
    unit = "ppm" if col.endswith("ppm") else "ppb"
    vals = composites_df[col].dropna()
    print(f"{col:<12} {vals.min():>8.1f} {vals.median():>8.1f} "
          f"{vals.mean():>8.1f} {vals.max():>8.1f}  {unit}")

in_zone = composites_df[
    (composites_df["MidDepth"] >= 80) &
    (composites_df["MidDepth"] <= 150)
]
print(f"\nMineralised zone (80-150m):")
print(f"  Composites in zone:    {len(in_zone):,}")
print(f"  Avg Cu in zone:        {in_zone['Cu_ppm'].mean():.0f} ppm")
print(f"  Max Cu in zone:        {in_zone['Cu_ppm'].max():.0f} ppm")
print(f"  Composites > 200 ppm:  {(in_zone['Cu_ppm'] > 200).sum():,}")

print(f"\nSaved: 04_composites.parquet")
print("=" * 55)

# COMMAND ----------

