# Databricks notebook source
# MAGIC %pip install pyarrow==16.1.0 scipy==1.13.1
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy import stats

SPATIAL_PATH  = "/Volumes/critical_minerals/geochem/data/04_spatial.parquet"
FEATURES_PATH = "/Volumes/critical_minerals/geochem/data/05_features.parquet"

df = pd.read_parquet(SPATIAL_PATH)
element_cols = [c for c in df.columns if c.endswith("_ppm")]

print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
print(f"Element columns: {element_cols}")

# COMMAND ----------

# WHY LOG TRANSFORM?
# Copper concentrations range from 0.1 ppm to 11,000 ppm in this dataset
# That's 5 orders of magnitude — ML models struggle with this
# log(0.1) = -2.3  |  log(100) = 4.6  |  log(11000) = 9.3
# After log transform the range is manageable and distribution is normal

print("Applying log1p transforms...")
print("(log1p means log(value + 1) — handles zero values safely)\n")

for col in element_cols:
    out_col       = col.replace("_ppm", "_log")
    df[out_col]   = np.log1p(df[col].clip(lower=0))

log_cols = [c for c in df.columns if c.endswith("_log")]
print(f"Created {len(log_cols)} log-transformed columns:")

# Show the effect — copper before and after
cu_raw = df["copper_ppm"].dropna()
cu_log = df["copper_log"].dropna()
print(f"\nCopper raw:  min={cu_raw.min():.2f}  max={cu_raw.max():.0f}  "
      f"mean={cu_raw.mean():.1f}  std={cu_raw.std():.1f}")
print(f"Copper log:  min={cu_log.min():.2f}  max={cu_log.max():.2f}  "
      f"mean={cu_log.mean():.2f}  std={cu_log.std():.2f}")
print("\nNotice: log transform compressed 5 orders of magnitude into a tight range")

# COMMAND ----------

# WHY Z-SCORES?
# A copper value of 100 ppm tells you nothing on its own
# But copper z-score of +3.5 tells you this sample is 3.5 standard
# deviations above the global average — that IS an anomaly worth drilling

# Z-score formula: z = (value - mean) / standard_deviation
# z > 2  = notable anomaly
# z > 3  = strong anomaly  
# z > 4  = exceptional — potential economic deposit

print("Computing global z-score anomalies...")
print("Formula: z = (log_value - global_mean) / global_std\n")

for col in log_cols:
    out_col     = col.replace("_log", "_zscore")
    mu          = df[col].mean()
    sigma       = df[col].std()
    df[out_col] = (df[col] - mu) / sigma if sigma > 0 else 0.0

zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
print(f"Created {len(zscore_cols)} z-score columns")

# Show distribution of copper anomalies
cu_z = df["copper_zscore"].dropna()
print(f"\nCopper z-score distribution:")
print(f"  Mean:              {cu_z.mean():.3f}  (should be ~0)")
print(f"  Std:               {cu_z.std():.3f}   (should be ~1)")
print(f"  Samples z > 2:     {(cu_z > 2).sum():,}  (notable anomaly)")
print(f"  Samples z > 3:     {(cu_z > 3).sum():,}  (strong anomaly)")
print(f"  Samples z > 4:     {(cu_z > 4).sum():,}  (exceptional)")

# COMMAND ----------

# WHY STRATIFIED Z-SCORES?
# Global z-scores have a problem:
# Granite naturally has high copper background
# Sedimentary rock naturally has low copper background
# A sample with 80 ppm Cu in sedimentary rock is MORE anomalous
# than 80 ppm Cu in granite — but global z-score treats them the same

# Solution: compute z-score WITHIN each geology unit
# This removes the regional background signal
# Much more sensitive to real anomalies

print("Computing geology-stratified z-scores...")
print("(removes regional lithological background — more sensitive)\n")

for col in log_cols:
    out_col              = col.replace("_log", "_zscore_local")
    group_mean           = df.groupby("geology_unit")[col].transform("mean")
    group_std            = df.groupby("geology_unit")[col].transform("std").fillna(1)
    df[out_col]          = (df[col] - group_mean) / group_std

local_zscore_cols = [c for c in df.columns if c.endswith("_zscore_local")]
print(f"Created {len(local_zscore_cols)} geology-stratified z-score columns")

# Compare global vs local for copper
print("\nGlobal vs local z-score for copper (top 5 anomalies):")
top5 = df.nlargest(5, "copper_zscore")[
    ["sample_id", "geology_unit", "copper_ppm",
     "copper_zscore", "copper_zscore_local"]
].round(2)
display(top5)

# COMMAND ----------

# WHY ELEMENT RATIOS?
# Certain element combinations are diagnostic of specific deposit types
# These come from 100 years of economic geology research:
#
# Cu/Mo > 10  → Porphyry copper deposit (like Bingham Canyon, Utah)
# Co/Ni > 1   → Laterite nickel deposit
# Cu/Zn > 5   → VMS (Volcanogenic Massive Sulphide) deposit
# As/Au        → Epithermal gold deposit pathfinder
#
# By engineering these ratios as features, we embed domain knowledge
# that would take a geologist years to learn

print("Computing pathfinder element ratios...")
print("(geochemical deposit-type indicators)\n")

RATIO_PAIRS = [
    ("copper_ppm",     "molybdenum_ppm", "ratio_cu_mo",  "Porphyry copper indicator"),
    ("cobalt_ppm",     "nickel_ppm",     "ratio_co_ni",  "Laterite nickel indicator"),
    ("copper_ppm",     "zinc_ppm",       "ratio_cu_zn",  "VMS deposit indicator"),
    ("arsenic_ppm",    "gold_ppm",       "ratio_as_au",  "Epithermal gold indicator"),
]

for elem_a, elem_b, out_col, description in RATIO_PAIRS:
    if elem_a in df.columns and elem_b in df.columns:
        df[out_col] = df[elem_a] / df[elem_b].replace(0, np.nan)
        valid        = df[out_col].notna().sum()
        print(f"  {out_col:<20} {description:<35} ({valid:,} valid values)")

# COMMAND ----------

# WHY GRID AGGREGATES?
# Individual samples are noisy — one high reading could be contamination
# But if an entire 10km grid cell has consistently high copper
# that's a much stronger signal
# Grid aggregates smooth out noise and capture the regional pattern

print("Computing grid cell aggregate statistics...")
print("(mean, max, std per element per 10km cell)\n")

agg_element_cols = ["copper_ppm", "nickel_ppm", "cobalt_ppm",
                    "gold_ppm",   "molybdenum_ppm"]

for col in agg_element_cols:
    if col not in df.columns:
        continue
    base = col.replace("_ppm", "")
    df[f"{base}_cell_mean"] = df.groupby("grid_id")[col].transform("mean")
    df[f"{base}_cell_max"]  = df.groupby("grid_id")[col].transform("max")
    df[f"{base}_cell_std"]  = df.groupby("grid_id")[col].transform("std").fillna(0)

cell_cols = [c for c in df.columns if "_cell_" in c]
print(f"Created {len(cell_cols)} grid aggregate columns")
print("\nExample — copper cell stats for first 5 rows:")
display(df[["sample_id", "grid_id", "copper_ppm",
            "copper_cell_mean", "copper_cell_max"]].head())

# COMMAND ----------

# THE FINAL SCORE — this is what Earth AI actually uses for drill targeting
#
# We combine multiple element z-scores into one number
# Higher score = higher probability of economic mineralisation
# This is the output the exploration team uses to rank drill targets
#
# Weights based on economic importance of each element:
# Copper:     35% — primary target for most Earth AI projects
# Nickel:     25% — critical battery metal
# Cobalt:     15% — critical battery metal
# Molybdenum: 10% — byproduct of porphyry copper
# Gold:       15% — high value byproduct

print("Computing composite mineralisation score...")
print("Weights: Cu=35%, Ni=25%, Co=15%, Mo=10%, Au=15%\n")

WEIGHTS = {
    "copper_zscore":     0.35,
    "nickel_zscore":     0.25,
    "cobalt_zscore":     0.15,
    "molybdenum_zscore": 0.10,
    "gold_zscore":       0.15,
}

score = sum(
    df[col].fillna(0) * weight
    for col, weight in WEIGHTS.items()
    if col in df.columns
)
df["mineralisation_score"] = score

# Rank every sample — rank 1 = best drill target
df["targeting_rank"] = df["mineralisation_score"].rank(
    ascending=False, method="min"
).astype(int)

# Flag top 2% as priority drill targets
threshold             = df["mineralisation_score"].quantile(0.98)
df["is_drill_target"] = df["mineralisation_score"] >= threshold

n_targets = df["is_drill_target"].sum()
print(f"Mineralisation score range: "
      f"{df['mineralisation_score'].min():.3f} to "
      f"{df['mineralisation_score'].max():.3f}")
print(f"Top 2% drill targets: {n_targets:,} samples")
print(f"\nTop 10 drill targets:")
display(df.nsmallest(10, "targeting_rank")[
    ["sample_id", "latitude", "longitude", "geology_unit",
     "copper_ppm", "mineralisation_score", "targeting_rank"]
].round(3))

# COMMAND ----------

feature_categories = {
    "Raw concentrations (ppm)":        [c for c in df.columns if c.endswith("_ppm")],
    "Log transforms":                  [c for c in df.columns if c.endswith("_log")],
    "Global z-scores":                 [c for c in df.columns if c.endswith("_zscore") and "local" not in c],
    "Local z-scores (geology)":        [c for c in df.columns if c.endswith("_zscore_local")],
    "Pathfinder ratios":               [c for c in df.columns if c.startswith("ratio_")],
    "Grid aggregates":                 [c for c in df.columns if "_cell_" in c],
    "Spatial features":                ["grid_id", "geology_unit", "lithology", "dist_to_boundary_km"],
    "Targeting output":                ["mineralisation_score", "targeting_rank", "is_drill_target"],
}

print("=" * 55)
print("FEATURE ENGINEERING REPORT")
print("=" * 55)
total_features = 0
for category, cols in feature_categories.items():
    existing = [c for c in cols if c in df.columns]
    total_features += len(existing)
    print(f"\n{category}:")
    for col in existing:
        print(f"  {col}")

print(f"\n{'='*55}")
print(f"Total features: {total_features}")
print(f"Total rows:     {len(df):,}")
print(f"Drill targets:  {df['is_drill_target'].sum():,} samples")
print("=" * 55)

# COMMAND ----------

import yaml
from datetime import datetime, timezone

# Save features as Parquet
df.to_parquet(FEATURES_PATH, index=False)
print(f"Saved: {FEATURES_PATH}")
print(f"Shape: {len(df):,} rows x {len(df.columns)} columns")

# Save metadata YAML — the data dictionary
# This is what the ML team reads to understand every feature
metadata = {
    "schema_version":   "1.0",
    "generated_at":     datetime.now(timezone.utc).isoformat(),
    "pipeline_version": "1.0.0",
    "source":           "USGS National Geochemical Survey",
    "total_rows":       len(df),
    "total_features":   len(df.columns),
    "drill_targets":    int(df["is_drill_target"].sum()),
    "feature_groups": {
        k: [c for c in v if c in df.columns]
        for k, v in feature_categories.items()
    }
}

METADATA_PATH = "/Volumes/critical_minerals/geochem/data/metadata.yaml"
with open(METADATA_PATH, "w") as f:
    yaml.dump(metadata, f, default_flow_style=False)

print(f"\nMetadata saved: {METADATA_PATH}")
print("\nStage 5 complete — ML-ready features produced.")
print(f"\nPipeline summary:")
print(f"  Raw samples in:      74,408")
print(f"  Clean samples out:   {len(df):,}")
print(f"  Features engineered: {len(df.columns)}")
print(f"  Drill targets found: {df['is_drill_target'].sum():,}")

# COMMAND ----------

