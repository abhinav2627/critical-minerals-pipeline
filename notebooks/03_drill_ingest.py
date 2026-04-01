# Databricks notebook source
# MAGIC %pip install numpy pandas pyarrow==16.1.0 matplotlib scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

DRILL_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/"
os.makedirs(DRILL_PATH, exist_ok=True)

print("Project 3 folder ready:")
print(f"  {DRILL_PATH}")

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

# ── Configuration ──────────────────────────────────────────────────
N_HOLES       = 50          # 50 drill holes across survey area
SURVEY_AREA_M = 5000        # 5km x 5km survey area
BASE_EASTING  = 550000      # UTM Zone 11N approximate Nevada coords
BASE_NORTHING = 4400000

# ── Generate collar positions ──────────────────────────────────────
# Holes are drilled on a rough grid with some randomness
hole_ids   = [f"DH{str(i+1).zfill(3)}" for i in range(N_HOLES)]
eastings   = BASE_EASTING  + rng.uniform(0, SURVEY_AREA_M, N_HOLES)
northings  = BASE_NORTHING + rng.uniform(0, SURVEY_AREA_M, N_HOLES)
elevations = rng.uniform(1800, 1950, N_HOLES)  # Nevada plateau elevation

# Drill depth varies — shallower at edges, deeper in centre (more interesting)
# Centre of survey area = 2500m from origin
dist_from_centre = np.sqrt(
    (eastings  - (BASE_EASTING  + 2500))**2 +
    (northings - (BASE_NORTHING + 2500))**2
)
max_dist   = np.max(dist_from_centre)
# Deeper holes near centre where the mineralisation target is
total_depth = 150 + 150 * (1 - dist_from_centre / max_dist)
total_depth = total_depth + rng.normal(0, 20, N_HOLES)
total_depth = total_depth.clip(100, 350).astype(int)

# Azimuth: mostly drilling northwest (315 degrees) — into the target
azimuth = rng.normal(315, 15, N_HOLES).clip(0, 360)

# Dip: mostly -55 to -65 degrees (angled, not straight down)
dip = rng.normal(-60, 5, N_HOLES).clip(-85, -30)

collar_df = pd.DataFrame({
    "HoleID":     hole_ids,
    "Easting":    eastings.round(2),
    "Northing":   northings.round(2),
    "Elevation":  elevations.round(2),
    "TotalDepth": total_depth,
    "Azimuth":    azimuth.round(1),
    "Dip":        dip.round(1),
    "DrillDate":  pd.date_range("2022-01-01", periods=N_HOLES, freq="3D")
                    .strftime("%Y-%m-%d").tolist(),
    "DrillType":  rng.choice(["RC", "DD", "RC"], N_HOLES),
})

print(f"Collar table: {len(collar_df)} drill holes")
print(f"\nDepth range: {collar_df['TotalDepth'].min()}m to {collar_df['TotalDepth'].max()}m")
print(f"Survey area: {SURVEY_AREA_M/1000:.0f}km x {SURVEY_AREA_M/1000:.0f}km")
print(f"\nDrill types:")
print(collar_df["DrillType"].value_counts())
display(collar_df.head(5))

# COMMAND ----------

# Survey table: records actual hole direction every 50 metres
# Holes drift slightly as they go deeper — this is realistic

survey_rows = []

for _, hole in collar_df.iterrows():
    hole_id    = hole["HoleID"]
    total_depth = hole["TotalDepth"]
    az_start   = hole["Azimuth"]
    dip_start  = hole["Dip"]

    # Survey stations every 50m
    depths = np.arange(0, total_depth + 1, 50)
    if depths[-1] < total_depth:
        depths = np.append(depths, total_depth)

    # Holes drift gradually — cumulative random walk
    n_stations  = len(depths)
    az_drift    = np.cumsum(rng.normal(0, 1.5, n_stations))
    dip_drift   = np.cumsum(rng.normal(0, 0.8, n_stations))

    azimuths = (az_start + az_drift).clip(0, 360)
    dips     = (dip_start + dip_drift).clip(-85, -30)

    for d, az, dp in zip(depths, azimuths, dips):
        survey_rows.append({
            "HoleID":  hole_id,
            "Depth":   round(float(d), 1),
            "Azimuth": round(float(az), 1),
            "Dip":     round(float(dp), 1),
        })

survey_df = pd.DataFrame(survey_rows)

print(f"Survey table: {len(survey_df):,} rows")
print(f"Avg surveys per hole: {len(survey_df)/N_HOLES:.1f}")
print(f"\nAzimuth range: {survey_df['Azimuth'].min():.1f} to {survey_df['Azimuth'].max():.1f}")
print(f"Dip range:     {survey_df['Dip'].min():.1f} to {survey_df['Dip'].max():.1f}")
display(survey_df.head(8))

# COMMAND ----------

# Assay table: element concentrations every 1 metre
# Key design: mineralised zone sits at 80-150m depth in centre of survey
# Holes near the centre intersect higher grades

assay_rows = []

# Mineralised zone centre point
TARGET_E = BASE_EASTING  + 2500# Assay table: element concentrations every 1 metre
# Key design: mineralised zone sits at 80-150m depth in centre of survey
# Holes near the centre intersect higher grades

assay_rows = []

# Mineralised zone centre point
TARGET_E = BASE_EASTING  + 2500
TARGET_N = BASE_NORTHING + 2500
TARGET_DEPTH_TOP = 80     # mineralisation starts at 80m
TARGET_DEPTH_BOT = 150    # mineralisation ends at 150m

for _, hole in collar_df.iterrows():
    hole_id    = hole["HoleID"]
    total_depth = hole["TotalDepth"]

    # How close is this hole to the target centre?
    dist = np.sqrt(
        (hole["Easting"]  - TARGET_E)**2 +
        (hole["Northing"] - TARGET_N)**2
    )
    # Grade multiplier — higher grades near centre (max 3x at centre)
    grade_multiplier = max(0.3, 3.0 * (1 - dist / (SURVEY_AREA_M * 0.7)))

    for depth_from in range(0, total_depth):
        depth_to = depth_from + 1

        # Is this interval in the mineralised zone?
        in_zone = TARGET_DEPTH_TOP <= depth_from <= TARGET_DEPTH_BOT

        # Background concentrations (log-normal everywhere)
        cu_bg  = rng.lognormal(2.5, 0.8)    # background ~12 ppm
        ni_bg  = rng.lognormal(2.8, 0.7)
        co_bg  = rng.lognormal(1.2, 0.6)
        au_bg  = rng.lognormal(-0.5, 1.2)   # ppb
        mo_bg  = rng.lognormal(0.8, 0.7)

        if in_zone:
            # Elevated grades in mineralised zone
            cu = cu_bg * grade_multiplier * rng.lognormal(3.5, 0.9)
            ni = ni_bg * grade_multiplier * rng.lognormal(2.0, 0.6)
            co = co_bg * grade_multiplier * rng.lognormal(2.5, 0.7)
            au = au_bg * grade_multiplier * rng.lognormal(3.0, 1.2)
            mo = mo_bg * grade_multiplier * rng.lognormal(2.8, 0.8)
        else:
            cu, ni, co, au, mo = cu_bg, ni_bg, co_bg, au_bg, mo_bg

        # Inject realistic messiness
        # ~5% below detection limit (stored as negative DL value)
        for val_name in ["cu", "ni", "co", "au", "mo"]:
            if rng.random() < 0.05:
                locals()[val_name] = -rng.choice([0.5, 1.0, 2.0, 5.0])

        # ~2% missing entirely (core not recovered)
        if rng.random() < 0.02:
            cu = ni = co = au = mo = np.nan

        # Core recovery percentage (occasionally low — realistic)
        recovery = rng.uniform(85, 100)
        if rng.random() < 0.05:
            recovery = rng.uniform(30, 70)  # poor recovery interval

        assay_rows.append({
            "HoleID":       hole_id,
            "From":         depth_from,
            "To":           depth_to,
            "Cu_ppm":       round(float(cu), 3) if not np.isnan(cu) else np.nan,
            "Ni_ppm":       round(float(ni), 3) if not np.isnan(ni) else np.nan,
            "Co_ppm":       round(float(co), 3) if not np.isnan(co) else np.nan,
            "Au_ppb":       round(float(au), 3) if not np.isnan(au) else np.nan,
            "Mo_ppm":       round(float(mo), 3) if not np.isnan(mo) else np.nan,
            "Recovery_pct": round(float(recovery), 1),
        })

assay_df = pd.DataFrame(assay_rows)

# Show data quality summary
print(f"Assay table: {len(assay_df):,} rows ({len(assay_df)/N_HOLES:.0f} avg per hole)")
print(f"\nData quality issues injected:")
for col in ["Cu_ppm", "Ni_ppm", "Co_ppm", "Au_ppb", "Mo_ppm"]:
    null_count = assay_df[col].isna().sum()
    bdl_count  = (assay_df[col] < 0).sum()
    print(f"  {col:<10} nulls={null_count:>4}  BDL={bdl_count:>4}")

print(f"\nCopper stats:")
print(f"  Background:    {assay_df[assay_df['Cu_ppm'] > 0]['Cu_ppm'].median():.1f} ppm (median)")
in_zone = assay_df[(assay_df["From"] >= 80) & (assay_df["To"] <= 150)]
print(f"  In zone (80-150m): {in_zone['Cu_ppm'].median():.1f} ppm (median)")
display(assay_df.head(5))
TARGET_N = BASE_NORTHING + 2500
TARGET_DEPTH_TOP = 80     # mineralisation starts at 80m
TARGET_DEPTH_BOT = 150    # mineralisation ends at 150m

for _, hole in collar_df.iterrows():
    hole_id    = hole["HoleID"]
    total_depth = hole["TotalDepth"]

    # How close is this hole to the target centre?
    dist = np.sqrt(
        (hole["Easting"]  - TARGET_E)**2 +
        (hole["Northing"] - TARGET_N)**2
    )
    # Grade multiplier — higher grades near centre (max 3x at centre)
    grade_multiplier = max(0.3, 3.0 * (1 - dist / (SURVEY_AREA_M * 0.7)))

    for depth_from in range(0, total_depth):
        depth_to = depth_from + 1

        # Is this interval in the mineralised zone?
        in_zone = TARGET_DEPTH_TOP <= depth_from <= TARGET_DEPTH_BOT

        # Background concentrations (log-normal everywhere)
        cu_bg  = rng.lognormal(2.5, 0.8)    # background ~12 ppm
        ni_bg  = rng.lognormal(2.8, 0.7)
        co_bg  = rng.lognormal(1.2, 0.6)
        au_bg  = rng.lognormal(-0.5, 1.2)   # ppb
        mo_bg  = rng.lognormal(0.8, 0.7)

        if in_zone:
            # Elevated grades in mineralised zone
            cu = cu_bg * grade_multiplier * rng.lognormal(3.5, 0.9)
            ni = ni_bg * grade_multiplier * rng.lognormal(2.0, 0.6)
            co = co_bg * grade_multiplier * rng.lognormal(2.5, 0.7)
            au = au_bg * grade_multiplier * rng.lognormal(3.0, 1.2)
            mo = mo_bg * grade_multiplier * rng.lognormal(2.8, 0.8)
        else:
            cu, ni, co, au, mo = cu_bg, ni_bg, co_bg, au_bg, mo_bg

        # Inject realistic messiness
        # ~5% below detection limit (stored as negative DL value)
        for val_name in ["cu", "ni", "co", "au", "mo"]:
            if rng.random() < 0.05:
                locals()[val_name] = -rng.choice([0.5, 1.0, 2.0, 5.0])

        # ~2% missing entirely (core not recovered)
        if rng.random() < 0.02:
            cu = ni = co = au = mo = np.nan

        # Core recovery percentage (occasionally low — realistic)
        recovery = rng.uniform(85, 100)
        if rng.random() < 0.05:
            recovery = rng.uniform(30, 70)  # poor recovery interval

        assay_rows.append({
            "HoleID":       hole_id,
            "From":         depth_from,
            "To":           depth_to,
            "Cu_ppm":       round(float(cu), 3) if not np.isnan(cu) else np.nan,
            "Ni_ppm":       round(float(ni), 3) if not np.isnan(ni) else np.nan,
            "Co_ppm":       round(float(co), 3) if not np.isnan(co) else np.nan,
            "Au_ppb":       round(float(au), 3) if not np.isnan(au) else np.nan,
            "Mo_ppm":       round(float(mo), 3) if not np.isnan(mo) else np.nan,
            "Recovery_pct": round(float(recovery), 1),
        })

assay_df = pd.DataFrame(assay_rows)

# Show data quality summary
print(f"Assay table: {len(assay_df):,} rows ({len(assay_df)/N_HOLES:.0f} avg per hole)")
print(f"\nData quality issues injected:")
for col in ["Cu_ppm", "Ni_ppm", "Co_ppm", "Au_ppb", "Mo_ppm"]:
    null_count = assay_df[col].isna().sum()
    bdl_count  = (assay_df[col] < 0).sum()
    print(f"  {col:<10} nulls={null_count:>4}  BDL={bdl_count:>4}")

print(f"\nCopper stats:")
print(f"  Background:    {assay_df[assay_df['Cu_ppm'] > 0]['Cu_ppm'].median():.1f} ppm (median)")
in_zone = assay_df[(assay_df["From"] >= 80) & (assay_df["To"] <= 150)]
print(f"  In zone (80-150m): {in_zone['Cu_ppm'].median():.1f} ppm (median)")
display(assay_df.head(5))

# COMMAND ----------

# Save as CSV (real-world format) AND Parquet (pipeline format)

COLLAR_CSV  = DRILL_PATH + "collar.csv"
SURVEY_CSV  = DRILL_PATH + "survey.csv"
ASSAY_CSV   = DRILL_PATH + "assay.csv"

collar_df.to_csv(COLLAR_CSV,  index=False)
survey_df.to_csv(SURVEY_CSV,  index=False)
assay_df.to_csv(ASSAY_CSV,    index=False)

# Also save raw Parquet for pipeline
collar_df.to_parquet(DRILL_PATH + "01_collar_raw.parquet", index=False)
survey_df.to_parquet(DRILL_PATH + "01_survey_raw.parquet", index=False)
assay_df.to_parquet(DRILL_PATH  + "01_assay_raw.parquet",  index=False)

print("Saved all 3 files:")
print(f"  collar.csv  — {len(collar_df):,} rows")
print(f"  survey.csv  — {len(survey_df):,} rows")
print(f"  assay.csv   — {len(assay_df):,} rows")

# Verify
print("\nVerification — files in project3_drilling:")
for f in dbutils.fs.ls(DRILL_PATH):
    size_kb = f.size / 1024
    print(f"  {f.name:<40} {size_kb:.0f} KB")

# COMMAND ----------

print("=" * 55)
print("STAGE 1 — INGESTION REPORT")
print("=" * 55)
print(f"Collar table:  {len(collar_df):>6,} rows  x  {len(collar_df.columns)} columns")
print(f"Survey table:  {len(survey_df):>6,} rows  x  {len(survey_df.columns)} columns")
print(f"Assay table:   {len(assay_df):>6,} rows  x  {len(assay_df.columns)} columns")
print(f"\nDrill holes:      {collar_df['HoleID'].nunique()}")
print(f"Survey stations:  {len(survey_df):,}")
print(f"Assay intervals:  {len(assay_df):,}")
print(f"Avg hole depth:   {collar_df['TotalDepth'].mean():.0f}m")
print(f"Total metres:     {collar_df['TotalDepth'].sum():,}m drilled")
print(f"\nData quality injected:")
print(f"  BDL values:     present in all elements (~5%)")
print(f"  Missing core:   present (~2% intervals)")
print(f"  Poor recovery:  present (~5% intervals)")
print(f"\nAll 3 files saved to: {DRILL_PATH}")
print("=" * 55)

# COMMAND ----------

