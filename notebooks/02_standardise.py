# Databricks notebook source
# MAGIC %pip install geopandas==0.14.4 pyproj==3.6.1 shapely==2.0.4 fiona==1.9.6 loguru==0.7.2 pyyaml==6.0.1 pyarrow==16.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import geopandas as gpd
import pandas as pd

SHAPEFILE_PATH = "/Volumes/critical_minerals/geochem/data/ngs.shp"

print("Loading real USGS National Geochemical Survey data...")
gdf = gpd.read_file(SHAPEFILE_PATH)

print(f"Loaded: {len(gdf):,} real samples")
print(f"Columns: {len(gdf.columns)}")
print(f"CRS: {gdf.crs}")
print(f"\nFirst 3 rows:")
display(gdf.head(3))

# COMMAND ----------

# Print every column name and its data type
print("ALL COLUMNS IN REAL USGS DATA:")
print("=" * 50)
for i, (col, dtype) in enumerate(zip(gdf.columns, gdf.dtypes)):
    print(f"{i:>2}. {col:<20} {dtype}")

print(f"\nSample values from first row:")
print(gdf.iloc[0])

# COMMAND ----------

# IMPORTANT: In this dataset negative values = below detection limit
# They are NOT real concentrations — they are coded as negative DL values
# e.g. AU_ICP40 = -8.0 means gold was below 8 ppb detection limit

print("UNDERSTANDING NEGATIVE VALUES (Below Detection Limit):")
print("=" * 55)

key_elements = ["CU_ICP40", "NI_ICP40", "CO_ICP40", "AU_ICP40", "MO_ICP40"]

for col in key_elements:
    total        = len(gdf)
    negative     = (gdf[col] < 0).sum()
    null         = gdf[col].isna().sum()
    positive     = (gdf[col] > 0).sum()
    
    print(f"\n{col}:")
    print(f"  Positive (real readings): {positive:>6,}  ({positive/total*100:.1f}%)")
    print(f"  Negative (below DL):      {negative:>6,}  ({negative/total*100:.1f}%)")
    print(f"  Null (not measured):      {null:>6,}  ({null/total*100:.1f}%)")
    print(f"  Min value: {gdf[col].min():.2f}   Max value: {gdf[col].max():.2f}")

# COMMAND ----------

# Map real USGS column names to our clean standard names
COLUMN_MAPPING = {
    "LABNO":    "sample_id",
    "CATEGORY": "category",
    "DATASET":  "dataset",
    "TYPEDESC": "sample_type",
}

# Element mapping — ICP40 suffix removed, standard names applied
ELEMENT_MAPPING = {
    "CU_ICP40": "copper_ppm",
    "NI_ICP40": "nickel_ppm",
    "CO_ICP40": "cobalt_ppm",
    "LI_ICP40": "lithium_ppm",
    "ZN_ICP40": "zinc_ppm",
    "PB_ICP40": "lead_ppm",
    "AU_ICP40": "gold_ppm",      # ppb — we'll convert later
    "MO_ICP40": "molybdenum_ppm",
    "MN_ICP40": "manganese_ppm",
    "AS_ICP40": "arsenic_ppm",   # bonus — pathfinder for gold deposits
    "CR_ICP40": "chromium_ppm",  # bonus — pathfinder for nickel deposits
}

# Apply column mapping
df = gdf.copy()
df = df.rename(columns=COLUMN_MAPPING)
df = df.rename(columns=ELEMENT_MAPPING)

print("Column mapping applied.")
print(f"\nOur key element columns:")
for col in ELEMENT_MAPPING.values():
    print(f"  {col}")

# COMMAND ----------

import numpy as np

# CRITICAL STEP — this is what separates a data engineer from a beginner
# Negative values = the instrument detected it was BELOW that detection limit
# Standard practice: substitute with absolute value / 2
# e.g. AU = -8.0 means below 8 ppb DL → substitute with 4.0 ppb

print("HANDLING BELOW-DETECTION-LIMIT VALUES:")
print("=" * 55)

element_cols = list(ELEMENT_MAPPING.values())

for col in element_cols:
    if col not in df.columns:
        continue
    
    before_neg = (df[col] < 0).sum()
    
    # Where negative: substitute with abs(value) / 2
    df[col] = df[col].apply(
        lambda x: abs(x) / 2.0 if pd.notna(x) and x < 0 else x
    )
    
    after_neg = (df[col] < 0).sum()
    print(f"{col:<20} BDL substituted: {before_neg:>5,} values → now {after_neg} negatives remain")

# Gold is in ppb → convert to ppm
df["gold_ppm"] = df["gold_ppm"] * 0.001
print(f"\nGold converted from ppb to ppm (÷ 1000)")

# COMMAND ----------

# Extract lat/lon from geometry column
# Current CRS: NAD27 → Target CRS: WGS84 (EPSG:4326)

print(f"Original CRS: {df.crs}")

# Reproject to WGS84
df = df.to_crs("EPSG:4326")
print(f"Reprojected to: {df.crs}")

# Extract coordinates into plain columns
df["longitude"] = df.geometry.x
df["latitude"]  = df.geometry.y

# Validate coordinate ranges — should be within US
print(f"\nCoordinate ranges after reprojection:")
print(f"  Latitude:  {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
print(f"  Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")

# Drop rows outside continental US + Alaska bounding box
before = len(df)
df = df[
    (df["latitude"]  >= 15.0) & (df["latitude"]  <= 72.0) &
    (df["longitude"] >= -180.0) & (df["longitude"] <= -50.0)
]
print(f"\nRows dropped outside US bounding box: {before - len(df):,}")
print(f"Remaining samples: {len(df):,}")

# COMMAND ----------

from datetime import datetime, timezone

# Lowercase string columns
for col in ["category", "sample_type", "dataset"]:
    if col in df.columns:
        df[col] = df[col].str.lower().str.strip()

# Add pipeline audit columns — always include these in production
df["_pipeline_version"] = "1.0.0"
df["_processed_at"]     = datetime.now(timezone.utc).isoformat()
df["_schema_version"]   = "1.0"
df["_source"]           = "USGS National Geochemical Survey (NGS)"

print("Categoricals standardised.")
print(f"\nSample types in dataset:")
print(df["sample_type"].value_counts())

# COMMAND ----------

# Keep only the columns we need — drop raw ICP40 columns we didn't map
cols_to_keep = (
    ["sample_id", "category", "sample_type", "dataset",
     "latitude", "longitude", "geometry"]
    + element_cols
    + ["_pipeline_version", "_processed_at", "_schema_version", "_source"]
)
cols_to_keep = [c for c in cols_to_keep if c in df.columns]
df_clean = df[cols_to_keep].copy()

CLEAN_PATH = "/Volumes/critical_minerals/geochem/data/02_standardised.parquet"
df_clean.to_parquet(CLEAN_PATH, index=False)

print(f"Saved: {CLEAN_PATH}")
print(f"Shape: {len(df_clean):,} rows x {len(df_clean.columns)} columns")
print(f"\nStage 2 complete with REAL USGS data.")

# COMMAND ----------

