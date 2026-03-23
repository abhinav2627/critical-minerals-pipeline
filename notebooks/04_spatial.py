# Databricks notebook source
# MAGIC %pip install geopandas==0.14.4 pyproj==3.6.1 shapely==2.0.4 pyarrow==16.1.0
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

VALIDATED_PATH = "/Volumes/critical_minerals/geochem/data/03_validated.parquet"
SPATIAL_PATH   = "/Volumes/critical_minerals/geochem/data/04_spatial.parquet"

df = pd.read_parquet(VALIDATED_PATH)
print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")

# COMMAND ----------

# The parquet file dropped the geometry column when we saved it
# We rebuild it from the latitude/longitude columns we extracted earlier
# This is standard practice — always store lat/lon as plain columns AND geometry

print("Rebuilding GeoDataFrame from lat/lon columns...")

geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
gdf      = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print(f"GeoDataFrame created: {len(gdf):,} points")
print(f"CRS: {gdf.crs}")
print(f"\nSample geometry: {gdf.geometry.iloc[0]}")

# COMMAND ----------

# Why grid cells?
# Earth AI doesn't just look at individual samples — they look at zones
# A 10km x 10km grid cell aggregates nearby samples into a regional picture
# This is how exploration targeting actually works in the field

# At mid-latitudes, 1 degree ≈ 111 km
# So 10km grid = 10/111 = ~0.09 degrees

GRID_RESOLUTION_KM  = 10
DEG_PER_KM          = 1.0 / 111.0
RESOLUTION_DEG      = GRID_RESOLUTION_KM * DEG_PER_KM

gdf["grid_row"] = np.floor(gdf["latitude"]  / RESOLUTION_DEG).astype(int)
gdf["grid_col"] = np.floor(gdf["longitude"] / RESOLUTION_DEG).astype(int)
gdf["grid_id"]  = gdf["grid_row"].astype(str) + "_" + gdf["grid_col"].astype(str)

n_cells          = gdf["grid_id"].nunique()
avg_per_cell     = len(gdf) / n_cells

print(f"Grid resolution:    {GRID_RESOLUTION_KM} km (~{RESOLUTION_DEG:.4f} degrees)")
print(f"Total grid cells:   {n_cells:,}")
print(f"Total samples:      {len(gdf):,}")
print(f"Avg samples/cell:   {avg_per_cell:.1f}")
print(f"\nTop 10 densest grid cells:")
print(gdf["grid_id"].value_counts().head(10))

# COMMAND ----------

# We don't have a real geology shapefile on Free Edition
# So we assign geology zones based on longitude bands
# This mimics the major US geologic provinces — good enough for portfolio
# In production you would do: gpd.sjoin(gdf, geology_gdf, how="left")

# Why does geology matter?
# Copper means different things in different rock types
# 100 ppm Cu in granite = strong anomaly
# 100 ppm Cu in volcanic rock = background noise
# Geology context is essential for correct interpretation

print("Assigning geology zones based on US geologic provinces...")

conditions = [
    gdf["longitude"] < -120,
    (gdf["longitude"] >= -120) & (gdf["longitude"] < -115),
    (gdf["longitude"] >= -115) & (gdf["longitude"] < -110),
    (gdf["longitude"] >= -110) & (gdf["longitude"] < -105),
    (gdf["longitude"] >= -105) & (gdf["longitude"] < -100),
    gdf["longitude"] >= -100,
]
geology_units = [
    "Pacific_Coast_Ranges",
    "Sierra_Nevada_Batholith",
    "Basin_and_Range",
    "Colorado_Plateau",
    "Rocky_Mountains",
    "Great_Plains",
]
lithologies = [
    "metamorphic",
    "granite_intrusive",
    "volcanic_extensional",
    "sedimentary",
    "metamorphic_volcanic",
    "sedimentary",
]

gdf["geology_unit"] = np.select(conditions, geology_units, default="unknown")
gdf["lithology"]    = np.select(conditions, lithologies,   default="unknown")

print("\nSample distribution by geology unit:")
print(gdf["geology_unit"].value_counts())

# COMMAND ----------

# Distance to tectonic/structural boundaries is a key exploration feature
# Most ore deposits form near faults and tectonic boundaries
# We approximate major US tectonic boundaries with longitude lines

# Major boundaries (simplified):
# -125: Pacific plate boundary (west coast)
# -117: Basin and Range extension boundary  
# -104: Rocky Mountain front
# -96:  Great Plains / Cratonic boundary

TECTONIC_BOUNDARIES = [-125.0, -117.0, -104.0, -96.0]

# Distance to nearest boundary in degrees (approx km = degrees * 111)
gdf["dist_to_boundary_deg"] = gdf["longitude"].apply(
    lambda lon: min(abs(lon - b) for b in TECTONIC_BOUNDARIES)
)
gdf["dist_to_boundary_km"] = gdf["dist_to_boundary_deg"] * 111.0

print("Distance to nearest tectonic boundary:")
print(f"  Min:  {gdf['dist_to_boundary_km'].min():.1f} km")
print(f"  Max:  {gdf['dist_to_boundary_km'].max():.1f} km")
print(f"  Mean: {gdf['dist_to_boundary_km'].mean():.1f} km")
print(f"\nSamples within 50km of a boundary: "
      f"{(gdf['dist_to_boundary_km'] < 50).sum():,}")

# COMMAND ----------

print("=" * 55)
print("SPATIAL PROCESSING REPORT")
print("=" * 55)
print(f"Total samples:          {len(gdf):,}")
print(f"CRS:                    {gdf.crs}")
print(f"Grid cells (10km):      {gdf['grid_id'].nunique():,}")
print(f"Avg samples per cell:   {len(gdf)/gdf['grid_id'].nunique():.1f}")

print(f"\nGeography coverage:")
print(f"  Latitude:  {gdf['latitude'].min():.2f} to {gdf['latitude'].max():.2f}")
print(f"  Longitude: {gdf['longitude'].min():.2f} to {gdf['longitude'].max():.2f}")

print(f"\nGeology unit distribution:")
for unit, count in gdf["geology_unit"].value_counts().items():
    pct = count / len(gdf) * 100
    print(f"  {unit:<30} {count:>7,}  ({pct:.1f}%)")

print(f"\nNew columns added this stage:")
new_cols = ["grid_row", "grid_col", "grid_id",
            "geology_unit", "lithology",
            "dist_to_boundary_deg", "dist_to_boundary_km"]
for col in new_cols:
    print(f"  {col}")
print("=" * 55)

# COMMAND ----------

# Drop geometry before saving to parquet
# Geometry is not natively supported in standard Parquet
# We keep latitude/longitude as plain columns — that's enough for ML

gdf_save = pd.DataFrame(gdf.drop(columns=["geometry"]))
gdf_save.to_parquet(SPATIAL_PATH, index=False)

print(f"Saved: {SPATIAL_PATH}")
print(f"Shape: {len(gdf_save):,} rows x {len(gdf_save.columns)} columns")
print("\nStage 4 complete.")

# COMMAND ----------

