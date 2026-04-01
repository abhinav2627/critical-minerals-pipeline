# Databricks notebook source
# MAGIC %pip install numpy pandas pyarrow==16.1.0 matplotlib scipy
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np

DRILL_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/"

collar_df = pd.read_parquet(DRILL_PATH + "02_collar_validated.parquet")
survey_df = pd.read_parquet(DRILL_PATH + "02_survey_validated.parquet")
assay_df  = pd.read_parquet(DRILL_PATH + "02_assay_validated.parquet")

print(f"Collar: {len(collar_df):,} holes")
print(f"Survey: {len(survey_df):,} stations")
print(f"Assay:  {len(assay_df):,} intervals")

# COMMAND ----------

# Before writing any code, let's visualise what desurveying does
# Look at one hole's survey data to understand the problem

hole_id = "DH001"
hole_collar = collar_df[collar_df["HoleID"] == hole_id].iloc[0]
hole_survey  = survey_df[survey_df["HoleID"] == hole_id].sort_values("Depth")

print(f"Hole: {hole_id}")
print(f"  Surface position: E={hole_collar['Easting']:.1f}, N={hole_collar['Northing']:.1f}, Elev={hole_collar['Elevation']:.1f}m")
print(f"  Total depth: {hole_collar['TotalDepth']}m")
print(f"  Survey stations:")
display(hole_survey)

print(f"\nNotice: azimuth and dip CHANGE with depth.")
print(f"This is drift. Without desurveying we'd assume a straight line.")
print(f"With desurveying we calculate the actual curved 3D path.")

# COMMAND ----------

def desurvey_hole(collar_row, survey_rows):
    """
    Convert depth measurements along a drill hole into 3D XYZ coordinates
    using the Minimum Curvature Method.
    
    WHY MINIMUM CURVATURE?
    The simplest approach (tangent method) assumes the hole goes straight
    between survey stations. Minimum curvature assumes it curves smoothly
    — this is more accurate and is the industry standard.
    
    INPUTS:
    - collar_row: one row from collar table (surface position)
    - survey_rows: all survey rows for this hole (depth, azimuth, dip)
    
    OUTPUT:
    - DataFrame with Depth, X (Easting), Y (Northing), Z (elevation)
    """
    
    # Start at the collar (surface position of the hole)
    x = collar_row["Easting"]     # Easting in UTM metres
    y = collar_row["Northing"]    # Northing in UTM metres
    z = collar_row["Elevation"]   # Elevation above sea level
    
    # Sort survey by depth
    survey = survey_rows.sort_values("Depth").reset_index(drop=True)
    
    # Store 3D coordinates at each survey station
    coords = [{"Depth": 0.0, "X": x, "Y": y, "Z": z}]
    
    # Loop through consecutive survey station pairs
    for i in range(len(survey) - 1):
        # Top of interval
        d1  = survey.loc[i,   "Depth"]
        az1 = survey.loc[i,   "Azimuth"]
        dp1 = survey.loc[i,   "Dip"]
        
        # Bottom of interval
        d2  = survey.loc[i+1, "Depth"]
        az2 = survey.loc[i+1, "Azimuth"]
        dp2 = survey.loc[i+1, "Dip"]
        
        # Length of this interval
        L = d2 - d1
        if L <= 0:
            continue
        
        # Convert angles from degrees to radians
        # WHY RADIANS? Python's trig functions (sin, cos) require radians
        # 1 degree = pi/180 radians
        az1_r = np.deg2rad(az1)
        az2_r = np.deg2rad(az2)
        dp1_r = np.deg2rad(dp1)
        dp2_r = np.deg2rad(dp2)
        
        # Minimum curvature ratio factor
        # This accounts for the curvature between stations
        # When the hole is perfectly straight, rf = 1.0
        # When it curves, rf adjusts the coordinate increments
        dl = np.arccos(
            max(-1, min(1,  # clamp to valid arccos range
                np.cos(dp2_r - dp1_r) -
                np.sin(dp1_r) * np.sin(dp2_r) * (1 - np.cos(az2_r - az1_r))
            ))
        )
        
        # Avoid division by zero when dl is near zero (hole is straight)
        if abs(dl) < 1e-10:
            rf = 1.0
        else:
            rf = 2.0 / dl * np.tan(dl / 2.0)
        
        # Calculate 3D coordinate increments
        # dN = change in Northing (Y direction)
        # dE = change in Easting  (X direction)  
        # dZ = change in elevation (Z direction, negative going down)
        dN = (L/2) * (np.cos(dp1_r)*np.cos(az1_r) + np.cos(dp2_r)*np.cos(az2_r)) * rf
        dE = (L/2) * (np.cos(dp1_r)*np.sin(az1_r) + np.cos(dp2_r)*np.sin(az2_r)) * rf
        dZ = (L/2) * (np.sin(dp1_r) + np.sin(dp2_r)) * rf
        
        # Update position
        x += dE
        y += dN
        z += dZ   # dZ is negative because we're going down
        
        coords.append({"Depth": d2, "X": x, "Y": y, "Z": z})
    
    return pd.DataFrame(coords)


# Test on one hole
hole_id     = "DH001"
collar_row  = collar_df[collar_df["HoleID"] == hole_id].iloc[0]
survey_rows = survey_df[survey_df["HoleID"] == hole_id]

result = desurvey_hole(collar_row, survey_rows)
print(f"Desurveyed {hole_id}:")
display(result)

print(f"\nHole started at:")
print(f"  E={collar_row['Easting']:.1f}  N={collar_row['Northing']:.1f}  Z={collar_row['Elevation']:.1f}")
print(f"\nHole ended at:")
print(f"  E={result['X'].iloc[-1]:.1f}  N={result['Y'].iloc[-1]:.1f}  Z={result['Z'].iloc[-1]:.1f}")
drift_e = result['X'].iloc[-1] - collar_row['Easting']
drift_n = result['Y'].iloc[-1] - collar_row['Northing']
print(f"\nTotal drift: {drift_e:.1f}m East, {drift_n:.1f}m North")
print(f"(This is how far the bottom of the hole is from directly below the collar)")

# COMMAND ----------

from scipy.interpolate import interp1d

def get_xyz_at_depths(collar_row, survey_rows, target_depths):
    """
    Get interpolated XYZ coordinates at specific depths.
    Used to get coordinates at every assay interval midpoint.
    
    WHY INTERPOLATE?
    Survey stations are every 50m. Assay intervals are every 1m.
    We need XYZ at every 1m depth, not just at survey stations.
    SciPy's interp1d fills in the gaps using linear interpolation.
    """
    # Get XYZ at survey stations
    station_coords = desurvey_hole(collar_row, survey_rows)
    
    if len(station_coords) < 2:
        return None
    
    depths = station_coords["Depth"].values
    
    # Create interpolation functions for X, Y, Z
    # interp1d creates a function that given any depth,
    # returns the interpolated X (or Y or Z)
    # bounds_error=False means don't crash if depth is outside range
    # fill_value="extrapolate" means extend beyond the last survey station
    interp_x = interp1d(depths, station_coords["X"].values,
                         bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(depths, station_coords["Y"].values,
                         bounds_error=False, fill_value="extrapolate")
    interp_z = interp1d(depths, station_coords["Z"].values,
                         bounds_error=False, fill_value="extrapolate")
    
    return pd.DataFrame({
        "Depth": target_depths,
        "X":     interp_x(target_depths),
        "Y":     interp_y(target_depths),
        "Z":     interp_z(target_depths),
    })


# Desurvey all holes and get XYZ at every assay midpoint
print("Desurveying all 50 drill holes...")
all_coords = []

for hole_id in collar_df["HoleID"]:
    collar_row  = collar_df[collar_df["HoleID"] == hole_id].iloc[0]
    survey_rows = survey_df[survey_df["HoleID"] == hole_id]
    hole_assays = assay_df[assay_df["HoleID"] == hole_id].copy()
    
    if len(survey_rows) < 2 or len(hole_assays) == 0:
        continue
    
    # Use midpoint of each assay interval
    # WHY MIDPOINT? The assay covers from 5m to 6m.
    # The representative depth is 5.5m (the middle)
    hole_assays["MidDepth"] = (hole_assays["From"] + hole_assays["To"]) / 2
    target_depths = hole_assays["MidDepth"].values
    
    coords = get_xyz_at_depths(collar_row, survey_rows, target_depths)
    if coords is None:
        continue
    
    coords["HoleID"] = hole_id
    all_coords.append(coords)

coords_df = pd.concat(all_coords, ignore_index=True)
print(f"Desurveyed: {len(coords_df):,} depth intervals across all holes")
print(f"\nXYZ coordinate ranges:")
print(f"  X (Easting):   {coords_df['X'].min():.0f} to {coords_df['X'].max():.0f} m")
print(f"  Y (Northing):  {coords_df['Y'].min():.0f} to {coords_df['Y'].max():.0f} m")
print(f"  Z (Elevation): {coords_df['Z'].min():.0f} to {coords_df['Z'].max():.0f} m")
display(coords_df.head(5))

# COMMAND ----------

# Clean fix — drop MidDepth if it already exists from a previous run
if "MidDepth" in assay_df.columns:
    assay_df = assay_df.drop(columns=["MidDepth"])

assay_df["MidDepth"] = (assay_df["From"] + assay_df["To"]) / 2

desurveyed_df = assay_df.merge(
    coords_df[["HoleID", "Depth", "X", "Y", "Z"]],
    left_on  = ["HoleID", "MidDepth"],
    right_on = ["HoleID", "Depth"],
    how      = "left"
).drop(columns=["Depth"])

# Verify no duplicates
before = len(desurveyed_df)
desurveyed_df = desurveyed_df.drop_duplicates(
    subset=["HoleID", "From", "To"]
)
after = len(desurveyed_df)
if before != after:
    print(f"Removed {before - after:,} duplicate rows")

print(f"Assay intervals with 3D coordinates: {len(desurveyed_df):,}")
print(f"\nSample — first 5 rows with XYZ:")
display(desurveyed_df[["HoleID", "From", "To", "MidDepth",
                         "X", "Y", "Z", "Cu_ppm", "Au_ppb"]].head(5))

matched = desurveyed_df["X"].notna().sum()
print(f"\nIntervals with XYZ coordinates: {matched:,} / {len(desurveyed_df):,}")

# COMMAND ----------

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

fig = plt.figure(figsize=(16, 10))
ax  = fig.add_subplot(111, projection="3d")

# Plot drill traces coloured by copper grade
valid = desurveyed_df.dropna(subset=["X", "Y", "Z", "Cu_ppm"])
valid = valid[valid["Cu_ppm"] > 0]

# Clip copper for colour scale
cu_clip = valid["Cu_ppm"].clip(upper=valid["Cu_ppm"].quantile(0.98))
norm    = mcolors.LogNorm(vmin=max(cu_clip.min(), 1), vmax=cu_clip.max())
cmap    = cm.get_cmap("YlOrRd")
colors  = cmap(norm(cu_clip.values))

sc = ax.scatter(
    valid["X"], valid["Y"], valid["Z"],
    c=cu_clip, cmap="YlOrRd", norm=norm,
    s=1, alpha=0.5, linewidths=0
)

# Plot collar positions as larger markers
ax.scatter(
    collar_df["Easting"], collar_df["Northing"], collar_df["Elevation"],
    c="black", s=30, marker="^", zorder=5, label="Collar (surface)"
)

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_zlabel("Elevation (m)")
ax.set_title(
    "3D Drill Hole Traces — Coloured by Copper Grade (ppm)\n"
    "50 holes · Nevada synthetic porphyry copper deposit",
    fontsize=12, fontweight="bold"
)
plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label="Cu (ppm)")
ax.legend(fontsize=9)

plt.tight_layout()

OUTPUT_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/outputs/"
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)
plt.savefig(OUTPUT_PATH + "01_drill_traces_3d.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_drill_traces_3d.png")

# COMMAND ----------

desurveyed_df.to_parquet(DRILL_PATH + "03_desurveyed.parquet", index=False)

print("=" * 55)
print("STAGE 3 — DESURVEYING COMPLETE")
print("=" * 55)
print(f"Input intervals:      {len(assay_df):,}")
print(f"Desurveyed intervals: {len(desurveyed_df):,}")
print(f"With XYZ coords:      {desurveyed_df['X'].notna().sum():,}")
print(f"\n3D extent of deposit:")
print(f"  Easting:   {desurveyed_df['X'].min():.0f} to {desurveyed_df['X'].max():.0f} m")
print(f"  Northing:  {desurveyed_df['Y'].min():.0f} to {desurveyed_df['Y'].max():.0f} m")
print(f"  Elevation: {desurveyed_df['Z'].min():.0f} to {desurveyed_df['Z'].max():.0f} m")
depth_range = desurveyed_df["Z"].min() - desurveyed_df["Z"].max()
print(f"  Depth below surface: up to {abs(depth_range):.0f}m")
print(f"\nSaved: 03_desurveyed.parquet")
print("=" * 55)

# COMMAND ----------

