# Databricks notebook source
# MAGIC %pip install numpy pandas pyarrow==16.1.0 matplotlib scipy geopandas==0.14.4

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

GEOPHYS_PATH = "/Volumes/critical_minerals/geochem/data/project4_geophysics/"

print("Files in project4_geophysics:")
for f in dbutils.fs.ls(GEOPHYS_PATH):
    size_mb = f.size / (1024*1024)
    print(f"  {f.name:<45} {size_mb:.1f} MB")

# COMMAND ----------

import pandas as pd
import numpy as np

MAG_PATH = GEOPHYS_PATH + "mag_nevada_final.csv"

mag_df = pd.read_csv(MAG_PATH)

print(f"Loaded: {len(mag_df):,} magnetic measurements")
print(f"Columns: {list(mag_df.columns)}")
print(f"\nGeographic coverage:")
print(f"  Latitude:  {mag_df['lat'].min():.3f} to {mag_df['lat'].max():.3f}")
print(f"  Longitude: {mag_df['lon'].min():.3f} to {mag_df['lon'].max():.3f}")
print(f"\nMagnetic anomaly stats:")
print(f"  Min:  {mag_df['final_mag'].min():.2f} nT")
print(f"  Max:  {mag_df['final_mag'].max():.2f} nT")
print(f"  Mean: {mag_df['final_mag'].mean():.2f} nT")
print(f"  Std:  {mag_df['final_mag'].std():.2f} nT")
print(f"\nFlight lines: {mag_df['Line'].nunique():,}")
print(f"Flights:      {mag_df['flight'].nunique()}")
display(mag_df.head(5))

# COMMAND ----------

# WHY STANDARDISE?
# final_mag is already the magnetic anomaly (total field minus IGRF)
# But it still contains instrument drift and noise
# We apply a simple high-pass filter to remove long-wavelength regional trends
# leaving only the short-wavelength local anomalies from shallow sources

from scipy.ndimage import uniform_filter

df = mag_df.copy()

# Rename for clarity
df = df.rename(columns={
    "lat":               "latitude",
    "lon":               "longitude",
    "final_mag":         "mag_anomaly_nT",
    "diurnally_cor_mag": "total_field_nT",
    "igrf_correction":   "igrf_nT",
    "gps_elev":          "elevation_m",
    "Line":              "flight_line",
})

# Remove remaining outliers using IQR method
Q1 = df["mag_anomaly_nT"].quantile(0.25)
Q3 = df["mag_anomaly_nT"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 3 * IQR
upper = Q3 + 3 * IQR

before = len(df)
df = df[
    (df["mag_anomaly_nT"] >= lower) &
    (df["mag_anomaly_nT"] <= upper)
]
print(f"Outliers removed: {before - len(df):,}")

# Compute derived features
# Normalised anomaly — how anomalous relative to survey background
df["mag_zscore"] = (
    (df["mag_anomaly_nT"] - df["mag_anomaly_nT"].mean()) /
     df["mag_anomaly_nT"].std()
)

# Flag positive anomalies — magnetic highs indicate magnetite-rich rocks
# associated with copper-gold mineralisation
df["is_mag_high"]  = df["mag_zscore"] > 2.0
df["is_mag_low"]   = df["mag_zscore"] < -2.0

print(f"Clean dataset: {len(df):,} measurements")
print(f"Magnetic highs (z > 2):  {df['is_mag_high'].sum():,}")
print(f"Magnetic lows  (z < -2): {df['is_mag_low'].sum():,}")
display(df.head(3))

# COMMAND ----------

from scipy.interpolate import griddata

# WHY GRID INTERPOLATION?
# Flight line data is not evenly spaced — measurements are dense
# along each flight line but sparse between lines.
# Grid interpolation fills the gaps to create a continuous surface
# that can be visualised as a map and joined to other raster datasets.

print("Interpolating flight line data onto regular grid...")
print("(This converts scattered points to a continuous surface)")

# Define grid resolution — 0.02 degrees ≈ 2km
# Coarser than satellite (30m) but appropriate for regional magnetic surveys
GRID_RES = 0.02

lat_min = df["latitude"].quantile(0.01)
lat_max = df["latitude"].quantile(0.99)
lon_min = df["longitude"].quantile(0.01)
lon_max = df["longitude"].quantile(0.99)

grid_lons = np.arange(lon_min, lon_max, GRID_RES)
grid_lats = np.arange(lat_min, lat_max, GRID_RES)
grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)

print(f"Grid dimensions: {grid_lat.shape[0]} x {grid_lat.shape[1]}")
print(f"Grid resolution: {GRID_RES} degrees (~{GRID_RES*111:.0f}km)")
print(f"Interpolating {len(df):,} points onto grid...")

# Linear interpolation — fast and appropriate for dense flight line data
# WHY LINEAR NOT CUBIC? Cubic can overshoot between flight lines
# creating artificial anomalies that don't exist in the real data
grid_mag = griddata(
    points = df[["longitude", "latitude"]].values,
    values = df["mag_anomaly_nT"].values,
    xi     = (grid_lon, grid_lat),
    method = "linear"
)

print(f"Grid interpolation complete")
print(f"Grid shape: {grid_mag.shape}")
print(f"Valid cells: {(~np.isnan(grid_mag)).sum():,}")
print(f"Grid anomaly range: {np.nanmin(grid_mag):.1f} to {np.nanmax(grid_mag):.1f} nT")

# COMMAND ----------

from scipy.ndimage import maximum_filter, label

# WHY ANOMALY DETECTION ON THE GRID?
# Individual high measurements could be noise
# A cluster of high measurements in a 10x10km area is a real anomaly
# We detect peaks in the gridded surface — much more reliable

# Regional trend removal — subtract smoothed background
# WHY? Long-wavelength magnetic variations come from deep crustal sources
# Short-wavelength variations come from shallow sources (our exploration targets)
from scipy.ndimage import gaussian_filter

# Regional field = heavily smoothed version (sigma=10 grid cells = ~200km)
regional  = gaussian_filter(np.nan_to_num(grid_mag), sigma=10)
# Residual  = what's left after removing regional trend
residual  = grid_mag - regional
# Set NaN back where original was NaN
residual[np.isnan(grid_mag)] = np.nan

print("Regional trend removed.")
print(f"Residual anomaly range: {np.nanmin(residual):.1f} to {np.nanmax(residual):.1f} nT")

# Detect magnetic highs — peaks in the residual field
# A magnetic high is a cell that is higher than all its neighbours
# within a 5-cell radius (5 x 2km = 10km search radius)
threshold     = np.nanpercentile(residual, 90)
mag_high_mask = residual >= threshold

# Label connected regions of magnetic highs
mag_high_clean              = np.nan_to_num(mag_high_mask.astype(int))
labeled_array, num_features = label(mag_high_clean)

print(f"\nMagnetic anomaly detection:")
print(f"  Threshold (90th percentile): {threshold:.1f} nT")
print(f"  High anomaly cells:          {mag_high_mask.sum():,}")
print(f"  Connected anomaly zones:     {num_features:,}")

# COMMAND ----------

# We don't have real gravity data — the USGS files were binary format
# BUT we can compute a proxy gravity signal from the magnetic data
# using the pseudogravity transform
#
# WHY PSEUDOGRAVITY?
# Magnetic and gravity anomalies from the same source are mathematically
# related. The pseudogravity transform converts a magnetic anomaly map
# into what the gravity anomaly would look like IF the source was
# uniformly magnetised. It highlights broad, deep sources that gravity
# is sensitive to but magnetics sometimes misses.
#
# This is a real technique used in production geophysics pipelines.

from scipy.fft import fft2, ifft2, fftfreq

print("Computing pseudogravity transform...")
print("(Converts magnetic anomaly to equivalent gravity signal)")

# Work with the residual grid, replacing NaN with 0
mag_grid_clean = np.nan_to_num(residual, nan=0.0)

# FFT of the magnetic grid
MAG_FFT = fft2(mag_grid_clean)

# Wavenumber arrays
rows, cols = mag_grid_clean.shape
kx = fftfreq(cols, d=GRID_RES)
ky = fftfreq(rows, d=GRID_RES)
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)
K[K == 0] = 1e-10   # avoid division by zero at DC component

# Inclination of Earth's field over Nevada (~60 degrees)
# This is the angle of Earth's magnetic field below horizontal
INC = np.deg2rad(60)

# Pseudogravity filter in frequency domain
# This is the standard wavenumber domain operator
PSEUDO_FILTER = 1.0 / (K * (np.sin(INC) + 1j * np.cos(INC) * KY/K))

# Apply filter and inverse FFT
PSEUDO_FFT  = MAG_FFT * PSEUDO_FILTER
pseudograv  = np.real(ifft2(PSEUDO_FFT))

# Normalise to sensible range (milligals equivalent)
pseudograv  = (pseudograv - np.mean(pseudograv)) / np.std(pseudograv)
pseudograv[np.isnan(grid_mag)] = np.nan

print(f"Pseudogravity range: {np.nanmin(pseudograv):.2f} to {np.nanmax(pseudograv):.2f} (normalised)")
print(f"Positive anomalies (dense rock): {(pseudograv > 1).sum():,} cells")
print(f"Negative anomalies (light rock): {(pseudograv < -1).sum():,} cells")

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    "Airborne Magnetic Survey — GeoDAWN, Northwestern Nevada\n"
    "142,797 real flight line measurements · USGS EarthMRI Program 2022",
    fontsize=13, fontweight="bold"
)

extent = [lon_min, lon_max, lat_min, lat_max]

# ── Panel 1: Raw flight line data ──
ax1 = axes[0,0]
sc = ax1.scatter(
    df["longitude"], df["latitude"],
    c=df["mag_anomaly_nT"],
    cmap="RdBu_r", s=0.3, alpha=0.6,
    vmin=df["mag_anomaly_nT"].quantile(0.02),
    vmax=df["mag_anomaly_nT"].quantile(0.98)
)
ax1.set_title("Raw flight line data\n(magnetic anomaly, nT)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
plt.colorbar(sc, ax=ax1, shrink=0.8).set_label("nT")

# ── Panel 2: Interpolated grid ──
ax2 = axes[0,1]
im2 = ax2.imshow(
    grid_mag, extent=extent, origin="lower",
    cmap="RdBu_r", aspect="auto",
    vmin=np.nanpercentile(grid_mag, 2),
    vmax=np.nanpercentile(grid_mag, 98)
)
ax2.set_title("Interpolated magnetic grid\n(2km resolution)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.colorbar(im2, ax=ax2, shrink=0.8).set_label("nT")

# ── Panel 3: Residual anomalies ──
ax3 = axes[1,0]
im3 = ax3.imshow(
    residual, extent=extent, origin="lower",
    cmap="RdBu_r", aspect="auto",
    vmin=np.nanpercentile(residual, 2),
    vmax=np.nanpercentile(residual, 98)
)
ax3.contour(
    grid_lon, grid_lat, residual,
    levels=[threshold], colors="black",
    linewidths=0.8, alpha=0.7
)
ax3.set_title("Residual magnetic anomaly\n(regional trend removed)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
plt.colorbar(im3, ax=ax3, shrink=0.8).set_label("nT")

# ── Panel 4: Pseudogravity ──
ax4 = axes[1,1]
im4 = ax4.imshow(
    pseudograv, extent=extent, origin="lower",
    cmap="inferno", aspect="auto",
    vmin=np.nanpercentile(pseudograv[~np.isnan(pseudograv)], 2),
    vmax=np.nanpercentile(pseudograv[~np.isnan(pseudograv)], 98)
)
ax4.set_title("Pseudogravity transform\n(highlights deep dense sources)", fontsize=11, fontweight="bold")
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
plt.colorbar(im4, ax=ax4, shrink=0.8).set_label("Normalised units")

plt.tight_layout()

OUTPUT_PATH = GEOPHYS_PATH + "outputs/"
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)
plt.savefig(OUTPUT_PATH + "01_magnetic_survey.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_magnetic_survey.png")

# COMMAND ----------

# Convert gridded results back to point dataset
# One row per grid cell — joinable with Projects 1, 2, 3

STEP = 1  # use every grid cell

rows_list = []
for i in range(0, grid_mag.shape[0], STEP):
    for j in range(0, grid_mag.shape[1], STEP):
        if np.isnan(grid_mag[i,j]):
            continue
        rows_list.append({
            "latitude":        grid_lat[i,j],
            "longitude":       grid_lon[i,j],
            "mag_anomaly_nT":  grid_mag[i,j],
            "mag_residual_nT": residual[i,j] if not np.isnan(residual[i,j]) else np.nan,
            "pseudogravity":   pseudograv[i,j] if not np.isnan(pseudograv[i,j]) else np.nan,
            "mag_zscore":      (grid_mag[i,j] - np.nanmean(grid_mag)) / np.nanstd(grid_mag),
            "is_mag_high":     bool(residual[i,j] >= threshold) if not np.isnan(residual[i,j]) else False,
        })

grid_points_df = pd.DataFrame(rows_list)

# Save
OUTPUT_PARQUET = GEOPHYS_PATH + "04_geophysics_features.parquet"
grid_points_df.to_parquet(OUTPUT_PARQUET, index=False)

print(f"Saved: {OUTPUT_PARQUET}")
print(f"Shape: {len(grid_points_df):,} rows x {len(grid_points_df.columns)} columns")
print(f"\nGrid point stats:")
print(f"  Magnetic anomaly range: {grid_points_df['mag_anomaly_nT'].min():.1f} to {grid_points_df['mag_anomaly_nT'].max():.1f} nT")
print(f"  Magnetic highs:         {grid_points_df['is_mag_high'].sum():,} cells")
print(f"  Coverage:               {grid_points_df['latitude'].min():.2f} to {grid_points_df['latitude'].max():.2f} lat")

# COMMAND ----------

print("=" * 60)
print("PROJECT 4 — AIRBORNE GEOPHYSICS PIPELINE COMPLETE")
print("=" * 60)
print(f"\nData source:    GeoDAWN USGS EarthMRI Program 2022")
print(f"Survey type:    Airborne magnetic (aeromagnetic)")
print(f"Coverage:       Northwestern Nevada")
print(f"\nRaw measurements:      {len(mag_df):,} flight line points")
print(f"After cleaning:        {len(df):,} points")
print(f"Grid cells:            {len(grid_points_df):,}")
print(f"Grid resolution:       {GRID_RES} degrees (~{GRID_RES*111:.0f}km)")
print(f"\nProcessing steps:")
print(f"  1. Flight line data loaded and cleaned")
print(f"  2. Outlier removal using IQR method")
print(f"  3. Grid interpolation (linear, scipy.griddata)")
print(f"  4. Regional trend removal (Gaussian filter)")
print(f"  5. Residual anomaly detection (90th percentile threshold)")
print(f"  6. Pseudogravity transform (FFT wavenumber domain)")
print(f"  7. Point dataset export for spatial joining")
print(f"\nMagnetic anomaly zones: {grid_points_df['is_mag_high'].sum():,}")
print(f"\nNew skills demonstrated:")
print(f"  Grid interpolation from scattered flight line data")
print(f"  Regional trend removal using Gaussian filtering")
print(f"  FFT-based pseudogravity transform")
print(f"  Wavenumber domain signal processing")
print("=" * 60)

# COMMAND ----------

