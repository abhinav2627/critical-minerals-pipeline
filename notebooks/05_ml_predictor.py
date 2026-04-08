# Databricks notebook source
# MAGIC %pip install xgboost mlflow scikit-learn pyarrow==16.1.0 matplotlib pandas numpy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np

P1 = "/Volumes/critical_minerals/geochem/data/project1_geochemistry/"
P2 = "/Volumes/critical_minerals/geochem/data/project2_satellite/"
P3 = "/Volumes/critical_minerals/geochem/data/project3_drilling/"
P4 = "/Volumes/critical_minerals/geochem/data/project4_geophysics/"

# Project 1 — geochemistry features
geo_df = pd.read_parquet(P1 + "05_features.parquet")

# Project 2 — satellite features
sat_df = pd.read_parquet(P2 + "satellite_features.parquet")

# Project 3 — drill composites
drill_df = pd.read_parquet(P3 + "04_composites.parquet")

# Project 4 — geophysics grid
geo4_df = pd.read_parquet(P4 + "04_geophysics_features.parquet")

print(f"Project 1 — Geochemistry:  {len(geo_df):,} rows x {len(geo_df.columns)} cols")
print(f"Project 2 — Satellite:     {len(sat_df):,} rows x {len(sat_df.columns)} cols")
print(f"Project 3 — Drill holes:   {len(drill_df):,} rows x {len(drill_df.columns)} cols")
print(f"Project 4 — Geophysics:    {len(geo4_df):,} rows x {len(geo4_df.columns)} cols")

# COMMAND ----------

# WHY BUILD FROM GEOCHEMISTRY AS THE BASE?
# Project 1 has the most rows (74,349) and the most complete
# spatial coverage across Nevada. We use it as the base and
# join satellite + geophysics features to each geochemistry point
# using nearest-neighbour spatial matching.

from sklearn.neighbors import BallTree

print("Building unified ML feature matrix...")

# ── Select features from each project ──────────────────────────────

# Project 1 features
geo_features = [
    "latitude", "longitude",
    "copper_zscore", "nickel_zscore", "cobalt_zscore",
    "gold_zscore", "molybdenum_zscore",
    "copper_zscore_local", "nickel_zscore_local",
    "mineralisation_score", "is_drill_target",
    "dist_to_boundary_km",
]
geo_features = [c for c in geo_features if c in geo_df.columns]
base_df = geo_df[geo_features].copy()
base_df = base_df.dropna(subset=["latitude","longitude"])

# Project 2 features — nearest satellite pixel to each geochem point
sat_features = ["alteration_score","iron_oxide_ratio",
                "clay_ratio","ferrous_index","is_high_alteration"]
sat_features = [c for c in sat_features if c in sat_df.columns]
sat_clean = sat_df[["easting","northing"] + sat_features].dropna()

# Convert satellite UTM to approximate degrees for matching
sat_clean["lon_approx"] = sat_clean["easting"] / 111000 - 118
sat_clean["lat_approx"] = sat_clean["northing"] / 111000 - (-39.7)

# BallTree nearest neighbour join
# WHY BALLTREE? Faster than brute force for large spatial datasets
sat_tree = BallTree(
    np.deg2rad(sat_clean[["lat_approx","lon_approx"]].values),
    metric="haversine"
)
base_coords = np.deg2rad(base_df[["latitude","longitude"]].values)
distances, indices = sat_tree.query(base_coords, k=1)

# Only join if nearest satellite pixel is within 50km
MAX_DIST_RAD = 50 / 6371  # 50km in radians
for feat in sat_features:
    values = sat_clean[feat].values[indices.flatten()]
    values[distances.flatten() > MAX_DIST_RAD] = np.nan
    base_df[f"sat_{feat}"] = values

print(f"Satellite features joined: {base_df['sat_alteration_score'].notna().sum():,} matches")

# Project 4 features — nearest geophysics grid cell
geo4_features = ["mag_anomaly_nT","mag_residual_nT",
                 "pseudogravity","mag_zscore","is_mag_high"]
geo4_features = [c for c in geo4_features if c in geo4_df.columns]
geo4_clean = geo4_df[["latitude","longitude"] + geo4_features].dropna()

geo4_tree = BallTree(
    np.deg2rad(geo4_clean[["latitude","longitude"]].values),
    metric="haversine"
)
distances4, indices4 = geo4_tree.query(base_coords, k=1)

MAX_DIST_RAD_GEO4 = 100 / 6371  # 100km — geophysics grid is coarser
for feat in geo4_features:
    values = geo4_clean[feat].values[indices4.flatten()]
    values[distances4.flatten() > MAX_DIST_RAD_GEO4] = np.nan
    base_df[f"geo4_{feat}"] = values

print(f"Geophysics features joined: {base_df['geo4_mag_anomaly_nT'].notna().sum():,} matches")
print(f"\nUnified feature matrix: {len(base_df):,} rows x {len(base_df.columns)} columns")
display(base_df.head(3))

# COMMAND ----------

# CORRECT LABELLING APPROACH
# Instead of using mineralisation_score (which uses z-scores)
# we use SPATIAL PROXIMITY to known mineral districts
# This is genuinely independent of our geochemical features

print("Creating labels from known Nevada mineral districts...")

# Known major copper/gold mining districts in Nevada
# These are real coordinates of historic mining areas
KNOWN_DISTRICTS = [
    # (lat, lon, name)
    (40.8758, -117.0688, "Carlin_Gold_Trend"),
    (39.5297, -119.8143, "Virginia_City_Comstock"),
    (38.0285, -117.1697, "Tonopah"),
    (37.9360, -117.2280, "Goldfield"),
    (41.0428, -114.8769, "Elko_County"),
    (40.6635, -116.6143, "Battle_Mountain"),
    (39.4680, -118.7760, "Fallon_Churchill"),
    (38.5382, -115.9677, "Eureka"),
    (36.2082, -115.9814, "Searchlight"),
    (41.1553, -117.7383, "Winnemucca"),
]

districts_df = pd.DataFrame(
    KNOWN_DISTRICTS,
    columns=["lat","lon","name"]
)

# Label = 1 if within 25km of a known district
# Label = 0 if more than 100km from any district
from sklearn.neighbors import BallTree

dist_tree = BallTree(
    np.deg2rad(districts_df[["lat","lon"]].values),
    metric="haversine"
)
base_coords = np.deg2rad(
    base_df[["latitude","longitude"]].values
)
distances_dist, _ = dist_tree.query(base_coords, k=1)
distances_km      = distances_dist.flatten() * 6371

base_df["dist_to_district_km"] = distances_km
base_df["label"]               = np.nan
base_df.loc[distances_km <= 15,  "label"] = 1   # near district
base_df.loc[distances_km >= 100, "label"] = 0   # far from district

n_pos = (base_df["label"] == 1).sum()
n_neg = (base_df["label"] == 0).sum()

print(f"Positive (within 25km of district):  {n_pos:,}")
print(f"Negative (beyond 100km of district): {n_neg:,}")
print(f"Excluded (25-100km buffer zone):     {base_df['label'].isna().sum():,}")
print(f"\nDistricts used:")
for _, row in districts_df.iterrows():
    nearby = (distances_km <= 25).sum()
    print(f"  {row['name']}")

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Explicitly list ONLY the features we want
# Nothing derived from the label, nothing from previous runs
# Add this to the feature_cols list in Cell 6
feature_cols = [
    "copper_zscore_local",
    "nickel_zscore_local",
    "geo4_mag_anomaly_nT",
    "geo4_mag_residual_nT",
    "geo4_pseudogravity",
    "geo4_mag_zscore",
]
feature_cols = [c for c in feature_cols if c in base_df.columns]
print(f"Final clean features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

print(f"Clean features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

train_df = base_df.dropna(subset=["label"]).copy()

X = train_df[feature_cols].values
y = train_df["label"].values.astype(int)

imputer = SimpleImputer(strategy="median")
X       = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train):,} samples")
print(f"Test set:     {len(X_test):,} samples")
print(f"Positive: {y_train.sum():,}  Negative: {(y_train==0).sum():,}")

# COMMAND ----------

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt

# Set MLflow experiment name
mlflow.set_experiment("/Users/abhinavmandal26@gmail.com/deposit_predictor")

print("Training XGBoost deposit predictor with MLflow tracking...")
print("=" * 55)

with mlflow.start_run(run_name="xgboost_v1"):

    # ── Hyperparameters ─────────────────────────────────────
    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (y_train==0).sum() / y_train.sum(),
        "random_state":     42,
        "eval_metric":      "auc",
    }

    # Log hyperparameters to MLflow
    mlflow.log_params(params)
    mlflow.log_param("n_features",        len(feature_cols))
    mlflow.log_param("n_train_samples",   len(X_train))
    mlflow.log_param("n_test_samples",    len(X_test))
    mlflow.log_param("feature_names",     str(feature_cols))

    # ── Train model ─────────────────────────────────────────
    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluate ─────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:,1]
    auc         = roc_auc_score(y_test, y_prob)
    accuracy    = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("auc",      auc)
    mlflow.log_metric("accuracy", accuracy)

    print(f"AUC Score:  {auc:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Background","Prospective"]))

    # Log model
    mlflow.xgboost.log_model(model, "deposit_predictor_model")
    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run ID: {run_id}")

# COMMAND ----------

import os

fig, ax = plt.subplots(figsize=(10, 8))

importances = pd.Series(
    model.feature_importances_,
    index=feature_cols[:len(model.feature_importances_)]
).sort_values(ascending=True)

colors = []
for feat in importances.index:
    if feat.startswith("sat_"):    colors.append("#3498db")
    elif feat.startswith("geo4_"): colors.append("#e74c3c")
    else:                          colors.append("#27ae60")

importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")

ax.set_title(
    "Feature Importance — XGBoost Deposit Predictor\n"
    "Green=Geochemistry  Blue=Satellite  Red=Geophysics",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Feature Importance (gain)")

# Fixed output path — create directly under project5_ml
OUTPUT_PATH = "/Volumes/critical_minerals/geochem/data/project5_ml/outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "01_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_feature_importance.png")

mlflow.log_artifact(OUTPUT_PATH + "01_feature_importance.png")

# COMMAND ----------

# Apply trained model to ALL geochemistry points
# This gives us a deposit probability for every location in Nevada

print("Scoring all 74,349 geochemistry locations...")

# Prepare full dataset
X_all = base_df[feature_cols].values
X_all = imputer.transform(X_all)

# Predict probability of being a deposit
base_df["deposit_probability"] = model.predict_proba(X_all)[:,1]
base_df["ml_prediction"]       = model.predict(X_all)

# Rank by deposit probability
base_df["ml_rank"] = base_df["deposit_probability"].rank(
    ascending=False, method="min"
).astype(int)

# Top targets
top_targets = base_df.nsmallest(10, "ml_rank")[
    ["latitude","longitude","deposit_probability",
     "mineralisation_score","copper_zscore","ml_rank"]
].round(4)

print(f"\nTop 10 ML-predicted deposit targets:")
display(top_targets)
print(f"\nHigh probability targets (>0.8): "
      f"{(base_df['deposit_probability'] > 0.8).sum():,}")

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "ML Deposit Predictor — XGBoost Deposit Probability\n"
    "Trained on geochemistry + satellite + geophysics features",
    fontsize=13, fontweight="bold"
)

import matplotlib.colors as mcolors

# ── Left: full probability map ──
ax1 = axes[0]
sc1 = ax1.scatter(
    base_df["longitude"], base_df["latitude"],
    c=base_df["deposit_probability"],
    cmap="RdYlGn", s=1.5, alpha=0.6,
    vmin=0, vmax=1
)
ax1.set_title("Deposit probability\n(all 74,349 locations)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
plt.colorbar(sc1, ax=ax1, shrink=0.8).set_label("P(deposit)")

# ── Right: high probability targets only ──
ax2 = axes[1]
background = base_df[base_df["deposit_probability"] < 0.7]
targets    = base_df[base_df["deposit_probability"] >= 0.7]

ax2.scatter(background["longitude"], background["latitude"],
            c="#2c3e50", s=1, alpha=0.2, label="Background")
sc2 = ax2.scatter(
    targets["longitude"], targets["latitude"],
    c=targets["deposit_probability"],
    cmap="YlOrRd", s=8, alpha=0.9, zorder=5,
    label=f"High probability (n={len(targets):,})"
)
ax2.set_title("High probability targets (P > 0.7)\nML-predicted exploration priorities",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Longitude")
ax2.set_facecolor("#1a1a2e")
plt.colorbar(sc2, ax=ax2, shrink=0.8).set_label("P(deposit)")
ax2.legend(fontsize=8, loc="lower right",
           facecolor="#2c3e50", labelcolor="white")

plt.tight_layout()
plt.savefig(OUTPUT_PATH + "02_deposit_probability_map.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_deposit_probability_map.png")
mlflow.log_artifact(OUTPUT_PATH + "02_deposit_probability_map.png")
mlflow.end_run()

# COMMAND ----------

# Save ML results
ML_PATH = "/Volumes/critical_minerals/geochem/data/project5_ml/"
os.makedirs(ML_PATH, exist_ok=True)

base_df.to_parquet(ML_PATH + "05_ml_predictions.parquet", index=False)

print("=" * 60)
print("PROJECT 5 — ML DEPOSIT PREDICTOR COMPLETE")
print("=" * 60)
print(f"\nModel:              XGBoost Classifier")
print(f"Features used:      {len(feature_cols)}")
print(f"Data sources:       Geochemistry + Satellite + Geophysics")
print(f"Training samples:   {len(X_train):,}")
print(f"Test samples:       {len(X_test):,}")
print(f"\nModel performance:")
print(f"  AUC:              {auc:.4f}")
print(f"  Accuracy:         {accuracy:.4f}")
print(f"\nPredictions:")
print(f"  Total scored:     {len(base_df):,} locations")
top = base_df.nsmallest(1,"ml_rank").iloc[0]
print(f"  High prob (>0.8): {(base_df['deposit_probability']>0.8).sum():,}")
print(f"\nTop ML target:")
print(f"  Lat/Lon:          {top['latitude']:.4f}, {top['longitude']:.4f}")
print(f"  P(deposit):       {top['deposit_probability']:.4f}")
print(f"  Mineralisation:   {top['mineralisation_score']:.3f}")
print(f"\nMLflow run ID:      {run_id}")
print(f"Saved:              05_ml_predictions.parquet")
print("=" * 60)

# COMMAND ----------

