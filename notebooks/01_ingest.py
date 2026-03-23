# Databricks notebook source
# MAGIC %pip install geopandas==0.14.4 pyproj==3.6.1 shapely==2.0.4 fiona==1.9.6 loguru==0.7.2 pyyaml==6.0.1 pyarrow==16.1.0 scipy==1.13.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import geopandas as gpd
import numpy as np
import yaml
import pyarrow

print(f"pandas:     {pd.__version__}")
print(f"geopandas:  {gpd.__version__}")
print(f"numpy:      {np.__version__}")
print(f"pyarrow:    {pyarrow.__version__}")
print("All good — ready to build!")

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS critical_minerals")
spark.sql("CREATE SCHEMA IF NOT EXISTS critical_minerals.geochem")
spark.sql("CREATE VOLUME IF NOT EXISTS critical_minerals.geochem.data")
print("Volume ready at: /Volumes/critical_minerals/geochem/data/")

# COMMAND ----------

import numpy as np
import pandas as pd

def generate_sample_data(n_rows=500, seed=42):
    rng = np.random.default_rng(seed)
    n = n_rows
    
    sample_ids = [f"USGS-{str(i).zfill(6)}" for i in range(n)]
    lats = rng.uniform(32.0, 49.0, n)
    lons = rng.uniform(-124.0, -104.0, n)
    states = rng.choice(["NV", "AZ", "NM", "MT", "ID", "CO", "CA"], n)
    rock_types = rng.choice(
        ["granite", "basalt", "schist", "andesite", "rhyolite", "unknown"], n,
        p=[0.25, 0.20, 0.15, 0.20, 0.15, 0.05]
    )

    # Log-normal concentrations — realistic geochemical distribution
    cu = rng.lognormal(mean=3.5, sigma=1.2, size=n)
    ni = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    co = rng.lognormal(mean=1.5, sigma=0.8, size=n)
    li = rng.lognormal(mean=2.5, sigma=0.9, size=n)
    zn = rng.lognormal(mean=3.8, sigma=1.1, size=n)
    pb = rng.lognormal(mean=3.2, sigma=1.3, size=n)
    au = rng.lognormal(mean=0.5, sigma=1.5, size=n)
    mo = rng.lognormal(mean=1.2, sigma=0.9, size=n)
    mn = rng.lognormal(mean=6.0, sigma=0.8, size=n)

    df = pd.DataFrame({
        "SAMPLE_ID": sample_ids,
        "LAT": lats, "LON": lons,
        "STATE": states, "ROCK_TYPE": rock_types,
        "SAMP_TYPE": rng.choice(["soil", "sediment", "rock", "water"], n),
        "ANAL_METH": rng.choice(["ICP-MS", "ICP-OES", "AAS", "XRF"], n),
        "COLL_DATE": pd.date_range("2000-01-01", periods=n, freq="12h").strftime("%Y-%m-%d").tolist(),
        "CU": cu, "NI": ni, "CO": co, "LI": li,
        "ZN": zn, "PB": pb, "AU": au, "MO": mo, "MN": mn,
    })

    # Inject realistic messiness — sentinel nulls and ND values
    for col in ["CU", "NI", "CO", "LI", "ZN", "PB", "AU", "MO", "MN"]:
        mask_sentinel = rng.random(n) < 0.08
        df.loc[mask_sentinel, col] = -9999
        mask_nd = rng.random(n) < 0.03
        df.loc[mask_nd, col] = "ND"

    return df

raw_df = generate_sample_data(n_rows=500)
print(f"Generated {len(raw_df):,} rows x {len(raw_df.columns)} columns")
print(f"\nFirst 3 rows:")
display(raw_df.head(3))


# COMMAND ----------

# This is what real geoscience data looks like — show the problems we need to fix
print("=== DATA QUALITY ISSUES IN RAW DATA ===\n")

for col in ["CU", "NI", "CO", "AU"]:
    sentinel_count = (raw_df[col].astype(str) == "-9999").sum()
    nd_count = (raw_df[col].astype(str) == "ND").sum()
    print(f"{col}: {sentinel_count} sentinel nulls (-9999), {nd_count} not-detected (ND)")

print(f"\nTotal rows: {len(raw_df):,}")
print(f"Columns: {list(raw_df.columns)}")
print(f"\nCU column sample values: {raw_df['CU'].head(10).tolist()}")

# COMMAND ----------

RAW_PATH = "/Volumes/critical_minerals/geochem/data/01_raw.parquet"

raw_df = raw_df.astype(str); raw_df.to_parquet(RAW_PATH, index=False)
print(f"Saved raw data to: {RAW_PATH}")

# Verify it saved correctly
verify = pd.read_parquet(RAW_PATH)
print(f"Verified: {len(verify):,} rows read back successfully")

# COMMAND ----------

