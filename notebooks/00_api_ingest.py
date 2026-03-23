# Databricks notebook source
# MAGIC %pip install requests==2.31.0 loguru==0.7.2 pyarrow==16.1.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import requests
import json

# USGS ScienceBase API — stable, documented, returns real data
# This fetches the EarthMRI geochemistry dataset metadata first
url = "https://www.sciencebase.gov/catalog/item/601963c6d34edf5c66f0d0e5"
params = {"format": "json"}

response = requests.get(url, params=params, timeout=30)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"\nDataset title: {data.get('title')}")
    print(f"Last updated:  {data.get('lastUpdated')}")
    print(f"\nAvailable files:")
    for f in data.get("files", [])[:10]:
        print(f"  {f.get('name')}  —  {f.get('size', 0) // 1024} KB")

# COMMAND ----------

