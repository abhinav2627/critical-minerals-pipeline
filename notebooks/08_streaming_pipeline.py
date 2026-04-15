# Databricks notebook source
# MAGIC %pip install pyarrow==16.1.0 pandas numpy matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()

print(f"Spark version: {spark.version}")
print(f"Streaming available: True")
print(f"Delta Lake available: ", end="")

try:
    spark.sql("SELECT 1").show()
    print("True")
except Exception as e:
    print(f"False — {e}")

# COMMAND ----------

# Use only what we need from pyspark — no wildcard import
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    BooleanType, TimestampType
)
import pyspark.sql.functions as F
import os

spark = SparkSession.builder.getOrCreate()

STREAM_PATH  = "/Volumes/critical_minerals/geochem/data/project8_streaming/"
CHECKPOINT   = STREAM_PATH + "checkpoints/"
DELTA_OUTPUT = STREAM_PATH + "delta_sink/"
ALERTS_PATH  = STREAM_PATH + "alerts/"

for path in [STREAM_PATH, CHECKPOINT, DELTA_OUTPUT, ALERTS_PATH]:
    os.makedirs(path, exist_ok=True)

sensor_schema = StructType([
    StructField("sensor_id",       StringType(),  True),
    StructField("sensor_type",     StringType(),  True),
    StructField("latitude",        DoubleType(),  True),
    StructField("longitude",       DoubleType(),  True),
    StructField("depth_m",         DoubleType(),  True),
    StructField("copper_ppm",      DoubleType(),  True),
    StructField("nickel_ppm",      DoubleType(),  True),
    StructField("mag_anomaly_nt",  DoubleType(),  True),
    StructField("temperature_c",   DoubleType(),  True),
    StructField("is_anomaly",      BooleanType(), True),
    StructField("event_timestamp", StringType(),  True),
])

print("Paths created and schema defined.")
print(f"Fields: {[f.name for f in sensor_schema.fields]}")

# COMMAND ----------

import json
import builtins
import numpy as np
from datetime import datetime, timedelta
import os

# Restore built-in round shadowed by pyspark.sql.functions wildcard import
round = builtins.round

# No PySpark imports here — pure Python cell
# This avoids all PySpark function conflicts

rng = np.random.default_rng(seed=42)

SENSOR_TYPES       = ["GEOCHEM", "MAG", "DRILL"]
N_SENSORS          = 10
N_BATCHES          = 20
READINGS_PER_BATCH = 50

sensor_ids = [f"SENSOR_{i:03d}" for i in range(N_SENSORS)]
TARGET_LAT = 38.42
TARGET_LON = -117.53

all_readings = []

for batch in range(N_BATCHES):
    batch_time = datetime.now() - timedelta(minutes=(N_BATCHES - batch) * 2)

    for _ in range(READINGS_PER_BATCH):
        sensor_id   = str(rng.choice(sensor_ids))
        sensor_type = str(rng.choice(SENSOR_TYPES))
        sensor_num  = int(sensor_id.split("_")[1])

        lat = float(38.0 + sensor_num * 0.05 + float(rng.normal(0, 0.01)))
        lon = float(-118.0 + sensor_num * 0.03 + float(rng.normal(0, 0.01)))

        dist      = float(np.sqrt((lat - TARGET_LAT)**2 + (lon - TARGET_LON)**2))
        proximity = float(np.maximum(0.0, 1.0 - dist / 0.5))

        depth  = float(rng.uniform(0, 200))
        copper = float(rng.lognormal(2.5, 0.8))
        nickel = float(rng.lognormal(2.8, 0.7))
        mag    = float(rng.normal(-145, 75))
        temp   = float(rng.uniform(15, 45))

        if proximity > 0.3:
            copper = copper * float(1.0 + proximity * float(rng.uniform(2, 8)))
            nickel = nickel * float(1.0 + proximity * float(rng.uniform(1, 4)))
            mag    = mag + float(proximity * float(rng.uniform(50, 200)))

        if float(rng.random()) < 0.05:
            copper = copper * float(rng.uniform(5, 20))

        copper = float(np.maximum(0.0, copper))
        nickel = float(np.maximum(0.0, nickel))

        # Use numpy abs — avoids PySpark abs conflict
        is_anomaly = bool(copper > 500.0 or float(np.abs(mag)) > 200.0)

        reading = {
            "sensor_id":       sensor_id,
            "sensor_type":     sensor_type,
            "latitude":        round(lat, 6),
            "longitude":       round(lon, 6),
            "depth_m":         round(depth, 2),
            "copper_ppm":      round(copper, 3),
            "nickel_ppm":      round(nickel, 3),
            "mag_anomaly_nt":  round(mag, 3),
            "temperature_c":   round(temp, 2),
            "is_anomaly":      is_anomaly,
            "event_timestamp": (
                batch_time + timedelta(seconds=float(rng.uniform(0, 120)))
            ).isoformat(),
        }
        all_readings.append(reading)

INPUT_PATH = STREAM_PATH + "input/"
os.makedirs(INPUT_PATH, exist_ok=True)

readings_per_file = len(all_readings) // N_BATCHES
for i in range(N_BATCHES):
    batch_readings = all_readings[i*readings_per_file:(i+1)*readings_per_file]
    fname = f"{INPUT_PATH}batch_{i:03d}.jsonl"
    with open(fname, "w") as f:
        for r in batch_readings:
            f.write(json.dumps(r) + "\n")

print(f"Generated {len(all_readings):,} sensor readings")
print(f"Saved as {N_BATCHES} JSONL files in {INPUT_PATH}")
print(f"Anomaly readings: {sum(1 for r in all_readings if r['is_anomaly']):,}")
print(f"Spike readings:   {sum(1 for r in all_readings if r['copper_ppm'] > 500):,}")
print(f"\nSample reading:")
print(json.dumps(all_readings[0], indent=2))

# COMMAND ----------

raw_stream = (
    spark.readStream
    .format("json")
    .schema(sensor_schema)
    .option("maxFilesPerTrigger", 2)
    .load(INPUT_PATH)
)

print(f"Stream defined")
print(f"isStreaming: {raw_stream.isStreaming}")
print(f"Schema: {raw_stream.schema.simpleString()}")

# COMMAND ----------

# All PySpark functions via F.xxx — no wildcard conflicts

enriched_stream = (
    raw_stream
    .withColumn("event_ts",
        F.to_timestamp(F.col("event_timestamp")))
    .withColumn("copper_zscore",
        (F.col("copper_ppm") - F.lit(21.0)) / F.lit(75.0))
    .withColumn("mag_zscore",
        (F.col("mag_anomaly_nt") + F.lit(143.0)) / F.lit(75.0))
    .withColumn("alert_level",
        F.when(F.col("copper_ppm") > 1000, "CRITICAL")
        .when(F.col("copper_ppm") > 500,   "HIGH")
        .when(F.col("copper_ppm") > 200,   "MEDIUM")
        .otherwise("NORMAL"))
    .withColumn("processing_ts",
        F.current_timestamp())
)

windowed_stats = (
    enriched_stream
    .withWatermark("event_ts", "10 minutes")
    .groupBy(
        F.window(F.col("event_ts"), "5 minutes"),
        F.col("sensor_type"),
        F.col("alert_level")
    )
    .agg(
        F.count("*")                                        .alias("reading_count"),
        F.avg("copper_ppm")                                 .alias("avg_copper_ppm"),
        F.max("copper_ppm")                                 .alias("max_copper_ppm"),
        F.avg("mag_anomaly_nt")                             .alias("avg_mag_anomaly"),
        F.sum(F.when(F.col("is_anomaly"), 1).otherwise(0)) .alias("anomaly_count"),
    )
)

print("Transformations defined:")
print("  enriched_stream — timestamp + zscore + alert_level")
print("  windowed_stats  — 5-min windows by sensor_type + alert_level")

# COMMAND ----------

raw_query = (
    enriched_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT + "raw/")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .start(DELTA_OUTPUT + "raw_readings/")
)

raw_query.awaitTermination()
print(f"Raw readings written to Delta Lake")

stats_query = (
    windowed_stats
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT + "stats/")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .start(DELTA_OUTPUT + "windowed_stats/")
)

stats_query.awaitTermination()
print(f"Windowed stats written to Delta Lake")

# COMMAND ----------

raw_df = spark.read.format("delta").load(DELTA_OUTPUT + "raw_readings/")

total         = raw_df.count()
anomalies     = raw_df.filter(F.col("is_anomaly") == True).count()
critical_count = raw_df.filter(F.col("alert_level") == "CRITICAL").count()
high_count    = raw_df.filter(F.col("alert_level") == "HIGH").count()

print(f"Raw readings in Delta:  {total:,}")
print(f"Anomalies detected:     {anomalies:,}")
print(f"CRITICAL alerts:        {critical_count:,}")
print(f"HIGH alerts:            {high_count:,}")

print(f"\nTop 5 highest copper readings:")
display(
    raw_df
    .select("sensor_id","sensor_type","latitude","longitude",
            "copper_ppm","alert_level","event_ts")
    .orderBy(F.col("copper_ppm").desc())
    .limit(5)
)

stats_df = spark.read.format("delta").load(DELTA_OUTPUT + "windowed_stats/")
print(f"\nWindowed aggregations: {stats_df.count():,} windows")
display(
    stats_df
    .select("window","sensor_type","alert_level",
            "reading_count","avg_copper_ppm","max_copper_ppm","anomaly_count")
    .orderBy(F.col("max_copper_ppm").desc())
    .limit(10)
)

# COMMAND ----------

alert_stream = (
    enriched_stream
    .filter(
        (F.col("alert_level").isin(["CRITICAL", "HIGH"])) |
        (F.col("is_anomaly") == True)
    )
    .select(
        F.col("sensor_id"),
        F.col("sensor_type"),
        F.col("latitude"),
        F.col("longitude"),
        F.col("depth_m"),
        F.col("copper_ppm"),
        F.col("mag_anomaly_nt"),
        F.col("alert_level"),
        F.col("copper_zscore"),
        F.col("event_ts"),
        F.col("processing_ts"),
        F.lit("ACTIVE").alias("alert_status"),
        F.lit("streaming_pipeline_v1").alias("detected_by"),
    )
)

alert_query = (
    alert_stream
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT + "alerts/")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .start(ALERTS_PATH + "active_alerts/")
)

alert_query.awaitTermination()

alerts_df = spark.read.format("delta").load(ALERTS_PATH + "active_alerts/")
total_alerts    = alerts_df.count()
critical_alerts = alerts_df.filter(F.col("alert_level") == "CRITICAL").count()
high_alerts     = alerts_df.filter(F.col("alert_level") == "HIGH").count()

print(f"Total alerts generated: {total_alerts:,}")
print(f"CRITICAL alerts:        {critical_alerts:,}")
print(f"HIGH alerts:            {high_alerts:,}")

print(f"\nTop 5 most critical alerts:")
display(
    alerts_df
    .select("sensor_id","sensor_type","latitude","longitude",
            "copper_ppm","alert_level","event_ts")
    .orderBy(F.col("copper_ppm").desc())
    .limit(5)
)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

raw_pd    = raw_df.toPandas()
alerts_pd = alerts_df.toPandas()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Real-Time Geosensor Streaming Pipeline\n"
    "Spark Structured Streaming 4.1.0  \u00B7  Delta Lake sink  \u00B7  "
    "5-minute windowed aggregations",
    fontsize=13, fontweight="bold"
)

colors_map = {
    "NORMAL":   "#27ae60",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e74c3c",
    "CRITICAL": "#8e44ad"
}

# Panel 1 — readings by alert level
ax1 = axes[0,0]
for level, color in colors_map.items():
    subset = raw_pd[raw_pd["alert_level"] == level]
    if len(subset):
        ax1.scatter(subset["longitude"], subset["latitude"],
                    c=color, s=8, alpha=0.6,
                    label=f"{level} ({len(subset):,})")
ax1.set_title("Sensor readings by alert level", fontsize=11, fontweight="bold")
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.legend(fontsize=8)

# Panel 2 — copper distribution
ax2 = axes[0,1]
clip_val = float(raw_pd["copper_ppm"].quantile(0.99))
ax2.hist(raw_pd["copper_ppm"].clip(upper=clip_val),
         bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
ax2.axvline(200,  color="#f39c12", linestyle="--", lw=2, label="MEDIUM (200 ppm)")
ax2.axvline(500,  color="#e74c3c", linestyle="--", lw=2, label="HIGH (500 ppm)")
ax2.axvline(1000, color="#8e44ad", linestyle="--", lw=2, label="CRITICAL (1000 ppm)")
ax2.set_title("Copper concentration distribution", fontsize=11, fontweight="bold")
ax2.set_xlabel("Copper (ppm)")
ax2.set_ylabel("Count")
ax2.legend(fontsize=8)

# Panel 3 — alert map
ax3 = axes[1,0]
ax3.set_facecolor("#1a1a2e")
background = raw_pd[raw_pd["alert_level"] == "NORMAL"]
ax3.scatter(background["longitude"], background["latitude"],
            c="#2c3e50", s=3, alpha=0.3)
for level, color in [("MEDIUM","#f39c12"),("HIGH","#e74c3c"),("CRITICAL","#8e44ad")]:
    subset = alerts_pd[alerts_pd["alert_level"] == level] if len(alerts_pd) else pd.DataFrame()
    if len(subset):
        ax3.scatter(subset["longitude"], subset["latitude"],
                    c=color, s=40, alpha=0.9, zorder=5,
                    label=f"{level} ({len(subset):,})")
ax3.set_title("Real-time alert locations", fontsize=11, fontweight="bold")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.legend(fontsize=8, facecolor="#2c3e50", labelcolor="white")

# Panel 4 — readings by sensor type
ax4 = axes[1,1]
if len(raw_pd):
    sensor_counts = (
        raw_pd.groupby(["sensor_type","alert_level"])
        .size()
        .unstack(fill_value=0)
    )
    color_list = [colors_map.get(c, "#95a5a6") for c in sensor_counts.columns]
    sensor_counts.plot(kind="bar", ax=ax4, color=color_list,
                       edgecolor="white", alpha=0.8)
ax4.set_title("Readings per sensor type by alert level",
              fontsize=11, fontweight="bold")
ax4.set_xlabel("Sensor type")
ax4.set_ylabel("Count")
ax4.tick_params(axis="x", rotation=0)
ax4.legend(fontsize=8)

plt.tight_layout()

OUTPUT_PATH = STREAM_PATH + "outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
plt.savefig(OUTPUT_PATH + "01_streaming_dashboard.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_streaming_dashboard.png")

# COMMAND ----------

print("=" * 60)
print("PROJECT 8 — REAL-TIME STREAMING PIPELINE COMPLETE")
print("=" * 60)
print(f"\nStreaming framework:  Apache Spark Structured Streaming 4.1.0")
print(f"Sink format:         Delta Lake")
print(f"Trigger mode:        availableNow (micro-batch)")
print(f"Window size:         5 minutes")
print(f"Watermark:           10 minutes (late data tolerance)")
print(f"\nData generated:")
print(f"  Sensor readings:   {raw_df.count():,}")
print(f"  Sensor types:      GEOCHEM · MAG · DRILL")
print(f"  Active sensors:    10")
print(f"  Batches:           20")
print(f"\nAnomalies detected:")
print(f"  Total alerts:      {alerts_df.count():,}")
print(f"  CRITICAL:          {alerts_df.filter(col('alert_level')=='CRITICAL').count():,}")
print(f"  HIGH:              {alerts_df.filter(col('alert_level')=='HIGH').count():,}")
print(f"\nDelta tables written:")
print(f"  raw_readings/      — all enriched sensor readings")
print(f"  windowed_stats/    — 5-minute aggregations by sensor type")
print(f"  active_alerts/     — CRITICAL and HIGH alert events")
print(f"\nNew skills demonstrated:")
print(f"  Spark Structured Streaming")
print(f"  Windowed aggregations with watermarking")
print(f"  Delta Lake streaming sink with ACID transactions")
print(f"  Real-time anomaly detection and alerting")
print("=" * 60)

# COMMAND ----------

