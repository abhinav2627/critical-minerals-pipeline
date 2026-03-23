# Databricks notebook source
# MAGIC %pip install pyarrow==16.1.0 geopandas==0.14.4

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np

CLEAN_PATH     = "/Volumes/critical_minerals/geochem/data/02_standardised.parquet"
VALIDATED_PATH = "/Volumes/critical_minerals/geochem/data/03_validated.parquet"

df = pd.read_parquet(CLEAN_PATH)
print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")

# COMMAND ----------

# A validation suite is a collection of named checks
# Each check has:
#   - A name (so you know what failed)
#   - A severity: CRITICAL (halts pipeline) or WARNING (logs but continues)
#   - A result: PASS or FAIL
#   - A count of affected rows

results = []

def check(name, passed, severity, message, rows_affected=0):
    status = "PASS" if passed else "FAIL"
    results.append({
        "check":         name,
        "status":        status,
        "severity":      severity,
        "message":       message,
        "rows_affected": rows_affected,
    })
    icon = "✓" if passed else ("⚠" if severity == "WARNING" else "✗")
    print(f"{icon} [{severity:<8}] {name:<40} {message}")

print("RUNNING VALIDATION SUITE")
print("=" * 75)

# ── Check 1: Minimum row count ──────────────────────────────────────────────
# Why: if we got fewer than 1000 rows something went badly wrong upstream
check(
    name          = "minimum_row_count",
    passed        = len(df) >= 1000,
    severity      = "CRITICAL",
    message       = f"{len(df):,} rows (min: 1,000)",
    rows_affected = 0 if len(df) >= 1000 else len(df),
)

# ── Check 2: Required columns present ───────────────────────────────────────
# Why: if a column is missing, every downstream stage silently breaks
required_cols = ["sample_id", "latitude", "longitude", "copper_ppm"]
missing       = [c for c in required_cols if c not in df.columns]
check(
    name          = "required_columns_present",
    passed        = len(missing) == 0,
    severity      = "CRITICAL",
    message       = f"Missing: {missing}" if missing else "All required columns present",
    rows_affected = 0,
)

# ── Check 3: No missing coordinates ─────────────────────────────────────────
# Why: a sample without coordinates is spatially useless
null_coords = df["latitude"].isna().sum() + df["longitude"].isna().sum()
check(
    name          = "no_missing_coordinates",
    passed        = null_coords == 0,
    severity      = "CRITICAL",
    message       = f"{null_coords} missing coordinate values",
    rows_affected = int(null_coords),
)

# ── Check 4: Coordinate bounds (must be within US) ──────────────────────────
# Why: a coordinate of (0, 0) or (999, 999) means the GPS failed
out_of_bounds = df[
    (df["latitude"]  < 15)   | (df["latitude"]  > 72) |
    (df["longitude"] < -180) | (df["longitude"] > -50)
]
check(
    name          = "coordinate_bounds_us",
    passed        = len(out_of_bounds) == 0,
    severity      = "WARNING",
    message       = f"{len(out_of_bounds)} samples outside US bounding box",
    rows_affected = len(out_of_bounds),
)

# ── Check 5: Duplicate sample IDs ───────────────────────────────────────────
# Why: duplicate IDs cause double-counting in aggregations and joins
n_dupes = df["sample_id"].duplicated().sum()
check(
    name          = "no_duplicate_sample_ids",
    passed        = n_dupes == 0,
    severity      = "WARNING",
    message       = f"{n_dupes:,} duplicate sample IDs",
    rows_affected = int(n_dupes),
)

# ── Check 6: Copper null rate ────────────────────────────────────────────────
# Why: if >50% of copper values are null, the ingestion likely failed
cu_null_rate = df["copper_ppm"].isna().mean()
check(
    name          = "copper_null_rate",
    passed        = cu_null_rate <= 0.50,
    severity      = "CRITICAL",
    message       = f"Copper null rate: {cu_null_rate:.1%} (max: 50%)",
    rows_affected = int(df["copper_ppm"].isna().sum()),
)

# ── Check 7: No negative element values ─────────────────────────────────────
# Why: we already substituted BDL values — any remaining negatives are errors
element_cols = [c for c in df.columns if c.endswith("_ppm")]
neg_counts   = {c: (df[c] < 0).sum() for c in element_cols if (df[c] < 0).sum() > 0}
check(
    name          = "no_negative_element_values",
    passed        = len(neg_counts) == 0,
    severity      = "WARNING",
    message       = f"Negative values in: {neg_counts}" if neg_counts else "No negative values",
    rows_affected = sum(neg_counts.values()),
)

# ── Check 8: Copper physically plausible ────────────────────────────────────
# Why: copper above 100,000 ppm is physically impossible (pure copper = 1M ppm)
cu_max       = df["copper_ppm"].max()
implausible  = (df["copper_ppm"] > 100000).sum()
check(
    name          = "copper_physically_plausible",
    passed        = implausible == 0,
    severity      = "WARNING",
    message       = f"Max copper: {cu_max:.1f} ppm | {implausible} values > 100,000 ppm",
    rows_affected = int(implausible),
)

# ── Check 9: Pipeline metadata columns present ───────────────────────────────
# Why: without these we can't trace which pipeline run produced this data
meta_cols   = ["_pipeline_version", "_processed_at", "_schema_version"]
missing_meta = [c for c in meta_cols if c not in df.columns]
check(
    name          = "pipeline_metadata_present",
    passed        = len(missing_meta) == 0,
    severity      = "WARNING",
    message       = f"Missing: {missing_meta}" if missing_meta else "All metadata columns present",
    rows_affected = 0,
)

# ── Check 10: Sample type populated ─────────────────────────────────────────
# Why: sample type (soil vs stream sediment) affects interpretation
null_type = df["sample_type"].isna().sum() if "sample_type" in df.columns else 0
check(
    name          = "sample_type_populated",
    passed        = null_type / len(df) < 0.10,
    severity      = "WARNING",
    message       = f"{null_type:,} samples missing sample type ({null_type/len(df):.1%})",
    rows_affected = int(null_type),
)

# COMMAND ----------

import json

results_df = pd.DataFrame(results)

# Count results
n_pass     = (results_df["status"] == "PASS").sum()
n_fail     = (results_df["status"] == "FAIL").sum()
n_critical = ((results_df["status"] == "FAIL") & 
              (results_df["severity"] == "CRITICAL")).sum()

print("\n" + "=" * 75)
print("VALIDATION SUITE SUMMARY")
print("=" * 75)
print(f"Total checks:     {len(results_df)}")
print(f"Passed:           {n_pass}")
print(f"Failed:           {n_fail}")
print(f"Critical failures:{n_critical}")
print(f"Suite result:     {'PASSED' if n_critical == 0 else 'FAILED'}")
print("=" * 75)

# Save validation report — this is your audit trail
report = {
    "timestamp":        pd.Timestamp.now().isoformat(),
    "total_rows":       len(df),
    "suite_passed":     bool(n_critical == 0),
    "checks_passed":    int(n_pass),
    "checks_failed":    int(n_fail),
    "critical_failures":int(n_critical),
    "results":          results,
}

REPORT_PATH = "/Volumes/critical_minerals/geochem/data/validation_report.json"
with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nValidation report saved: {REPORT_PATH}")

# HALT if critical checks failed
if n_critical > 0:
    failed = results_df[
        (results_df["status"] == "FAIL") & 
        (results_df["severity"] == "CRITICAL")
    ]["check"].tolist()
    raise Exception(
        f"PIPELINE HALTED — {n_critical} critical validation failure(s): {failed}"
    )

print("\nAll critical checks passed — safe to proceed to Stage 4.")

# COMMAND ----------

df.to_parquet(VALIDATED_PATH, index=False)
print(f"Saved validated data: {VALIDATED_PATH}")
print(f"Shape: {len(df):,} rows x {len(df.columns)} columns")
print("\nStage 3 complete.")

# COMMAND ----------

