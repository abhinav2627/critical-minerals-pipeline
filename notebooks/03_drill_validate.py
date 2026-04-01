# Databricks notebook source
pip install pyarrow==16.1.0 numpy pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np

DRILL_PATH = "/Volumes/critical_minerals/geochem/data/project3_drilling/"

collar_df = pd.read_parquet(DRILL_PATH + "01_collar_raw.parquet")
survey_df = pd.read_parquet(DRILL_PATH + "01_survey_raw.parquet")
assay_df  = pd.read_parquet(DRILL_PATH + "01_assay_raw.parquet")

print(f"Collar: {len(collar_df):,} rows")
print(f"Survey: {len(survey_df):,} rows")
print(f"Assay:  {len(assay_df):,} rows")

# COMMAND ----------

results = []

def check(name, passed, severity, message, rows=0):
    results.append({"check": name, "passed": passed,
                    "severity": severity, "message": message, "rows": rows})
    icon = "✓" if passed else ("⚠" if severity == "WARNING" else "✗")
    print(f"{icon} [{severity:<8}] {name:<45} {message}")

print("COLLAR VALIDATION")
print("=" * 75)

# Check 1 — required columns
required = ["HoleID", "Easting", "Northing", "Elevation", "TotalDepth", "Azimuth", "Dip"]
missing  = [c for c in required if c not in collar_df.columns]
check("collar_required_columns", len(missing) == 0, "CRITICAL",
      f"Missing: {missing}" if missing else "All required columns present")

# Check 2 — no duplicate HoleIDs
dupes = collar_df["HoleID"].duplicated().sum()
check("collar_no_duplicate_holes", dupes == 0, "CRITICAL",
      f"{dupes} duplicate HoleIDs" if dupes else "No duplicates", dupes)

# Check 3 — no missing values in key columns
for col in ["HoleID", "Easting", "Northing", "TotalDepth"]:
    nulls = collar_df[col].isna().sum()
    check(f"collar_{col.lower()}_not_null", nulls == 0, "CRITICAL",
          f"{nulls} nulls" if nulls else "No nulls", nulls)

# Check 4 — depth must be positive
neg_depth = (collar_df["TotalDepth"] <= 0).sum()
check("collar_positive_depth", neg_depth == 0, "CRITICAL",
      f"{neg_depth} non-positive depths" if neg_depth else "All depths positive", neg_depth)

# Check 5 — azimuth range 0-360
bad_az = ((collar_df["Azimuth"] < 0) | (collar_df["Azimuth"] > 360)).sum()
check("collar_azimuth_range", bad_az == 0, "WARNING",
      f"{bad_az} values outside 0-360" if bad_az else "All azimuths valid", bad_az)

# Check 6 — dip range -90 to 0
bad_dip = ((collar_df["Dip"] < -90) | (collar_df["Dip"] > 0)).sum()
check("collar_dip_range", bad_dip == 0, "WARNING",
      f"{bad_dip} dips outside -90 to 0" if bad_dip else "All dips valid", bad_dip)

# COMMAND ----------

print("\nSURVEY VALIDATION")
print("=" * 75)

# Check 1 — all HoleIDs exist in collar
collar_ids  = set(collar_df["HoleID"])
survey_ids  = set(survey_df["HoleID"])
orphan_ids  = survey_ids - collar_ids
check("survey_holeid_in_collar", len(orphan_ids) == 0, "CRITICAL",
      f"{len(orphan_ids)} HoleIDs not in collar: {orphan_ids}" if orphan_ids
      else "All HoleIDs match collar")

# Check 2 — every collar hole has at least one survey
holes_with_survey = set(survey_df["HoleID"].unique())
holes_missing     = collar_ids - holes_with_survey
check("survey_all_holes_covered", len(holes_missing) == 0, "CRITICAL",
      f"{len(holes_missing)} holes have no survey" if holes_missing
      else "All holes have survey data")

# Check 3 — depths must be non-negative
neg_depths = (survey_df["Depth"] < 0).sum()
check("survey_no_negative_depths", neg_depths == 0, "CRITICAL",
      f"{neg_depths} negative depths" if neg_depths else "All depths non-negative", neg_depths)

# Check 4 — survey depth within hole total depth
merged = survey_df.merge(collar_df[["HoleID", "TotalDepth"]], on="HoleID")
beyond = (merged["Depth"] > merged["TotalDepth"] + 1).sum()
check("survey_depth_within_total", beyond == 0, "WARNING",
      f"{beyond} survey depths exceed total depth" if beyond
      else "All survey depths within total depth", beyond)

# Check 5 — each hole has survey at depth 0
has_zero = survey_df.groupby("HoleID")["Depth"].min()
missing_zero = (has_zero > 1).sum()
check("survey_starts_at_zero", missing_zero == 0, "WARNING",
      f"{missing_zero} holes missing survey at depth 0" if missing_zero
      else "All holes have survey at depth 0", missing_zero)

# COMMAND ----------

print("\nASSAY VALIDATION")
print("=" * 75)

# Check 1 — all HoleIDs in collar
assay_ids   = set(assay_df["HoleID"])
orphan_assay = assay_ids - collar_ids
check("assay_holeid_in_collar", len(orphan_assay) == 0, "CRITICAL",
      f"{len(orphan_assay)} HoleIDs not in collar" if orphan_assay
      else "All HoleIDs match collar")

# Check 2 — From < To (interval must be positive direction)
bad_intervals = (assay_df["From"] >= assay_df["To"]).sum()
check("assay_from_less_than_to", bad_intervals == 0, "CRITICAL",
      f"{bad_intervals} intervals where From >= To" if bad_intervals
      else "All intervals valid (From < To)", bad_intervals)

# Check 3 — no gaps between consecutive intervals
assay_sorted = assay_df.sort_values(["HoleID", "From"])
assay_sorted["prev_To"] = assay_sorted.groupby("HoleID")["To"].shift(1)
gaps = assay_sorted[
    (assay_sorted["prev_To"].notna()) &
    (abs(assay_sorted["From"] - assay_sorted["prev_To"]) > 0.01)
]
check("assay_no_depth_gaps", len(gaps) == 0, "WARNING",
      f"{len(gaps)} depth gaps detected between intervals" if len(gaps)
      else "No depth gaps detected", len(gaps))

# Check 4 — null rates per element
element_cols = ["Cu_ppm", "Ni_ppm", "Co_ppm", "Au_ppb", "Mo_ppm"]
for col in element_cols:
    null_rate = assay_df[col].isna().mean()
    check(f"assay_null_rate_{col.lower()}", null_rate < 0.15, "WARNING",
          f"Null rate = {null_rate:.1%}", int(assay_df[col].isna().sum()))

# Check 5 — BDL values (negative) detected
for col in element_cols:
    bdl_count = (assay_df[col] < 0).sum()
    check(f"assay_bdl_detected_{col.lower()}", True, "INFO",
          f"{bdl_count:,} BDL values detected — will substitute with abs(val)/2", bdl_count)

# Check 6 — recovery percentage in valid range
bad_recovery = ((assay_df["Recovery_pct"] < 0) |
                (assay_df["Recovery_pct"] > 100)).sum()
check("assay_recovery_range", bad_recovery == 0, "WARNING",
      f"{bad_recovery} recovery values outside 0-100" if bad_recovery
      else "All recovery values valid", bad_recovery)

# COMMAND ----------

import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

results_df = pd.DataFrame(results)
n_pass     = (results_df["passed"] == True).sum()
n_fail     = (results_df["passed"] == False).sum()
n_critical = ((results_df["passed"] == False) &
              (results_df["severity"] == "CRITICAL")).sum()

print("\n" + "=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)
print(f"Total checks:      {len(results_df)}")
print(f"Passed:            {n_pass}")
print(f"Failed:            {n_fail}")
print(f"Critical failures: {n_critical}")
print(f"Suite result:      {'PASSED' if n_critical == 0 else 'FAILED'}")
print("=" * 55)

# Save clean validated tables
collar_df.to_parquet(DRILL_PATH + "02_collar_validated.parquet", index=False)
survey_df.to_parquet(DRILL_PATH + "02_survey_validated.parquet", index=False)
assay_df.to_parquet(DRILL_PATH  + "02_assay_validated.parquet",  index=False)

# Save report
report = {
    "timestamp":        pd.Timestamp.now().isoformat(),
    "suite_passed":     bool(n_critical == 0),
    "checks_passed":    int(n_pass),
    "checks_failed":    int(n_fail),
    "critical_failures":int(n_critical),
    "results":          results
}
with open(DRILL_PATH + "validation_report.json", "w") as f:
    json.dump(report, f, indent=2, cls=NpEncoder)

print(f"\nSaved validated files and validation report.")
print("\nStage 2 complete — all 3 tables validated.")

if n_critical > 0:
    raise Exception(f"PIPELINE HALTED — {n_critical} critical failures")

# COMMAND ----------

