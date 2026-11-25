import os
import json
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

# Java 17
os.environ["JAVA_HOME"] = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)

# Hadoop Windows utilities
os.environ["HADOOP_HOME"] = r"C:\hadoop"

# Add Java and Hadoop to PATH
os.environ["PATH"] = (
    os.path.join(os.environ["JAVA_HOME"], "bin")
    + os.pathsep
    + os.path.join(os.environ["HADOOP_HOME"], "bin")
    + os.pathsep
    + os.environ["PATH"]
)

# Verify required executables
java_exe = Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe"
winutils_exe = Path(os.environ["HADOOP_HOME"]) / "bin" / "winutils.exe"

if not java_exe.is_file():
    raise FileNotFoundError(f"Java not found: {java_exe}")

if not winutils_exe.is_file():
    raise FileNotFoundError(f"Hadoop winutils not found: {winutils_exe}")


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import kagglehub

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 3. INITIALIZE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("Flight Delay - Preprocessing")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark initialized successfully!")


# ============================================================
# 4. LOAD DATASET
# ============================================================

dataset_path = Path(
    kagglehub.dataset_download("usdot/flight-delays")
)

csv_path = dataset_path / "flights.csv"

df = spark.read.csv(
    str(csv_path),
    header=True,
    inferSchema=True
)

print("\n=== ORIGINAL DATASET ===")

original_count = df.count()

print("Original records:", original_count)
print("Original columns:", len(df.columns))


# ============================================================
# 5. REMOVE CANCELLED AND DIVERTED FLIGHTS
# ============================================================

df = df.filter(
    (F.col("CANCELLED") == 0) &
    (F.col("DIVERTED") == 0)
)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

required_columns = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "ARRIVAL_DELAY"
]

df = df.dropna(subset=required_columns)


# ============================================================
# 7. VALIDATE DEPARTURE TIMES
# ============================================================

# Valid scheduled departure: 0000 - 2400
df = df.filter(
    F.col("SCHEDULED_DEPARTURE").between(0, 2400)
)

# Minutes must be between 00 and 59
df = df.filter(
    (F.col("SCHEDULED_DEPARTURE") % 100) < 60
)


# ============================================================
# 8. REMOVE EXACT DUPLICATES
# ============================================================

valid_count = df.count()

df = df.dropDuplicates()


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

# Extract scheduled departure hour
df = df.withColumn(
    "DEP_HOUR",
    (
        F.floor(F.col("SCHEDULED_DEPARTURE") / 100) % 24
    ).cast("integer")
)

# Create classification target
# 1 = Flight delayed by 15+ minutes
# 0 = Flight not delayed

df = df.withColumn(
    "IS_DELAYED",
    F.when(
        F.col("ARRIVAL_DELAY") >= 15, 1
    ).otherwise(0)
)


# ============================================================
# 10. SELECT RELEVANT COLUMNS
# ============================================================

columns = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DISTANCE",
    "ARRIVAL_DELAY",
    "DEP_HOUR",
    "IS_DELAYED"
]

df = df.select(*columns)


# ============================================================
# 11. CALCULATE FINAL STATISTICS
# ============================================================

print("\n=== CLEANED DATASET ===")

stats = df.agg(
    F.count("*").alias("total"),
    F.sum("IS_DELAYED").alias("delayed")
).first()

clean_count = stats["total"]
delayed_count = stats["delayed"]

removed_invalid = original_count - valid_count
removed_duplicates = valid_count - clean_count

delay_rate = round(
    delayed_count / clean_count * 100, 2
)

print("Clean records:", clean_count)
print("Removed invalid/cancelled/diverted:", removed_invalid)
print("Removed exact duplicates:", removed_duplicates)
print("Delayed flights:", delayed_count)
print("Delay rate:", delay_rate, "%")

print("\nSample records:")
df.show(5)


# ============================================================
# 12. SAVE CLEANED DATASET AS PARQUET
# ============================================================

print("\n=== SAVING PARQUET ===")

output_path = Path(
    "data/processed/flights_clean.parquet"
).resolve()

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.write \
    .mode("overwrite") \
    .parquet(str(output_path))

print("Parquet saved successfully!")
print("Location:", output_path)


# ============================================================
# 13. VERIFY SAVED DATASET
# ============================================================

print("\n=== VERIFYING PARQUET ===")

verified_df = spark.read.parquet(str(output_path))

verified_count = verified_df.count()

print("Saved records:", verified_count)
print("Saved columns:", len(verified_df.columns))

if verified_count != clean_count:
    raise RuntimeError(
        "Parquet verification failed: record count mismatch!"
    )

print("Parquet verification successful!")


# ============================================================
# 14. SAVE PREPROCESSING STATISTICS
# ============================================================

metrics_path = Path("results/metrics")
metrics_path.mkdir(parents=True, exist_ok=True)

summary = {
    "original_records": original_count,
    "valid_records_before_deduplication": valid_count,
    "clean_records": clean_count,
    "removed_invalid_cancelled_diverted": removed_invalid,
    "removed_exact_duplicates": removed_duplicates,
    "delayed_flights": delayed_count,
    "delay_rate_percent": delay_rate,
    "output_columns": columns,
    "parquet_verified": True
}

with open(
    metrics_path / "preprocessing_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 15. FINISH
# ============================================================

print("\n====================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("====================================")

print("Clean records:", clean_count)
print("Delayed flights:", delayed_count)
print("Delay rate:", delay_rate, "%")

print("\nGenerated files:")
print("1. data/processed/flights_clean.parquet")
print("2. results/metrics/preprocessing_summary.json")

spark.stop()