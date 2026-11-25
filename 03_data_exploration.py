import os
import json
from pathlib import Path

# Configure Java 17
os.environ["JAVA_HOME"] = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import kagglehub


# 1. Initialize Spark
spark = (
    SparkSession.builder
    .appName("Flight Delay - Data Exploration")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# 2. Load dataset
path = Path(kagglehub.dataset_download("usdot/flight-delays"))

df = spark.read.csv(
    str(path / "flights.csv"),
    header=True,
    inferSchema=True
)

print("\n=== DATASET INFORMATION ===")

total = df.count()

print("Total records:", total)
print("Total columns:", len(df.columns))


# 3. Missing values
print("\n=== MISSING VALUES ===")

missing = df.agg(
    *[
        F.sum(F.col(c).isNull().cast("long")).alias(c)
        for c in df.columns
    ]
).first().asDict()

for column, count in missing.items():
    if count:
        percentage = count / total * 100
        print(f"{column}: {count:,} ({percentage:.2f}%)")


# 4. Potential duplicate flight records
print("\n=== DUPLICATE CHECK ===")

flight_keys = [
    "YEAR", "MONTH", "DAY",
    "AIRLINE", "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT", "SCHEDULED_DEPARTURE"
]

duplicates = (
    df.groupBy(*flight_keys)
    .count()
    .filter(F.col("count") > 1)
    .agg(F.sum(F.col("count") - 1).alias("duplicates"))
    .first()["duplicates"]
) or 0

print("Potential duplicate flight keys:", duplicates)


# 5. Flight statistics
print("\n=== FLIGHT STATISTICS ===")

stats = df.agg(
    F.sum(F.when(F.col("CANCELLED") == 1, 1).otherwise(0))
        .alias("cancelled"),

    F.sum(F.when(F.col("DIVERTED") == 1, 1).otherwise(0))
        .alias("diverted"),

    F.count("ARRIVAL_DELAY").alias("valid_arrivals"),

    F.sum(F.when(F.col("ARRIVAL_DELAY") >= 15, 1).otherwise(0))
        .alias("delayed"),

    F.avg("ARRIVAL_DELAY").alias("average_delay"),
    F.min("ARRIVAL_DELAY").alias("minimum_delay"),
    F.max("ARRIVAL_DELAY").alias("maximum_delay")
).first().asDict()

for key, value in stats.items():
    print(f"{key}: {value}")

delay_rate = 100 * stats["delayed"] / stats["valid_arrivals"]

print(f"Arrival delay rate: {delay_rate:.2f}%")


# 6. Save results
output = Path("results/metrics")
output.mkdir(parents=True, exist_ok=True)

summary = {
    "total_records": total,
    "total_columns": len(df.columns),
    "column_types": dict(df.dtypes),
    "missing_values": missing,
    "potential_duplicate_keys": duplicates,
    "statistics": stats,
    "delay_rate_percent": delay_rate
}

with open(output / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\nResults saved to results/metrics/dataset_summary.json")

spark.stop()