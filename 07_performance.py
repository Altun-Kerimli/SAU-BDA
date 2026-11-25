
import os
import json
import time
import statistics
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

os.environ["JAVA_HOME"] = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)

os.environ["HADOOP_HOME"] = r"C:\hadoop"

os.environ["PATH"] = (
    os.path.join(os.environ["JAVA_HOME"], "bin")
    + os.pathsep
    + os.path.join(os.environ["HADOOP_HOME"], "bin")
    + os.pathsep
    + os.environ["PATH"]
)

# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel

# ============================================================
# 3. INITIALIZE SPARK
# ============================================================

PARTITIONS = 8
REPETITIONS = 3

spark = (
    SparkSession.builder
    .appName("Flight Delay - Scalability Analysis")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", str(PARTITIONS))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data/processed/flights_clean.parquet"
METRICS_DIR = BASE_DIR / "results/metrics"
FIGURES_DIR = BASE_DIR / "results/figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. LOAD DATASET
# ============================================================

df = spark.read.parquet(str(DATA_PATH))

# Only columns required for our benchmark
source = df.select("AIRLINE", "IS_DELAYED")

total_records = source.count()

print("\n=== PERFORMANCE EXPERIMENT ===")
print("Spark version:", spark.version)
print("Total records:", total_records)
print("Partitions:", PARTITIONS)
print("Repetitions:", REPETITIONS)

# ============================================================
# 5. DATASET SIZES
# ============================================================

dataset_sizes = [
    100_000,
    500_000,
    1_000_000,
    total_records
]

dataset_sizes = sorted(set(dataset_sizes))

# ============================================================
# 6. ANALYTICAL OPERATION
# ============================================================

def run_analysis(dataframe):
    """
    Calculate airline-level flight counts and delay rates.

    Spark operations:
    - GroupBy
    - Count
    - Average
    - Shuffle aggregation

    collect() triggers execution.
    """

    return (
        dataframe
        .groupBy("AIRLINE")
        .agg(
            F.count("*").alias("total_flights"),
            F.avg("IS_DELAYED").alias("delay_rate")
        )
        .collect()
    )


# ============================================================
# 7. BENCHMARK FUNCTION
# ============================================================

def benchmark(target_size):

    print("\n====================================")
    print(f"TESTING: {target_size:,} RECORDS")
    print("====================================")

    # Select the requested number of records.
    # Use the complete DataFrame for the full-dataset test.

    if target_size == total_records:
        subset = source
    else:
        subset = source.limit(target_size)

    # Use an equal partition count for all experiments.
    # Materialization is excluded from measured runtime.

    subset = (
        subset
        .repartition(PARTITIONS)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    try:
        actual_count = subset.count()

        print("Prepared records:", actual_count)

        # Warm-up run: Spark initialization and query planning
        # effects are excluded from the measured trials.

        warmup_result = run_analysis(subset)

        verified_count = sum(
            row["total_flights"] for row in warmup_result
        )

        if verified_count != actual_count:
            raise RuntimeError(
                "Benchmark aggregation record count mismatch."
            )

        execution_times = []

        # Measure the same operation multiple times.

        for repetition in range(1, REPETITIONS + 1):

            start_time = time.perf_counter()

            run_analysis(subset)

            elapsed = time.perf_counter() - start_time

            execution_times.append(elapsed)

            print(
                f"Run {repetition}: {elapsed:.4f} seconds"
            )

        median_time = statistics.median(execution_times)
        average_time = statistics.mean(execution_times)

        throughput = actual_count / median_time

        print(f"Median time: {median_time:.4f} seconds")
        print(f"Average time: {average_time:.4f} seconds")
        print(f"Throughput: {throughput:,.0f} rows/sec")

        return {
            "dataset_size": actual_count,
            "partitions": PARTITIONS,
            "run_1_seconds": round(execution_times[0], 4),
            "run_2_seconds": round(execution_times[1], 4),
            "run_3_seconds": round(execution_times[2], 4),
            "average_seconds": round(average_time, 4),
            "median_seconds": round(median_time, 4),
            "rows_per_second": round(throughput, 2)
        }

    finally:
        # Release this subset before preparing the next one.
        subset.unpersist(blocking=True)


# ============================================================
# 8. EXECUTE SCALABILITY EXPERIMENTS
# ============================================================

results = []

for size in dataset_sizes:
    results.append(benchmark(size))

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

csv_path = METRICS_DIR / "07_performance.csv"

results_df.to_csv(
    csv_path,
    index=False
)

json_path = METRICS_DIR / "07_performance.json"

summary = {
    "experiment": "Spark scalability by dataset size",
    "spark_version": spark.version,
    "master": "local[2]",
    "driver_memory": "4g",
    "partitions": PARTITIONS,
    "repetitions": REPETITIONS,
    "warmup_runs": 1,
    "storage_level": "MEMORY_AND_DISK",
    "timing_scope": "Cached airline aggregation",
    "results": results
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# ============================================================
# 10. GENERATE SCALABILITY CHART
# ============================================================

plt.figure(figsize=(11, 6))

x_positions = range(len(results_df))

plt.plot(
    list(x_positions),
    results_df["median_seconds"],
    marker="o",
    linewidth=2
)

labels = [
    f"{size / 1_000_000:.2f}M"
    for size in results_df["dataset_size"]
]

plt.xticks(list(x_positions), labels)

plt.xlabel("Dataset Size (Million Flights)")
plt.ylabel("Median Execution Time (Seconds)")

plt.title(
    "Apache Spark Scalability Analysis\n"
    "Airline Delay Aggregation"
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

figure_path = FIGURES_DIR / "07_scalability.png"

plt.savefig(figure_path, dpi=200)
plt.close()

# ============================================================
# 11. FINAL RESULTS
# ============================================================

print("\n====================================")
print("PERFORMANCE ANALYSIS COMPLETED")
print("====================================")

print("\nPerformance comparison:")

print(results_df.to_string(index=False))

print("\nGenerated files:")
print("1. results/metrics/07_performance.csv")
print("2. results/metrics/07_performance.json")
print("3. results/figures/07_scalability.png")

spark.stop()
