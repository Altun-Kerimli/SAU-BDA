
"""
SAU Big Data Analytics
07b - Apache Spark Partition Performance Benchmark

Compare 2, 4, 8 and 16 partitions using the same full
cleaned flight dataset and the same Spark SQL aggregation.

Method:
    - Local Spark execution with 2 worker threads
    - MEMORY_AND_DISK caching
    - Cache preparation measured separately
    - Adaptive Query Execution disabled for consistency
    - One warm-up query per configuration
    - Five timed executions per configuration
    - Median, mean, standard deviation and throughput

Outputs:
    results/metrics/07b_partition_comparison.csv
    results/metrics/07b_partition_comparison.json
    results/figures/07b_partition_comparison.png

Run:
    python 07b_partition_benchmark.py
"""

import os
import sys
import json
import time
import statistics

from pathlib import Path

# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

JAVA_HOME = Path(
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)

HADOOP_HOME = Path(r"C:\hadoop")

for name, path in [
    ("JAVA_HOME", JAVA_HOME),
    ("HADOOP_HOME", HADOOP_HOME)
]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing {name}: {path}"
        )

    os.environ[name] = str(path)

    os.environ["PATH"] = (
        str(path / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ============================================================
# 2. LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ============================================================
# 3. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT / "data/processed/flights_clean.parquet"
)

METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

PARTITION_COUNTS = [2, 4, 8, 16]

REPETITIONS = 5
WARMUP_RUNS = 1

MASTER = "local[2]"
DRIVER_MEMORY = "4g"

# ============================================================
# 4. SPARK SESSION
# ============================================================

def create_spark():

    return (
        SparkSession.builder
        .appName("SAU BDA - Partition Benchmark")
        .master(MASTER)
        .config(
            "spark.driver.memory",
            DRIVER_MEMORY
        )
        .config(
            "spark.sql.shuffle.partitions",
            "8"
        )
        .config(
            "spark.sql.adaptive.enabled",
            "false"
        )
        .getOrCreate()
    )


# ============================================================
# 5. ANALYTICAL QUERY
# ============================================================

def build_query(df):
    """
    Analyze airline-level arrival delay rates.

    The same aggregation is executed for every
    partition configuration.

    collect() triggers the actual Spark computation.
    """

    return (
        df.groupBy("AIRLINE")
        .agg(
            F.count("*").alias("total_flights"),
            F.sum("IS_DELAYED").alias("delayed_flights"),
            F.avg("IS_DELAYED").alias("delay_rate")
        )
    )


# ============================================================
# 6. VERIFY QUERY OUTPUT
# ============================================================

def verify_result(rows, expected_count):

    actual_count = sum(
        int(row["total_flights"])
        for row in rows
    )

    if actual_count != expected_count:
        raise RuntimeError(
            f"Aggregation mismatch: "
            f"{actual_count} != {expected_count}"
        )

    if not rows:
        raise RuntimeError(
            "Aggregation returned no airlines."
        )

    return len(rows)


# ============================================================
# 7. BENCHMARK ONE PARTITION CONFIGURATION
# ============================================================

def benchmark(spark, source, partitions, total_records):

    print("\n==========================================")
    print(f"TESTING {partitions} PARTITIONS")
    print("==========================================", flush=True)

    # Set shuffle partitions explicitly.
    # Disable AQE so Spark does not automatically
    # coalesce partitions during the comparison.

    spark.conf.set(
        "spark.sql.shuffle.partitions",
        str(partitions)
    )

    # Redistribute the SAME dataset for each test.
    # This avoids changing sample sizes between runs.

    prepared = (
        source
        .repartition(partitions)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    try:

        # --------------------------------------------
        # A. MATERIALIZE CACHED INPUT
        # --------------------------------------------

        preparation_start = time.perf_counter()

        cached_records = prepared.count()

        preparation_time = (
            time.perf_counter() - preparation_start
        )

        if cached_records != total_records:
            raise RuntimeError(
                "Cached dataset count mismatch."
            )

        print(
            "Prepared records:",
            f"{cached_records:,}"
        )

        print(
            "Cache preparation:",
            f"{preparation_time:.4f} s"
        )

        # --------------------------------------------
        # B. CONSTRUCT IDENTICAL ANALYTICAL QUERY
        # --------------------------------------------

        query = build_query(prepared)

        # --------------------------------------------
        # C. WARM-UP
        # --------------------------------------------

        for _ in range(WARMUP_RUNS):
            warmup_rows = query.collect()
            verify_result(
                warmup_rows,
                total_records
            )

        # --------------------------------------------
        # D. TIMED EXECUTIONS
        # --------------------------------------------

        times = []

        airline_groups = None

        for run_number in range(
            1, REPETITIONS + 1
        ):

            start = time.perf_counter()

            # Action: executes the entire aggregation.
            rows = query.collect()

            elapsed = (
                time.perf_counter() - start
            )

            # Validate outside the measured interval.
            airline_groups = verify_result(
                rows,
                total_records
            )

            times.append(elapsed)

            print(
                f"Run {run_number}: "
                f"{elapsed:.4f} s",
                flush=True
            )

        # --------------------------------------------
        # E. SUMMARY STATISTICS
        # --------------------------------------------

        median_time = statistics.median(times)
        mean_time = statistics.mean(times)

        std_time = (
            statistics.stdev(times)
            if len(times) > 1 else 0.0
        )

        throughput = (
            total_records / median_time
        )

        print(
            f"Median: {median_time:.4f} s"
        )

        print(
            f"Mean:   {mean_time:.4f} s"
        )

        print(
            f"Std:    {std_time:.4f} s"
        )

        print(
            f"Throughput: {throughput:,.0f} rows/s"
        )

        return {
            "partitions": partitions,
            "records": total_records,
            "airline_groups": airline_groups,
            "cache_preparation_seconds": round(
                preparation_time, 4
            ),
            "median_seconds": round(
                median_time, 4
            ),
            "mean_seconds": round(
                mean_time, 4
            ),
            "std_seconds": round(
                std_time, 4
            ),
            "min_seconds": round(
                min(times), 4
            ),
            "max_seconds": round(
                max(times), 4
            ),
            "rows_per_second": round(
                throughput, 2
            ),
            "run_times_seconds": [
                round(t, 4)
                for t in times
            ]
        }

    finally:

        # Release the previous cached configuration
        # before benchmarking the next one.

        prepared.unpersist(
            blocking=True
        )


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

def main():

    if not DATASET.is_dir():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {DATASET}"
        )

    print("\n=== ENVIRONMENT ===")
    print("Python:", sys.executable)

    spark = create_spark()
    spark.sparkContext.setLogLevel("ERROR")

    try:

        print("\n=== BENCHMARK CONFIGURATION ===")
        print("Spark version:", spark.version)
        print("Execution mode:", MASTER)
        print("Driver memory:", DRIVER_MEMORY)
        print("Partition counts:", PARTITION_COUNTS)
        print("Timed repetitions:", REPETITIONS)
        print("Warm-up runs:", WARMUP_RUNS)
        print("Caching: MEMORY_AND_DISK")
        print("Adaptive Query Execution: disabled")

        # --------------------------------------------
        # LOAD SOURCE DATA
        # --------------------------------------------

        source = (
            spark.read.parquet(str(DATASET))
            .select(
                "AIRLINE",
                "IS_DELAYED"
            )
        )

        total_records = source.count()

        print(
            "\nTotal flights:",
            f"{total_records:,}"
        )

        results = []

        # --------------------------------------------
        # EXECUTE EXPERIMENTS
        # --------------------------------------------

        for partitions in PARTITION_COUNTS:

            result = benchmark(
                spark,
                source,
                partitions,
                total_records
            )

            results.append(result)

        # --------------------------------------------
        # COMPARE RESULTS
        # --------------------------------------------

        baseline = next(
            row["median_seconds"]
            for row in results
            if row["partitions"] == 2
        )

        for row in results:

            row["speedup_vs_2_partitions"] = round(
                baseline / row["median_seconds"],
                3
            )

        best = min(
            results,
            key=lambda row: row["median_seconds"]
        )

        # ====================================================
        # 9. SAVE CSV
        # ====================================================

        results_df = pd.DataFrame(results)

        csv_columns = [
            "partitions",
            "records",
            "airline_groups",
            "cache_preparation_seconds",
            "median_seconds",
            "mean_seconds",
            "std_seconds",
            "min_seconds",
            "max_seconds",
            "rows_per_second",
            "speedup_vs_2_partitions"
        ]

        output_csv = (
            METRICS / "07b_partition_comparison.csv"
        )

        results_df[csv_columns].to_csv(
            output_csv,
            index=False
        )

        # ====================================================
        # 10. SAVE JSON
        # ====================================================

        summary = {
            "experiment": (
                "Local Spark partition performance comparison"
            ),

            "dataset": (
                "flights_clean.parquet"
            ),

            "spark_version": spark.version,

            "execution_mode": MASTER,
            "driver_memory": DRIVER_MEMORY,

            "records": total_records,

            "partition_counts": PARTITION_COUNTS,

            "warmup_runs": WARMUP_RUNS,
            "timed_repetitions": REPETITIONS,

            "storage_level": "MEMORY_AND_DISK",

            "adaptive_query_execution": False,

            "operation": (
                "Group flights by airline; "
                "count flights, sum delays, "
                "calculate average delay rate"
            ),

            "timing_scope": (
                "Cached Spark aggregation only. "
                "Excludes Parquet reading, repartitioning "
                "and cache preparation."
            ),

            "best_measured_partition_count": (
                best["partitions"]
            ),

            "results": results,

            "limitations": [
                "Single-machine Spark local mode",
                "Only two local worker threads",
                "One aggregation workload",
                "Cached input rather than end-to-end processing",
                "Short timings are sensitive to runtime variation",
                "Results do not establish multi-machine scalability"
            ]
        }

        output_json = (
            METRICS / "07b_partition_comparison.json"
        )

        output_json.write_text(
            json.dumps(
                summary,
                indent=2
            ),
            encoding="utf-8"
        )

        # ====================================================
        # 11. GENERATE CHART
        # ====================================================

        x = [
            row["partitions"]
            for row in results
        ]

        y = [
            row["median_seconds"]
            for row in results
        ]

        yerr = [
            [
                row["median_seconds"]
                - row["min_seconds"]
                for row in results
            ],
            [
                row["max_seconds"]
                - row["median_seconds"]
                for row in results
            ]
        ]

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=5
        )

        plt.xticks(x)

        plt.xlabel(
            "Spark Partition Count"
        )

        plt.ylabel(
            "Cached Aggregation Time (Seconds)"
        )

        plt.title(
            "Apache Spark Partition Performance\n"
            f"{total_records:,} Flights - {MASTER}"
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        output_chart = (
            FIGURES / "07b_partition_comparison.png"
        )

        plt.savefig(
            output_chart,
            dpi=200
        )

        plt.close()

        # ====================================================
        # 12. RESULTS
        # ====================================================

        print("\n==========================================")
        print("PARTITION BENCHMARK COMPLETED")
        print("==========================================")

        print(
            results_df[csv_columns].to_string(
                index=False
            )
        )

        print(
            "\nFastest measured configuration:",
            best["partitions"],
            "partitions"
        )

        print(
            "Median:",
            f"{best['median_seconds']:.4f} s"
        )

        print("\nGenerated files:")
        print(output_csv)
        print(output_json)
        print(output_chart)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
