import os
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

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ============================================================
# 3. INITIALIZE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("Flight Delay - EDA")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data/processed/flights_clean.parquet"
FIGURES_DIR = BASE_DIR / "results/figures"
METRICS_DIR = BASE_DIR / "results/metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. LOAD CLEANED DATASET
# ============================================================

df = spark.read.parquet(str(DATA_PATH))

print("\n=== DATASET LOADED ===")
print("Total records:", df.count())
print("Total columns:", len(df.columns))

# ============================================================
# 5. REUSABLE ANALYSIS FUNCTION
# ============================================================

def analyze(dataframe, group_column):
    """
    Group flights and calculate:
    - Total flights
    - Delayed flights
    - Delay percentage
    - Average arrival delay
    """

    return dataframe.groupBy(group_column).agg(
        F.count("*").alias("total_flights"),

        F.sum("IS_DELAYED").alias("delayed_flights"),

        F.round(
            F.avg("IS_DELAYED") * 100, 2
        ).alias("delay_rate"),

        F.round(
            F.avg("ARRIVAL_DELAY"), 2
        ).alias("average_delay")
    )


# ============================================================
# 6. REUSABLE EXPORT FUNCTION
# ============================================================

def save_analysis(data, filename, x, chart_type="bar"):

    # Save statistical results
    csv_path = METRICS_DIR / f"{filename}.csv"
    data.to_csv(csv_path, index=False)

    # Generate visualization
    plt.figure(figsize=(11, 6))

    if chart_type == "line":
        plt.plot(
            data[x].astype(str),
            data["delay_rate"],
            marker="o"
        )
    else:
        plt.bar(
            data[x].astype(str),
            data["delay_rate"]
        )

    plt.title(filename.replace("_", " ").title())
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel("Delayed Flights (%)")

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save chart
    figure_path = FIGURES_DIR / f"{filename}.png"

    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"\nSaved: {filename}")
    print(data.to_string(index=False))


# ============================================================
# SCENARIO 1 — AIRLINES
# ============================================================

print("\n=== SCENARIO 1: AIRLINE DELAYS ===")

airlines = (
    analyze(df, "AIRLINE")
    .orderBy(F.desc("delay_rate"))
    .limit(10)
    .toPandas()
)

save_analysis(
    airlines,
    "01_airline_delays",
    "AIRLINE"
)


# ============================================================
# SCENARIO 2 — AIRPORTS
# ============================================================

print("\n=== SCENARIO 2: AIRPORT DELAYS ===")

airports = (
    analyze(df, "ORIGIN_AIRPORT")
    .filter(F.col("total_flights") >= 10000)
    .orderBy(F.desc("delay_rate"))
    .limit(10)
    .toPandas()
)

# Minimum 10,000 flights avoids ranking tiny airports
# with unreliable delay percentages.

save_analysis(
    airports,
    "02_airport_delays",
    "ORIGIN_AIRPORT"
)


# ============================================================
# SCENARIO 3 — MONTHLY DELAYS
# ============================================================

print("\n=== SCENARIO 3: MONTHLY DELAYS ===")

months = (
    analyze(df, "MONTH")
    .orderBy("MONTH")
    .toPandas()
)

save_analysis(
    months,
    "03_monthly_delays",
    "MONTH",
    chart_type="line"
)


# ============================================================
# SCENARIO 4 — DEPARTURE HOUR
# ============================================================

print("\n=== SCENARIO 4: DEPARTURE TIME ===")

hours = (
    analyze(df, "DEP_HOUR")
    .orderBy("DEP_HOUR")
    .toPandas()
)

save_analysis(
    hours,
    "04_departure_hour_delays",
    "DEP_HOUR",
    chart_type="line"
)


# ============================================================
# SCENARIO 5 — FLIGHT DISTANCE
# ============================================================

print("\n=== SCENARIO 5: FLIGHT DISTANCE ===")

# Divide flights into distance categories (miles)

df_distance = df.withColumn(
    "DISTANCE_GROUP",

    F.when(
        F.col("DISTANCE") < 500, "0-499"
    ).when(
        F.col("DISTANCE") < 1000, "500-999"
    ).when(
        F.col("DISTANCE") < 1500, "1000-1499"
    ).when(
        F.col("DISTANCE") < 2000, "1500-1999"
    ).otherwise("2000+")
)

distance = analyze(
    df_distance,
    "DISTANCE_GROUP"
)

# Maintain logical category order
distance = distance.withColumn(
    "sort_order",

    F.when(F.col("DISTANCE_GROUP") == "0-499", 1)
    .when(F.col("DISTANCE_GROUP") == "500-999", 2)
    .when(F.col("DISTANCE_GROUP") == "1000-1499", 3)
    .when(F.col("DISTANCE_GROUP") == "1500-1999", 4)
    .otherwise(5)
)

distance = (
    distance
    .orderBy("sort_order")
    .drop("sort_order")
    .toPandas()
)

save_analysis(
    distance,
    "05_distance_delays",
    "DISTANCE_GROUP"
)


# ============================================================
# FINISH
# ============================================================

print("\n====================================")
print("ALL 5 EDA SCENARIOS COMPLETED")
print("====================================")

print("\nGenerated:")
print("5 statistical CSV files")
print("5 visualization PNG files")

spark.stop()