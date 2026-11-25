import os

os.environ["JAVA_HOME"] = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)

from pathlib import Path
import kagglehub
from pyspark.sql import SparkSession

# Initialize Spark
spark = (
    SparkSession.builder
    .appName("SAU Big Data Analytics")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Locate downloaded dataset
path = Path(kagglehub.dataset_download("usdot/flight-delays"))
csv_file = path / "flights.csv"

# Load data with Spark
df = spark.read.csv(
    str(csv_file),
    header=True,
    inferSchema=True
)

# Basic dataset information
print("Total records:", df.count())
print("Total columns:", len(df.columns))

print("\nDataset schema:")
df.printSchema()

print("\nSample flights:")
df.select(
    "YEAR", "MONTH", "AIRLINE",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "ARRIVAL_DELAY"
).show(5)

spark.stop()
