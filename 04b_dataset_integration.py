
"""
SAU Big Data Analytics - Dataset Integration

Correct October 2015 numeric airport identifiers and enrich
the cleaned flight dataset with airline and airport information.

Original flight records and airport identifiers are preserved.
Unknown airport IDs are never guessed or removed.

Run:
    python 04b_dataset_integration.py
"""

import os
import sys
import csv
import json
import urllib.request
from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
HADOOP_HOME = r"C:\hadoop"

for variable, folder in [
    ("JAVA_HOME", JAVA_HOME),
    ("HADOOP_HOME", HADOOP_HOME)
]:
    if not Path(folder).is_dir():
        raise FileNotFoundError(
            f"Missing {variable}: {folder}"
        )

    os.environ[variable] = folder

    os.environ["PATH"] = (
        str(Path(folder) / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

# Use the same Python executable for Spark workers.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import kagglehub
import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 3. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CLEAN = ROOT / "data/processed/flights_clean.parquet"
ENRICHED = ROOT / "data/processed/flights_enriched.parquet"

REFERENCE_DIR = ROOT / "data/reference"
METRICS_DIR = ROOT / "results/metrics"

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LOOKUP_URLS = {
    "L_AIRPORT_ID.csv": (
        "https://raw.githubusercontent.com/ActianCorp/"
        "vectormpp-cloud-prep/master/"
        "airline-sample-data/data/l_airport_id.csv"
    ),
    "L_AIRPORT.csv": (
        "https://raw.githubusercontent.com/ActianCorp/"
        "vectormpp-cloud-prep/master/"
        "airline-sample-data/data/l_airport.csv"
    )
}


# ============================================================
# 4. DOWNLOAD REFERENCE TABLES
# ============================================================

def get_reference(filename):

    destination = REFERENCE_DIR / filename

    # Use previously downloaded references.
    if (
        destination.is_file()
        and destination.stat().st_size > 1000
    ):
        return destination

    print(
        f"Downloading reference: {filename}",
        flush=True
    )

    request = urllib.request.Request(
        LOOKUP_URLS[filename],
        headers={"User-Agent": "SAU-BDA/1.0"}
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=45
        ) as response:

            content = response.read()

    except Exception as exc:
        raise RuntimeError(
            f"Cannot download {filename}.\n"
            f"URL: {LOOKUP_URLS[filename]}"
        ) from exc

    # Validate CSV header.
    if not content.lstrip(b"\xef\xbb\xbf").startswith(
        b"Code,Description"
    ):
        raise ValueError(
            f"Invalid reference file: {filename}"
        )

    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)

    return destination


# ============================================================
# 5. READ REFERENCE TABLES
# ============================================================

def read_reference(path):

    records = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not {"Code", "Description"}.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"Invalid reference columns: {path}"
            )

        for row in reader:

            code = row["Code"].strip().upper()
            description = row["Description"].strip()

            if code and description:
                records.append(
                    (code, description)
                )

    return records


# ============================================================
# 6. BUILD VERIFIED AIRPORT CROSSWALK
# ============================================================

def build_crosswalk(id_path, code_path, valid_codes):

    id_by_description = defaultdict(set)
    code_by_description = defaultdict(set)

    # Numeric airport IDs
    for identifier, description in read_reference(id_path):

        if identifier.isdigit() and len(identifier) == 5:
            id_by_description[description].add(identifier)

    # Three-letter airport codes
    for code, description in read_reference(code_path):

        if code in valid_codes:
            code_by_description[description].add(code)

    candidates = defaultdict(set)

    # Match using identical airport descriptions.
    for description, identifiers in id_by_description.items():

        codes = code_by_description.get(
            description, set()
        )

        # Accept only unambiguous matches.
        if len(identifiers) == 1 and len(codes) == 1:

            identifier = next(iter(identifiers))
            airport_code = next(iter(codes))

            candidates[identifier].add(airport_code)

    mapping = {
        identifier: next(iter(codes))
        for identifier, codes in candidates.items()
        if len(codes) == 1
    }

    # Validate known airport identifiers.
    expected = {
        "10397": "ATL",
        "13930": "ORD"
    }

    for identifier, airport_code in expected.items():

        actual = mapping.get(identifier)

        if actual != airport_code:
            raise RuntimeError(
                f"Mapping validation failed: "
                f"{identifier} -> {actual}, "
                f"expected {airport_code}"
            )

    return mapping


# ============================================================
# 7. CREATE SPARK SESSION
# ============================================================

def main():

    if not CLEAN.is_dir():
        raise FileNotFoundError(
            f"Missing cleaned dataset: {CLEAN}"
        )

    print("\n=== ENVIRONMENT ===")
    print("Python:", sys.executable)
    print("PySpark:", pyspark.__file__)

    id_path = get_reference("L_AIRPORT_ID.csv")
    code_path = get_reference("L_AIRPORT.csv")

    spark = (
        SparkSession.builder
        .appName("SAU BDA - Airport Integration")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.sql.execution.pyspark.udf.faulthandler.enabled",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        # ====================================================
        # 8. LOAD DATASETS
        # ====================================================

        flights = spark.read.parquet(
            str(CLEAN)
        )

        original_count = flights.count()

        dataset_path = Path(
            kagglehub.dataset_download(
                "usdot/flight-delays"
            )
        )

        airlines = (
            spark.read.csv(
                str(dataset_path / "airlines.csv"),
                header=True
            )
            .select(
                F.col("IATA_CODE").alias("AIRLINE_KEY"),
                F.col("AIRLINE").alias("AIRLINE_NAME")
            )
            .dropDuplicates(["AIRLINE_KEY"])
        )

        airports = (
            spark.read.csv(
                str(dataset_path / "airports.csv"),
                header=True
            )
            .select(
                F.col("IATA_CODE").alias("AIRPORT_KEY"),
                "AIRPORT",
                "CITY",
                "STATE",
                "COUNTRY"
            )
            .dropDuplicates(["AIRPORT_KEY"])
        )

        print("\n=== SOURCE DATASET ===")
        print("Flight records:", original_count)

        # ====================================================
        # 9. BUILD AIRPORT MAPPING
        # ====================================================

        valid_codes = {
            row["AIRPORT_KEY"]
            for row in airports.select(
                "AIRPORT_KEY"
            ).collect()
            if row["AIRPORT_KEY"]
        }

        mapping = build_crosswalk(
            id_path,
            code_path,
            valid_codes
        )

        print(
            "Verified airport mappings:",
            len(mapping),
            flush=True
        )

        if len(mapping) < 100:
            raise RuntimeError(
                "Insufficient verified airport mappings."
            )

        # ====================================================
        # 10. SAVE CROSSWALK AS CSV
        # ====================================================

        # IMPORTANT FIX:
        # Instead of spark.createDataFrame(mapping_rows),
        # we save the mapping to CSV and load it natively.
        #
        # This avoids the Python worker previously involved
        # in constructing a Spark DataFrame from Python data.

        mapping_rows = sorted(mapping.items())

        crosswalk_path = (
            REFERENCE_DIR / "airport_crosswalk.csv"
        )

        with crosswalk_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "NUMERIC_ID",
                "MAPPED_CODE"
            ])

            writer.writerows(mapping_rows)

        print(
            "Crosswalk CSV saved:",
            crosswalk_path,
            flush=True
        )

        # ====================================================
        # 11. LOAD CROSSWALK USING NATIVE SPARK CSV
        # ====================================================

        crosswalk = (
            spark.read
            .option("header", True)
            .schema(
                "NUMERIC_ID STRING, MAPPED_CODE STRING"
            )
            .csv(str(crosswalk_path))
        )

        crosswalk_count = crosswalk.count()

        print(
            "Crosswalk loaded:",
            crosswalk_count,
            flush=True
        )

        if crosswalk_count != len(mapping):
            raise RuntimeError(
                "Crosswalk record count mismatch."
            )

        # Separate origin and destination lookup columns.

        origin_mapping = crosswalk.select(
            F.col("NUMERIC_ID").alias("ORIGIN_ID_KEY"),
            F.col("MAPPED_CODE").alias("ORIGIN_MAPPED")
        )

        destination_mapping = crosswalk.select(
            F.col("NUMERIC_ID").alias("DEST_ID_KEY"),
            F.col("MAPPED_CODE").alias("DEST_MAPPED")
        )

        # ====================================================
        # 12. PREPARE AIRPORT METADATA
        # ====================================================

        origins = airports.select(
            F.col("AIRPORT_KEY").alias("ORIGIN_KEY"),
            F.col("AIRPORT").alias("ORIGIN_AIRPORT_NAME"),
            F.col("CITY").alias("ORIGIN_CITY"),
            F.col("STATE").alias("ORIGIN_STATE"),
            F.col("COUNTRY").alias("ORIGIN_COUNTRY")
        )

        destinations = airports.select(
            F.col("AIRPORT_KEY").alias("DESTINATION_KEY"),
            F.col("AIRPORT").alias(
                "DESTINATION_AIRPORT_NAME"
            ),
            F.col("CITY").alias("DESTINATION_CITY"),
            F.col("STATE").alias("DESTINATION_STATE"),
            F.col("COUNTRY").alias("DESTINATION_COUNTRY")
        )

        # ====================================================
        # 13. DATASET INTEGRATION
        # ====================================================

        print(
            "\nIntegrating airline and airport metadata...",
            flush=True
        )

        enriched = (
            flights

            # Airline names
            .join(
                F.broadcast(airlines),
                F.col("AIRLINE") == F.col("AIRLINE_KEY"),
                "left"
            )
            .drop("AIRLINE_KEY")

            # Numeric origin ID mapping
            .join(
                F.broadcast(origin_mapping),
                F.col("ORIGIN_AIRPORT")
                == F.col("ORIGIN_ID_KEY"),
                "left"
            )
            .drop("ORIGIN_ID_KEY")

            # Numeric destination ID mapping
            .join(
                F.broadcast(destination_mapping),
                F.col("DESTINATION_AIRPORT")
                == F.col("DEST_ID_KEY"),
                "left"
            )
            .drop("DEST_ID_KEY")

            # Normalize origin airport identifier
            .withColumn(
                "ORIGIN_AIRPORT_CODE",
                F.coalesce(
                    F.col("ORIGIN_MAPPED"),
                    F.col("ORIGIN_AIRPORT")
                )
            )

            # Normalize destination airport identifier
            .withColumn(
                "DESTINATION_AIRPORT_CODE",
                F.coalesce(
                    F.col("DEST_MAPPED"),
                    F.col("DESTINATION_AIRPORT")
                )
            )

            # Track where origin code came from
            .withColumn(
                "ORIGIN_CODE_SOURCE",
                F.when(
                    F.col("ORIGIN_MAPPED").isNotNull(),
                    "BTS_CROSSWALK"
                )
                .when(
                    F.col("ORIGIN_AIRPORT").rlike(
                        "^[0-9]{5}$"
                    ),
                    "UNMAPPED_ID"
                )
                .otherwise("ORIGINAL_CODE")
            )

            # Track where destination code came from
            .withColumn(
                "DESTINATION_CODE_SOURCE",
                F.when(
                    F.col("DEST_MAPPED").isNotNull(),
                    "BTS_CROSSWALK"
                )
                .when(
                    F.col("DESTINATION_AIRPORT").rlike(
                        "^[0-9]{5}$"
                    ),
                    "UNMAPPED_ID"
                )
                .otherwise("ORIGINAL_CODE")
            )

            .drop(
                "ORIGIN_MAPPED",
                "DEST_MAPPED"
            )

            # Origin airport metadata
            .join(
                F.broadcast(origins),
                F.col("ORIGIN_AIRPORT_CODE")
                == F.col("ORIGIN_KEY"),
                "left"
            )
            .drop("ORIGIN_KEY")

            # Destination airport metadata
            .join(
                F.broadcast(destinations),
                F.col("DESTINATION_AIRPORT_CODE")
                == F.col("DESTINATION_KEY"),
                "left"
            )
            .drop("DESTINATION_KEY")
        )

        # ====================================================
        # 14. VERIFY MAPPING AND INTEGRATION
        # ====================================================

        print(
            "\nVerifying integration...",
            flush=True
        )

        stats = enriched.agg(

            F.count("*").alias("records"),

            F.sum(
                (F.col("MONTH") == 10).cast("long")
            ).alias("october_flights"),

            F.sum(
                (
                    F.col("ORIGIN_CODE_SOURCE")
                    == "BTS_CROSSWALK"
                ).cast("long")
            ).alias("mapped_origins"),

            F.sum(
                (
                    F.col("DESTINATION_CODE_SOURCE")
                    == "BTS_CROSSWALK"
                ).cast("long")
            ).alias("mapped_destinations"),

            F.sum(
                (
                    F.col("ORIGIN_CODE_SOURCE")
                    == "UNMAPPED_ID"
                ).cast("long")
            ).alias("unmapped_origin_ids"),

            F.sum(
                (
                    F.col("DESTINATION_CODE_SOURCE")
                    == "UNMAPPED_ID"
                ).cast("long")
            ).alias("unmapped_destination_ids"),

            F.sum(
                F.col("AIRLINE_NAME")
                .isNull()
                .cast("long")
            ).alias("unmatched_airlines"),

            F.sum(
                F.col("ORIGIN_AIRPORT_NAME")
                .isNull()
                .cast("long")
            ).alias("unmatched_origins"),

            F.sum(
                F.col("DESTINATION_AIRPORT_NAME")
                .isNull()
                .cast("long")
            ).alias("unmatched_destinations")

        ).first().asDict()

        # Confirm no flight records were lost or duplicated.
        if stats["records"] != original_count:
            raise RuntimeError(
                "Integration changed flight record count."
            )

        october = stats["october_flights"]

        origin_coverage = (
            100 * stats["mapped_origins"] / october
            if october else 0
        )

        destination_coverage = (
            100 * stats["mapped_destinations"] / october
            if october else 0
        )

        print("\n=== OCTOBER MAPPING COVERAGE ===")

        print(
            f"Origin: {origin_coverage:.2f}%"
        )

        print(
            f"Destination: {destination_coverage:.2f}%"
        )

        print(
            "Unmatched origin rows:",
            stats["unmatched_origins"]
        )

        print(
            "Unmatched destination rows:",
            stats["unmatched_destinations"]
        )

        # Do not overwrite an existing enriched dataset if the
        # new mapping provides insufficient coverage.

        if min(
            origin_coverage,
            destination_coverage
        ) < 90:

            raise RuntimeError(
                "Airport mapping coverage below 90%. "
                "Existing enriched dataset was not overwritten."
            )

        # ====================================================
        # 15. EXPORT PARQUET
        # ====================================================

        ENRICHED.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "\nSaving enriched Parquet...",
            flush=True
        )

        enriched.write.mode("overwrite").parquet(
            str(ENRICHED)
        )

        # ====================================================
        # 16. VERIFY SAVED PARQUET
        # ====================================================

        print(
            "\nVerifying saved Parquet...",
            flush=True
        )

        verified = spark.read.parquet(
            str(ENRICHED)
        )

        saved_count = verified.count()

        if saved_count != original_count:
            raise RuntimeError(
                "Parquet verification failed: "
                "record count mismatch."
            )

        print(
            "Saved records:",
            saved_count
        )

        print(
            "Saved columns:",
            len(verified.columns)
        )

        # ====================================================
        # 17. SAVE SUMMARY
        # ====================================================

        summary = {
            "source_dataset": "US DOT 2015 Flight Delays",

            "integration_method": (
                "BTS exact-description crosswalk "
                "and Spark broadcast left joins"
            ),

            "original_flights": original_count,
            "enriched_flights": saved_count,
            "enriched_columns": len(verified.columns),

            "verified_airport_mappings": len(mapping),
            "october_flights": october,

            "october_origin_coverage_percent": round(
                origin_coverage, 2
            ),

            "october_destination_coverage_percent": round(
                destination_coverage, 2
            ),

            **stats,

            "original_airport_columns_preserved": True,
            "parquet_verified": True
        }

        output_json = (
            METRICS_DIR / "04b_integration_summary.json"
        )

        output_json.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8"
        )

        # ====================================================
        # 18. FINISH
        # ====================================================

        print("\n====================================")
        print("DATASET INTEGRATION COMPLETED")
        print("====================================")

        print(
            json.dumps(summary, indent=2)
        )

        print("\nParquet:", ENRICHED)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
