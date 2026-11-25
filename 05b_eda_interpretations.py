
"""
SAU Big Data Analytics
05b - EDA Verification and Interpretations

Recalculates airport EDA using normalized airport identifiers
from flights_enriched.parquet, then generates a report covering
all five analytical scenarios.

Run:
    python 05b_eda_interpretations.py
"""

import os
import sys
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT
# ============================================================

JAVA_HOME = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)
HADOOP_HOME = r"C:\hadoop"

for variable, directory in [
    ("JAVA_HOME", JAVA_HOME),
    ("HADOOP_HOME", HADOOP_HOME),
]:
    if Path(directory).is_dir():
        os.environ[variable] = directory
        os.environ["PATH"] = (
            str(Path(directory) / "bin")
            + os.pathsep
            + os.environ.get("PATH", "")
        )

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ============================================================
# 2. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

ENRICHED = (
    ROOT / "data/processed/flights_enriched.parquet"
)

METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"

REPORT = ROOT / "results/05_eda_interpretations.md"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def load(filename):
    path = METRICS / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing EDA results: {path}"
        )

    return pd.read_csv(path)


def difference(a, b):
    return f"{a - b:.2f} percentage points"


# ============================================================
# 3. CORRECT AIRPORT EDA USING ENRICHED DATA
# ============================================================

def regenerate_airport_eda():

    if not ENRICHED.is_dir():
        raise FileNotFoundError(
            "Enriched dataset not found. "
            "Run 04b_dataset_integration.py first."
        )

    spark = (
        SparkSession.builder
        .appName("SAU BDA - Corrected Airport EDA")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df = spark.read.parquet(str(ENRICHED))

        required = {
            "ORIGIN_AIRPORT_CODE",
            "IS_DELAYED",
            "ARRIVAL_DELAY",
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"Missing required columns: {missing}"
            )

        print("\n=== CORRECTED AIRPORT EDA ===")

        # Group flights by normalized origin airport code.
        # October numeric IDs have already been mapped by 04b.

        grouped = (
            df.groupBy("ORIGIN_AIRPORT_CODE")
            .agg(
                F.count("*").alias("total_flights"),

                F.sum(
                    F.col("IS_DELAYED")
                ).alias("delayed_flights"),

                F.round(
                    F.avg("IS_DELAYED") * 100,
                    2
                ).alias("delay_rate"),

                F.round(
                    F.avg("ARRIVAL_DELAY"),
                    2
                ).alias("avg_arrival_delay")
            )
        )

        # Exclude small groups from the airport ranking.
        # Keep the ten highest delay rates.

        top_airports = (
            grouped
            .filter(F.col("total_flights") >= 10000)
            .orderBy(
                F.desc("delay_rate"),
                F.desc("total_flights")
            )
            .limit(10)
            .collect()
        )

        if len(top_airports) == 0:
            raise RuntimeError(
                "No airports satisfied the flight threshold."
            )

        # Preserve the existing CSV column name expected
        # by our interpretation generator.

        airport_rows = []

        for row in top_airports:
            airport_rows.append({
                "ORIGIN_AIRPORT": row[
                    "ORIGIN_AIRPORT_CODE"
                ],
                "total_flights": int(
                    row["total_flights"]
                ),
                "delayed_flights": int(
                    row["delayed_flights"]
                ),
                "delay_rate": float(
                    row["delay_rate"]
                ),
                "avg_arrival_delay": float(
                    row["avg_arrival_delay"]
                )
            })

        airport_results = pd.DataFrame(airport_rows)

        output_csv = (
            METRICS / "02_airport_delays.csv"
        )

        airport_results.to_csv(
            output_csv,
            index=False
        )

        # ====================================================
        # 4. REGENERATE AIRPORT CHART
        # ====================================================

        chart_data = airport_results.sort_values(
            "delay_rate",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            chart_data["ORIGIN_AIRPORT"],
            chart_data["delay_rate"]
        )

        plt.xlabel("Arrival Delay Rate (%)")

        plt.ylabel("Origin Airport")

        plt.title(
            "Top 10 Origin Airports by Arrival Delay Rate\n"
            "(Normalized Airport Codes, >=10,000 Flights)"
        )

        plt.grid(
            axis="x",
            alpha=0.3
        )

        plt.tight_layout()

        output_png = (
            FIGURES / "02_airport_delays.png"
        )

        plt.savefig(
            output_png,
            dpi=200
        )

        plt.close()

        # ====================================================
        # 5. VERIFY DATASET TOTALS
        # ====================================================

        verification = df.agg(
            F.count("*").alias("flights"),
            F.sum("IS_DELAYED").alias("delayed")
        ).first()

        total = int(verification["flights"])
        delayed = int(verification["delayed"])

        print("Records:", total)
        print("Delayed flights:", delayed)

        print("\nCorrected airport ranking:")
        print(
            airport_results.to_string(index=False)
        )

        print("\nUpdated:", output_csv)
        print("Updated:", output_png)

        return total, delayed

    finally:
        spark.stop()


# ============================================================
# 6. GENERATE FIVE SCENARIO INTERPRETATIONS
# ============================================================

def generate_report(expected_total, expected_delayed):

    airlines = load("01_airline_delays.csv")
    airports = load("02_airport_delays.csv")
    months = load("03_monthly_delays.csv")

    hours = load(
        "04_departure_hour_delays.csv"
    )

    distances = load(
        "05_distance_delays.csv"
    )

    for name, data in [
        ("airlines", airlines),
        ("airports", airports),
        ("months", months),
        ("hours", hours),
        ("distances", distances)
    ]:
        if data.empty:
            raise RuntimeError(
                f"Empty EDA results: {name}"
            )

    # Monthly, hourly and distance categories are exhaustive.

    n = int(months["total_flights"].sum())

    delayed = int(
        months["delayed_flights"].sum()
    )

    if n != expected_total:
        raise RuntimeError(
            "EDA totals do not match the enriched dataset."
        )

    if delayed != expected_delayed:
        raise RuntimeError(
            "Delayed-flight totals do not match."
        )

    for name, data in [
        ("hours", hours),
        ("distances", distances)
    ]:
        if (
            int(data["total_flights"].sum()) != n
            or int(data["delayed_flights"].sum()) != delayed
        ):
            raise RuntimeError(
                f"{name} totals are inconsistent."
            )

    overall_rate = 100 * delayed / n

    # Highest-ranked scenarios

    top_airline = airlines.loc[
        airlines["delay_rate"].idxmax()
    ]

    top_airport = airports.loc[
        airports["delay_rate"].idxmax()
    ]

    max_month = months.loc[
        months["delay_rate"].idxmax()
    ]

    min_month = months.loc[
        months["delay_rate"].idxmin()
    ]

    # Exclude very small hourly samples.

    regular_hours = hours[
        hours["total_flights"] >= 10000
    ]

    max_hour = regular_hours.loc[
        regular_hours["delay_rate"].idxmax()
    ]

    min_hour = regular_hours.loc[
        regular_hours["delay_rate"].idxmin()
    ]

    max_distance = distances.loc[
        distances["delay_rate"].idxmax()
    ]

    min_distance = distances.loc[
        distances["delay_rate"].idxmin()
    ]

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    # ========================================================
    # 7. WRITE REPORT
    # ========================================================

    report = f"""# Exploratory Data Analysis — Five Research Scenarios

## Dataset Overview

- Dataset: US DOT Flight Delays (2015)
- Cleaned flights: **{n:,}**
- Delayed flights: **{delayed:,}**
- Overall delay rate: **{overall_rate:.2f}%**
- Delay definition: Arrival delay of at least 15 minutes.
- Main processing technology: Apache Spark.

The five scenarios investigate observational relationships.
They do not establish causality.

## Scenario 1 — Airline Delay Analysis

**Research question:** Which airlines experience the highest
arrival delay rates?

**Method:** Spark grouped completed flights by `AIRLINE`.
Counts, delayed-flight totals and delay rates were calculated.
The ten highest-ranking airlines were selected.

**Results:** Airline **{top_airline['AIRLINE']}** had the highest
displayed delay rate at **{top_airline['delay_rate']:.2f}%** across
**{int(top_airline['total_flights']):,}** completed flights.

This is **{difference(top_airline['delay_rate'], overall_rate)}**
above the overall dataset rate.

**Interpretation:** Airlines have different observed delay rates.
Differences may reflect route networks, schedules, operational
practices and airport conditions. The data cannot identify the
specific cause of the variation.

**Limitations:** Only one year was analyzed. Airline comparisons
were not adjusted for route, weather or airport characteristics.
Cancelled and diverted flights were excluded.

**Visualization:** `results/figures/01_airline_delays.png`

---

## Scenario 2 — Origin Airport Delay Analysis

**Research question:** Which frequently used origin airports
have the highest arrival delay rates?

**Method:** Spark grouped flights by normalized
`ORIGIN_AIRPORT_CODE`. The airport identifiers were corrected
during dataset integration: October's numeric airport IDs were
mapped to three-letter codes using verified BTS reference
tables. Airports with fewer than 10,000 completed flights
were excluded from the ranking.

**Results:** Airport **{top_airport['ORIGIN_AIRPORT']}** had the
highest observed delay rate at **{top_airport['delay_rate']:.2f}%**
across **{int(top_airport['total_flights']):,}** completed flights.

**Interpretation:** Flights originating from different airports
have different probabilities of arriving late. Congestion,
weather, connecting-flight schedules and airport operations
are possible contributing factors.

**Data-quality improvement:** October 2015 used numeric airport
identifiers, whereas other months used airport codes.
The integration step normalized most of these identifiers,
allowing October flights to be grouped with the corresponding
airports across the rest of the year.

**Limitations:** Some October identifiers remain unresolved.
They were preserved rather than assigned an uncertain airport
code. The outcome measures arrival delays, not departure delays.
Airports below the flight-count threshold are excluded.

**Visualization:** `results/figures/02_airport_delays.png`

---

## Scenario 3 — Monthly Delay Analysis

**Research question:** How does arrival delay frequency vary
across the months of 2015?

**Method:** Spark grouped flights by `MONTH` and calculated
total flights, delayed flights and delay rates.

**Results:** The highest delay rate occurred in
**{month_names[int(max_month['MONTH'])]}**, at
**{max_month['delay_rate']:.2f}%** across
**{int(max_month['total_flights']):,}** completed flights.

The lowest rate occurred in
**{month_names[int(min_month['MONTH'])]}**, at
**{min_month['delay_rate']:.2f}%**.

Difference: **{difference(max_month['delay_rate'], min_month['delay_rate'])}**.

**Interpretation:** Flight delays exhibit variation throughout
the calendar year. Seasonal weather conditions, travel demand
and operational factors may contribute.

**Limitations:** The analysis covers only 2015, so it cannot
demonstrate that the same seasonal pattern persists across years.

**Visualization:** `results/figures/03_monthly_delays.png`

---

## Scenario 4 — Scheduled Departure Hour Analysis

**Research question:** How does scheduled departure time relate
to observed arrival delay frequency?

**Method:** The `DEP_HOUR` feature was derived during
preprocessing. Spark grouped flights by scheduled departure hour
and calculated arrival delay rates.

**Results:** Among hours containing at least 10,000 flights,
**{int(max_hour['DEP_HOUR']):02d}:00** had the highest delay rate:
**{max_hour['delay_rate']:.2f}%**, across
**{int(max_hour['total_flights']):,}** completed flights.

The lowest rate occurred at
**{int(min_hour['DEP_HOUR']):02d}:00**, with
**{min_hour['delay_rate']:.2f}%**.

Difference: **{difference(max_hour['delay_rate'], min_hour['delay_rate'])}**.

**Interpretation:** Flight delays may accumulate throughout
the operating day. However, airport and airline scheduling
differences may also contribute to the observed pattern.

**Limitations:** Departure hours represent local airport times.
Multiple time zones are combined. Hours with very small samples
were excluded from the main comparison.

**Visualization:** `results/figures/04_departure_hour_delays.png`

---

## Scenario 5 — Flight Distance Analysis

**Research question:** How does flight distance relate to
arrival delay rates?

**Method:** Flight distances were divided into mile-based
categories using Spark conditional expressions.
Counts and delay rates were calculated for each category.

**Results:** The **{max_distance['DISTANCE_GROUP']} mile**
category had the highest delay rate at
**{max_distance['delay_rate']:.2f}%**, across
**{int(max_distance['total_flights']):,}** completed flights.

The lowest rate occurred in the
**{min_distance['DISTANCE_GROUP']} mile** category,
at **{min_distance['delay_rate']:.2f}%**.

Difference: **{difference(max_distance['delay_rate'], min_distance['delay_rate'])}**.

**Interpretation:** Flight distance is associated with some
variation in arrival delay frequency, but distance alone
cannot explain delays. Route characteristics, traffic,
aircraft operations and weather may also matter.

**Limitations:** Distance categories conceal variation
within each band. No causal analysis was performed.

**Visualization:** `results/figures/05_distance_delays.png`

---

## Overall Interpretation

The five scenarios show that airline, origin airport, month,
scheduled departure hour and flight distance are associated
with differences in arrival delay frequency.

The airport analysis now uses normalized airport identifiers,
correcting the October 2015 inconsistency discovered during
dataset integration.

The EDA findings support our selection of scheduling and
airline-related features for subsequent machine-learning
experiments.

These associations do not establish causality or guarantee
predictive performance on future flights.

## Verification

- Cleaned/enriched flights: {n:,}
- Delayed flights: {delayed:,}
- Overall delay rate: {overall_rate:.2f}%
- Monthly, hourly and distance aggregate totals verified.
- Airport rankings recalculated from enriched Parquet.
- October airport identifiers normalized where possible.
- Original source records preserved.
"""

    REPORT.write_text(
        report,
        encoding="utf-8"
    )

    print("\n=== EDA INTERPRETATIONS COMPLETED ===")
    print("Report:", REPORT)

    print(
        f"Verified total = {n:,} flights, "
        f"delayed = {delayed:,} "
        f"({overall_rate:.2f}%)"
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    total, delayed = regenerate_airport_eda()

    generate_report(total, delayed)

    print("\nAll five EDA scenarios are ready for the report.")


if __name__ == "__main__":
    main()
