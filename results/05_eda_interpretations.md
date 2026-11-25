# Exploratory Data Analysis — Five Research Scenarios

## Dataset Overview

- Dataset: US DOT Flight Delays (2015)
- Cleaned flights: **5,714,008**
- Delayed flights: **1,063,439**
- Overall delay rate: **18.61%**
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

**Results:** Airline **NK** had the highest
displayed delay rate at **29.71%** across
**115,193** completed flights.

This is **11.10 percentage points**
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

**Results:** Airport **LGA** had the
highest observed delay rate at **23.39%**
across **103,281** completed flights.

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
**June**, at
**23.48%** across
**492,847** completed flights.

The lowest rate occurred in
**October**, at
**12.44%**.

Difference: **11.04 percentage points**.

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
**20:00** had the highest delay rate:
**26.32%**, across
**253,612** completed flights.

The lowest rate occurred at
**05:00**, with
**7.27%**.

Difference: **19.05 percentage points**.

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

**Results:** The **1000-1499 mile**
category had the highest delay rate at
**19.72%**, across
**845,177** completed flights.

The lowest rate occurred in the
**0-499 mile** category,
at **18.21%**.

Difference: **1.51 percentage points**.

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

- Cleaned/enriched flights: 5,714,008
- Delayed flights: 1,063,439
- Overall delay rate: 18.61%
- Monthly, hourly and distance aggregate totals verified.
- Airport rankings recalculated from enriched Parquet.
- October airport identifiers normalized where possible.
- Original source records preserved.
