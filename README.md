# Flight Delay Analytics and Prediction

Big Data Analytics course project
Sakarya University, 2026

Author: Altun Karimli

## Project Overview

Analysis of 2015 US domestic flight delays using
Apache Spark and machine learning.

## Technologies

- Python
- Apache Spark 4.1.1
- PySpark
- Spark MLlib
- Apache Parquet
- Pandas
- Matplotlib

## Dataset

2015 Flight Delays and Cancellations

Source:
https://www.kaggle.com/datasets/usdot/flight-delays

Original records: 5,819,079
Cleaned records: 5,714,008

## Project Structure

- 01-03: Dataset inspection and Spark setup
- 04-04b: Preprocessing and integration
- 05-05b: Exploratory analysis
- 06-06c: Machine learning experiments
- 07-07b: Performance benchmarks
- results/figures: Generated visualizations
- results/metrics: Experimental results

## Machine Learning

Algorithms:
- Logistic Regression
- Random Forest
- Gradient-Boosted Trees

Best model: Random Forest

F1-score: 0.3309
ROC-AUC: 0.6035

## Execution Environment

Apache Spark local mode on Windows.

The performance results measure cached aggregation
operations on one computer, not a distributed cluster.
