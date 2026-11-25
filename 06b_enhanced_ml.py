
"""
SAU Big Data Analytics - Enhanced Machine Learning

Uses the enriched flight dataset with corrected October airport IDs.

Models:
    1. Logistic Regression
    2. Random Forest

Chronological split:
    Training:   January - August 2015
    Validation: September - October 2015
    Testing:    November - December 2015

Features:
    Airline, origin airport, destination airport,
    weekday, cyclical month/hour, distance, scheduled duration.

Run:
    python 06b_enhanced_ml.py
"""

import json
import math
import os
import sys
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

# Use the same Python interpreter for Spark workers.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark import StorageLevel

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

from pyspark.ml.functions import vector_to_array

# ============================================================
# 3. PROJECT PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT / "data/processed/flights_enriched.parquet"
)

METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"

SEED = 42

# Approximately 285,000 flights by default.
SAMPLE_FRACTION = float(
    os.getenv("ML_SAMPLE_FRACTION", "0.05")
)

TOP_AIRPORTS = int(
    os.getenv("ML_TOP_AIRPORTS", "25")
)

FOREST_TREES = int(
    os.getenv("ML_FOREST_TREES", "25")
)

THRESHOLDS = [
    i / 100 for i in range(15, 86, 5)
]

# ============================================================
# 4. CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    actual,
    probabilities,
    threshold
):
    """
    Calculate classification metrics for delayed flights.

    Positive class:
        IS_DELAYED = 1
    """

    actual = np.asarray(
        actual,
        dtype=np.int8
    )

    predicted = (
        np.asarray(probabilities) >= threshold
    ).astype(np.int8)

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    tn = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": (tp + tn) / len(actual),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# 5. SELECT FREQUENT AIRPORTS
# ============================================================

def top_airport_codes(train, field):
    """
    Select the most frequent airports using training data.

    Numeric unresolved identifiers are excluded.
    Only valid three-letter codes can become categories.

    This prevents learning category frequencies from
    validation or test data.
    """

    rows = (
        train
        .filter(
            F.col(field).rlike("^[A-Z]{3}$")
        )
        .groupBy(field)
        .count()
        .orderBy(
            F.desc("count"),
            F.asc(field)
        )
        .limit(TOP_AIRPORTS)
        .collect()
    )

    return [
        row[field]
        for row in rows
    ]


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

def make_features(
    df,
    origins,
    destinations
):
    """
    Create categorical and cyclical features.

    Unknown and less frequent airports become OTHER.
    """

    return (
        df

        .withColumn(
            "ORIGIN_GROUP",
            F.when(
                F.col("ORIGIN_AIRPORT_CODE").isin(*origins),
                F.col("ORIGIN_AIRPORT_CODE")
            ).otherwise(F.lit("OTHER"))
        )

        .withColumn(
            "DESTINATION_GROUP",
            F.when(
                F.col("DESTINATION_AIRPORT_CODE").isin(
                    *destinations
                ),
                F.col("DESTINATION_AIRPORT_CODE")
            ).otherwise(F.lit("OTHER"))
        )

        .withColumn(
            "WEEKDAY_CATEGORY",
            F.col("DAY_OF_WEEK").cast("string")
        )

        # Cyclical month encoding
        .withColumn(
            "MONTH_SIN",
            F.sin(
                (F.col("MONTH") - 1)
                * (2.0 * math.pi / 12.0)
            )
        )

        .withColumn(
            "MONTH_COS",
            F.cos(
                (F.col("MONTH") - 1)
                * (2.0 * math.pi / 12.0)
            )
        )

        # Cyclical scheduled departure hour
        .withColumn(
            "HOUR_SIN",
            F.sin(
                F.col("DEP_HOUR")
                * (2.0 * math.pi / 24.0)
            )
        )

        .withColumn(
            "HOUR_COS",
            F.cos(
                F.col("DEP_HOUR")
                * (2.0 * math.pi / 24.0)
            )
        )

        .withColumn(
            "label",
            F.col("IS_DELAYED").cast("double")
        )
    )


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    if not SOURCE.is_dir():
        raise FileNotFoundError(
            "Enriched dataset missing. "
            "Run 04b_dataset_integration.py first.\n"
            f"Expected: {SOURCE}"
        )

    if not 0 < SAMPLE_FRACTION <= 1:
        raise ValueError(
            "ML_SAMPLE_FRACTION must be between 0 and 1."
        )

    if TOP_AIRPORTS < 1 or FOREST_TREES < 1:
        raise ValueError(
            "Airport count and tree count must be positive."
        )

    METRICS.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n=== ENVIRONMENT ===")
    print("Python:", sys.executable)

    spark = (
        SparkSession.builder
        .appName("SAU BDA - Enhanced Chronological ML")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            "false"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    cached = []

    try:

        # ====================================================
        # 8. LOAD ENRICHED DATASET
        # ====================================================

        required = [
            "MONTH",
            "DAY_OF_WEEK",
            "DEP_HOUR",
            "AIRLINE",
            "ORIGIN_AIRPORT_CODE",
            "DESTINATION_AIRPORT_CODE",
            "DISTANCE",
            "SCHEDULED_TIME",
            "IS_DELAYED"
        ]

        source = spark.read.parquet(
            str(SOURCE)
        )

        missing = sorted(
            set(required) - set(source.columns)
        )

        if missing:
            raise RuntimeError(
                f"Missing dataset columns: {missing}"
            )

        df = (
            source
            .select(*required)
            .dropna(subset=required)

            .filter(
                F.col("MONTH").between(1, 12)
            )

            .filter(
                F.col("DAY_OF_WEEK").between(1, 7)
            )

            .filter(
                F.col("DEP_HOUR").between(0, 23)
            )

            .filter(
                (F.col("DISTANCE") > 0)
                & (F.col("SCHEDULED_TIME") > 0)
            )

            .filter(
                F.col("IS_DELAYED").isin(0, 1)
            )

            .sample(
                withReplacement=False,
                fraction=SAMPLE_FRACTION,
                seed=SEED
            )
        )

        print("\n=== DATASET ===")
        print(
            "Source: flights_enriched.parquet"
        )

        print(
            "Sampling fraction:",
            SAMPLE_FRACTION
        )

        # ====================================================
        # 9. CHRONOLOGICAL TRAIN/VALIDATION/TEST
        # ====================================================

        train_raw = (
            df
            .filter(F.col("MONTH") <= 8)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        valid_raw = (
            df
            .filter(
                F.col("MONTH").between(9, 10)
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        test_raw = (
            df
            .filter(F.col("MONTH") >= 11)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached.extend([
            train_raw,
            valid_raw,
            test_raw
        ])

        sizes = {
            "train_Jan_Aug": train_raw.count(),
            "validation_Sep_Oct": valid_raw.count(),
            "test_Nov_Dec": test_raw.count()
        }

        print(
            "\nChronological split:",
            sizes,
            flush=True
        )

        if min(sizes.values()) < 1000:
            raise RuntimeError(
                "Not enough sampled records for evaluation."
            )

        # ====================================================
        # 10. BUILD TRAINING-ONLY AIRPORT VOCABULARY
        # ====================================================

        origins = top_airport_codes(
            train_raw,
            "ORIGIN_AIRPORT_CODE"
        )

        destinations = top_airport_codes(
            train_raw,
            "DESTINATION_AIRPORT_CODE"
        )

        print(
            "Frequent origin airports:",
            len(origins)
        )

        print(
            "Frequent destination airports:",
            len(destinations)
        )

        # Apply identical feature engineering to all splits.

        train = make_features(
            train_raw, origins, destinations
        )

        valid = make_features(
            valid_raw, origins, destinations
        )

        test = make_features(
            test_raw, origins, destinations
        )

        # ====================================================
        # 11. PREPROCESSING PIPELINE
        # ====================================================

        categorical = [
            "AIRLINE",
            "ORIGIN_GROUP",
            "DESTINATION_GROUP",
            "WEEKDAY_CATEGORY"
        ]

        index_columns = [
            "IDX_" + column
            for column in categorical
        ]

        vector_columns = [
            "VEC_" + column
            for column in categorical
        ]

        numerical = [
            "MONTH_SIN",
            "MONTH_COS",
            "HOUR_SIN",
            "HOUR_COS",
            "DISTANCE",
            "SCHEDULED_TIME"
        ]

        indexers = [
            StringIndexer(
                inputCol=column,
                outputCol=index,
                handleInvalid="keep"
            )
            for column, index in zip(
                categorical,
                index_columns
            )
        ]

        encoder = OneHotEncoder(
            inputCols=index_columns,
            outputCols=vector_columns,
            handleInvalid="keep",
            dropLast=False
        )

        assembler = VectorAssembler(
            inputCols=vector_columns + numerical,
            outputCol="unscaled_features"
        )

        scaler = StandardScaler(
            inputCol="unscaled_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )

        # Fit category encoders and scaler ONLY on training data.

        preprocessing = Pipeline(
            stages=[
                *indexers,
                encoder,
                assembler,
                scaler
            ]
        ).fit(train)

        train_features = (
            preprocessing
            .transform(train)
            .select("features", "label")
        )

        valid_features = (
            preprocessing
            .transform(valid)
            .select("features", "label")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        test_features = (
            preprocessing
            .transform(test)
            .select("features", "label")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached.extend([
            valid_features,
            test_features
        ])

        # ====================================================
        # 12. HANDLE CLASS IMBALANCE
        # ====================================================

        distribution = {
            int(row["label"]): int(row["count"])
            for row in (
                train_features
                .groupBy("label")
                .count()
                .collect()
            )
        }

        negatives = distribution.get(0, 0)
        positives = distribution.get(1, 0)

        if negatives == 0 or positives == 0:
            raise RuntimeError(
                "Training data must contain both classes."
            )

        positive_weight = negatives / positives

        train_features = (
            train_features
            .withColumn(
                "class_weight",
                F.when(
                    F.col("label") == 1.0,
                    positive_weight
                ).otherwise(1.0)
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached.append(train_features)

        train_features.count()

        print("\n=== CLASS DISTRIBUTION ===")
        print("Not delayed:", negatives)
        print("Delayed:", positives)

        print(
            "Positive-class weight:",
            round(positive_weight, 4)
        )

        # ====================================================
        # 13. DEFINE ML ALGORITHMS
        # ====================================================

        algorithms = [
            (
                "Logistic Regression",
                LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    maxIter=40,
                    regParam=0.05
                )
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    numTrees=FOREST_TREES,
                    maxDepth=6,
                    maxBins=32,
                    seed=SEED
                )
            )
        ]

        roc_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        pr_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        validation_rows = []
        output_rows = []

        # ====================================================
        # 14. TRAIN AND EVALUATE BOTH MODELS
        # ====================================================

        for name, algorithm in algorithms:

            print(
                "\n===================================="
            )

            print(
                "Training:",
                name,
                flush=True
            )

            # Fit only on Jan-Aug data.
            model = algorithm.fit(train_features)

            # =================================================
            # 15. VALIDATION THRESHOLD OPTIMIZATION
            # =================================================

            validation_predictions = (
                model
                .transform(valid_features)
                .select(
                    "label",
                    vector_to_array("probability")[1]
                    .alias("probability")
                )
                .toPandas()
            )

            scores = [
                classification_metrics(
                    validation_predictions["label"],
                    validation_predictions["probability"],
                    threshold
                )
                for threshold in THRESHOLDS
            ]

            # Select threshold maximizing validation F1.
            # Ties prefer threshold closest to 0.50.

            chosen = max(
                scores,
                key=lambda result: (
                    result["f1_score"],
                    -abs(result["threshold"] - 0.5)
                )
            )

            for score in scores:
                validation_rows.append({
                    "model": name,
                    **score
                })

            print(
                f"Validation threshold: "
                f"{chosen['threshold']:.2f}"
            )

            print(
                f"Validation F1: "
                f"{chosen['f1_score']:.4f}"
            )

            # =================================================
            # 16. FINAL CHRONOLOGICAL TEST
            # =================================================

            predictions = (
                model
                .transform(test_features)
                .persist(StorageLevel.MEMORY_AND_DISK)
            )

            try:

                roc_auc = float(
                    roc_evaluator.evaluate(predictions)
                )

                pr_auc = float(
                    pr_evaluator.evaluate(predictions)
                )

                test_predictions = (
                    predictions
                    .select(
                        "label",
                        vector_to_array("probability")[1]
                        .alias("probability")
                    )
                    .toPandas()
                )

                for setting, threshold in [
                    ("default_0.50", 0.50),
                    (
                        "validation_tuned",
                        chosen["threshold"]
                    )
                ]:

                    metrics = classification_metrics(
                        test_predictions["label"],
                        test_predictions["probability"],
                        threshold
                    )

                    result = {
                        "model": name,
                        "setting": setting,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "test_rows": len(test_predictions),
                        **metrics
                    }

                    output_rows.append(result)

                    print(
                        f"\n{setting}"
                    )

                    print(
                        f"Accuracy: {metrics['accuracy']:.4f}"
                    )

                    print(
                        f"Precision: {metrics['precision']:.4f}"
                    )

                    print(
                        f"Recall: {metrics['recall']:.4f}"
                    )

                    print(
                        f"F1: {metrics['f1_score']:.4f}"
                    )

                    print(
                        f"ROC-AUC: {roc_auc:.4f}"
                    )

                    print(
                        f"PR-AUC: {pr_auc:.4f}"
                    )

            finally:
                predictions.unpersist(blocking=True)

        # ====================================================
        # 17. SAVE VALIDATION AND TEST RESULTS
        # ====================================================

        validation_df = pd.DataFrame(
            validation_rows
        )

        results_df = pd.DataFrame(
            output_rows
        )

        validation_df.to_csv(
            METRICS / "06b_validation_thresholds.csv",
            index=False
        )

        results_df.to_csv(
            METRICS / "06b_chronological_model_comparison.csv",
            index=False
        )

        # ====================================================
        # 18. SAVE JSON SUMMARY
        # ====================================================

        summary = {
            "dataset": (
                "flights_enriched.parquet "
                "(corrected October airport identifiers)"
            ),

            "split": (
                "Train Jan-Aug 2015; "
                "validation Sep-Oct 2015; "
                "test Nov-Dec 2015"
            ),

            "sample_fraction": SAMPLE_FRACTION,
            "seed": SEED,
            "split_counts": sizes,

            "training_class_counts": distribution,
            "positive_class_weight": positive_weight,

            "top_airports_per_direction": TOP_AIRPORTS,
            "forest_trees": FOREST_TREES,

            "feature_columns": (
                categorical + numerical
            ),

            "threshold_selection": (
                "Maximum validation F1; "
                "test set remains untouched"
            ),

            "comparison_warning": (
                "Results are not directly comparable "
                "with the original random-split experiment."
            ),

            "results": output_rows
        }

        output_json = (
            METRICS
            / "06b_chronological_model_comparison.json"
        )

        output_json.write_text(
            json.dumps(
                summary,
                indent=2
            ),
            encoding="utf-8"
        )

        # ====================================================
        # 19. GENERATE MODEL COMPARISON CHART
        # ====================================================

        tuned_results = results_df[
            results_df["setting"] == "validation_tuned"
        ]

        plot_data = (
            tuned_results
            .set_index("model")[
                [
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "pr_auc"
                ]
            ]
            .T
        )

        ax = plot_data.plot.bar(
            figsize=(11, 5),
            rot=0
        )

        ax.set_title(
            "Chronological Test (Nov-Dec 2015)\n"
            "Validation-Tuned ML Models"
        )

        ax.set_xlabel("Evaluation Metric")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

        ax.grid(
            axis="y",
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES / "06b_chronological_model_comparison.png",
            dpi=180
        )

        plt.close()

        # ====================================================
        # 20. FINAL OUTPUT
        # ====================================================

        print("\n====================================")
        print("ENHANCED ML EXPERIMENT COMPLETED")
        print("====================================")

        print(
            results_df.to_string(index=False)
        )

        print("\nGenerated files:")

        print(
            "results/metrics/06b_validation_thresholds.csv"
        )

        print(
            "results/metrics/"
            "06b_chronological_model_comparison.csv"
        )

        print(
            "results/metrics/"
            "06b_chronological_model_comparison.json"
        )

        print(
            "results/figures/"
            "06b_chronological_model_comparison.png"
        )

    finally:

        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=True)
            except Exception:
                pass

        spark.stop()


if __name__ == "__main__":
    main()
