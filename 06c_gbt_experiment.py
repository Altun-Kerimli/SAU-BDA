
"""SAU BDA - Gradient-Boosted Trees flight-delay experiment.

Run after 06b_enhanced_ml.py:
    python 06c_gbt_experiment.py

Uses the same 06b features, sample and chronological split.
Train: Jan-Aug; validate: Sep-Oct; test: Nov-Dec 2015.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Environment
for name, folder in (
    ("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"),
    ("HADOOP_HOME", r"C:\hadoop"),
):
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"Missing {name}: {folder}")
    os.environ[name] = folder
    os.environ["PATH"] = (
        str(Path(folder) / "bin")
        + os.pathsep + os.environ.get("PATH", "")
    )

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
import matplotlib.pyplot as plt

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.functions import vector_to_array

# ============================================================
# 1. REUSE THE EXISTING 06b IMPLEMENTATION
# ============================================================

base_path = ROOT / "06b_enhanced_ml.py"

if not base_path.is_file():
    raise FileNotFoundError(
        f"Required baseline script missing: {base_path}"
    )

spec = importlib.util.spec_from_file_location(
    "sau_06b_baseline", base_path
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEED = 42
SAMPLE_FRACTION = 0.05
TOP_AIRPORTS = 25

SOURCE = ROOT / "data/processed/flights_enriched.parquet"
METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"

METRICS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def probabilities(predictions):
    return predictions.select(
        "label",
        vector_to_array("probability")[1].alias("probability")
    ).toPandas()


# ============================================================
# 2. RUN EXPERIMENT
# ============================================================

def main():

    if not SOURCE.is_dir():
        raise FileNotFoundError(
            "Run 04b_dataset_integration.py first."
        )

    print("Python:", sys.executable, flush=True)

    spark = (
        SparkSession.builder
        .appName("SAU BDA - GBT Experiment")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    cached = []

    try:
        # ====================================================
        # 3. LOAD EXACTLY THE SAME FEATURES AS 06b
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
            "IS_DELAYED",
        ]

        source = spark.read.parquet(str(SOURCE))
        missing = set(required) - set(source.columns)

        if missing:
            raise RuntimeError(
                f"Missing columns: {sorted(missing)}"
            )

        sample = (
            source
            .select(*required)
            .dropna(subset=required)
            .filter(F.col("MONTH").between(1, 12))
            .filter(F.col("DAY_OF_WEEK").between(1, 7))
            .filter(F.col("DEP_HOUR").between(0, 23))
            .filter(
                (F.col("DISTANCE") > 0)
                & (F.col("SCHEDULED_TIME") > 0)
            )
            .filter(F.col("IS_DELAYED").isin(0, 1))
            .sample(
                withReplacement=False,
                fraction=SAMPLE_FRACTION,
                seed=SEED
            )
        )

        # ====================================================
        # 4. CHRONOLOGICAL SPLIT
        # ====================================================

        train_raw = (
            sample.filter(F.col("MONTH") <= 8)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        valid_raw = (
            sample.filter(F.col("MONTH").between(9, 10))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        test_raw = (
            sample.filter(F.col("MONTH") >= 11)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached += [train_raw, valid_raw, test_raw]

        counts = {
            "train_Jan_Aug": train_raw.count(),
            "validation_Sep_Oct": valid_raw.count(),
            "test_Nov_Dec": test_raw.count(),
        }

        print("Chronological split:", counts, flush=True)

        if min(counts.values()) < 1000:
            raise RuntimeError(
                "Not enough records in the chronological splits."
            )

        # ====================================================
        # 5. CHECK COMPARABILITY WITH 06b
        # ====================================================

        baseline_json = (
            METRICS / "06b_chronological_model_comparison.json"
        )

        comparable = False

        if baseline_json.is_file():
            old = json.loads(
                baseline_json.read_text(encoding="utf-8")
            )

            comparable = (
                old.get("split_counts") == counts
                and old.get("seed") == SEED
                and old.get("sample_fraction") == SAMPLE_FRACTION
                and old.get("top_airports_per_direction")
                == TOP_AIRPORTS
            )

        if not comparable:
            print(
                "WARNING: 06b settings/counts differ or its "
                "results are missing. Do not treat the scores "
                "as a controlled model comparison.",
                flush=True
            )

        # ====================================================
        # 6. REUSE 06b FEATURE ENGINEERING
        # ====================================================

        origins = base.top_airport_codes(
            train_raw, "ORIGIN_AIRPORT_CODE"
        )

        destinations = base.top_airport_codes(
            train_raw, "DESTINATION_AIRPORT_CODE"
        )

        train = base.make_features(
            train_raw, origins, destinations
        )

        valid = base.make_features(
            valid_raw, origins, destinations
        )

        test = base.make_features(
            test_raw, origins, destinations
        )

        categories = [
            "AIRLINE",
            "ORIGIN_GROUP",
            "DESTINATION_GROUP",
            "WEEKDAY_CATEGORY",
        ]

        indexes = ["IDX_" + x for x in categories]
        encoded = ["VEC_" + x for x in categories]

        numerical = [
            "MONTH_SIN",
            "MONTH_COS",
            "HOUR_SIN",
            "HOUR_COS",
            "DISTANCE",
            "SCHEDULED_TIME",
        ]

        preprocessing = Pipeline(stages=[
            *[
                StringIndexer(
                    inputCol=c,
                    outputCol=i,
                    handleInvalid="keep"
                )
                for c, i in zip(categories, indexes)
            ],
            OneHotEncoder(
                inputCols=indexes,
                outputCols=encoded,
                handleInvalid="keep",
                dropLast=False
            ),
            VectorAssembler(
                inputCols=encoded + numerical,
                outputCol="unscaled_features"
            ),
            StandardScaler(
                inputCol="unscaled_features",
                outputCol="features",
                withStd=True,
                withMean=False
            ),
        ]).fit(train)

        train_x = preprocessing.transform(train).select(
            "features", "label"
        )

        valid_x = (
            preprocessing.transform(valid)
            .select("features", "label")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        test_x = (
            preprocessing.transform(test)
            .select("features", "label")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached += [valid_x, test_x]

        # ====================================================
        # 7. CLASS IMBALANCE
        # ====================================================

        labels = {
            int(row["label"]): int(row["count"])
            for row in train_x.groupBy("label").count().collect()
        }

        if not labels.get(0) or not labels.get(1):
            raise RuntimeError(
                "Both classes are required for training."
            )

        positive_weight = labels[0] / labels[1]

        train_x = (
            train_x
            .withColumn(
                "class_weight",
                F.when(
                    F.col("label") == 1.0,
                    positive_weight
                ).otherwise(1.0)
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached.append(train_x)
        train_x.count()

        print(
            f"\nTraining GBT: weight={positive_weight:.4f}, "
            "30 iterations, depth=5",
            flush=True
        )

        # ====================================================
        # 8. GRADIENT-BOOSTED TREES
        # ====================================================

        algorithm = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=30,
            maxDepth=5,
            maxBins=32,
            stepSize=0.1,
            subsamplingRate=0.8,
            minInstancesPerNode=10,
            seed=SEED
        )

        model = algorithm.fit(train_x)

        # ====================================================
        # 9. TUNE THRESHOLD ON VALIDATION ONLY
        # ====================================================

        validation = probabilities(
            model.transform(valid_x)
        )

        scores = [
            base.classification_metrics(
                validation["label"],
                validation["probability"],
                threshold
            )
            for threshold in base.THRESHOLDS
        ]

        chosen = max(
            scores,
            key=lambda result: (
                result["f1_score"],
                -abs(result["threshold"] - 0.5)
            )
        )

        print(
            f"Validation threshold: {chosen['threshold']:.2f}"
        )
        print(
            f"Validation F1: {chosen['f1_score']:.4f}"
        )

        # ====================================================
        # 10. FINAL TEST
        # ====================================================

        predictions = (
            model.transform(test_x)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        cached.append(predictions)

        roc_auc = BinaryClassificationEvaluator(
            labelCol="label",
            metricName="areaUnderROC"
        ).evaluate(predictions)

        pr_auc = BinaryClassificationEvaluator(
            labelCol="label",
            metricName="areaUnderPR"
        ).evaluate(predictions)

        test_probabilities = probabilities(predictions)

        results = []

        for setting, threshold in [
            ("default_0.50", 0.5),
            ("validation_tuned", chosen["threshold"])
        ]:
            metrics = base.classification_metrics(
                test_probabilities["label"],
                test_probabilities["probability"],
                threshold
            )

            results.append({
                "model": "Gradient-Boosted Trees",
                "setting": setting,
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "test_rows": len(test_probabilities),
                **metrics
            })

        # ====================================================
        # 11. SAVE RESULTS
        # ====================================================

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            METRICS / "06c_gbt_results.csv",
            index=False
        )

        summary = {
            "model": "Spark GBTClassifier",
            "spark_version": spark.version,
            "dataset": "flights_enriched.parquet",
            "seed": SEED,
            "sample_fraction": SAMPLE_FRACTION,
            "split_counts": counts,
            "training_class_counts": labels,
            "positive_class_weight": positive_weight,
            "top_airports_per_direction": TOP_AIRPORTS,
            "hyperparameters": {
                "maxIter": 30,
                "maxDepth": 5,
                "stepSize": 0.1,
                "subsamplingRate": 0.8,
                "minInstancesPerNode": 10
            },
            "baseline_comparison_settings_match": comparable,
            "threshold_selection": (
                "Maximum validation F1; test not used for tuning"
            ),
            "validation_best": chosen,
            "test_results": results
        }

        (
            METRICS / "06c_gbt_results.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8"
        )

        # ====================================================
        # 12. COMPARE WITH EXISTING 06b RESULTS
        # ====================================================

        comparison = results_df[
            results_df["setting"] == "validation_tuned"
        ].copy()

        baseline_csv = (
            METRICS
            / "06b_chronological_model_comparison.csv"
        )

        if comparable and baseline_csv.is_file():

            previous = pd.read_csv(baseline_csv)

            previous = previous[
                previous["setting"] == "validation_tuned"
            ]

            comparison = pd.concat(
                [previous, comparison],
                ignore_index=True
            )

        columns = [
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc"
        ]

        ax = (
            comparison
            .set_index("model")[columns]
            .T
            .plot.bar(figsize=(11, 6), rot=0)
        )

        ax.set_title(
            "Chronological Test (Nov-Dec 2015)\n"
            "Validation-Tuned Model Comparison"
        )

        ax.set_xlabel("Evaluation Metric")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURES / "06c_gbt_comparison.png",
            dpi=180
        )

        plt.close()

        # ====================================================
        # 13. FINAL OUTPUT
        # ====================================================

        print("\n=== GRADIENT-BOOSTED TREES RESULTS ===")

        print(results_df.to_string(index=False))

        print(
            "\nComparison with 06b:",
            "Settings verified" if comparable
            else "Not verified"
        )

        print("\nGenerated files:")
        print("results/metrics/06c_gbt_results.csv")
        print("results/metrics/06c_gbt_results.json")
        print("results/figures/06c_gbt_comparison.png")

    finally:

        for frame in reversed(cached):
            try:
                frame.unpersist(blocking=True)
            except Exception:
                pass

        spark.stop()


if __name__ == "__main__":
    main()
