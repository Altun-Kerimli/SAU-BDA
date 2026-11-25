import os
import json
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

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ============================================================
# 3. INITIALIZE SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("Flight Delay - Machine Learning")
    .master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
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

print("\n=== DATASET INFORMATION ===")
print("Total records:", df.count())

# Only use features available before departure.
# ARRIVAL_DELAY is excluded to prevent data leakage.

required_columns = [
    "IS_DELAYED",
    "AIRLINE",
    "MONTH",
    "DAY_OF_WEEK",
    "DEP_HOUR",
    "DISTANCE",
    "SCHEDULED_TIME"
]

df = (
    df.select(*required_columns)
    .dropna()
    .filter(F.col("DISTANCE") > 0)
    .filter(F.col("SCHEDULED_TIME") > 0)
)


# ============================================================
# 5. SAMPLE DATASET
# ============================================================

# Use 5% to reduce training time on the local computer.
# Increase this value later if necessary.

SAMPLE_FRACTION = 0.05
SEED = 42

df = df.sample(
    withReplacement=False,
    fraction=SAMPLE_FRACTION,
    seed=SEED
)

# 80% training, 20% testing

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=SEED
)

print("\n=== DATASET SPLIT ===")

print("Training records:", train_df.count())
print("Testing records:", test_df.count())


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# Convert airline codes to numerical indexes

indexer = StringIndexer(
    inputCol="AIRLINE",
    outputCol="AIRLINE_IDX",
    handleInvalid="keep"
)

# One-hot encoding for airline categories

encoder = OneHotEncoder(
    inputCols=["AIRLINE_IDX"],
    outputCols=["AIRLINE_VEC"]
)

# Combine all features into a single vector

assembler = VectorAssembler(
    inputCols=[
        "AIRLINE_VEC",
        "MONTH",
        "DAY_OF_WEEK",
        "DEP_HOUR",
        "DISTANCE",
        "SCHEDULED_TIME"
    ],
    outputCol="features"
)

# Create preprocessing pipeline

preprocessing = Pipeline(
    stages=[
        indexer,
        encoder,
        assembler
    ]
)

# Fit preprocessing ONLY on training data

preprocessing_model = preprocessing.fit(train_df)

training = (
    preprocessing_model.transform(train_df)
    .select(
        "features",
        F.col("IS_DELAYED").cast("double").alias("label")
    )
)

testing = (
    preprocessing_model.transform(test_df)
    .select(
        "features",
        F.col("IS_DELAYED").cast("double").alias("label")
    )
)


# ============================================================
# 7. CLASS IMBALANCE HANDLING
# ============================================================

print("\n=== CLASS DISTRIBUTION ===")

# Count delayed and non-delayed flights in training data

class_counts = {
    int(row["label"]): row["count"]
    for row in training.groupBy("label").count().collect()
}

negative_count = class_counts.get(0, 0)
positive_count = class_counts.get(1, 0)

train_count = negative_count + positive_count

if positive_count == 0 or negative_count == 0:
    raise RuntimeError(
        "Training dataset must contain both delayed "
        "and non-delayed flights."
    )

print("Non-delayed flights:", negative_count)
print("Delayed flights:", positive_count)

negative_percentage = negative_count / train_count * 100
positive_percentage = positive_count / train_count * 100

print(f"Non-delayed percentage: {negative_percentage:.2f}%")
print(f"Delayed percentage: {positive_percentage:.2f}%")


# ============================================================
# 8. CALCULATE CLASS WEIGHTS
# ============================================================

# The minority class (delayed flights) receives higher weight.
#
# Example:
# 80% non-delayed / 20% delayed = weight of 4.0
#
# Each delayed flight contributes approximately four times
# as much to the weighted training objective.

positive_weight = negative_count / positive_count

training = training.withColumn(
    "class_weight",
    F.when(
        F.col("label") == 1.0,
        F.lit(float(positive_weight))
    ).otherwise(F.lit(1.0))
)

# Cache processed data for repeated ML operations

training = training.cache()
testing = testing.cache()

training.count()
test_count = testing.count()

print("\n=== CLASS WEIGHTING ===")

print(f"Positive class weight: {positive_weight:.4f}")
print("Negative class weight: 1.0000")

print("\nTraining dataset prepared:", train_count)
print("Testing dataset prepared:", test_count)


# ============================================================
# 9. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    # Model 1: Logistic Regression

    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        maxIter=30,
        regParam=0.1
    ),

    # Model 2: Random Forest

    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        numTrees=25,
        maxDepth=6,
        maxBins=32,
        seed=SEED
    )
}


# ============================================================
# 10. EVALUATION CONFIGURATION
# ============================================================

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

results = []


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

for name, algorithm in models.items():

    print("\n====================================")
    print(f"TRAINING: {name}")
    print("====================================")

    # Train model using weighted training data

    model = algorithm.fit(training)

    # Predict on the original, unweighted test data

    predictions = model.transform(testing)

    # Confusion matrix

    confusion = {
        (int(row["label"]), int(row["prediction"])): row["count"]
        for row in (
            predictions
            .groupBy("label", "prediction")
            .count()
            .collect()
        )
    }

    # True Negative
    tn = confusion.get((0, 0), 0)

    # False Positive
    fp = confusion.get((0, 1), 0)

    # False Negative
    fn = confusion.get((1, 0), 0)

    # True Positive
    tp = confusion.get((1, 1), 0)


    # ========================================================
    # 12. CALCULATE EVALUATION METRICS
    # ========================================================

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    auc = evaluator.evaluate(predictions)


    # ========================================================
    # 13. STORE RESULTS
    # ========================================================

    result = {
        "model": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

    results.append(result)


    # ========================================================
    # 14. PRINT MODEL PERFORMANCE
    # ========================================================

    print("\n=== MODEL PERFORMANCE ===")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\n=== CONFUSION MATRIX ===")

    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    print("\n=== DELAY DETECTION ===")

    actual_delayed = tp + fn
    predicted_delayed = tp + fp

    print("Actual delayed flights:", actual_delayed)
    print("Predicted delayed flights:", predicted_delayed)
    print("Correctly detected delays:", tp)

    if tp == 0:
        print(
            "WARNING: Model failed to identify delayed flights."
        )


# ============================================================
# 15. SAVE MODEL COMPARISON RESULTS
# ============================================================

results_df = pd.DataFrame(results)

# Save CSV

results_df.to_csv(
    METRICS_DIR / "06_model_comparison.csv",
    index=False
)

# Save JSON

summary = {
    "sample_fraction": SAMPLE_FRACTION,
    "seed": SEED,
    "training_records": train_count,
    "testing_records": test_count,
    "training_class_distribution": {
        "non_delayed": negative_count,
        "delayed": positive_count
    },
    "positive_class_weight": round(positive_weight, 4),
    "models": results
}

with open(
    METRICS_DIR / "06_model_comparison.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 16. GENERATE MODEL COMPARISON CHART
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]

chart_data = (
    results_df
    .set_index("model")[metrics]
    .T
)

ax = chart_data.plot(
    kind="bar",
    figsize=(11, 6),
    rot=0
)

ax.set_title(
    "Machine Learning Model Comparison - Class Weighted"
)

ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")

ax.set_ylim(0, 1)

ax.legend(
    title="Algorithm"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "06_model_comparison.png",
    dpi=200
)

plt.close()


# ============================================================
# 17. FINISH
# ============================================================

print("\n====================================")
print("MACHINE LEARNING COMPLETED")
print("====================================")

print("\n=== FINAL MODEL COMPARISON ===")

print(
    results_df.to_string(index=False)
)

print("\nGenerated files:")
print("1. results/metrics/06_model_comparison.csv")
print("2. results/metrics/06_model_comparison.json")
print("3. results/figures/06_model_comparison.png")

training.unpersist()
testing.unpersist()

spark.stop()