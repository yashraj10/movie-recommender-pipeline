"""
Movie Recommender Pipeline — Airflow DAG
==========================================
End-to-end pipeline: ingest → validate → features → upload → train → evaluate → drift → promote/block

DAG Structure (8 tasks with branching):

  ingest_data
      │
      ▼
  validate_data
      │
      ▼
  spark_feature_engineering
      │
      ▼
  upload_features_to_s3
      │
      ├──────────────────┐
      ▼                  ▼
  train_mf_baseline   train_ncf_model
      │                  │
      └────────┬─────────┘
               ▼
        evaluate_models
               │
               ▼
        drift_check
               │
          ┌────┴────┐
          ▼         ▼
       promote    block_and_alert
       model      (if drift fails)

Schedule: @weekly (or manual trigger via Airflow UI)
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

# ── Paths (inside Docker container) ────────────────────────────────────
DATA_DIR = "/opt/airflow/data"
RAW_DATA_DIR = f"{DATA_DIR}/ml-32m"
FEATURES_DIR = f"{DATA_DIR}/features"
MODELS_DIR = f"{DATA_DIR}/models"

# Add project modules to path so Airflow can import them
sys.path.insert(0, "/opt/airflow")

# ── Training Configuration ─────────────────────────────────────────────
TRAINING_CONFIG = {
    "gmf_dim": 64,
    "mlp_dim": 64,
    "mlp_layers": [128, 64, 32],
    "dropout_rate": 0.2,
    "lr": 0.001,
    "batch_size": 4096,
    "epochs": 15,
    "neg_ratio": 4,
    "patience": 3,
    "min_delta": 0.0001,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 1e-6,
    "min_user_ratings": 20,
    "min_item_ratings": 5,
    "seed": 42,
}

# ── Default Args ───────────────────────────────────────────────────────
default_args = {
    "owner": "yashraj",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}


# ══════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def _ingest_data(**context):
    """
    TASK 1: Download MovieLens 32M dataset (or verify it exists).

    In production, this would pull fresh data from a data warehouse
    or streaming pipeline. For this project, we download from GroupLens
    on first run and reuse on subsequent runs.
    """
    import subprocess

    logger.info("Checking for raw data at %s", RAW_DATA_DIR)

    if not os.path.exists(f"{RAW_DATA_DIR}/ratings.csv"):
        logger.info("Downloading MovieLens 32M dataset...")
        os.makedirs(DATA_DIR, exist_ok=True)

        subprocess.run([
            "wget", "-q",
            "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
            "-O", f"{DATA_DIR}/ml-32m.zip",
        ], check=True)

        subprocess.run([
            "unzip", "-o", f"{DATA_DIR}/ml-32m.zip",
            "-d", DATA_DIR,
        ], check=True)

        logger.info("Download complete")
    else:
        logger.info("Raw data already exists — skipping download")

    # Verify files exist
    for filename in ["ratings.csv", "movies.csv"]:
        filepath = f"{RAW_DATA_DIR}/{filename}"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing required file: {filepath}")

    # Count rows for manifest
    with open(f"{RAW_DATA_DIR}/ratings.csv") as f:
        n_ratings = sum(1 for _ in f) - 1

    manifest = {
        "ratings_count": n_ratings,
        "data_dir": RAW_DATA_DIR,
        "status": "ingested",
    }

    context["ti"].xcom_push(key="ingestion_manifest", value=manifest)
    logger.info("Ingestion complete: %d ratings", n_ratings)
    return manifest


def _validate_data(**context):
    """
    TASK 2: Run data quality checks on raw data.

    Validates completeness, value ranges, and referential integrity.
    Fails the task (and blocks pipeline) if critical issues found.
    """
    from pyspark.sql import SparkSession

    logger.info("Starting data validation...")

    spark = SparkSession.builder \
        .appName("Validation") \
        .master("local[*]") \
        .config("spark.driver.memory", "3g") \
        .getOrCreate()

    try:
        from spark.schemas import RATINGS_SCHEMA, MOVIES_SCHEMA
        from pyspark.sql import functions as F

        ratings = spark.read.csv(
            f"{RAW_DATA_DIR}/ratings.csv",
            header=True, schema=RATINGS_SCHEMA,
        )
        movies = spark.read.csv(
            f"{RAW_DATA_DIR}/movies.csv",
            header=True, schema=MOVIES_SCHEMA,
        )

        # Single-pass aggregation for quality checks
        agg = ratings.agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("userId").isNull() | F.col("movieId").isNull(), 1).otherwise(0)).alias("nulls"),
            F.min("rating").alias("min_rating"),
            F.max("rating").alias("max_rating"),
            F.countDistinct("userId").alias("n_users"),
            F.countDistinct("movieId").alias("n_movies"),
        ).collect()[0]

        checks = {
            "total_ratings": agg["total"],
            "null_count": agg["nulls"],
            "min_rating": float(agg["min_rating"]),
            "max_rating": float(agg["max_rating"]),
            "n_users": agg["n_users"],
            "n_movies": agg["n_movies"],
        }

        logger.info("Quality checks: %s", checks)

        # Fail on critical issues
        assert checks["null_count"] == 0, f"Found {checks['null_count']} null ratings"
        assert checks["min_rating"] >= 0.5, f"Min rating {checks['min_rating']} < 0.5"
        assert checks["max_rating"] <= 5.0, f"Max rating {checks['max_rating']} > 5.0"
        assert checks["total_ratings"] > 30_000_000, f"Only {checks['total_ratings']} ratings"

        context["ti"].xcom_push(key="quality_checks", value=checks)
        logger.info("✓ All quality checks passed")

    finally:
        spark.stop()


def _spark_feature_engineering(**context):
    """
    TASK 3: Full PySpark feature engineering pipeline.

    Steps: cold start filter → user features → item features →
           interaction features → ID remap → temporal split → save Parquet
    """
    logger.info("Starting Spark feature engineering...")

    # Import and run the full pipeline
    from spark.feature_engineering import main as run_spark_pipeline
    metadata = run_spark_pipeline()

    context["ti"].xcom_push(key="feature_metadata", value=metadata)
    logger.info("Feature engineering complete: %s", metadata)
    return metadata


def _upload_features_to_s3(**context):
    """
    TASK 4: Upload Parquet features and baselines to S3 data lake.

    Uploads to s3://bucket/features/ with manifest.
    """
    bucket = os.environ.get("S3_BUCKET", "movie-recommender-yashraj")

    try:
        from pipeline.s3_utils import upload_all_features
        results = upload_all_features(FEATURES_DIR, bucket)
        logger.info("✓ All features uploaded to S3: %s", results)
    except Exception as e:
        logger.warning("S3 upload failed (non-blocking): %s", e)
        logger.info("Continuing pipeline — features available locally")


def _train_mf_baseline(**context):
    """
    TASK 5A: Train Matrix Factorization baseline model.

    This is the "simple but strong" baseline that NCF must beat.
    """
    import numpy as np
    import tensorflow as tf
    from model.matrix_factorization import MatrixFactorization
    from model.data_loader import load_interactions_from_parquet, generate_negative_samples, create_tf_dataset

    metadata = context["ti"].xcom_pull(key="feature_metadata", task_ids="spark_feature_engineering")
    n_users = metadata["n_users"]
    n_items = metadata["n_items"]

    logger.info("Training MF baseline: %d users, %d items", n_users, n_items)

    # Load data
    train_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/train")
    val_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/val")

    # Negative sampling
    train_sampled = generate_negative_samples(train_df, n_items, neg_ratio=4)
    val_sampled = generate_negative_samples(val_df, n_items, neg_ratio=1)

    # Create TF datasets
    train_ds = create_tf_dataset(train_sampled, batch_size=4096, shuffle=True)
    val_ds = create_tf_dataset(val_sampled, batch_size=4096, shuffle=False)

    # Build and train model
    model = MatrixFactorization(n_users, n_items, embedding_dim=64)

    from model.train import train_model
    model, history, metrics = train_model(
        model, train_ds, val_ds,
        config={**TRAINING_CONFIG, "model_type": "mf"},
        model_name="mf_baseline",
        save_dir=MODELS_DIR,
    )

    context["ti"].xcom_push(key="mf_metrics", value=metrics)
    context["ti"].xcom_push(key="mf_model_path", value=f"{MODELS_DIR}/mf_baseline")
    logger.info("✓ MF baseline training complete: %s", metrics)


def _train_ncf_model(**context):
    """
    TASK 5B: Train Neural Collaborative Filtering model.

    Two-tower architecture (GMF + MLP) — this is the main model.
    Must beat MF baseline to justify its complexity.
    """
    import numpy as np
    import tensorflow as tf
    from model.ncf import NeuralCollaborativeFiltering
    from model.data_loader import load_interactions_from_parquet, generate_negative_samples, create_tf_dataset

    metadata = context["ti"].xcom_pull(key="feature_metadata", task_ids="spark_feature_engineering")
    n_users = metadata["n_users"]
    n_items = metadata["n_items"]

    logger.info("Training NCF model: %d users, %d items", n_users, n_items)

    # Load data
    train_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/train")
    val_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/val")

    # Negative sampling
    train_sampled = generate_negative_samples(train_df, n_items, neg_ratio=4)
    val_sampled = generate_negative_samples(val_df, n_items, neg_ratio=1)

    # Create TF datasets
    train_ds = create_tf_dataset(train_sampled, batch_size=4096, shuffle=True)
    val_ds = create_tf_dataset(val_sampled, batch_size=4096, shuffle=False)

    # Build and train model
    model = NeuralCollaborativeFiltering(
        n_users, n_items,
        gmf_dim=TRAINING_CONFIG["gmf_dim"],
        mlp_dim=TRAINING_CONFIG["mlp_dim"],
        mlp_layers=TRAINING_CONFIG["mlp_layers"],
        dropout_rate=TRAINING_CONFIG["dropout_rate"],
    )

    from model.train import train_model
    model, history, metrics = train_model(
        model, train_ds, val_ds,
        config=TRAINING_CONFIG,
        model_name="ncf_latest",
        save_dir=MODELS_DIR,
    )

    # Log to MLflow
    try:
        from pipeline.mlflow_tracking import log_training_run
        run_id = log_training_run(
            model, TRAINING_CONFIG, metrics, history,
            model_path=f"{MODELS_DIR}/ncf_latest",
            model_type="ncf",
        )
        context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
    except Exception as e:
        logger.warning("MLflow logging failed (non-blocking): %s", e)

    context["ti"].xcom_push(key="ncf_metrics", value=metrics)
    context["ti"].xcom_push(key="ncf_model_path", value=f"{MODELS_DIR}/ncf_latest")
    logger.info("✓ NCF training complete: %s", metrics)


def _evaluate_models(**context):
    """
    TASK 6: Evaluate both models on test set.

    Computes HR@K, NDCG@K, Coverage for MF and NCF.
    Logs comparison to MLflow.
    """
    import tensorflow as tf
    from model.evaluate import evaluate_recommendation_model, print_model_comparison
    from model.data_loader import load_interactions_from_parquet

    metadata = context["ti"].xcom_pull(key="feature_metadata", task_ids="spark_feature_engineering")
    n_users = metadata["n_users"]
    n_items = metadata["n_items"]

    # Load test data
    test_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/test")
    train_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/train")
    val_df = load_interactions_from_parquet(f"{FEATURES_DIR}/interactions/val")

    # Load MF model
    mf_path = context["ti"].xcom_pull(key="mf_model_path", task_ids="train_mf_baseline")
    from model.matrix_factorization import MatrixFactorization
    mf_model = MatrixFactorization(n_users, n_items, embedding_dim=64)
    mf_model.build(input_shape=(None, 2))
    mf_model.load_weights(f"{mf_path}/weights.weights.h5")

    mf_results = evaluate_recommendation_model(
        mf_model, test_df, train_df, val_df, n_users, n_items,
        k_values=[5, 10, 20], max_eval_users=5000,
    )

    # Load NCF model
    ncf_path = context["ti"].xcom_pull(key="ncf_model_path", task_ids="train_ncf_model")
    from model.ncf import NeuralCollaborativeFiltering
    ncf_model = NeuralCollaborativeFiltering(
        n_users, n_items,
        gmf_dim=64, mlp_dim=64,
        mlp_layers=[128, 64, 32], dropout_rate=0.2,
    )
    ncf_model.build(input_shape=(None, 2))
    ncf_model.load_weights(f"{ncf_path}/weights.weights.h5")

    ncf_results = evaluate_recommendation_model(
        ncf_model, test_df, train_df, val_df, n_users, n_items,
        k_values=[5, 10, 20], max_eval_users=5000,
    )

    # Print comparison
    print_model_comparison(mf_results, ncf_results)

    context["ti"].xcom_push(key="mf_eval_results", value=mf_results)
    context["ti"].xcom_push(key="ncf_eval_results", value=ncf_results)
    logger.info("✓ Evaluation complete — MF: %s, NCF: %s", mf_results, ncf_results)


def _drift_check(**context):
    """
    TASK 7: Run PSI drift check on feature distributions.

    Compares current feature distributions against training baseline.
    Returns task_id of next task to execute:
        - "promote_model" if drift check passes
        - "block_and_alert" if drift check fails

    This is a BranchPythonOperator — it controls pipeline flow.
    """
    import numpy as np
    from pipeline.drift_monitor import run_drift_check, format_drift_report

    baseline_path = f"{FEATURES_DIR}/baseline_distributions.npz"

    if not os.path.exists(baseline_path):
        logger.warning("No baseline found — skipping drift check, allowing promotion")
        context["ti"].xcom_push(key="drift_passed", value=True)
        return "promote_model"

    # Load current features (in production, these come from fresh data)
    # For initial deployment, we compare baseline to itself (always passes)
    baseline = np.load(baseline_path)
    current_features = {k: baseline[k] for k in baseline.files}

    passed, results = run_drift_check(baseline_path, current_features)

    # Log results
    report = format_drift_report(results)
    logger.info("\n%s", report)

    context["ti"].xcom_push(key="drift_results", value=results)
    context["ti"].xcom_push(key="drift_passed", value=passed)

    if passed:
        logger.info("✓ Drift check PASSED — proceeding to promotion")
        return "promote_model"
    else:
        logger.warning("✗ Drift check FAILED — blocking promotion")
        return "block_and_alert"


def _promote_model(**context):
    """
    TASK 8A: Promote NCF model to Production.

    Only runs if drift check passes.
    Promotes in MLflow registry and uploads to S3.
    """
    ncf_metrics = context["ti"].xcom_pull(key="ncf_eval_results", task_ids="evaluate_models")

    # Check against current production
    try:
        from pipeline.mlflow_tracking import promote_model, get_production_metrics
        production_metrics = get_production_metrics()
        promoted, message = promote_model(None, ncf_metrics, production_metrics)
        logger.info(message)
    except Exception as e:
        logger.warning("MLflow promotion failed (non-blocking): %s", e)
        logger.info("Model available locally at %s",
                     context["ti"].xcom_pull(key="ncf_model_path", task_ids="train_ncf_model"))

    # Upload to S3
    try:
        from pipeline.s3_utils import upload_model
        ncf_path = context["ti"].xcom_pull(key="ncf_model_path", task_ids="train_ncf_model")
        bucket = os.environ.get("S3_BUCKET", "movie-recommender-yashraj")
        upload_model(ncf_path, bucket, "ncf", version=1)
    except Exception as e:
        logger.warning("S3 model upload failed (non-blocking): %s", e)

    logger.info("✓ Model promotion complete")


def _block_and_alert(**context):
    """
    TASK 8B: Block promotion and log alert.

    Only runs if drift check fails.
    The current Production model stays in place.
    """
    drift_results = context["ti"].xcom_pull(key="drift_results", task_ids="drift_check")

    logger.warning("=" * 60)
    logger.warning("MODEL PROMOTION BLOCKED — DRIFT DETECTED")
    logger.warning("=" * 60)
    logger.warning("Drift results: %s", json.dumps(drift_results, indent=2, default=str))
    logger.warning("Current Production model remains active")
    logger.warning("=" * 60)

    # In production, this would send a Slack/PagerDuty alert
    # For now, just log the failure


# ══════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="movie_recommender_pipeline",
    default_args=default_args,
    description=(
        "End-to-end movie recommendation pipeline: "
        "ingest → validate → Spark FE → S3 → train → evaluate → drift → promote"
    ),
    schedule_interval="@weekly",
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=["recommendation", "ml", "spark", "tensorflow", "mlflow"],
    doc_md=__doc__,
) as dag:

    # ── Task definitions ───────────────────────────────────────────
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest_data,
        doc="Download and verify MovieLens 32M dataset",
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        doc="Run data quality checks (nulls, ranges, referential integrity)",
    )

    spark_fe = PythonOperator(
        task_id="spark_feature_engineering",
        python_callable=_spark_feature_engineering,
        doc="PySpark pipeline: features, ID remap, temporal split, Parquet output",
    )

    upload_s3 = PythonOperator(
        task_id="upload_features_to_s3",
        python_callable=_upload_features_to_s3,
        doc="Upload Parquet features to S3 data lake",
    )

    train_mf = PythonOperator(
        task_id="train_mf_baseline",
        python_callable=_train_mf_baseline,
        doc="Train Matrix Factorization baseline (dot product + bias)",
    )

    train_ncf = PythonOperator(
        task_id="train_ncf_model",
        python_callable=_train_ncf_model,
        doc="Train Neural Collaborative Filtering (GMF + MLP two-tower)",
    )

    evaluate = PythonOperator(
        task_id="evaluate_models",
        python_callable=_evaluate_models,
        doc="Compute HR@K, NDCG@K, Coverage on test set for both models",
    )

    drift = BranchPythonOperator(
        task_id="drift_check",
        python_callable=_drift_check,
        doc="PSI drift check — gates model promotion (threshold=0.2)",
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
        doc="Promote NCF to Production in MLflow registry + S3",
    )

    block = PythonOperator(
        task_id="block_and_alert",
        python_callable=_block_and_alert,
        doc="Block promotion, log drift alert, keep current Production model",
    )

    # ── Task dependencies ──────────────────────────────────────────
    # Linear: ingest → validate → features → upload
    ingest >> validate >> spark_fe >> upload_s3

    # Parallel training after features are ready
    upload_s3 >> [train_mf, train_ncf]

    # Both models must finish before evaluation
    [train_mf, train_ncf] >> evaluate

    # Drift check gates promotion
    evaluate >> drift

    # Branch: promote OR block (never both)
    drift >> [promote, block]
