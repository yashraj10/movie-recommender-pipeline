"""
MLflow Experiment Tracking & Model Registry
=============================================
Centralized MLflow integration for:
    - Logging hyperparameters, metrics, and training curves
    - Registering models with version control
    - Promoting models to Production based on metric gates
    - Comparing experiment runs

MLflow server runs as a Docker container alongside Airflow
(see docker-compose.yml) at http://mlflow:5000 (internal)
or http://localhost:5000 (local development).
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "movie-recommender"
REGISTERED_MODEL_NAME = "MovieRecommenderNCF"

# Promotion gate: new model must beat current Production on this metric
PROMOTION_METRIC = "NDCG@10"
PROMOTION_DIRECTION = "higher"  # higher is better


def _get_git_commit_hash() -> str:
    """Get current git commit hash for experiment tagging."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
) -> None:
    """
    Configure MLflow tracking URI and experiment.

    Args:
        tracking_uri: MLflow server URL
        experiment_name: Name of the experiment to log to
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow configured: uri=%s, experiment=%s", tracking_uri, experiment_name)
    except ImportError:
        logger.warning("mlflow not installed — tracking disabled")
    except Exception as e:
        logger.warning("MLflow setup failed: %s — tracking disabled", e)


def log_training_run(
    model: Any,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    history: Optional[Any] = None,
    model_path: Optional[str] = None,
    model_type: str = "ncf",
    register: bool = True,
) -> Optional[str]:
    """
    Log a complete training run to MLflow.

    Logs:
        - All hyperparameters from config
        - All evaluation metrics (HR@K, NDCG@K, etc.)
        - Training curves (loss per epoch) if history provided
        - Git commit hash for reproducibility
        - Model artifact for registry

    Args:
        model: Trained TensorFlow model
        config: Training configuration dict
        metrics: Evaluation metrics dict
        history: Keras training history (optional)
        model_path: Local path to saved model (optional)
        model_type: "ncf" or "mf" for tagging
        register: Whether to register model in MLflow registry

    Returns:
        MLflow run_id or None if tracking is disabled
    """
    try:
        import mlflow
        import mlflow.tensorflow
    except ImportError:
        logger.warning("mlflow not installed — skipping logging")
        return None

    try:
        setup_mlflow()

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("MLflow run started: %s", run_id)

            # ── Log parameters ─────────────────────────────────
            safe_params = {}
            for k, v in config.items():
                if isinstance(v, (list, dict)):
                    safe_params[k] = json.dumps(v)
                else:
                    safe_params[k] = v
            mlflow.log_params(safe_params)

            # ── Log metrics ────────────────────────────────────
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            # ── Log training curves ────────────────────────────
            if history is not None and hasattr(history, "history"):
                for epoch_idx in range(len(history.history.get("loss", []))):
                    for key in history.history:
                        mlflow.log_metric(
                            f"epoch_{key}",
                            history.history[key][epoch_idx],
                            step=epoch_idx,
                        )

            # ── Tags ───────────────────────────────────────────
            mlflow.set_tags({
                "dataset": "movielens-32m",
                "task": "recommendation",
                "model_type": model_type,
                "split_strategy": "temporal_per_user",
                "git_commit": _get_git_commit_hash(),
                "framework": "tensorflow",
            })

            # ── Log model artifact ─────────────────────────────
            if model_path and os.path.exists(model_path):
                mlflow.log_artifacts(model_path, artifact_path="model")

            # ── Register model ─────────────────────────────────
            if register:
                try:
                    model_uri = f"runs:/{run_id}/model"
                    mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
                    logger.info(
                        "Model registered: %s (run: %s)",
                        REGISTERED_MODEL_NAME, run_id,
                    )
                except Exception as e:
                    logger.warning("Model registration failed: %s", e)

            logger.info("MLflow run complete: %s", run_id)
            return run_id

    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        return None


def promote_model(
    run_id: Optional[str],
    current_metrics: Dict[str, float],
    production_metrics: Dict[str, float],
    metric_name: str = PROMOTION_METRIC,
) -> Tuple[bool, str]:
    """
    Promote model to Production ONLY if it beats current Production model.

    This is the quality gate that prevents bad models from reaching production.
    The Airflow DAG calls this after evaluation + drift check pass.

    Args:
        run_id: MLflow run ID of candidate model
        current_metrics: Metrics of the candidate model
        production_metrics: Metrics of the current Production model
        metric_name: Metric to use for comparison

    Returns:
        Tuple of (promoted: bool, message: str)
    """
    current_value = current_metrics.get(metric_name)
    production_value = production_metrics.get(metric_name, 0)

    if current_value is None:
        return False, f"Blocked: metric '{metric_name}' not found in current metrics"

    # Check if candidate beats production
    if PROMOTION_DIRECTION == "higher":
        should_promote = current_value > production_value
    else:
        should_promote = current_value < production_value

    if should_promote:
        message = (
            f"✓ PROMOTED: {metric_name} {current_value:.4f} > "
            f"production {production_value:.4f}"
        )
        logger.info(message)

        # Attempt MLflow registry transition
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest.version,
                    stage="Production",
                )
                message += f" (v{latest.version} → Production)"
        except Exception as e:
            logger.warning("MLflow registry update failed: %s", e)
            message += " (MLflow registry update skipped)"

        return True, message
    else:
        message = (
            f"✗ BLOCKED: {metric_name} {current_value:.4f} <= "
            f"production {production_value:.4f}"
        )
        logger.info(message)
        return False, message


def get_production_metrics() -> Dict[str, float]:
    """
    Retrieve metrics from the current Production model in MLflow registry.

    Returns:
        Dict of metric name → value, or empty dict if no Production model
    """
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        # Find Production version
        for v in versions:
            if v.current_stage == "Production":
                run = client.get_run(v.run_id)
                return {k: float(v) for k, v in run.data.metrics.items()}

        logger.info("No Production model found — first deployment")
        return {}

    except Exception as e:
        logger.warning("Could not retrieve production metrics: %s", e)
        return {}


def list_experiments() -> List[Dict]:
    """
    List all runs in the experiment with key metrics.

    Returns:
        List of dicts with run_id, model_type, and metrics
    """
    try:
        import mlflow

        setup_mlflow()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        if experiment is None:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.`NDCG@10` DESC"],
        )

        results = []
        for _, row in runs.iterrows():
            results.append({
                "run_id": row["run_id"],
                "model_type": row.get("tags.model_type", "unknown"),
                "NDCG@10": row.get("metrics.NDCG@10"),
                "HR@10": row.get("metrics.HR@10"),
                "status": row["status"],
            })

        return results

    except Exception as e:
        logger.warning("Could not list experiments: %s", e)
        return []


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("MLflow Tracking — Validation")
    print("=" * 50)

    # Test promotion logic (no MLflow server needed)
    promoted, msg = promote_model(
        run_id=None,
        current_metrics={"NDCG@10": 0.45, "HR@10": 0.68},
        production_metrics={"NDCG@10": 0.40, "HR@10": 0.65},
    )
    print(f"Test 1 (should promote):  {msg}")
    assert promoted, "Should have promoted — NDCG improved"

    promoted, msg = promote_model(
        run_id=None,
        current_metrics={"NDCG@10": 0.38, "HR@10": 0.60},
        production_metrics={"NDCG@10": 0.40, "HR@10": 0.65},
    )
    print(f"Test 2 (should block):    {msg}")
    assert not promoted, "Should have blocked — NDCG worse"

    promoted, msg = promote_model(
        run_id=None,
        current_metrics={"NDCG@10": 0.50},
        production_metrics={},  # No production model yet
    )
    print(f"Test 3 (first deploy):    {msg}")
    assert promoted, "Should promote — first deployment"

    print(f"\n✓ Git commit: {_get_git_commit_hash()}")
    print(f"✓ MLflow tracking validation passed")
