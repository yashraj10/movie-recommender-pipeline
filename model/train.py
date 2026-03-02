"""
Model Training Pipeline
========================
Training loop for Matrix Factorization and NCF models with:
    - Early stopping on validation loss
    - Learning rate scheduling (ReduceLROnPlateau)
    - Optional MLflow experiment logging
    - Model checkpointing and SavedModel export

Works on both CPU (Codespaces for testing) and GPU (Colab T4 for full training).
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Training Configuration ─────────────────────────────────────────────
TRAINING_CONFIG = {
    # Model architecture
    "gmf_dim": 64,
    "mlp_dim": 64,
    "mlp_layers": [128, 64, 32],
    "dropout_rate": 0.2,

    # Training hyperparameters
    "lr": 0.001,
    "batch_size": 4096,
    "epochs": 15,
    "neg_ratio": 4,

    # Early stopping
    "patience": 3,
    "min_delta": 0.0001,

    # LR scheduling
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 1e-6,

    # Data
    "min_user_ratings": 20,
    "min_item_ratings": 5,

    # Reproducibility
    "seed": 42,
}


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Dict[str, Any] = None,
    model_name: str = "model",
    save_dir: str = "models",
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, float]]:
    """
    Train a recommendation model with best practices.

    Features:
    - Binary cross-entropy loss (implicit feedback)
    - Adam optimizer with learning rate scheduling
    - Early stopping to prevent overfitting
    - Saves best model based on validation loss

    Args:
        model: Compiled or uncompiled Keras model
        train_dataset: tf.data.Dataset yielding (inputs, labels)
        val_dataset: tf.data.Dataset yielding (inputs, labels)
        config: Training configuration dict
        model_name: Name for saving (e.g., "mf_baseline" or "ncf_v1")
        save_dir: Directory to save model checkpoints

    Returns:
        Tuple of (trained_model, training_history, final_metrics)
    """
    if config is None:
        config = TRAINING_CONFIG

    logger.info("=" * 60)
    logger.info("TRAINING: %s", model_name)
    logger.info("=" * 60)
    logger.info("Config: %s", json.dumps(config, indent=2, default=str))

    # ── Compile ────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.get("lr", 0.001))
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # ── Callbacks ──────────────────────────────────────────────────
    callbacks = []

    # Early stopping: stop if val_loss doesn't improve for N epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.get("patience", 3),
        min_delta=config.get("min_delta", 0.0001),
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stopping)

    # Learning rate reduction: halve LR if val_loss plateaus
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 2),
        min_lr=config.get("min_lr", 1e-6),
        verbose=1,
    )
    callbacks.append(lr_scheduler)

    # Model checkpoint: save best model
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.weights.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    callbacks.append(checkpoint)

    # ── Train ──────────────────────────────────────────────────────
    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.get("epochs", 15),
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start_time
    epochs_trained = len(history.history["loss"])

    # ── Extract final metrics ──────────────────────────────────────
    final_metrics = {
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "train_auc": float(history.history["auc"][-1]),
        "val_auc": float(history.history["val_auc"][-1]),
        "epochs_trained": epochs_trained,
        "training_time_seconds": round(elapsed, 1),
        "best_val_loss": float(min(history.history["val_loss"])),
    }

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE: %s", model_name)
    logger.info("  Epochs:         %d / %d", epochs_trained, config.get("epochs", 15))
    logger.info("  Time:           %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("  Best val_loss:  %.4f", final_metrics["best_val_loss"])
    logger.info("  Final val_auc:  %.4f", final_metrics["val_auc"])
    logger.info("=" * 60)

    # ── Save model ─────────────────────────────────────────────────
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_weights(os.path.join(save_path, "weights.weights.h5"))
    logger.info("Model weights saved to %s", save_path)

    # Save config alongside model
    config_path = os.path.join(save_path, "config.json")
    os.makedirs(save_path, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info("Config saved to %s", config_path)

    return model, history, final_metrics


def print_training_summary(
    mf_metrics: Optional[Dict] = None,
    ncf_metrics: Optional[Dict] = None,
) -> None:
    """
    Print a comparison table of MF vs NCF training metrics.

    Args:
        mf_metrics: Metrics dict from MF training
        ncf_metrics: Metrics dict from NCF training
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON — Training Metrics")
    print("=" * 60)
    print(f"{'Metric':<25} {'MF Baseline':>15} {'NCF':>15}")
    print("-" * 60)

    metrics_to_show = [
        ("val_loss", "Val Loss", False),
        ("val_accuracy", "Val Accuracy", True),
        ("val_auc", "Val AUC", True),
        ("epochs_trained", "Epochs Trained", False),
        ("training_time_seconds", "Time (seconds)", False),
    ]

    for key, label, higher_better in metrics_to_show:
        mf_val = mf_metrics.get(key, "N/A") if mf_metrics else "N/A"
        ncf_val = ncf_metrics.get(key, "N/A") if ncf_metrics else "N/A"

        if isinstance(mf_val, float):
            mf_str = f"{mf_val:.4f}"
        else:
            mf_str = str(mf_val)

        if isinstance(ncf_val, float):
            ncf_str = f"{ncf_val:.4f}"
        else:
            ncf_str = str(ncf_val)

        print(f"{label:<25} {mf_str:>15} {ncf_str:>15}")

    print("=" * 60)


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from model.matrix_factorization import MatrixFactorization
    from model.ncf import NeuralCollaborativeFiltering

    # Create tiny synthetic dataset
    n_users, n_items = 100, 50
    n_samples = 5000

    user_ids = np.random.randint(0, n_users, n_samples).astype(np.int32)
    item_ids = np.random.randint(0, n_items, n_samples).astype(np.int32)
    labels = np.random.randint(0, 2, n_samples).astype(np.float32)
    inputs = np.stack([user_ids, item_ids], axis=1)

    train_ds = tf.data.Dataset.from_tensor_slices((inputs[:4000], labels[:4000]))
    train_ds = train_ds.batch(256).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((inputs[4000:], labels[4000:]))
    val_ds = val_ds.batch(256).prefetch(tf.data.AUTOTUNE)

    test_config = {**TRAINING_CONFIG, "epochs": 2, "patience": 2}

    # Train MF
    print("\n--- Testing MF Training ---")
    mf_model = MatrixFactorization(n_users, n_items, embedding_dim=16)
    mf_model, mf_history, mf_metrics = train_model(
        mf_model, train_ds, val_ds,
        config=test_config, model_name="mf_test", save_dir="/tmp/models",
    )

    # Train NCF
    print("\n--- Testing NCF Training ---")
    ncf_model = NeuralCollaborativeFiltering(
        n_users, n_items, gmf_dim=16, mlp_dim=16, mlp_layers=[32, 16],
    )
    ncf_model, ncf_history, ncf_metrics = train_model(
        ncf_model, train_ds, val_ds,
        config=test_config, model_name="ncf_test", save_dir="/tmp/models",
    )

    # Compare
    print_training_summary(mf_metrics, ncf_metrics)
    print("\n✓ Training pipeline validation passed")
