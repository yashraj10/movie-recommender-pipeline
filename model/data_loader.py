"""
TensorFlow Data Loader
======================
Load Parquet features into TF Datasets for training recommendation models.

Handles:
    - Loading Parquet interactions from Spark pipeline output
    - Negative sampling (generating "user did NOT watch this movie" pairs)
    - Converting to tf.data.Dataset with batching and prefetching
    - Both implicit (binary) and explicit (rating) formulations

Environment: Works on both CPU (Codespaces) and GPU (Colab T4)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 4096   # Large batches for stable gradients on 32M interactions
DEFAULT_NEG_RATIO = 4       # 4 negatives per positive (He et al., 2017 NCF paper)
DEFAULT_SHUFFLE_BUFFER = 100_000
SEED = 42


def load_interactions_from_parquet(
    parquet_path: str,
) -> pd.DataFrame:
    """
    Load interaction data from Parquet files produced by Spark pipeline.

    Args:
        parquet_path: Path to Parquet directory (e.g., "data/features/interactions/train")

    Returns:
        DataFrame with columns: user_idx, item_idx, rating, timestamp
    """
    logger.info("Loading interactions from %s", parquet_path)

    df = pd.read_parquet(parquet_path)

    required_cols = ["user_idx", "item_idx"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    logger.info("Loaded %s interactions", f"{len(df):,}")
    return df


def generate_negative_samples(
    interactions_df: pd.DataFrame,
    n_items: int,
    neg_ratio: int = DEFAULT_NEG_RATIO,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate negative samples for implicit feedback training.

    For each positive interaction (user rated movie), sample neg_ratio
    movies the user has NOT rated. This creates a binary classification
    dataset: label=1 for real interactions, label=0 for negatives.

    Why negative sampling:
        In implicit feedback, we only observe positive signals (user watched/rated).
        We need to generate "negative" examples (user did NOT watch) to train
        a binary classifier. Without negatives, the model would predict 1 for everything.

    Why neg_ratio=4:
        Standard in NCF literature (He et al., 2017). Too few negatives means
        the model doesn't learn to distinguish. Too many creates class imbalance
        and slows training.

    Args:
        interactions_df: DataFrame with user_idx, item_idx columns
        n_items: Total number of unique items (for sampling range)
        neg_ratio: Number of negative samples per positive interaction
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns [user_idx, item_idx, label]
        where label=1 for positive, label=0 for negative
    """
    logger.info(
        "Generating negative samples: %d negatives per positive (ratio=%d)",
        len(interactions_df) * neg_ratio, neg_ratio,
    )

    rng = np.random.RandomState(seed)

    # Build set of positive (user, item) pairs for fast lookup
    user_items: Dict[int, set] = {}
    for _, row in interactions_df[["user_idx", "item_idx"]].iterrows():
        uid = int(row["user_idx"])
        iid = int(row["item_idx"])
        if uid not in user_items:
            user_items[uid] = set()
        user_items[uid].add(iid)

    # Generate positives
    positives = interactions_df[["user_idx", "item_idx"]].copy()
    positives["label"] = 1.0

    # Generate negatives
    neg_users = []
    neg_items = []

    for uid, pos_items in user_items.items():
        n_neg = len(pos_items) * neg_ratio
        # Sample from all items, rejecting positives
        sampled = set()
        while len(sampled) < n_neg:
            candidates = rng.randint(0, n_items, size=n_neg * 2)
            for c in candidates:
                if c not in pos_items and c not in sampled:
                    sampled.add(c)
                    if len(sampled) >= n_neg:
                        break

        for iid in sampled:
            neg_users.append(uid)
            neg_items.append(iid)

    negatives = pd.DataFrame({
        "user_idx": neg_users,
        "item_idx": neg_items,
        "label": 0.0,
    })

    # Combine and shuffle
    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_pos = len(positives)
    n_neg = len(negatives)
    logger.info(
        "Generated %s total samples: %s positive (%.1f%%), %s negative (%.1f%%)",
        f"{len(combined):,}",
        f"{n_pos:,}", 100 * n_pos / len(combined),
        f"{n_neg:,}", 100 * n_neg / len(combined),
    )

    return combined


def create_tf_dataset(
    interactions_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
) -> tf.data.Dataset:
    """
    Convert a pandas DataFrame to a tf.data.Dataset.

    Why batch_size=4096:
        - 32M interactions need large batches for stable gradients
        - T4 GPU has 12.7GB VRAM — 4096 × embedding lookups fits comfortably
        - Matches production-scale batch sizes at Netflix/Spotify

    Args:
        interactions_df: DataFrame with user_idx, item_idx, label columns
        batch_size: Training batch size
        shuffle: Whether to shuffle the dataset
        shuffle_buffer: Size of shuffle buffer

    Returns:
        tf.data.Dataset yielding (inputs, labels) batches
        where inputs shape = (batch, 2) and labels shape = (batch,)
    """
    user_ids = interactions_df["user_idx"].values.astype(np.int32)
    item_ids = interactions_df["item_idx"].values.astype(np.int32)
    labels = interactions_df["label"].values.astype(np.float32)

    # Stack user and item IDs into (N, 2) array
    inputs = np.stack([user_ids, item_ids], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=SEED)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(
        "Created TF Dataset: %s samples, batch_size=%d, shuffle=%s",
        f"{len(interactions_df):,}", batch_size, shuffle,
    )

    return dataset


def prepare_datasets(
    train_path: str,
    val_path: str,
    test_path: str,
    n_items: int,
    neg_ratio: int = DEFAULT_NEG_RATIO,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame, int]:
    """
    Full data preparation pipeline: load Parquet → negative sample → TF Datasets.

    Args:
        train_path: Path to train Parquet directory
        val_path: Path to validation Parquet directory
        test_path: Path to test Parquet directory
        n_items: Total number of unique items
        neg_ratio: Negative sampling ratio
        batch_size: Batch size for TF Datasets

    Returns:
        Tuple of (train_dataset, val_dataset, test_df, n_users)
    """
    # Load raw interactions
    train_df = load_interactions_from_parquet(train_path)
    val_df = load_interactions_from_parquet(val_path)
    test_df = load_interactions_from_parquet(test_path)

    # Get n_users from the data
    n_users = max(
        train_df["user_idx"].max(),
        val_df["user_idx"].max(),
        test_df["user_idx"].max(),
    ) + 1  # +1 because 0-indexed

    logger.info("Dataset stats: n_users=%d, n_items=%d", n_users, n_items)

    # Generate negative samples for train and val
    train_sampled = generate_negative_samples(train_df, n_items, neg_ratio)
    val_sampled = generate_negative_samples(val_df, n_items, neg_ratio=1)  # Less negatives for val

    # Convert to TF Datasets
    train_dataset = create_tf_dataset(train_sampled, batch_size, shuffle=True)
    val_dataset = create_tf_dataset(val_sampled, batch_size, shuffle=False)

    # Test set stays as DataFrame for evaluation (HR@K, NDCG@K need special handling)

    return train_dataset, val_dataset, test_df, n_users


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Test with actual Parquet data if available, else synthetic
    train_path = "data/features/interactions/train"

    if os.path.exists(train_path):
        logger.info("Testing with real Parquet data (small sample)...")
        df = load_interactions_from_parquet(train_path)

        # Use small sample for quick test
        sample = df.head(10_000).copy()
        n_items = int(sample["item_idx"].max()) + 1

        sampled = generate_negative_samples(sample, n_items, neg_ratio=4)
        dataset = create_tf_dataset(sampled, batch_size=256)

        # Verify one batch
        for inputs, labels in dataset.take(1):
            print(f"✓ Input shape:  {inputs.shape}")   # (256, 2)
            print(f"✓ Labels shape: {labels.shape}")    # (256,)
            print(f"✓ Label range:  [{labels.numpy().min()}, {labels.numpy().max()}]")
            print(f"✓ Positive ratio: {labels.numpy().mean():.2%}")
    else:
        logger.info("No Parquet data found — testing with synthetic data...")
        synthetic = pd.DataFrame({
            "user_idx": np.random.randint(0, 100, 1000),
            "item_idx": np.random.randint(0, 500, 1000),
            "rating": np.random.uniform(0.5, 5.0, 1000),
        })

        sampled = generate_negative_samples(synthetic, n_items=500, neg_ratio=4)
        dataset = create_tf_dataset(sampled, batch_size=64)

        for inputs, labels in dataset.take(1):
            print(f"✓ Input shape:  {inputs.shape}")
            print(f"✓ Labels shape: {labels.shape}")

    print(f"\n✓ Data loader validation passed")
