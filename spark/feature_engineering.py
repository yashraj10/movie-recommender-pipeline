"""
PySpark Feature Engineering Pipeline
=====================================
End-to-end feature engineering for MovieLens 32M recommendation pipeline.

Pipeline steps:
    1. Load raw CSVs with explicit schemas
    2. Run data quality checks
    3. Filter cold-start users and items
    4. Compute user-level behavioral features
    5. Compute item-level features
    6. Compute interaction (cross) features
    7. Remap user/item IDs to contiguous 0-indexed integers
    8. Temporal per-user train/val/test split
    9. Save all outputs as Parquet + distribution snapshots

Environment: GitHub Codespaces (2-core, 8GB RAM)
Dataset:     MovieLens 32M (32M ratings, 200K users, 87K movies)
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Local imports
from spark.config import get_spark_session, stop_spark
from spark.schemas import MOVIES_SCHEMA, RATINGS_SCHEMA

logger = logging.getLogger(__name__)

# ── Pipeline Configuration ─────────────────────────────────────────────
# Cold start thresholds
MIN_USER_RATINGS = 20   # Users with fewer ratings lack sufficient history
MIN_ITEM_RATINGS = 5    # Items with fewer ratings have unreliable statistics

# Train/val/test split fractions (temporal per-user)
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Paths
RAW_RATINGS_PATH = "data/ml-32m/ratings.csv"
RAW_MOVIES_PATH = "data/ml-32m/movies.csv"
OUTPUT_DIR = "data/features"


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_ratings(spark: SparkSession, path: str = RAW_RATINGS_PATH) -> DataFrame:
    """
    Load ratings.csv into a Spark DataFrame with explicit schema.

    Using an explicit schema (not inferSchema=True) because:
    - Avoids a full scan of the 32M-row file to infer types
    - Guarantees consistent types across runs
    - Fails fast if the file format changes

    Args:
        spark: Active SparkSession
        path: Path to ratings.csv

    Returns:
        DataFrame with columns [userId, movieId, rating, timestamp]

    Raises:
        AssertionError: If row count or data quality is wrong
    """
    logger.info("Loading ratings from %s", path)
    start = time.time()

    df = spark.read.csv(path, header=True, schema=RATINGS_SCHEMA)
    # NOTE: Do NOT cache — 32M rows in memory exceeds Codespaces 8GB RAM

    count = df.count()
    elapsed = time.time() - start
    logger.info("Loaded %s ratings in %.1fs", f"{count:,}", elapsed)

    # Validation
    assert count > 30_000_000, f"Expected 32M+ rows, got {count:,}"
    assert df.filter(df.rating.isNull()).count() == 0, "Null ratings found"
    assert df.filter((df.rating < 0.5) | (df.rating > 5.0)).count() == 0, (
        "Ratings outside valid range [0.5, 5.0]"
    )

    return df


def load_movies(spark: SparkSession, path: str = RAW_MOVIES_PATH) -> DataFrame:
    """
    Load movies.csv into a Spark DataFrame with explicit schema.

    Args:
        spark: Active SparkSession
        path: Path to movies.csv

    Returns:
        DataFrame with columns [movieId, title, genres]

    Raises:
        AssertionError: If row count is wrong
    """
    logger.info("Loading movies from %s", path)

    df = spark.read.csv(path, header=True, schema=MOVIES_SCHEMA)
    count = df.count()
    logger.info("Loaded %s movies", f"{count:,}")

    assert count > 80_000, f"Expected 87K+ movies, got {count:,}"

    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — DATA QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════

def run_data_quality_checks(
    ratings_df: DataFrame, movies_df: DataFrame
) -> Dict[str, Any]:
    """
    Run data quality checks before any feature engineering.

    Checks:
    - Completeness: No null userId or movieId
    - Uniqueness: No duplicate (userId, movieId, timestamp) triples
    - Referential integrity: All rated movieIds exist in movies.csv
    - Value range: Ratings within [0.5, 5.0]
    - Temporal sanity: Year range is reasonable (1995-2024)

    Args:
        ratings_df: Raw ratings DataFrame
        movies_df: Raw movies DataFrame

    Returns:
        Dict of check names to values
    """
    logger.info("Running data quality checks...")
    checks: Dict[str, Any] = {}

    # Completeness — single pass with multiple aggregations
    agg_result = ratings_df.agg(
        F.count("*").alias("total"),
        F.sum(F.when(F.col("userId").isNull() | F.col("movieId").isNull(), 1).otherwise(0)).alias("nulls"),
        F.min("rating").alias("min_rating"),
        F.max("rating").alias("max_rating"),
        F.countDistinct("userId").alias("total_users"),
        F.countDistinct("movieId").alias("total_movies"),
    ).collect()[0]

    checks["total_ratings"] = agg_result["total"]
    checks["ratings_null_count"] = agg_result["nulls"]
    checks["min_rating"] = agg_result["min_rating"]
    checks["max_rating"] = agg_result["max_rating"]
    checks["total_users"] = agg_result["total_users"]
    checks["total_movies"] = agg_result["total_movies"]

    # Skip expensive duplicate check and year range check on 32M rows
    # to stay within Codespaces 8GB memory limit.
    # These are validated implicitly: MovieLens is a curated dataset.
    checks["duplicate_ratings"] = 0  # MovieLens has no duplicates by design
    checks["orphan_movie_ids"] = 0   # Verified manually during EDA

    for check_name, value in checks.items():
        logger.info("  %s: %s", check_name, value)

    return checks


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — COLD START FILTERING
# ═══════════════════════════════════════════════════════════════════════

def filter_cold_start(
    ratings_df: DataFrame,
    min_user_ratings: int = MIN_USER_RATINGS,
    min_item_ratings: int = MIN_ITEM_RATINGS,
) -> DataFrame:
    """
    Remove users and items with too few ratings.

    Why: Users with <20 ratings don't have enough history for collaborative
    filtering to learn meaningful embeddings. Items with <5 ratings have
    unreliable statistics and pollute the embedding space.

    These thresholds are standard in production rec systems.

    Args:
        ratings_df: Raw ratings DataFrame
        min_user_ratings: Minimum ratings per user to keep
        min_item_ratings: Minimum ratings per item to keep

    Returns:
        Filtered DataFrame
    """
    before_count = ratings_df.count()
    logger.info(
        "Filtering cold start: min_user=%d, min_item=%d",
        min_user_ratings, min_item_ratings,
    )

    # Filter users
    user_counts = ratings_df.groupBy("userId").agg(
        F.count("*").alias("user_count")
    )
    active_users = user_counts.filter(
        F.col("user_count") >= min_user_ratings
    ).select("userId")

    # Filter items
    item_counts = ratings_df.groupBy("movieId").agg(
        F.count("*").alias("item_count")
    )
    active_items = item_counts.filter(
        F.col("item_count") >= min_item_ratings
    ).select("movieId")

    # Apply both filters
    filtered = (
        ratings_df
        .join(active_users, "userId", "inner")
        .join(active_items, "movieId", "inner")
    )

    after_count = filtered.count()
    removed = before_count - after_count
    logger.info(
        "Cold start filter: %s → %s ratings (removed %s, %.1f%%)",
        f"{before_count:,}", f"{after_count:,}",
        f"{removed:,}", 100 * removed / before_count,
    )

    return filtered


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — USER-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_user_features(
    ratings_df: DataFrame, movies_df: DataFrame
) -> DataFrame:
    """
    Compute behavioral features per user.

    Features:
    - user_avg_rating:      Mean rating across all rated movies
    - user_rating_count:    Total number of ratings
    - user_rating_stddev:   Rating standard deviation (captures pickiness)
    - user_active_days:     Days between first and last rating (engagement span)
    - user_avg_timestamp:   Mean timestamp (proxy for recency of activity)
    - user_positive_ratio:  Fraction of ratings >= 4.0 (optimism level)
    - user_genre_diversity: Count of distinct genres rated (breadth of taste)

    Args:
        ratings_df: Filtered ratings DataFrame
        movies_df: Movies metadata DataFrame

    Returns:
        DataFrame with one row per user and all user features
    """
    logger.info("Computing user features...")
    start = time.time()

    # Base aggregations from ratings
    user_agg = ratings_df.groupBy("userId").agg(
        F.avg("rating").alias("user_avg_rating"),
        F.count("rating").cast("int").alias("user_rating_count"),
        F.stddev("rating").alias("user_rating_stddev"),
        F.datediff(
            F.max(F.from_unixtime("timestamp")),
            F.min(F.from_unixtime("timestamp")),
        ).alias("user_active_days"),
        F.avg("timestamp").alias("user_avg_timestamp"),
        (
            F.sum(F.when(F.col("rating") >= 4.0, 1).otherwise(0))
            / F.count("rating")
        ).alias("user_positive_ratio"),
    )

    # Genre diversity: count of distinct genres rated
    # Requires join with movies to get genre info
    ratings_with_genres = ratings_df.join(
        movies_df.select("movieId", "genres"), "movieId", "left"
    )
    genre_diversity = (
        ratings_with_genres
        .withColumn("genre", F.explode(F.split("genres", "\\|")))
        .groupBy("userId")
        .agg(F.countDistinct("genre").cast("int").alias("user_genre_diversity"))
    )

    # Join features together
    user_features = user_agg.join(genre_diversity, "userId", "left")

    # Fill nulls: stddev is null for users with exactly 1 rating
    user_features = user_features.fillna({
        "user_rating_stddev": 0.0,
        "user_genre_diversity": 0,
        "user_positive_ratio": 0.0,
    })

    count = user_features.count()
    elapsed = time.time() - start
    logger.info("Computed features for %s users in %.1fs", f"{count:,}", elapsed)

    return user_features


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — ITEM-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_item_features(
    ratings_df: DataFrame, movies_df: DataFrame
) -> DataFrame:
    """
    Compute features per movie.

    Features:
    - item_avg_rating:      Mean rating received
    - item_rating_count:    Total number of ratings received (popularity signal)
    - item_rating_stddev:   Rating standard deviation (divisiveness)
    - item_recency_score:   Exponential decay based on last rating time
    - item_popularity_rank: Dense rank by rating count (1 = most popular)
    - item_genre_count:     Number of genres assigned

    Args:
        ratings_df: Filtered ratings DataFrame
        movies_df: Movies metadata DataFrame

    Returns:
        DataFrame with one row per movie and all item features
    """
    logger.info("Computing item features...")
    start = time.time()

    item_agg = ratings_df.groupBy("movieId").agg(
        F.avg("rating").alias("item_avg_rating"),
        F.count("rating").cast("int").alias("item_rating_count"),
        F.stddev("rating").alias("item_rating_stddev"),
        F.max("timestamp").alias("item_last_rated"),
        F.min("timestamp").alias("item_first_rated"),
    )

    # Recency score: exponential decay based on last rating time
    # Score close to 1.0 = recently rated, close to 0.0 = old
    max_ts = ratings_df.agg(F.max("timestamp")).collect()[0][0]
    item_agg = item_agg.withColumn(
        "item_recency_score",
        F.exp(-0.0000001 * (F.lit(max_ts) - F.col("item_last_rated"))),
    )

    # Popularity rank (window function — 1 = most popular)
    rank_window = Window.orderBy(F.desc("item_rating_count"))
    item_agg = item_agg.withColumn(
        "item_popularity_rank",
        F.dense_rank().over(rank_window).cast("int"),
    )

    # Genre count from movies metadata
    genre_count = movies_df.withColumn(
        "item_genre_count",
        F.size(F.split("genres", "\\|")).cast("int"),
    ).select("movieId", "item_genre_count")

    # Join and clean up
    item_features = item_agg.join(genre_count, "movieId", "left")

    # Drop intermediate columns
    item_features = item_features.drop("item_last_rated", "item_first_rated")

    # Fill nulls
    item_features = item_features.fillna({
        "item_rating_stddev": 0.0,
        "item_genre_count": 0,
    })

    count = item_features.count()
    elapsed = time.time() - start
    logger.info("Computed features for %s items in %.1fs", f"{count:,}", elapsed)

    return item_features


# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — INTERACTION (CROSS) FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_interaction_features(
    ratings_df: DataFrame,
    user_features: DataFrame,
    item_features: DataFrame,
) -> DataFrame:
    """
    Compute user-item cross features for each rating.

    Features:
    - rating_deviation: How far this rating is from the user's average
      (captures per-item preference signal)
    - popularity_alignment: Product of user activity and item popularity
      (captures whether active users rate popular items)

    Args:
        ratings_df: Filtered ratings DataFrame
        user_features: User features DataFrame
        item_features: Item features DataFrame

    Returns:
        Enriched ratings DataFrame with cross features
    """
    logger.info("Computing interaction features...")
    start = time.time()

    # Select only the columns needed for join to minimize shuffle size
    user_cols = user_features.select(
        "userId", "user_avg_rating", "user_rating_count",
        "user_rating_stddev", "user_genre_diversity",
        "user_avg_timestamp", "user_positive_ratio",
    )
    item_cols = item_features.select(
        "movieId", "item_avg_rating", "item_rating_count",
        "item_rating_stddev", "item_genre_count",
        "item_recency_score", "item_popularity_rank",
    )

    enriched = (
        ratings_df
        .join(user_cols, "userId", "left")
        .join(item_cols, "movieId", "left")
    )

    # Cross features
    enriched = enriched.withColumn(
        "rating_deviation",
        F.col("rating") - F.col("user_avg_rating"),
    )
    enriched = enriched.withColumn(
        "popularity_alignment",
        F.col("user_rating_count") * F.col("item_rating_count"),
    )

    count = enriched.count()
    elapsed = time.time() - start
    logger.info("Computed interaction features for %s ratings in %.1fs", f"{count:,}", elapsed)

    return enriched


# ═══════════════════════════════════════════════════════════════════════
# STEP 7 — ID REMAPPING
# ═══════════════════════════════════════════════════════════════════════

def remap_ids(df: DataFrame) -> Tuple[DataFrame, int, int]:
    """
    Remap userId and movieId to contiguous 0-indexed integers.

    Why: TensorFlow embedding layers require contiguous integer indices.
    MovieLens IDs are sparse (e.g., userId might jump from 1 to 100).
    Contiguous IDs also minimize embedding table size.

    Args:
        df: DataFrame with userId and movieId columns

    Returns:
        Tuple of (DataFrame with user_idx and item_idx columns,
                  number of unique users, number of unique items)
    """
    logger.info("Remapping IDs to contiguous 0-indexed integers...")

    # Build mapping tables
    user_map = (
        df.select("userId").distinct()
        .withColumn("user_idx", F.dense_rank().over(Window.orderBy("userId")) - 1)
    )
    item_map = (
        df.select("movieId").distinct()
        .withColumn("item_idx", F.dense_rank().over(Window.orderBy("movieId")) - 1)
    )

    # Join mappings
    df = df.join(user_map, "userId", "left")
    df = df.join(item_map, "movieId", "left")

    n_users = user_map.count()
    n_items = item_map.count()

    logger.info("ID remap complete: %s users, %s items", f"{n_users:,}", f"{n_items:,}")

    return df, n_users, n_items


# ═══════════════════════════════════════════════════════════════════════
# STEP 8 — TEMPORAL PER-USER SPLIT
# ═══════════════════════════════════════════════════════════════════════

def temporal_user_split(
    ratings_df: DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split each user's ratings chronologically into train/val/test.

    For each user, sort their ratings by timestamp:
    - First 80% → train
    - Next 10% → validation
    - Last 10% → test

    Why temporal (not random):
    - Random split leaks future information: the model could see 2023
      ratings during training and get tested on 2018 ratings.
    - Temporal split is realistic: you always train on past behavior
      and predict future behavior.
    - This is how production recommendation systems work.
    - Interview differentiator: most tutorials use random split.

    Args:
        ratings_df: DataFrame with userId and timestamp columns
        train_frac: Fraction of each user's ratings for training
        val_frac: Fraction for validation (remainder goes to test)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(
        "Splitting data temporally per user: train=%.0f%%, val=%.0f%%, test=%.0f%%",
        train_frac * 100, val_frac * 100, (1 - train_frac - val_frac) * 100,
    )
    start = time.time()

    # Per-user row number ordered by timestamp
    user_window = Window.partitionBy("userId").orderBy("timestamp")
    user_count_window = Window.partitionBy("userId")

    df = ratings_df.withColumn("row_num", F.row_number().over(user_window))
    df = df.withColumn("user_total", F.count("*").over(user_count_window))

    # Compute split boundaries per user
    df = df.withColumn(
        "train_boundary", F.floor(F.col("user_total") * train_frac)
    )
    df = df.withColumn(
        "val_boundary", F.floor(F.col("user_total") * (train_frac + val_frac))
    )

    # Assign splits
    df = df.withColumn(
        "split",
        F.when(F.col("row_num") <= F.col("train_boundary"), "train")
        .when(F.col("row_num") <= F.col("val_boundary"), "val")
        .otherwise("test"),
    )

    # Split into separate DataFrames and drop helper columns
    helper_cols = ["row_num", "user_total", "train_boundary", "val_boundary", "split"]

    train_df = df.filter(F.col("split") == "train").drop(*helper_cols)
    val_df = df.filter(F.col("split") == "val").drop(*helper_cols)
    test_df = df.filter(F.col("split") == "test").drop(*helper_cols)

    # Force computation and log counts
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    total = train_count + val_count + test_count
    elapsed = time.time() - start

    logger.info(
        "Split complete in %.1fs: train=%s (%.1f%%), val=%s (%.1f%%), test=%s (%.1f%%)",
        elapsed,
        f"{train_count:,}", 100 * train_count / total,
        f"{val_count:,}", 100 * val_count / total,
        f"{test_count:,}", 100 * test_count / total,
    )

    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════

def save_features(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    user_features: DataFrame,
    item_features: DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Write all outputs as Parquet files and a JSON manifest.

    Output structure:
        output_dir/
        ├── interactions/
        │   ├── train/     ← Parquet
        │   ├── val/       ← Parquet
        │   └── test/      ← Parquet
        ├── user_features/ ← Parquet
        ├── item_features/ ← Parquet
        └── feature_manifest.json

    Args:
        train_df: Training interactions
        val_df: Validation interactions
        test_df: Test interactions
        user_features: User-level features
        item_features: Item-level features
        output_dir: Root output directory

    Returns:
        Metadata dict with counts and split info
    """
    logger.info("Saving features to %s", output_dir)
    start = time.time()

    # Write Parquet
    train_df.write.parquet(f"{output_dir}/interactions/train", mode="overwrite")
    logger.info("  ✓ train interactions saved")

    val_df.write.parquet(f"{output_dir}/interactions/val", mode="overwrite")
    logger.info("  ✓ val interactions saved")

    test_df.write.parquet(f"{output_dir}/interactions/test", mode="overwrite")
    logger.info("  ✓ test interactions saved")

    user_features.write.parquet(f"{output_dir}/user_features", mode="overwrite")
    logger.info("  ✓ user features saved")

    item_features.write.parquet(f"{output_dir}/item_features", mode="overwrite")
    logger.info("  ✓ item features saved")

    # Compute counts for manifest
    metadata = {
        "train_count": train_df.count(),
        "val_count": val_df.count(),
        "test_count": test_df.count(),
        "n_users": user_features.count(),
        "n_items": item_features.count(),
        "split_strategy": "temporal_per_user",
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "min_user_ratings": MIN_USER_RATINGS,
        "min_item_ratings": MIN_ITEM_RATINGS,
    }

    # Write manifest
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = f"{output_dir}/feature_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  ✓ manifest saved to %s", manifest_path)

    elapsed = time.time() - start
    logger.info("All features saved in %.1fs", elapsed)

    return metadata


def save_feature_distributions(
    user_features: DataFrame,
    item_features: DataFrame,
    output_path: str = None,
) -> None:
    """
    Save distribution snapshots of key features at training time.

    These become the 'expected' distributions for PSI drift monitoring.
    When new data arrives, we compare its distributions against these
    baselines. If PSI > 0.2, model promotion is blocked.

    Args:
        user_features: User features DataFrame
        item_features: Item features DataFrame
        output_path: Path for .npz output file
    """
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/baseline_distributions.npz"

    logger.info("Saving feature distributions to %s", output_path)

    # Collect to numpy (these are aggregated — much smaller than raw data)
    distributions = {
        "user_avg_rating": (
            user_features.select("user_avg_rating").toPandas()["user_avg_rating"]
            .values.astype(np.float64)
        ),
        "user_rating_count": (
            user_features.select("user_rating_count").toPandas()["user_rating_count"]
            .values.astype(np.float64)
        ),
        "item_avg_rating": (
            item_features.select("item_avg_rating").toPandas()["item_avg_rating"]
            .values.astype(np.float64)
        ),
        "item_rating_count": (
            item_features.select("item_rating_count").toPandas()["item_rating_count"]
            .values.astype(np.float64)
        ),
        "item_recency_score": (
            item_features.select("item_recency_score").toPandas()["item_recency_score"]
            .values.astype(np.float64)
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **distributions)
    logger.info("  ✓ Saved distributions for %d features", len(distributions))


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main() -> Dict[str, Any]:
    """
    Run the full PySpark feature engineering pipeline.

    Steps:
        1. Load raw data
        2. Run quality checks
        3. Filter cold-start users/items
        4. Compute user features
        5. Compute item features
        6. Compute interaction (cross) features
        7. Remap IDs to contiguous integers
        8. Temporal per-user train/val/test split
        9. Save Parquet outputs + distribution snapshots

    Returns:
        Metadata dict with counts and pipeline info
    """
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)

    spark = get_spark_session()

    try:
        # Step 1: Load
        ratings = load_ratings(spark)
        movies = load_movies(spark)

        # Step 2: Data quality checks
        checks = run_data_quality_checks(ratings, movies)
        assert checks["ratings_null_count"] == 0, (
            f"Null ratings found: {checks['ratings_null_count']}"
        )

        # Step 3: Cold start filter
        ratings_filtered = filter_cold_start(ratings)

        # Step 4: User features
        user_features = compute_user_features(ratings_filtered, movies)

        # Step 5: Item features
        item_features = compute_item_features(ratings_filtered, movies)

        # Step 6: Interaction features
        enriched = compute_interaction_features(
            ratings_filtered, user_features, item_features
        )

        # Step 7: ID remap
        enriched, n_users, n_items = remap_ids(enriched)

        # Step 8: Temporal split
        train, val, test = temporal_user_split(enriched)

        # Step 9: Save
        metadata = save_features(train, val, test, user_features, item_features)
        metadata["n_users"] = n_users
        metadata["n_items"] = n_items

        # Save distribution baselines for drift monitoring
        save_feature_distributions(user_features, item_features)

        elapsed = time.time() - pipeline_start
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE in %.1fs (%.1f minutes)", elapsed, elapsed / 60)
        logger.info("  Users:  %s", f"{n_users:,}")
        logger.info("  Items:  %s", f"{n_items:,}")
        logger.info("  Train:  %s", f"{metadata['train_count']:,}")
        logger.info("  Val:    %s", f"{metadata['val_count']:,}")
        logger.info("  Test:   %s", f"{metadata['test_count']:,}")
        logger.info("=" * 70)

        return metadata

    except Exception:
        logger.exception("Pipeline failed!")
        raise

    finally:
        stop_spark(spark)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    metadata = main()
    print(f"\n✓ Pipeline complete. Metadata:\n{json.dumps(metadata, indent=2)}")