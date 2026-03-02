"""
Feature Engineering Tests
==========================
Unit tests for the PySpark feature engineering pipeline.
Tests data quality, feature computation, temporal split, and ID remapping.

Run: python -m pytest tests/test_features.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd


# ── Test: Cold Start Filtering ─────────────────────────────────────────
class TestColdStartFiltering:

    def test_removes_low_activity_users(self, sample_ratings_data):
        """Users with fewer than min_user_ratings should be removed."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import filter_cold_start

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            filtered = filter_cold_start(df, min_user_ratings=5, min_item_ratings=1)

            # Check no user has fewer than 5 ratings
            from pyspark.sql import functions as F
            user_counts = filtered.groupBy("userId").agg(F.count("*").alias("cnt"))
            min_count = user_counts.agg(F.min("cnt")).collect()[0][0]

            assert min_count >= 5, f"Found user with only {min_count} ratings"
        finally:
            spark.stop()

    def test_removes_low_activity_items(self, sample_ratings_data):
        """Items with fewer than min_item_ratings should be removed."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import filter_cold_start

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            filtered = filter_cold_start(df, min_user_ratings=1, min_item_ratings=3)

            from pyspark.sql import functions as F
            item_counts = filtered.groupBy("movieId").agg(F.count("*").alias("cnt"))
            min_count = item_counts.agg(F.min("cnt")).collect()[0][0]

            assert min_count >= 3, f"Found item with only {min_count} ratings"
        finally:
            spark.stop()


# ── Test: Temporal Split ───────────────────────────────────────────────
class TestTemporalSplit:

    def test_no_temporal_leakage(self, sample_ratings_data):
        """
        For each user, max train timestamp must be < min val timestamp,
        and max val timestamp must be < min test timestamp.
        This is the KEY design decision — prevents data leakage.
        """
        from pyspark.sql import SparkSession
        from spark.feature_engineering import temporal_user_split

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            train, val, test = temporal_user_split(df, train_frac=0.8, val_frac=0.1)

            from pyspark.sql import functions as F

            # Get per-user timestamp ranges for each split
            train_max = train.groupBy("userId").agg(F.max("timestamp").alias("train_max"))
            val_min = val.groupBy("userId").agg(F.min("timestamp").alias("val_min"))
            val_max = val.groupBy("userId").agg(F.max("timestamp").alias("val_max"))
            test_min = test.groupBy("userId").agg(F.min("timestamp").alias("test_min"))

            # Check train < val for users in both splits
            train_val = train_max.join(val_min, "userId", "inner")
            leaks_train_val = train_val.filter(
                F.col("train_max") > F.col("val_min")
            ).count()

            assert leaks_train_val == 0, f"Temporal leakage: {leaks_train_val} users have train > val"

            # Check val < test for users in both splits
            val_test = val_max.join(test_min, "userId", "inner")
            leaks_val_test = val_test.filter(
                F.col("val_max") > F.col("test_min")
            ).count()

            assert leaks_val_test == 0, f"Temporal leakage: {leaks_val_test} users have val > test"

        finally:
            spark.stop()

    def test_split_proportions(self, sample_ratings_data):
        """Train ~80%, Val ~10%, Test ~10% within reasonable tolerance."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import temporal_user_split

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            train, val, test = temporal_user_split(df, train_frac=0.8, val_frac=0.1)

            total = df.count()
            train_pct = train.count() / total
            val_pct = val.count() / total
            test_pct = test.count() / total

            # Allow 10% tolerance because per-user split can't be exact
            assert 0.65 <= train_pct <= 0.90, f"Train {train_pct:.2%} outside expected range"
            assert 0.03 <= val_pct <= 0.20, f"Val {val_pct:.2%} outside expected range"
            assert 0.03 <= test_pct <= 0.20, f"Test {test_pct:.2%} outside expected range"

        finally:
            spark.stop()

    def test_no_data_loss(self, sample_ratings_data):
        """Total rows across splits should equal original count."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import temporal_user_split

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            train, val, test = temporal_user_split(df, train_frac=0.8, val_frac=0.1)

            total = df.count()
            split_total = train.count() + val.count() + test.count()

            assert split_total == total, f"Data loss: {total} → {split_total}"

        finally:
            spark.stop()


# ── Test: ID Remapping ─────────────────────────────────────────────────
class TestIDRemapping:

    def test_ids_are_contiguous(self, sample_ratings_data):
        """user_idx and item_idx should be 0-indexed and contiguous."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import remap_ids

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            df = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            remapped, n_users, n_items = remap_ids(df)

            pdf = remapped.select("user_idx").distinct().toPandas()
            user_ids = sorted(pdf["user_idx"].tolist())

            assert user_ids[0] == 0, f"user_idx should start at 0, got {user_ids[0]}"
            assert user_ids[-1] == n_users - 1, f"user_idx should end at {n_users - 1}, got {user_ids[-1]}"
            assert len(user_ids) == n_users, f"Expected {n_users} unique user_idx, got {len(user_ids)}"

        finally:
            spark.stop()


# ── Test: User Features ────────────────────────────────────────────────
class TestUserFeatures:

    def test_no_nulls_after_computation(self, sample_ratings_data, sample_movies_data):
        """All user features should have zero nulls after fillna."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import compute_user_features

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            ratings = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            movies = spark.createDataFrame(pd.DataFrame(sample_movies_data))

            user_features = compute_user_features(ratings, movies)
            pdf = user_features.toPandas()

            null_counts = pdf.isnull().sum()
            total_nulls = null_counts.sum()

            assert total_nulls == 0, f"Found nulls in user features:\n{null_counts[null_counts > 0]}"

        finally:
            spark.stop()


# ── Test: Item Features ────────────────────────────────────────────────
class TestItemFeatures:

    def test_no_nulls_after_computation(self, sample_ratings_data, sample_movies_data):
        """All item features should have zero nulls after fillna."""
        from pyspark.sql import SparkSession
        from spark.feature_engineering import compute_item_features

        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

        try:
            ratings = spark.createDataFrame(pd.DataFrame(sample_ratings_data))
            movies = spark.createDataFrame(pd.DataFrame(sample_movies_data))

            item_features = compute_item_features(ratings, movies)
            pdf = item_features.toPandas()

            null_counts = pdf.isnull().sum()
            total_nulls = null_counts.sum()

            assert total_nulls == 0, f"Found nulls in item features:\n{null_counts[null_counts > 0]}"

        finally:
            spark.stop()
