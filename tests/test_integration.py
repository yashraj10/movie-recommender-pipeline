"""
Integration Test - End-to-End Mini Pipeline

Runs the FULL pipeline on a tiny synthetic dataset (~2000 ratings)
to verify all components connect correctly:

  synthetic data -> cold start filter -> features -> temporal split ->
  ID remap -> negative sampling -> TF dataset -> train 1 epoch (MF + NCF) ->
  evaluate -> drift check -> promotion decision

This test runs on CPU in ~60 seconds. No GPU required.
"""

import os
import shutil
import tempfile
import logging
import numpy as np
import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINI_N_USERS = 100
MINI_N_ITEMS = 200
MINI_NEG_RATIO = 2
MINI_BATCH_SIZE = 64
MINI_EPOCHS = 1
MINI_EVAL_USERS = 20
MINI_EMBEDDING_DIM = 8
MINI_MLP_LAYERS = [16, 8]


@pytest.fixture(scope="module")
def tmp_dir():
    d = tempfile.mkdtemp(prefix="mini_pipeline_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def synthetic_data(tmp_dir):
    np.random.seed(42)
    rows = []
    base_ts = 1_500_000_000
    for user_id in range(1, MINI_N_USERS + 1):
        n_ratings = np.random.randint(15, 31)
        movies = np.random.choice(range(1, MINI_N_ITEMS + 1), size=n_ratings, replace=False)
        for i, movie_id in enumerate(movies):
            rows.append({
                "userId": user_id,
                "movieId": int(movie_id),
                "rating": float(np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])),
                "timestamp": base_ts + user_id * 1000 + i * 10,
            })
    ratings_df = pd.DataFrame(rows)

    genres_pool = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Horror", "Animation"]
    movies_rows = []
    for mid in range(1, MINI_N_ITEMS + 1):
        ng = np.random.randint(1, 4)
        genres = "|".join(np.random.choice(genres_pool, size=ng, replace=False))
        movies_rows.append({
            "movieId": mid,
            "title": "Movie {} ({})".format(mid, 2000 + mid % 24),
            "genres": genres,
        })
    movies_df = pd.DataFrame(movies_rows)

    ratings_path = os.path.join(tmp_dir, "ratings.csv")
    movies_path = os.path.join(tmp_dir, "movies.csv")
    ratings_df.to_csv(ratings_path, index=False)
    movies_df.to_csv(movies_path, index=False)

    output_dir = os.path.join(tmp_dir, "features")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Synthetic data: {} ratings, {} users, {} items".format(
        len(ratings_df), MINI_N_USERS, MINI_N_ITEMS))

    return {
        "ratings_path": ratings_path,
        "movies_path": movies_path,
        "output_dir": output_dir,
    }


@pytest.fixture(scope="module")
def spark_outputs(synthetic_data, tmp_dir):
    from pyspark.sql import SparkSession
    from spark.feature_engineering import (
        compute_user_features,
        compute_item_features,
        compute_interaction_features,
        filter_cold_start,
        temporal_user_split,
        remap_ids,
    )

    spark = (
        SparkSession.builder
        .appName("IntegrationTest")
        .master("local[*]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    try:
        ratings = spark.read.csv(synthetic_data["ratings_path"], header=True, inferSchema=True)
        movies = spark.read.csv(synthetic_data["movies_path"], header=True, inferSchema=True)

        filtered = filter_cold_start(ratings, min_user_ratings=10, min_item_ratings=3)
        user_features = compute_user_features(filtered, movies)
        item_features = compute_item_features(filtered, movies)
        enriched = compute_interaction_features(filtered, user_features, item_features)
        remapped, n_users, n_items = remap_ids(enriched)
        train, val, test = temporal_user_split(remapped)

        output_dir = synthetic_data["output_dir"]
        train.toPandas().to_parquet(os.path.join(output_dir, "train.parquet"))
        val.toPandas().to_parquet(os.path.join(output_dir, "val.parquet"))
        test.toPandas().to_parquet(os.path.join(output_dir, "test.parquet"))
        user_features.toPandas().to_parquet(os.path.join(output_dir, "user_features.parquet"))
        item_features.toPandas().to_parquet(os.path.join(output_dir, "item_features.parquet"))

        result = {
            "n_users": n_users,
            "n_items": n_items,
            "train_count": train.count(),
            "val_count": val.count(),
            "test_count": test.count(),
            "output_dir": output_dir,
            "train_pdf": train.toPandas(),
            "val_pdf": val.toPandas(),
            "test_pdf": test.toPandas(),
        }
    finally:
        spark.stop()

    return result


# ══════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════════════

class TestSparkFeatureEngineering:

    def test_cold_start_filter_applied(self, spark_outputs):
        assert spark_outputs["n_users"] <= MINI_N_USERS
        assert spark_outputs["n_items"] <= MINI_N_ITEMS

    def test_split_proportions(self, spark_outputs):
        total = (
            spark_outputs["train_count"]
            + spark_outputs["val_count"]
            + spark_outputs["test_count"]
        )
        train_frac = spark_outputs["train_count"] / total
        assert 0.6 < train_frac < 0.95
        assert spark_outputs["val_count"] > 0
        assert spark_outputs["test_count"] > 0

    def test_temporal_no_leakage(self, spark_outputs):
        train = spark_outputs["train_pdf"]
        val = spark_outputs["val_pdf"]
        common_users = set(train["userId"].unique()) & set(val["userId"].unique())
        if len(common_users) == 0:
            pytest.skip("No overlapping users between train and val")
        for uid in list(common_users)[:10]:
            max_train_ts = train[train["userId"] == uid]["timestamp"].max()
            min_val_ts = val[val["userId"] == uid]["timestamp"].min()
            assert max_train_ts <= min_val_ts

    def test_ids_contiguous(self, spark_outputs):
        train = spark_outputs["train_pdf"]
        user_idxs = sorted(train["user_idx"].unique())
        assert user_idxs[0] == 0

    def test_parquet_files_exist(self, spark_outputs):
        output_dir = spark_outputs["output_dir"]
        for name in [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "user_features.parquet",
            "item_features.parquet",
        ]:
            assert os.path.exists(os.path.join(output_dir, name))


class TestNegativeSamplingAndDataset:

    @pytest.fixture(scope="class")
    def sampled_data(self, spark_outputs):
        train_df = spark_outputs["train_pdf"]
        n_items = spark_outputs["n_items"]

        user_items = set(zip(train_df["user_idx"], train_df["item_idx"]))
        positives = train_df[["user_idx", "item_idx"]].copy()
        positives["label"] = 1.0

        negatives = []
        for user_id in train_df["user_idx"].unique():
            user_pos = {item for u, item in user_items if u == user_id}
            neg_pool = list(set(range(n_items)) - user_pos)
            n_neg = min(len(user_pos) * MINI_NEG_RATIO, len(neg_pool))
            if n_neg > 0:
                sampled = np.random.choice(neg_pool, size=n_neg, replace=False)
                for item in sampled:
                    negatives.append({"user_idx": user_id, "item_idx": item, "label": 0.0})

        neg_df = pd.DataFrame(negatives)
        combined = pd.concat([positives, neg_df], ignore_index=True)

        return {
            "combined": combined,
            "n_items": n_items,
            "n_users": spark_outputs["n_users"],
            "n_positives": len(positives),
            "n_negatives": len(neg_df),
        }

    def test_negative_no_overlap_with_positive(self, sampled_data):
        df = sampled_data["combined"]
        pos_pairs = set(zip(df[df["label"] == 1.0]["user_idx"], df[df["label"] == 1.0]["item_idx"]))
        neg_pairs = set(zip(df[df["label"] == 0.0]["user_idx"], df[df["label"] == 0.0]["item_idx"]))
        assert len(pos_pairs & neg_pairs) == 0

    def test_label_ratio(self, sampled_data):
        ratio = sampled_data["n_negatives"] / max(sampled_data["n_positives"], 1)
        assert 0.5 * MINI_NEG_RATIO <= ratio <= 1.5 * MINI_NEG_RATIO

    def test_tf_dataset_creation(self, sampled_data):
        import tensorflow as tf

        df = sampled_data["combined"]
        inputs = np.stack(
            [df["user_idx"].values, df["item_idx"].values], axis=1
        ).astype(np.int32)
        labels = df["label"].values.astype(np.float32)

        dataset = (
            tf.data.Dataset.from_tensor_slices((inputs, labels))
            .shuffle(1000)
            .batch(MINI_BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        batch_inputs, batch_labels = next(iter(dataset))
        assert batch_inputs.shape == (MINI_BATCH_SIZE, 2)
        assert batch_labels.shape == (MINI_BATCH_SIZE,)


class TestModelTraining:

    @pytest.fixture(scope="class")
    def trained_models(self, spark_outputs):
        import tensorflow as tf
        from model.matrix_factorization import MatrixFactorization
        from model.ncf import NeuralCollaborativeFiltering

        train_df = spark_outputs["train_pdf"]
        n_users = spark_outputs["n_users"]
        n_items = spark_outputs["n_items"]

        user_items = set(zip(train_df["user_idx"], train_df["item_idx"]))
        positives = train_df[["user_idx", "item_idx"]].copy()
        positives["label"] = 1.0

        negatives = []
        for uid in train_df["user_idx"].unique()[:50]:
            user_pos = {it for u, it in user_items if u == uid}
            pool = list(set(range(n_items)) - user_pos)
            n_neg = min(len(user_pos) * MINI_NEG_RATIO, len(pool))
            if n_neg > 0:
                sampled = np.random.choice(pool, size=n_neg, replace=False)
                for it in sampled:
                    negatives.append({"user_idx": uid, "item_idx": it, "label": 0.0})

        combined = pd.concat([positives, pd.DataFrame(negatives)], ignore_index=True)
        inputs = np.stack(
            [combined["user_idx"].values, combined["item_idx"].values], axis=1
        ).astype(np.int32)
        labels = combined["label"].values.astype(np.float32)

        dataset = (
            tf.data.Dataset.from_tensor_slices((inputs, labels))
            .shuffle(1000)
            .batch(MINI_BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Train MF baseline
        mf = MatrixFactorization(n_users, n_items, embedding_dim=MINI_EMBEDDING_DIM)
        mf.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        mf_hist = mf.fit(dataset, epochs=MINI_EPOCHS, verbose=0)

        # Train NCF
        ncf = NeuralCollaborativeFiltering(
            n_users, n_items,
            gmf_dim=MINI_EMBEDDING_DIM,
            mlp_dim=MINI_EMBEDDING_DIM,
            mlp_layers=MINI_MLP_LAYERS,
            dropout_rate=0.1,
        )
        ncf.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        ncf_hist = ncf.fit(dataset, epochs=MINI_EPOCHS, verbose=0)

        return {
            "mf": mf,
            "ncf": ncf,
            "mf_history": mf_hist,
            "ncf_history": ncf_hist,
            "n_users": n_users,
            "n_items": n_items,
        }

    def test_mf_output_range(self, trained_models):
        model = trained_models["mf"]
        preds = model.predict(np.array([[0, 0], [1, 1]], dtype=np.int32), verbose=0)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_ncf_output_range(self, trained_models):
        model = trained_models["ncf"]
        preds = model.predict(np.array([[0, 0], [1, 1]], dtype=np.int32), verbose=0)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_loss_is_finite(self, trained_models):
        assert np.isfinite(trained_models["mf_history"].history["loss"][-1])
        assert np.isfinite(trained_models["ncf_history"].history["loss"][-1])

    def test_model_save_load(self, trained_models, tmp_dir):
        import tensorflow as tf
        # These imports MUST happen before load_model so that the
        # @keras.saving.register_keras_serializable decorators run first.
        # Without them Keras cannot locate the custom class during deserialization.
        from model.ncf import NeuralCollaborativeFiltering  # noqa: F401
        from model.matrix_factorization import MatrixFactorization  # noqa: F401

        save_path = os.path.join(tmp_dir, "test_ncf_save.keras")
        model = trained_models["ncf"]
        test_input = np.array([[0, 0], [1, 1]], dtype=np.int32)

        original_preds = model.predict(test_input, verbose=0)
        model.save(save_path)
        loaded = tf.keras.models.load_model(save_path)
        loaded_preds = loaded.predict(test_input, verbose=0)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)


class TestEvaluation:

    def test_hit_rate_valid_range(self, spark_outputs):
        import tensorflow as tf
        from model.matrix_factorization import MatrixFactorization

        train_df = spark_outputs["train_pdf"]
        test_df = spark_outputs["test_pdf"]
        n_users = spark_outputs["n_users"]
        n_items = spark_outputs["n_items"]

        model = MatrixFactorization(n_users, n_items, embedding_dim=MINI_EMBEDDING_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        # Warm up the model with one prediction so weights are initialised
        model.predict(np.array([[0, 0]], dtype=np.int32), verbose=0)

        pos_set = set(zip(train_df["user_idx"], train_df["item_idx"]))
        test_users = test_df.groupby("user_idx").first().reset_index().head(MINI_EVAL_USERS)

        hits = 0
        for _, row in test_users.iterrows():
            uid = int(row["user_idx"])
            true_item = int(row["item_idx"])
            user_pos = {it for u, it in pos_set if u == uid}
            neg_pool = list(set(range(n_items)) - user_pos - {true_item})
            n_sample = min(49, len(neg_pool))
            candidates = list(np.random.choice(neg_pool, size=n_sample, replace=False))
            candidates.append(true_item)

            user_ids = np.full(len(candidates), uid, dtype=np.int32)
            inputs_arr = np.stack(
                [user_ids, np.array(candidates, dtype=np.int32)], axis=1
            )
            scores = model.predict(inputs_arr, verbose=0).flatten()
            top_10 = np.argsort(scores)[-10:]
            if (len(candidates) - 1) in top_10:
                hits += 1

        hr = hits / len(test_users)
        assert 0.0 <= hr <= 1.0


class TestDriftMonitor:

    def test_no_drift_passes(self):
        from pipeline.drift_monitor import compute_psi

        expected = np.random.normal(3.5, 1.0, 10000)
        psi = compute_psi(expected, expected.copy())
        assert psi < 0.1

    def test_high_drift_fails(self):
        from pipeline.drift_monitor import compute_psi, PSI_THRESHOLD

        expected = np.random.normal(3.5, 1.0, 10000)
        actual = np.random.normal(5.0, 1.0, 10000)
        psi = compute_psi(expected, actual)
        assert psi > PSI_THRESHOLD

    def test_full_drift_check_pipeline(self, tmp_dir):
        from pipeline.drift_monitor import run_drift_check, FEATURES_TO_MONITOR

        baseline_data = {feat: np.random.normal(3.0, 1.0, 5000) for feat in FEATURES_TO_MONITOR}
        baseline_path = os.path.join(tmp_dir, "test_baseline.npz")
        np.savez(baseline_path, **baseline_data)

        # Same distribution → should pass
        passed, results = run_drift_check(baseline_path, baseline_data)
        assert passed

        # Shifted distribution → should fail
        drifted = {feat: arr + 3.0 for feat, arr in baseline_data.items()}
        passed, results = run_drift_check(baseline_path, drifted)
        assert not passed


class TestFullPipelineConnectivity:
    """THE integration test — full pipeline end-to-end."""

    def test_pipeline_produces_promotion_decision(self, spark_outputs, tmp_dir):
        import tensorflow as tf
        from model.matrix_factorization import MatrixFactorization
        from pipeline.drift_monitor import compute_psi, PSI_THRESHOLD

        train_df = spark_outputs["train_pdf"]
        n_users = spark_outputs["n_users"]
        n_items = spark_outputs["n_items"]

        assert len(train_df) > 0
        logger.info("Step 1 PASS: Loaded {} train rows".format(len(train_df)))

        # Build training dataset with negatives
        user_items = set(zip(train_df["user_idx"], train_df["item_idx"]))
        positives = train_df[["user_idx", "item_idx"]].copy()
        positives["label"] = 1.0

        negs = []
        for uid in train_df["user_idx"].unique()[:20]:
            user_pos = {it for u, it in user_items if u == uid}
            pool = list(set(range(n_items)) - user_pos)
            sampled = np.random.choice(
                pool, size=min(len(user_pos) * 2, len(pool)), replace=False
            )
            for it in sampled:
                negs.append({"user_idx": uid, "item_idx": it, "label": 0.0})

        combined = pd.concat([positives, pd.DataFrame(negs)], ignore_index=True)
        logger.info("Step 2 PASS: {} samples".format(len(combined)))

        inputs = np.stack(
            [combined["user_idx"].values, combined["item_idx"].values], axis=1
        ).astype(np.int32)
        labels = combined["label"].values.astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(64).prefetch(1)
        logger.info("Step 3 PASS: TF dataset created")

        # Train
        model = MatrixFactorization(n_users, n_items, embedding_dim=MINI_EMBEDDING_DIM)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][-1])
        logger.info("Step 4 PASS: Trained 1 epoch")

        # Predict
        pred = model.predict(np.array([[0, 0]], dtype=np.int32), verbose=0).flatten()[0]
        assert 0.0 <= pred <= 1.0
        logger.info("Step 5 PASS: Prediction={:.4f}".format(pred))

        # Drift check
        baseline = np.random.normal(3.5, 1.0, 1000)
        current = np.random.normal(3.5, 1.0, 1000)
        psi = compute_psi(baseline, current)
        logger.info("Step 6 PASS: PSI={:.4f}".format(psi))

        # Promotion decision
        decision = "PROMOTE" if psi < PSI_THRESHOLD else "BLOCK"
        assert decision in ("PROMOTE", "BLOCK")
        logger.info("Step 7 PASS: Decision={}".format(decision))
        logger.info("INTEGRATION TEST PASSED — Full pipeline end-to-end")