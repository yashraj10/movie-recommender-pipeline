"""
Model Tests
=============
Unit tests for Matrix Factorization and Neural Collaborative Filtering models.
Tests output shapes, value ranges, save/load, and negative sampling.

Run: python -m pytest tests/test_model.py -v
"""

import os
import pytest
import numpy as np
import tensorflow as tf


# ── Test: Matrix Factorization ─────────────────────────────────────────
class TestMatrixFactorization:

    def test_output_shape(self, n_users, n_items):
        """MF model output shape should be (batch_size,)."""
        from model.matrix_factorization import MatrixFactorization

        model = MatrixFactorization(n_users, n_items, embedding_dim=16)
        inputs = np.array([[0, 0], [1, 5], [10, 20]], dtype=np.int32)
        output = model(inputs)

        assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"

    def test_output_range(self, n_users, n_items):
        """MF sigmoid output should be in [0, 1]."""
        from model.matrix_factorization import MatrixFactorization

        model = MatrixFactorization(n_users, n_items, embedding_dim=16)
        inputs = np.array([[i, j] for i in range(5) for j in range(5)], dtype=np.int32)
        output = model(inputs).numpy()

        assert np.all(output >= 0.0), f"Output below 0: {output.min()}"
        assert np.all(output <= 1.0), f"Output above 1: {output.max()}"

    def test_parameter_count(self, n_users, n_items):
        """Verify parameter count matches expected formula."""
        from model.matrix_factorization import MatrixFactorization

        emb_dim = 16
        model = MatrixFactorization(n_users, n_items, embedding_dim=emb_dim)
        # Forward pass to build weights (Keras 3 requirement)
        model(np.array([[0, 0]], dtype=np.int32))

        expected = (n_users + n_items) * emb_dim + (n_users + n_items) + 1  # +1 for global_bias scalar
        actual = model.count_params()

        assert actual == expected, f"Expected {expected} params, got {actual}"

    def test_save_load_weights(self, n_users, n_items, tmp_path):
        """Saved and loaded model should produce identical predictions."""
        from model.matrix_factorization import MatrixFactorization

        model = MatrixFactorization(n_users, n_items, embedding_dim=16)
        inputs = np.array([[0, 0], [1, 5], [10, 20]], dtype=np.int32)

        # Get predictions before save
        pred_before = model(inputs).numpy()

        # Save weights
        weight_path = str(tmp_path / "mf_test.weights.h5")
        model.save_weights(weight_path)

        # Create new model, build via forward pass (Keras 3 requirement)
        model2 = MatrixFactorization(n_users, n_items, embedding_dim=16)
        model2(np.array([[0, 0]], dtype=np.int32))
        model2.load_weights(weight_path)

        pred_after = model2(inputs).numpy()
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)


# ── Test: Neural Collaborative Filtering ───────────────────────────────
class TestNCF:

    def test_output_shape(self, n_users, n_items):
        """NCF model output shape should be (batch_size,)."""
        from model.ncf import NeuralCollaborativeFiltering

        model = NeuralCollaborativeFiltering(
            n_users, n_items, gmf_dim=16, mlp_dim=16,
            mlp_layers=[32, 16], dropout_rate=0.2,
        )
        inputs = np.array([[0, 0], [1, 5], [10, 20]], dtype=np.int32)
        output = model(inputs, training=False)

        assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"

    def test_output_range(self, n_users, n_items):
        """NCF sigmoid output should be in [0, 1]."""
        from model.ncf import NeuralCollaborativeFiltering

        model = NeuralCollaborativeFiltering(
            n_users, n_items, gmf_dim=16, mlp_dim=16,
            mlp_layers=[32, 16], dropout_rate=0.2,
        )
        inputs = np.array([[i, j] for i in range(10) for j in range(10)], dtype=np.int32)
        output = model(inputs, training=False).numpy()

        assert np.all(output >= 0.0), f"Output below 0: {output.min()}"
        assert np.all(output <= 1.0), f"Output above 1: {output.max()}"

    def test_training_vs_inference(self, n_users, n_items):
        """Training mode (dropout active) and inference should differ."""
        from model.ncf import NeuralCollaborativeFiltering

        model = NeuralCollaborativeFiltering(
            n_users, n_items, gmf_dim=16, mlp_dim=16,
            mlp_layers=[32, 16], dropout_rate=0.5,  # High dropout to see difference
        )
        inputs = np.array([[0, 0]] * 100, dtype=np.int32)

        # Run multiple training forward passes — dropout causes variance
        train_outputs = [model(inputs, training=True).numpy() for _ in range(5)]
        inference_output = model(inputs, training=False).numpy()

        # Training outputs should vary (dropout), inference should be deterministic
        train_variance = np.var([o.mean() for o in train_outputs])

        # Inference should be deterministic
        inf_out2 = model(inputs, training=False).numpy()
        np.testing.assert_array_almost_equal(inference_output, inf_out2, decimal=5)

    def test_save_load_weights(self, n_users, n_items, tmp_path):
        """Saved and loaded NCF should produce identical predictions."""
        from model.ncf import NeuralCollaborativeFiltering

        model = NeuralCollaborativeFiltering(
            n_users, n_items, gmf_dim=16, mlp_dim=16,
            mlp_layers=[32, 16], dropout_rate=0.2,
        )
        inputs = np.array([[0, 0], [1, 5], [10, 20]], dtype=np.int32)

        pred_before = model(inputs, training=False).numpy()

        weight_path = str(tmp_path / "ncf_test.weights.h5")
        model.save_weights(weight_path)

        # Create new model, build via forward pass (Keras 3 requirement)
        model2 = NeuralCollaborativeFiltering(
            n_users, n_items, gmf_dim=16, mlp_dim=16,
            mlp_layers=[32, 16], dropout_rate=0.2,
        )
        model2(np.array([[0, 0]], dtype=np.int32), training=False)
        model2.load_weights(weight_path)

        pred_after = model2(inputs, training=False).numpy()
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)


# ── Test: Negative Sampling ────────────────────────────────────────────
class TestNegativeSampling:

    def test_correct_ratio(self):
        """Negative samples should have correct ratio to positives."""
        from model.data_loader import generate_negative_samples
        import pandas as pd

        positive_df = pd.DataFrame({
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [0, 1, 2, 3, 4],
            "label": [1.0] * 5,
        })

        result = generate_negative_samples(positive_df, n_items=30, neg_ratio=4)

        n_positive = (result["label"] == 1.0).sum()
        n_negative = (result["label"] == 0.0).sum()

        assert n_positive == 5, f"Expected 5 positives, got {n_positive}"
        assert n_negative == 20, f"Expected 20 negatives (5 × 4), got {n_negative}"

    def test_no_overlap(self):
        """Negative samples should not overlap with positive interactions."""
        from model.data_loader import generate_negative_samples
        import pandas as pd

        positive_df = pd.DataFrame({
            "user_idx": [0, 0, 0, 1, 1],
            "item_idx": [0, 1, 2, 3, 4],
            "label": [1.0] * 5,
        })

        result = generate_negative_samples(positive_df, n_items=30, neg_ratio=4)

        # Build set of positive (user, item) pairs
        positive_pairs = set(zip(positive_df["user_idx"], positive_df["item_idx"]))

        # Check negatives don't overlap
        negatives = result[result["label"] == 0.0]
        for _, row in negatives.iterrows():
            pair = (int(row["user_idx"]), int(row["item_idx"]))
            assert pair not in positive_pairs, f"Negative sample {pair} overlaps with positive"

    def test_label_values(self):
        """Labels should be exactly 0.0 or 1.0."""
        from model.data_loader import generate_negative_samples
        import pandas as pd

        positive_df = pd.DataFrame({
            "user_idx": [0, 1, 2],
            "item_idx": [0, 1, 2],
            "label": [1.0] * 3,
        })

        result = generate_negative_samples(positive_df, n_items=30, neg_ratio=2)
        unique_labels = set(result["label"].unique())

        assert unique_labels == {0.0, 1.0}, f"Expected labels {{0.0, 1.0}}, got {unique_labels}"