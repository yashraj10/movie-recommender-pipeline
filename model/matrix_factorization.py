"""
Matrix Factorization Baseline Model
====================================
Dot-product matrix factorization with bias terms in TensorFlow/Keras.

This is the "simple but strong" baseline for recommendation.
The NCF model (ncf.py) must beat this to justify its complexity.

Architecture:
    predicted_score = global_bias + user_bias + item_bias + dot(user_emb, item_emb)
    output = sigmoid(predicted_score)  → probability of interaction

Reference:
    Koren et al., "Matrix Factorization Techniques for Recommender Systems" (2009)
"""

import logging
from typing import Dict

import tensorflow as tf

logger = logging.getLogger(__name__)


class MatrixFactorization(tf.keras.Model):
    """
    Dot-product matrix factorization with bias terms.

    For implicit feedback (binary classification):
        P(interaction) = sigmoid(global_bias + user_bias + item_bias + dot(u, v))

    Why this architecture:
        - Dot product captures linear user-item affinity
        - Bias terms handle user-level optimism and item-level popularity
        - L2 regularization prevents overfitting on sparse data
        - This is the standard baseline in recommendation literature

    Args:
        n_users: Number of unique users
        n_items: Number of unique items
        embedding_dim: Dimensionality of user/item embeddings
        l2_reg: L2 regularization strength for embeddings
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        l2_reg: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embedding layers
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=n_users,
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
            name="user_embedding",
        )
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
            name="item_embedding",
        )

        # Bias terms (one scalar per user/item)
        self.user_bias = tf.keras.layers.Embedding(
            input_dim=n_users,
            output_dim=1,
            embeddings_initializer="zeros",
            name="user_bias",
        )
        self.item_bias = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=1,
            embeddings_initializer="zeros",
            name="item_bias",
        )

        # Global bias
        self.global_bias = tf.Variable(0.0, name="global_bias")

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (batch_size, 2) where
                    inputs[:, 0] = user_ids, inputs[:, 1] = item_ids
            training: Whether in training mode (unused here, but required by Keras)

        Returns:
            Tensor of shape (batch_size,) with predicted interaction probabilities
        """
        user_ids = inputs[:, 0]  # (batch,)
        item_ids = inputs[:, 1]  # (batch,)

        # Embeddings: (batch, embedding_dim)
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Dot product: (batch,)
        dot = tf.reduce_sum(user_emb * item_emb, axis=1)

        # Biases: (batch,)
        u_bias = tf.squeeze(self.user_bias(user_ids), axis=1)
        i_bias = tf.squeeze(self.item_bias(item_ids), axis=1)

        # Score = dot + biases
        score = dot + u_bias + i_bias + self.global_bias

        # Sigmoid for implicit feedback (probability of interaction)
        return tf.sigmoid(score)

    def get_config(self) -> Dict:
        """Serialization config for model saving."""
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim,
        }

    def summary_stats(self) -> str:
        """Human-readable model summary."""
        n_emb_params = (self.n_users + self.n_items) * self.embedding_dim
        n_bias_params = self.n_users + self.n_items + 1
        total = n_emb_params + n_bias_params
        return (
            f"MatrixFactorization(\n"
            f"  users={self.n_users:,}, items={self.n_items:,}, "
            f"dim={self.embedding_dim}\n"
            f"  embedding params: {n_emb_params:,}\n"
            f"  bias params:      {n_bias_params:,}\n"
            f"  total params:     {total:,}\n"
            f")"
        )


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Create model with small dimensions for testing
    model = MatrixFactorization(n_users=1000, n_items=500, embedding_dim=64)

    # Test forward pass
    import numpy as np
    test_input = tf.constant(np.array([[0, 1], [2, 3], [999, 499]]), dtype=tf.int32)
    output = model(test_input)

    print(f"✓ Model created")
    print(model.summary_stats())
    print(f"✓ Forward pass output shape: {output.shape}")  # (3,)
    print(f"✓ Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
    assert output.shape == (3,), f"Expected (3,), got {output.shape}"
    assert tf.reduce_all(output >= 0) and tf.reduce_all(output <= 1), "Output not in [0,1]"
    print(f"✓ All assertions passed — MF model is valid")
