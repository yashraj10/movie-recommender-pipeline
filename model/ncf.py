"""
Neural Collaborative Filtering (NCF) Model
============================================
Two-tower architecture combining Generalized Matrix Factorization (GMF)
and Multi-Layer Perceptron (MLP) towers.

Architecture:
    GMF Tower: Element-wise product of user and item embeddings
               → captures linear interaction patterns (like classic MF)

    MLP Tower: Concatenate user and item embeddings → feed through MLP
               → captures non-linear interaction patterns

    Fusion: Concatenate GMF output + MLP output → Dense(1) → Sigmoid
            → predicts probability of interaction

Why two towers:
    GMF and MLP learn DIFFERENT representations. GMF captures multiplicative
    (linear) patterns while MLP captures complex non-linear patterns.
    Separate embedding spaces let each tower specialize.

Reference:
    He et al., "Neural Collaborative Filtering" (WWW 2017)
"""

import logging
from typing import Dict, List

import keras
import tensorflow as tf

logger = logging.getLogger(__name__)


@keras.saving.register_keras_serializable(package="MovieRecommender")
class NeuralCollaborativeFiltering(tf.keras.Model):
    """
    Two-tower NCF: GMF + MLP with fusion layer.

    Architecture diagram:
        GMF Tower:                           MLP Tower:
        User ID → Embed(64) ─┐              User ID → Embed(64) ─┐
                              ├→ Hadamard                          ├→ Concat(128)
        Item ID → Embed(64) ─┘              Item ID → Embed(64) ─┘
                                                    │
                                             Dense(128) + BN + Dropout
                                             Dense(64)  + BN + Dropout
                                             Dense(32)  + BN + Dropout
                                                    │
                GMF output(64) ──── Concat ──── MLP output(32)
                                      │
                               Dense(1, sigmoid)
                                      │
                               P(interaction)

    Args:
        n_users: Number of unique users
        n_items: Number of unique items
        gmf_dim: Embedding dimension for GMF tower
        mlp_dim: Embedding dimension for MLP tower
        mlp_layers: List of hidden layer sizes for MLP tower
        dropout_rate: Dropout rate between MLP layers
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 64,
        mlp_dim: int = 64,
        mlp_layers: List[int] = None,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        # Store all constructor args — needed by get_config()
        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.mlp_layer_sizes = mlp_layers
        self.dropout_rate = dropout_rate

        # ── GMF Tower ──────────────────────────────────────────────
        self.gmf_user_emb = tf.keras.layers.Embedding(
            n_users, gmf_dim,
            embeddings_initializer="he_normal",
            name="gmf_user_embedding",
        )
        self.gmf_item_emb = tf.keras.layers.Embedding(
            n_items, gmf_dim,
            embeddings_initializer="he_normal",
            name="gmf_item_embedding",
        )

        # ── MLP Tower ──────────────────────────────────────────────
        self.mlp_user_emb = tf.keras.layers.Embedding(
            n_users, mlp_dim,
            embeddings_initializer="he_normal",
            name="mlp_user_embedding",
        )
        self.mlp_item_emb = tf.keras.layers.Embedding(
            n_items, mlp_dim,
            embeddings_initializer="he_normal",
            name="mlp_item_embedding",
        )

        # MLP hidden layers: Dense → BatchNorm → ReLU → Dropout
        self.mlp_dense_layers = []
        self.mlp_bn_layers = []
        self.mlp_dropout_layers = []

        for i, units in enumerate(mlp_layers):
            self.mlp_dense_layers.append(
                tf.keras.layers.Dense(units, activation="relu", name=f"mlp_dense_{i}")
            )
            self.mlp_bn_layers.append(
                tf.keras.layers.BatchNormalization(name=f"mlp_bn_{i}")
            )
            self.mlp_dropout_layers.append(
                tf.keras.layers.Dropout(dropout_rate, name=f"mlp_dropout_{i}")
            )

        # ── Fusion Layer ───────────────────────────────────────────
        # GMF output (gmf_dim) + MLP output (mlp_layers[-1]) → Dense(1)
        self.output_layer = tf.keras.layers.Dense(
            1, activation="sigmoid", name="prediction"
        )

    def call(self, inputs, training=False):
        """
        Forward pass through both towers with fusion.

        Args:
            inputs: Tensor of shape (batch_size, 2)
                    inputs[:, 0] = user_ids, inputs[:, 1] = item_ids
            training: Whether in training mode (affects BatchNorm and Dropout)

        Returns:
            Tensor of shape (batch_size,) with predicted interaction probabilities
        """
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]

        # GMF Tower
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item  # Hadamard product

        # MLP Tower
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        x = tf.concat([mlp_user, mlp_item], axis=1)

        for dense, bn, dropout in zip(
            self.mlp_dense_layers, self.mlp_bn_layers, self.mlp_dropout_layers
        ):
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)

        # Fusion
        combined = tf.concat([gmf_output, x], axis=1)
        output = self.output_layer(combined)

        return tf.squeeze(output, axis=1)

    def get_config(self) -> Dict:
        """
        All __init__ args — required for .keras format save/load.
        Keras calls this to serialize the model architecture.
        from_config() uses this dict to reconstruct the model exactly.
        """
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "gmf_dim": self.gmf_dim,
            "mlp_dim": self.mlp_dim,
            "mlp_layers": self.mlp_layer_sizes,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        """Reconstruct model from get_config() dict."""
        return cls(**config)

    def summary_stats(self) -> str:
        """Human-readable model summary."""
        gmf_params = (self.n_users + self.n_items) * self.gmf_dim
        mlp_emb_params = (self.n_users + self.n_items) * self.mlp_dim

        mlp_dense_params = 0
        prev_dim = self.mlp_dim * 2
        for units in self.mlp_layer_sizes:
            mlp_dense_params += prev_dim * units + units
            mlp_dense_params += units * 4  # BatchNorm: gamma, beta, mean, var
            prev_dim = units

        fusion_dim = self.gmf_dim + self.mlp_layer_sizes[-1]
        output_params = fusion_dim + 1
        total = gmf_params + mlp_emb_params + mlp_dense_params + output_params

        return (
            f"NeuralCollaborativeFiltering(\n"
            f"  users={self.n_users:,}, items={self.n_items:,}\n"
            f"  GMF: dim={self.gmf_dim}, params={gmf_params:,}\n"
            f"  MLP: dim={self.mlp_dim}, layers={self.mlp_layer_sizes}, "
            f"params={mlp_emb_params + mlp_dense_params:,}\n"
            f"  Fusion: {fusion_dim} → 1, params={output_params:,}\n"
            f"  Dropout: {self.dropout_rate}\n"
            f"  Total params: ~{total:,}\n"
            f")"
        )


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    import numpy as np

    model = NeuralCollaborativeFiltering(
        n_users=1000, n_items=500,
        gmf_dim=64, mlp_dim=64,
        mlp_layers=[128, 64, 32],
        dropout_rate=0.2,
    )

    test_input = tf.constant(np.array([[0, 1], [2, 3], [999, 499]]), dtype=tf.int32)

    out_train = model(test_input, training=True)
    out_infer = model(test_input, training=False)

    print(model.summary_stats())
    print(f"✓ Training output shape: {out_train.shape}")
    print(f"✓ Inference output shape: {out_infer.shape}")
    print(f"✓ Output range: [{out_infer.numpy().min():.4f}, {out_infer.numpy().max():.4f}]")
    assert out_infer.shape == (3,), f"Expected (3,), got {out_infer.shape}"
    assert tf.reduce_all(out_infer >= 0) and tf.reduce_all(out_infer <= 1), "Output not in [0,1]"
    print("✓ All assertions passed — NCF model is valid")