"""
Shared Pytest Fixtures
=======================
Reusable test data and configurations for unit + integration tests.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_ratings_data():
    """Small synthetic ratings dataset for fast testing."""
    np.random.seed(42)
    n_users = 50
    n_items = 30
    n_ratings = 500

    data = {
        "userId": np.random.randint(1, n_users + 1, n_ratings),
        "movieId": np.random.randint(1, n_items + 1, n_ratings),
        "rating": np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_ratings),
        "timestamp": np.sort(np.random.randint(946684800, 1704067200, n_ratings)),  # 2000-2024
    }
    return data


@pytest.fixture
def sample_movies_data():
    """Small synthetic movies dataset."""
    movies = {
        "movieId": list(range(1, 31)),
        "title": [f"Movie {i} ({2000 + i})" for i in range(1, 31)],
        "genres": [
            "Action|Adventure", "Comedy|Romance", "Drama", "Horror|Thriller",
            "Sci-Fi|Action", "Animation|Children", "Documentary", "Crime|Drama",
            "Fantasy|Adventure", "Musical|Comedy",
        ] * 3,
    }
    return movies


@pytest.fixture
def n_users():
    return 50


@pytest.fixture
def n_items():
    return 30


@pytest.fixture
def sample_interactions():
    """Interaction DataFrame with user_idx, item_idx, label columns."""
    import pandas as pd
    np.random.seed(42)

    n_positive = 200
    n_negative = 800

    positive = pd.DataFrame({
        "user_idx": np.random.randint(0, 50, n_positive),
        "item_idx": np.random.randint(0, 30, n_positive),
        "label": np.ones(n_positive, dtype=np.float32),
    })

    negative = pd.DataFrame({
        "user_idx": np.random.randint(0, 50, n_negative),
        "item_idx": np.random.randint(0, 30, n_negative),
        "label": np.zeros(n_negative, dtype=np.float32),
    })

    return pd.concat([positive, negative], ignore_index=True)


@pytest.fixture
def baseline_distributions(tmp_path):
    """Create a temporary baseline distributions file for drift testing."""
    np.random.seed(42)
    distributions = {
        "user_avg_rating": np.random.normal(3.5, 0.8, 10000),
        "user_rating_count": np.random.exponential(100, 10000),
        "item_avg_rating": np.random.normal(3.2, 1.0, 10000),
        "item_rating_count": np.random.exponential(500, 10000),
        "item_recency_score": np.random.uniform(0, 1, 10000),
    }

    filepath = tmp_path / "baseline_distributions.npz"
    np.savez(filepath, **distributions)
    return str(filepath)


@pytest.fixture
def training_config():
    """Standard training config for tests."""
    return {
        "gmf_dim": 16,
        "mlp_dim": 16,
        "mlp_layers": [32, 16],
        "dropout_rate": 0.2,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 2,
        "neg_ratio": 4,
        "patience": 2,
        "min_delta": 0.0001,
        "lr_factor": 0.5,
        "lr_patience": 1,
        "min_lr": 1e-6,
        "min_user_ratings": 20,
        "min_item_ratings": 5,
        "seed": 42,
    }