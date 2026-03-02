"""
Recommendation Model Evaluation
================================
Compute ranking metrics for recommendation evaluation.

Metrics:
    1. Hit Rate @ K (HR@K):  Did the model rank the actual item in top K?
    2. NDCG @ K:             How HIGH did the model rank the actual item?
    3. AUC:                  Area under ROC curve for binary interaction prediction
    4. Coverage:             What % of items ever get recommended?

Evaluation protocol (standard in RecSys literature):
    For each user in the test set:
    1. Take their actual test item (positive)
    2. Sample 99 random items they haven't interacted with (negatives)
    3. Score all 100 items with the model
    4. Check if the positive item appears in top K

Reference:
    He et al., "Neural Collaborative Filtering" (WWW 2017) — Section 4.1
"""

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────
N_EVAL_NEGATIVES = 99       # Standard: 99 negatives + 1 positive = 100 candidates
DEFAULT_K_VALUES = [5, 10, 20]
MAX_EVAL_USERS = 5000       # Cap evaluation users for speed (sample if more)
EVAL_BATCH_SIZE = 4096
SEED = 42


def _get_user_positive_items(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[int, Set[int]]:
    """
    Build a lookup of all items each user has interacted with.
    Used to ensure negative samples don't overlap with known positives.

    Args:
        train_df: Training interactions DataFrame
        val_df: Validation interactions DataFrame

    Returns:
        Dict mapping user_idx → set of item_idx they've interacted with
    """
    user_items: Dict[int, Set[int]] = {}

    for df in [train_df, val_df]:
        for _, row in df[["user_idx", "item_idx"]].iterrows():
            uid = int(row["user_idx"])
            iid = int(row["item_idx"])
            if uid not in user_items:
                user_items[uid] = set()
            user_items[uid].add(iid)

    return user_items


def _sample_negative_candidates(
    user_id: int,
    n_items: int,
    n_negatives: int,
    positive_items: Set[int],
    rng: np.random.RandomState,
) -> List[int]:
    """
    Sample random items the user hasn't interacted with.

    Args:
        user_id: User ID
        n_items: Total number of items
        n_negatives: Number of negatives to sample
        positive_items: Set of items this user has interacted with
        rng: Random state for reproducibility

    Returns:
        List of negative item IDs
    """
    negatives = []
    while len(negatives) < n_negatives:
        candidates = rng.randint(0, n_items, size=n_negatives * 2)
        for c in candidates:
            if c not in positive_items and c not in negatives:
                negatives.append(c)
                if len(negatives) >= n_negatives:
                    break
    return negatives


def hit_rate_at_k(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    n_items: int,
    user_positive_items: Dict[int, Set[int]],
    k: int = 10,
    max_users: int = MAX_EVAL_USERS,
) -> float:
    """
    Hit Rate @ K: fraction of users whose actual test item is in the model's top K.

    Protocol:
        For each user, score 100 items (1 positive + 99 negatives).
        HR@10 of 0.65 means: for 65% of users, their actual next item
        was in the model's top 10 predictions out of 100 candidates.

    Args:
        model: Trained recommendation model
        test_df: Test interactions with user_idx, item_idx
        n_items: Total number of items
        user_positive_items: Dict of user → set of known positive items
        k: Top-K cutoff
        max_users: Maximum users to evaluate (for speed)

    Returns:
        HR@K score (0 to 1)
    """
    rng = np.random.RandomState(SEED)

    # Get unique test users and their test items
    test_users = test_df.groupby("user_idx")["item_idx"].first().reset_index()
    if len(test_users) > max_users:
        test_users = test_users.sample(max_users, random_state=SEED)

    hits = 0
    total = 0

    for _, row in test_users.iterrows():
        user_id = int(row["user_idx"])
        true_item = int(row["item_idx"])
        positives = user_positive_items.get(user_id, set())

        # Sample 99 negatives + 1 positive = 100 candidates
        negatives = _sample_negative_candidates(
            user_id, n_items, N_EVAL_NEGATIVES, positives | {true_item}, rng
        )
        candidates = negatives + [true_item]  # True item at index 99

        # Score all 100 candidates
        user_ids = np.full(len(candidates), user_id, dtype=np.int32)
        item_ids = np.array(candidates, dtype=np.int32)
        inputs = np.stack([user_ids, item_ids], axis=1)
        scores = model.predict(inputs, batch_size=EVAL_BATCH_SIZE, verbose=0).flatten()

        # Check if true item (last position) is in top K
        top_k_indices = np.argsort(scores)[-k:]
        if (len(candidates) - 1) in top_k_indices:
            hits += 1
        total += 1

    hr = hits / total if total > 0 else 0.0
    return hr


def ndcg_at_k(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    n_items: int,
    user_positive_items: Dict[int, Set[int]],
    k: int = 10,
    max_users: int = MAX_EVAL_USERS,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Measures ranking quality — rewards placing the relevant item higher.

    NDCG@K = DCG@K / IDCG@K
    DCG@K = 1 / log2(rank + 1)
    IDCG@K = 1 / log2(2) = 1.0 (since we have 1 relevant item)

    Values range [0, 1]. NDCG@10 > 0.4 is typically good for
    collaborative filtering on MovieLens.

    Args:
        model: Trained recommendation model
        test_df: Test interactions with user_idx, item_idx
        n_items: Total number of items
        user_positive_items: Dict of user → set of known positive items
        k: Top-K cutoff
        max_users: Maximum users to evaluate

    Returns:
        NDCG@K score (0 to 1)
    """
    rng = np.random.RandomState(SEED)

    test_users = test_df.groupby("user_idx")["item_idx"].first().reset_index()
    if len(test_users) > max_users:
        test_users = test_users.sample(max_users, random_state=SEED)

    ndcg_scores = []

    for _, row in test_users.iterrows():
        user_id = int(row["user_idx"])
        true_item = int(row["item_idx"])
        positives = user_positive_items.get(user_id, set())

        negatives = _sample_negative_candidates(
            user_id, n_items, N_EVAL_NEGATIVES, positives | {true_item}, rng
        )
        candidates = negatives + [true_item]

        user_ids = np.full(len(candidates), user_id, dtype=np.int32)
        item_ids = np.array(candidates, dtype=np.int32)
        inputs = np.stack([user_ids, item_ids], axis=1)
        scores = model.predict(inputs, batch_size=EVAL_BATCH_SIZE, verbose=0).flatten()

        # Find rank of true item (descending order)
        ranked = np.argsort(scores)[::-1]
        true_item_idx = len(candidates) - 1  # True item is last
        position = np.where(ranked == true_item_idx)[0][0]

        if position < k:
            ndcg = 1.0 / np.log2(position + 2)  # +2 because position is 0-indexed
        else:
            ndcg = 0.0

        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def compute_coverage(
    model: tf.keras.Model,
    n_users: int,
    n_items: int,
    k: int = 10,
    sample_users: int = 500,
) -> float:
    """
    Item coverage: what fraction of items ever appear in any user's top-K?

    Low coverage = popularity bias (only recommends blockbusters).
    Good models surface long-tail items too.

    Args:
        model: Trained recommendation model
        n_users: Total number of users
        n_items: Total number of items
        k: Top-K for recommendations
        sample_users: Number of users to sample (scoring all items × all users is expensive)

    Returns:
        Coverage score (0 to 1)
    """
    rng = np.random.RandomState(SEED)
    sampled_users = rng.choice(n_users, size=min(sample_users, n_users), replace=False)

    recommended_items = set()

    all_items = np.arange(n_items, dtype=np.int32)

    for user_id in sampled_users:
        user_ids = np.full(n_items, user_id, dtype=np.int32)
        inputs = np.stack([user_ids, all_items], axis=1)
        scores = model.predict(inputs, batch_size=EVAL_BATCH_SIZE, verbose=0).flatten()

        top_k = np.argsort(scores)[-k:]
        recommended_items.update(top_k.tolist())

    coverage = len(recommended_items) / n_items
    return coverage


def evaluate_recommendation_model(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_users: int,
    n_items: int,
    k_values: List[int] = None,
    max_eval_users: int = MAX_EVAL_USERS,
    compute_coverage_flag: bool = True,
) -> Dict[str, float]:
    """
    Full evaluation of a recommendation model.

    Computes HR@K and NDCG@K for multiple K values, plus coverage.

    Args:
        model: Trained recommendation model
        test_df: Test interactions DataFrame
        train_df: Training interactions (for building positive item sets)
        val_df: Validation interactions (for building positive item sets)
        n_users: Total number of users
        n_items: Total number of items
        k_values: List of K values for top-K metrics
        max_eval_users: Max users for evaluation (caps computation time)
        compute_coverage_flag: Whether to compute coverage (slower)

    Returns:
        Dict of metric names to values
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    logger.info("Evaluating model on %d test users (max %d)", len(test_df), max_eval_users)
    start = time.time()

    # Build positive item lookup
    user_positive_items = _get_user_positive_items(train_df, val_df)

    results = {}

    # HR@K and NDCG@K for each K
    for k in k_values:
        logger.info("Computing HR@%d and NDCG@%d...", k, k)

        hr = hit_rate_at_k(
            model, test_df, n_items, user_positive_items, k, max_eval_users
        )
        ndcg = ndcg_at_k(
            model, test_df, n_items, user_positive_items, k, max_eval_users
        )

        results[f"HR@{k}"] = round(hr, 4)
        results[f"NDCG@{k}"] = round(ndcg, 4)

        logger.info("  HR@%d = %.4f, NDCG@%d = %.4f", k, hr, k, ndcg)

    # Coverage
    if compute_coverage_flag:
        logger.info("Computing coverage...")
        cov = compute_coverage(model, n_users, n_items, k=10, sample_users=500)
        results["Coverage@10"] = round(cov, 4)
        logger.info("  Coverage@10 = %.4f", cov)

    elapsed = time.time() - start
    results["eval_time_seconds"] = round(elapsed, 1)

    logger.info("Evaluation complete in %.1fs", elapsed)
    return results


def print_model_comparison(
    mf_results: Optional[Dict] = None,
    ncf_results: Optional[Dict] = None,
) -> None:
    """
    Print a formatted model comparison table (for README).

    Args:
        mf_results: Evaluation results for MF baseline
        ncf_results: Evaluation results for NCF model
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Recommendation Metrics")
    print("=" * 70)
    print(f"{'Metric':<20} {'MF Baseline':>15} {'NCF':>15} {'Δ':>10}")
    print("-" * 70)

    metrics = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "Coverage@10"]

    for metric in metrics:
        mf_val = mf_results.get(metric, None) if mf_results else None
        ncf_val = ncf_results.get(metric, None) if ncf_results else None

        mf_str = f"{mf_val:.4f}" if mf_val is not None else "N/A"
        ncf_str = f"{ncf_val:.4f}" if ncf_val is not None else "N/A"

        if mf_val is not None and ncf_val is not None:
            delta = ncf_val - mf_val
            delta_pct = 100 * delta / mf_val if mf_val > 0 else 0
            delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
        else:
            delta_str = "N/A"

        print(f"{metric:<20} {mf_str:>15} {ncf_str:>15} {delta_str:>10}")

    print("=" * 70)


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from model.matrix_factorization import MatrixFactorization

    # Create tiny synthetic data
    n_users, n_items = 50, 30
    rng = np.random.RandomState(SEED)

    train_data = pd.DataFrame({
        "user_idx": rng.randint(0, n_users, 500),
        "item_idx": rng.randint(0, n_items, 500),
    })
    val_data = pd.DataFrame({
        "user_idx": rng.randint(0, n_users, 100),
        "item_idx": rng.randint(0, n_items, 100),
    })
    test_data = pd.DataFrame({
        "user_idx": rng.randint(0, n_users, 100),
        "item_idx": rng.randint(0, n_items, 100),
    })

    # Train a tiny MF model
    model = MatrixFactorization(n_users, n_items, embedding_dim=8)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # Quick train
    inputs = np.stack([train_data["user_idx"].values, train_data["item_idx"].values], axis=1).astype(np.int32)
    labels = np.random.randint(0, 2, len(train_data)).astype(np.float32)
    model.fit(inputs, labels, epochs=2, batch_size=64, verbose=0)

    # Evaluate
    results = evaluate_recommendation_model(
        model, test_data, train_data, val_data,
        n_users, n_items, k_values=[5, 10],
        max_eval_users=50, compute_coverage_flag=True,
    )

    print(f"\n✓ Evaluation results: {results}")
    assert "HR@5" in results, "Missing HR@5"
    assert "NDCG@10" in results, "Missing NDCG@10"
    assert "Coverage@10" in results, "Missing Coverage@10"
    print(f"✓ All evaluation metrics computed successfully")
