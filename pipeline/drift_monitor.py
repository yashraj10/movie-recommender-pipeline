"""
PSI Drift Monitoring
=====================
Population Stability Index (PSI) monitors feature distribution shifts
between training time and inference time.

Why drift monitoring matters:
    If user behavior changes (e.g., rating inflation, new genre popularity),
    the model's training data no longer represents reality. PSI detects this
    shift BEFORE the model makes bad recommendations.

PSI thresholds (industry standard):
    PSI < 0.1:   No significant shift    → PASS
    PSI 0.1–0.2: Moderate shift          → PASS with WARNING
    PSI > 0.2:   Significant shift       → FAIL, block promotion

The Airflow DAG uses this as a gate: if ANY monitored feature fails PSI,
model promotion is blocked and the current Production model stays.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PSI_THRESHOLD = 0.2  # PSI > 0.2 = significant drift, block promotion
# ── Configuration ──────────────────────────────────────────────────────
PSI_THRESHOLD_FAIL = 0.2      # Above this → block promotion
PSI_THRESHOLD_WARN = 0.1      # Above this → log warning
PSI_BINS = 10                 # Number of bins for histogram comparison

FEATURES_TO_MONITOR = [
    "user_avg_rating",
    "user_rating_count",
    "item_avg_rating",
    "item_rating_count",
    "item_recency_score",
]


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

    How it works:
    1. Bin the expected distribution into N equal-frequency bins
    2. Count what percentage of actual values fall into each bin
    3. Compare the two percentage distributions

    A small PSI means the distributions look similar (stable).
    A large PSI means something changed (drift).

    Args:
        expected: Feature values from training time (baseline)
        actual: Feature values from current data
        bins: Number of histogram bins

    Returns:
        PSI score (0 = identical distributions, higher = more drift)
    """
    # Create bins from expected distribution (percentile-based)
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 2:
        logger.warning("Not enough unique breakpoints — returning PSI=0")
        return 0.0

    # Count values in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to percentages with smoothing (avoid division by zero)
    n_bins = len(expected_counts)
    expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
    actual_pct = (actual_counts + 1) / (len(actual) + n_bins)

    # PSI formula
    psi = float(np.sum(
        (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    ))

    return psi


def run_drift_check(
    baseline_path: str,
    current_features: Dict[str, np.ndarray],
    features_to_monitor: List[str] = None,
    threshold: float = PSI_THRESHOLD_FAIL,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Compare current feature distributions against training baseline.

    This is called by the Airflow DAG's drift_check task.
    If ANY feature fails (PSI > threshold), promotion is blocked.

    Args:
        baseline_path: Path to .npz file with training-time distributions
        current_features: Dict mapping feature name → numpy array of current values
        features_to_monitor: List of feature names to check (default: all 5)
        threshold: PSI threshold for failure

    Returns:
        Tuple of (all_passed: bool, results: dict per feature)
    """
    if features_to_monitor is None:
        features_to_monitor = FEATURES_TO_MONITOR

    logger.info("Running drift check on %d features (threshold=%.2f)",
                len(features_to_monitor), threshold)

    # Load baseline distributions
    baseline = np.load(baseline_path)
    results = {}
    all_passed = True

    for feature in features_to_monitor:
        if feature not in baseline:
            logger.warning("Feature '%s' not in baseline — skipping", feature)
            results[feature] = {
                "psi": None,
                "status": "SKIP",
                "passed": True,
                "message": "Not in baseline",
            }
            continue

        if feature not in current_features:
            logger.warning("Feature '%s' not in current data — skipping", feature)
            results[feature] = {
                "psi": None,
                "status": "SKIP",
                "passed": True,
                "message": "Not in current data",
            }
            continue

        expected = baseline[feature]
        actual = current_features[feature]

        psi = compute_psi(expected, actual)
        passed = psi < threshold

        if psi < PSI_THRESHOLD_WARN:
            status = "PASS"
        elif psi < PSI_THRESHOLD_FAIL:
            status = "WARNING"
        else:
            status = "FAIL"

        results[feature] = {
            "psi": round(psi, 4),
            "status": status,
            "passed": passed,
            "expected_mean": round(float(np.mean(expected)), 4),
            "actual_mean": round(float(np.mean(actual)), 4),
            "expected_std": round(float(np.std(expected)), 4),
            "actual_std": round(float(np.std(actual)), 4),
        }

        if not passed:
            all_passed = False

        logger.info(
            "  %s: PSI=%.4f [%s] (expected_mean=%.2f, actual_mean=%.2f)",
            feature, psi, status,
            float(np.mean(expected)), float(np.mean(actual)),
        )

    logger.info(
        "Drift check %s: %d/%d features passed",
        "PASSED" if all_passed else "FAILED",
        sum(1 for r in results.values() if r["passed"]),
        len(results),
    )

    return all_passed, results


def simulate_no_drift(baseline_path: str) -> Tuple[bool, Dict]:
    """
    Simulate a clean run where distributions haven't changed.
    Should always PASS. Used for pipeline validation.

    Args:
        baseline_path: Path to .npz baseline file

    Returns:
        Tuple of (passed, results)
    """
    logger.info("Simulating NO drift scenario...")
    baseline = np.load(baseline_path)
    current = {k: baseline[k] for k in baseline.files}

    return run_drift_check(baseline_path, current)


def simulate_high_drift(baseline_path: str) -> Tuple[bool, Dict]:
    """
    Simulate a significant distribution shift.
    Should always FAIL. Used to validate the drift monitor catches problems.

    Simulates: average ratings shift up by 1.0 (rating inflation)
    This would happen if users started being more generous with ratings.

    Args:
        baseline_path: Path to .npz baseline file

    Returns:
        Tuple of (passed, results)
    """
    logger.info("Simulating HIGH drift scenario (rating inflation)...")
    baseline = np.load(baseline_path)

    rng = np.random.RandomState(42)
    drifted = {}
    for feature in FEATURES_TO_MONITOR:
        if feature in baseline.files:
            # Shift distribution significantly
            original = baseline[feature]
            shift = rng.normal(loc=1.0, scale=0.5, size=len(original))
            drifted[feature] = original + shift

    return run_drift_check(baseline_path, drifted)


def format_drift_report(results: Dict[str, Dict]) -> str:
    """
    Format drift check results as a human-readable report.

    Args:
        results: Dict from run_drift_check

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DRIFT MONITORING REPORT")
    lines.append("=" * 60)
    lines.append(f"{'Feature':<25} {'PSI':>8} {'Status':>10} {'Result':>8}")
    lines.append("-" * 60)

    for feature, data in results.items():
        psi_str = f"{data['psi']:.4f}" if data["psi"] is not None else "N/A"
        lines.append(
            f"{feature:<25} {psi_str:>8} {data['status']:>10} "
            f"{'✓' if data['passed'] else '✗':>8}"
        )

    n_passed = sum(1 for r in results.values() if r["passed"])
    n_total = len(results)
    lines.append("-" * 60)
    lines.append(f"Overall: {n_passed}/{n_total} passed")
    lines.append("=" * 60)

    return "\n".join(lines)


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("PSI Drift Monitor — Validation")
    print("=" * 50)

    # Test 1: PSI of identical distributions should be ~0
    rng = np.random.RandomState(42)
    dist_a = rng.normal(3.5, 1.0, 10000)
    psi_same = compute_psi(dist_a, dist_a)
    print(f"Test 1 — identical distributions:  PSI = {psi_same:.6f}")
    assert psi_same < 0.01, f"PSI should be ~0 for identical distributions, got {psi_same}"

    # Test 2: PSI of similar distributions should be small
    dist_b = rng.normal(3.5, 1.0, 10000)  # Same params, different samples
    psi_similar = compute_psi(dist_a, dist_b)
    print(f"Test 2 — similar distributions:    PSI = {psi_similar:.6f}")
    assert psi_similar < 0.1, f"PSI should be <0.1 for similar distributions, got {psi_similar}"

    # Test 3: PSI of shifted distributions should be high
    dist_c = rng.normal(5.0, 1.0, 10000)  # Mean shifted from 3.5 to 5.0
    psi_shifted = compute_psi(dist_a, dist_c)
    print(f"Test 3 — shifted distributions:    PSI = {psi_shifted:.6f}")
    assert psi_shifted > 0.2, f"PSI should be >0.2 for shifted distributions, got {psi_shifted}"

    # Test 4: Test with real baseline if available
    baseline_path = "data/features/baseline_distributions.npz"
    if os.path.exists(baseline_path):
        print(f"\nTesting with real baseline: {baseline_path}")

        # No drift (same data)
        passed, results = simulate_no_drift(baseline_path)
        print(format_drift_report(results))
        assert passed, "No-drift simulation should PASS"
        print("✓ No-drift simulation: PASSED (correct)")

        # High drift (shifted data)
        passed, results = simulate_high_drift(baseline_path)
        print(format_drift_report(results))
        assert not passed, "High-drift simulation should FAIL"
        print("✓ High-drift simulation: FAILED (correct — drift detected)")
    else:
        print(f"\n⚠ Baseline not found at {baseline_path} — skipping real data tests")

    print(f"\n✓ Drift monitor validation passed")
